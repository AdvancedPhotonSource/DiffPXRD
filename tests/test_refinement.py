"""Profile objectives and small end-to-end single/shared parameter fits."""

import numpy as np
import torch

from diffpxrd import fit_weighted_profile_numpy, solved_profile_wls_loss_torch
from diffpxrd.trajectory import CubicSplineTrajectory


def test_profile_objective_recovers_scale_and_background():
    x = np.linspace(10., 80., 128)
    calculated = np.exp(-0.5 * ((x - 35) / 2)**2)
    observed = 2.5 * calculated + 0.17
    numpy_fit = fit_weighted_profile_numpy(
        observed, calculated, weights=np.ones_like(observed), fit_scale=True, fit_background=True,
    )
    np.testing.assert_allclose([numpy_fit.scale, numpy_fit.background], [2.5, 0.17], atol=1e-10)
    loss, diagnostics = solved_profile_wls_loss_torch(
        torch.tensor(observed).unsqueeze(0), torch.tensor(calculated).unsqueeze(0),
        weights=torch.ones((1, 128), dtype=torch.float64), fit_scale=True, fit_background=True,
    )
    assert loss.shape == (1,)
    assert loss.item() < 1e-20
    torch.testing.assert_close(diagnostics["fitted_profile"], torch.tensor(observed).unsqueeze(0))


def _fit(parameters, render, observed):
    optimizer = torch.optim.LBFGS(
        [parameters], max_iter=60, tolerance_grad=1e-11,
        tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        per_profile, _ = solved_profile_wls_loss_torch(
            observed, render(), weights=torch.ones_like(observed), fit_scale=True, fit_background=True,
        )
        loss = per_profile.mean()
        loss.backward()
        return loss

    initial = closure().detach()
    optimizer.step(closure)
    final = closure().detach()
    assert torch.isfinite(final) and final < initial


def test_single_lattice_parameter_is_recovered_from_a_profile(lattice_profiles):
    render = lattice_profiles(1)
    target = torch.tensor([0.001], dtype=torch.float64)
    observed = render(target).detach()
    observed /= observed.amax(dim=1, keepdim=True)
    expansion = torch.nn.Parameter(torch.zeros_like(target))
    _fit(expansion, lambda: render(expansion), observed)
    torch.testing.assert_close(expansion, target, rtol=0, atol=1e-6)


def test_shared_evolution_coefficients_are_recovered_from_multiple_profiles(lattice_profiles):
    render = lattice_profiles(3)
    state = torch.linspace(0., 1., 3, dtype=torch.float64)
    design = torch.stack((torch.ones_like(state), state), dim=1)
    target = torch.tensor([0.0005, 0.001], dtype=torch.float64)
    observed = render(design @ target).detach()
    observed /= observed.amax(dim=1, keepdim=True)
    coefficients = torch.nn.Parameter(torch.zeros_like(target))
    _fit(coefficients, lambda: render(design @ coefficients), observed)
    torch.testing.assert_close(design @ coefficients, design @ target, rtol=0, atol=1e-6)


def test_spline_partition_of_unity_and_coefficient_gradients():
    spline = CubicSplineTrajectory.from_state_values(
        np.linspace(0., 1., 7), full_knots=[0., 0., 0., 0., 0.5, 1., 1., 1., 1.],
        input_min=0., input_max=1.,
    )
    np.testing.assert_allclose(spline.design_matrix.sum(axis=1), 1., atol=1e-12)
    coefficients = torch.ones((1, 5), dtype=torch.float64, requires_grad=True)
    values = spline.amplitudes(coefficients)
    torch.testing.assert_close(values, torch.ones((7, 1), dtype=torch.float64))
    assert torch.autograd.gradcheck(spline.amplitudes, (coefficients,), eps=1e-6, atol=1e-8)
