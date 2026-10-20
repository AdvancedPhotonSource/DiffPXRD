"""Finite-difference checks and physically important zero/boundary cases."""

from types import SimpleNamespace

from pymatgen.core import Lattice, Structure
import pytest
import torch

from diffpxrd import compute_fcj_profile_histogram
from diffpxrd.cw.contracts import (
    apply_compiled_reflection_contract_single,
    build_reflection_contract,
    compile_reflection_contract,
)


def test_cell_coordinates_occupancies_and_displacement_pass_gradcheck(model):
    structure = Structure(
        Lattice.from_parameters(4.1, 5.2, 6.3, 83.17, 101.23, 92.31),
        ["Si", "O"], [[0.13, 0.27, 0.39], [0.43, 0.51, 0.67]],
        site_properties={"u_iso": [0.004, 0.006]},
    )
    batch = model.structure_to_batch(structure, dtype=torch.float64)
    fields = ("lengths", "angles", "frac_coords", "site_species_occs", "site_uiso")
    inputs = tuple(batch[key].clone().requires_grad_() for key in fields)
    hkls = torch.tensor([[1, 0, 0], [1, 2, 1], [2, 1, -1]], dtype=torch.float64)

    def evaluate(*parameters):
        current = dict(batch, **dict(zip(fields, parameters, strict=True)))
        reflections = model.compute_reflection_layer_from_hkls(current, hkls)
        scattering = model.compute_scattering_layer(current, reflections)
        correction = model.compute_correction_layer(reflections, scattering)
        return correction.pattern.square().mean()

    assert torch.autograd.gradcheck(evaluate, inputs, eps=1e-6, atol=1e-3, rtol=2e-5)


def test_tch_position_and_width_derivatives_pass_gradcheck(model):
    domain = torch.linspace(31.9, 32.1, 101, dtype=torch.float64)

    def render(parameters):
        position, variance, gamma = parameters.unbind()
        pattern = torch.stack((position.reshape(1, 1), torch.ones((1, 1), dtype=torch.float64)))
        return model.compute_tch_profile_layer_from_pattern(
            pattern, domain=domain, return_diagnostics=False,
            instrument={"U": 0., "V": 0., "W": variance, "X": 0., "Y": 0., "Z": gamma},
        ).histogram

    parameters = torch.tensor([32.0037, 4., 3.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(render, (parameters,), eps=1e-5, atol=2e-5, rtol=2e-4)


def test_fcj_axial_derivatives_pass_gradcheck():
    domain = torch.linspace(12.28, 12.40, 65, dtype=torch.float64)
    weights = torch.sin(torch.arange(65, dtype=torch.float64) * 0.017) + 0.25

    def evaluate(parameters):
        position, intensity, variance, gamma, shl = parameters.unbind()
        result = compute_fcj_profile_histogram(
            position.reshape(1, 1), intensity.reshape(1, 1), variance.reshape(1, 1),
            gamma.reshape(1, 1), domain, shl, backend="bucketed",
        )
        return (result * weights).sum()

    parameters = torch.tensor([12.345123, 2., 1.3, 0.015, 0.004], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-6, atol=2e-5, rtol=5e-5)


def test_zero_intensity_families_keep_their_derivative_but_absences_do_not():
    model = SimpleNamespace(device="cpu")
    hkls = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    positions = torch.tensor([10., 10., 20., 0.], dtype=torch.float64)
    intensities = torch.tensor([2., 3., 0., 0.], dtype=torch.float64, requires_grad=True)
    reference = build_reflection_contract(
        model, hkls, positions, intensities, spacegroup_symbol="P 1", include_overlaps=False,
    )
    compiled = compile_reflection_contract(model, hkls, spacegroup_symbol="P 1", static_full_families=True)
    actual = apply_compiled_reflection_contract_single(model, compiled, positions, intensities)
    for value, expected in zip(actual, reference[:5], strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert actual[1].tolist() == [10., 20.]
    assert actual[2].tolist() == [5., 0.]
    assert actual[4].tolist() == [2, 1]
    actual[2].sum().backward()
    assert intensities.grad.tolist() == [1., 1., 1., 0.]
    absent = build_reflection_contract(
        model, torch.tensor([[1, 0, 0], [2, 0, 0]]), torch.tensor([10., 20.]),
        torch.zeros(2), spacegroup_symbol="I m -3 m", include_overlaps=False,
    )
    assert absent[0].tolist() == [[-2, 0, 0]]


def test_non_bragg_reflections_do_not_acquire_intensity(model, nacl):
    batch = model.structure_to_batch(nacl, dtype=torch.float64)
    hkls = torch.tensor([[2, 0, 0], [30, 0, 0]], dtype=torch.float64)
    reflections = model.compute_reflection_layer_from_hkls(batch, hkls)
    scattering = model.compute_scattering_layer(batch, reflections)
    corrected = model.compute_correction_layer(reflections, scattering)
    assert corrected.two_thetas[0, 0] > 0
    assert corrected.two_thetas[0, 1] == corrected.corrected_intensities[0, 1] == 0


def test_absorption_at_zero_has_finite_derivatives(model):
    positions = torch.linspace(5., 60., 32, dtype=torch.float64).unsqueeze(0).requires_grad_()
    value = model.calculate_cw_absorption_factor(
        positions, sample_parameters={"geometry": "Debye-Scherrer", "mu_r": 0.},
    )
    derivative, = torch.autograd.grad(value.sum(), positions)
    assert torch.isfinite(value).all() and torch.isfinite(derivative).all()


def test_uniaxial_width_and_texture_cell_derivatives(model):
    parameters = torch.tensor([5.123456789, 6.217891234, 7.311234567, 83.17, 101.23, 92.31],
                              dtype=torch.float64, requires_grad=True)

    def evaluate(cell):
        reflections = model.compute_reflection_layer_from_hkls(
            {"lengths": cell[:3].unsqueeze(0), "angles": cell[3:].unsqueeze(0)},
            [[0, 0, 2], [2, 0, 0], [1, 2, 3]],
        )
        positions = (360 / torch.pi) * torch.asin(model.wavelength * reflections.recip_lengths / 2)
        widths = model.get_cw_sample_profile_terms(
            positions, wavelength=model.wavelength, peak_hkls=reflections.hkl_pts,
            cell_matrix=reflections.cell_matrix,
            sample_profile={"size_type": "uniaxial", "size_i": 0.7, "size_a": 2.3,
                            "size_mix": 0.25, "size_axis_hkl": [0, 0, 1]},
        )
        texture = model.calculate_march_dollase_factor(
            reflections.hkl_pts, reflections.cell_matrix,
            sample_parameters={"preferred_orientation_model": "march_dollase",
                               "preferred_orientation_axis_hkl": [0, 0, 1],
                               "preferred_orientation_ratio": 0.73},
        )
        return torch.cat((*widths, texture), dim=1)

    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-6, atol=1e-8, rtol=5e-6)


def test_31m_signed_coefficient_has_nonzero_derivative_at_zero(model):
    positions = torch.tensor([[20., 40., 60.]], dtype=torch.float64)
    hkls = torch.tensor([[[1, 0, 0], [1, 1, 1], [1, -2, 2]]], dtype=torch.float64)
    parameters = torch.tensor([140., 80., 55., 0.], dtype=torch.float64, requires_grad=True)

    def evaluate(values):
        return model.get_cw_sample_profile_terms(
            positions, wavelength=model.wavelength, peak_hkls=hkls,
            sample_profile={"mustrain_type": "generalized", "mustrain_laue_group": "31m",
                            "mustrain_coefficients": values},
        )[1].sum()

    derivative, = torch.autograd.grad(evaluate(parameters), parameters)
    assert derivative[3] != 0 and torch.isfinite(derivative).all()
    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-3, atol=1e-10, rtol=1e-6)
    with pytest.raises(ValueError, match="31m"):
        evaluate({"S400": 140., "S004": 80., "S202": 55., "S301": 1.})
