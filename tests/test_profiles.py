"""Line shapes, fixed-reflection equivalence, and the FCJ precision boundary."""

import math

import numpy as np
import pytest
import torch

from diffpxrd import ModularCWForwardModel, compute_fcj_profile_histogram


def test_fixed_reflections_preserve_the_forward_pattern(model, nacl):
    batch = model.structure_to_batch(nacl, dtype=torch.float64)
    frozen = model.compile_reflections(batch)
    direct = model.forward_with_layers(batch)
    fixed = model.forward_with_layers_fixed_hkls(batch, frozen)
    torch.testing.assert_close(direct["correction"].pattern, fixed["correction"].pattern)
    other = ModularCWForwardModel(q_min=0.5, q_max=5.5, wavelength=1.5406, device="cpu")
    with pytest.raises(ValueError, match="q-bounds"):
        other.forward_with_layers_fixed_hkls(batch, frozen)


@pytest.mark.parametrize("kind", ["simple", "tch", "fcj"])
def test_supported_profile_shapes_preserve_domain_and_scale(model, kind):
    centers = torch.tensor([[32.0037]], dtype=torch.float64)
    pattern = torch.stack((centers, torch.ones_like(centers)))
    domain = torch.linspace(31.5, 32.5, 301, dtype=torch.float64)
    instrument = {"U": 0., "V": 0., "W": 4., "X": 0., "Y": 0., "Z": 3., "SH/L": 0.002}
    def render(value):
        if kind == "simple":
            return model.compute_profile_layer_from_pattern(
                value, uvw=torch.tensor([[0., 0., 0.004]], dtype=torch.float64), domain=domain)
        return getattr(model, f"compute_{kind}_profile_layer_from_pattern")(
            value, instrument=instrument, domain=domain)

    result = render(pattern)
    scaled = render(torch.stack((centers, 2 * torch.ones_like(centers))))
    assert result.histogram.shape == (1, domain.numel())
    assert result.histogram.dtype == torch.float64
    assert torch.isfinite(result.histogram).all() and torch.all(result.histogram >= 0)
    torch.testing.assert_close(result.pattern_domain, domain, rtol=0, atol=0)
    torch.testing.assert_close(scaled.histogram, 2 * result.histogram)


@pytest.mark.parametrize("variance,gamma", [(16., 0.001), (0.001, 6.), (4., 3.)])
def test_tch_components_share_the_reported_full_width(model, variance, gamma):
    pattern = torch.tensor([[[32.]], [[1.]]], dtype=torch.float64)
    instrument = {"U": 0., "V": 0., "W": variance, "X": 0., "Y": 0., "Z": gamma}
    pilot = model.compute_tch_profile_layer_from_pattern(pattern, instrument=instrument, domain=[32.])
    half = pilot.fwhm.item() / 2
    result = model.compute_tch_profile_layer_from_pattern(
        pattern, instrument=instrument, domain=[32 - half, 32., 32 + half])
    for component in (result.gaussian_peak, result.lorentz_peak):
        ratios = component[0, [0, 2], 0] / component[0, 1, 0]
        torch.testing.assert_close(ratios, torch.full_like(ratios, 0.5), rtol=2e-10, atol=0)


def test_tch_area_matches_analytic_gaussian_lorentzian_integrals(model):
    domain = torch.linspace(31.5, 32.5, 10001, dtype=torch.float64)
    result = model.compute_tch_profile_layer_from_pattern(
        torch.tensor([[[32.]], [[1.]]], dtype=torch.float64), domain=domain,
        instrument={"U": 0., "V": 0., "W": 4., "X": 0., "Y": 0., "Z": 3.},
    )
    fwhm, eta = result.fwhm.squeeze(), result.eta.squeeze()
    sigma = fwhm / math.sqrt(8 * math.log(2))
    expected = (1 - eta) * torch.erf(0.5 / (math.sqrt(2) * sigma))
    expected += eta * (2 / math.pi) * torch.atan(1 / fwhm)
    torch.testing.assert_close(torch.trapezoid(result.histogram[0], domain), expected, rtol=2e-7, atol=0)


def _numpy_fcj(position, intensity, variance, gamma, domain, shl):
    """Independent factored radicand for Finger-Cox-Jephcoat quadrature."""
    theta = np.deg2rad(position)
    minimum = np.rad2deg(np.arccos(np.clip(np.sqrt(1 + shl**2) * np.cos(theta), -1, 1)))
    orders = (*range(2, 32, 2), *range(40, 130, 10), *range(140, 320, 20), 400)
    count = next((n for n in orders if int(300 * abs(position - minimum) / gamma) <= n), 400)
    points, weights = np.polynomial.legendre.leggauss(count)
    delta = minimum + (position - minimum) * points[count // 2:]
    angle = np.deg2rad(delta)
    radicand = np.sin(theta + angle) * np.sin(theta - angle)
    weights = weights[count // 2:] * (shl * abs(np.cos(theta)) / np.sqrt(radicand) - 1) / np.cos(angle)
    gaussian_width = 2.35482005 * np.sqrt(variance)
    width = (gaussian_width**5 + 2.69269 * gaussian_width**4 * gamma
             + 2.42843 * gaussian_width**3 * gamma**2 + 4.47163 * gaussian_width**2 * gamma**3
             + 0.07842 * gaussian_width * gamma**4 + gamma**5)**0.2
    fraction = gamma / width
    eta = 1.36603 * fraction - 0.47719 * fraction**2 + 0.11116 * fraction**3
    offsets = (domain[:, None] - delta) * 100
    lorentzian = 2 * width / (np.pi * (width**2 + 4 * offsets**2))
    gaussian = 2.35482005 / (2.506628275 * width) * np.exp(
        np.maximum(-0.5 * offsets**2 / (width / 2.35482005)**2, -20))
    return intensity * ((eta * lorentzian + (1 - eta) * gaussian) @ weights) / weights.sum()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fcj_matches_factored_numpy_quadrature(dtype):
    parameters = torch.tensor([12.345123, 2., 1.3, 0.015, 0.004], dtype=dtype)
    domain = torch.linspace(12.265123, 12.425123, 257, dtype=dtype)
    position, intensity, variance, gamma, shl = parameters.unbind()
    actual = compute_fcj_profile_histogram(
        position.reshape(1, 1), intensity.reshape(1, 1), variance.reshape(1, 1),
        gamma.reshape(1, 1), domain, shl, backend="bucketed",
    )
    values = parameters.double().numpy()
    expected = _numpy_fcj(*values[:4], domain.double().numpy(), values[4])
    rtol, normalized_atol = (5e-5, 5e-7) if dtype == torch.float32 else (5e-9, 5e-11)
    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=rtol,
                               atol=normalized_atol * np.max(np.abs(expected)))
    assert actual.dtype == dtype


def test_fcj_rejects_invalid_axial_lengths():
    peak = torch.tensor([[32.]], dtype=torch.float64)
    domain = torch.linspace(31.5, 32.5, 33, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compute_fcj_profile_histogram(peak, peak, peak, peak, domain, torch.tensor(-0.002))
