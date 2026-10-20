"""Batched/serial equivalence, mixed structures, and optional CUDA parity."""

from pymatgen.core import Lattice, Structure
import pytest
import torch

from diffpxrd import ModularCWForwardModel


def test_heterogeneous_structure_batches_match_serial_profiles_and_gradients(model):
    structures = [
        Structure(Lattice.cubic(4.2), [{"Li": 0.5, "Na": 0.5}, "O"],
                  [[0., 0., 0.], [0.5, 0.5, 0.5]], site_properties={"u_iso": [0.004, 0.006]}),
        Structure(Lattice.tetragonal(4., 5.1), ["Si"], [[0.25, 0.25, 0.125]],
                  site_properties={"u_aniso": [[0.010, 0.002, 0.003, 0., 0., 0.]]}),
    ]
    hkls = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float64)
    domain = torch.linspace(10., 70., 401, dtype=torch.float64)
    widths = torch.tensor([4., 6.], dtype=torch.float64, requires_grad=True)

    def render(inputs, width):
        batch = model.structures_to_batch(inputs, dtype=torch.float64)
        reflections = model.compute_reflection_layer_from_hkls(batch, hkls)
        scattering = model.compute_scattering_layer(batch, reflections)
        correction = model.compute_correction_layer(reflections, scattering)
        return model.compute_fcj_profile_layer(
            correction, domain=domain,
            instrument={"U": 0., "V": 0., "W": width, "X": 0., "Y": 0., "Z": 2., "SH/L": 0.002},
        ).histogram

    together = render(structures, widths)
    separate = torch.cat([render([structure], widths[index]) for index, structure in enumerate(structures)])
    torch.testing.assert_close(together, separate, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(together.sum(), widths)[0],
                               torch.autograd.grad(separate.sum(), widths)[0], rtol=1e-9, atol=1e-9)


def test_combining_batches_promotes_precision_and_pads_species(model, nacl):
    first = model.structure_to_batch(nacl, dtype=torch.float32)
    mixed = Structure(Lattice.cubic(4.123456789), [{"Li": 0.7, "Na": 0.3}], [[0.1, 0.2, 0.3]])
    second = model.structure_to_batch(mixed, dtype=torch.float64)
    combined = model.combine_batches([first, second])
    assert combined["num_atoms"].tolist() == [len(nacl), 1]
    assert combined["site_species_occs"].shape == (len(nacl) + 1, 2)
    assert all(value.dtype == torch.float64 for value in combined.values() if value.is_floating_point())
    torch.testing.assert_close(combined["lengths"][1], second["lengths"][0], rtol=0, atol=0)
    torch.testing.assert_close(combined["site_species_occs"][-1], second["site_species_occs"][0], rtol=0, atol=0)


def test_mixed_symmetries_preserve_strain_coefficient_order(model):
    positions = torch.tensor([[20., 40., 60.], [25., 45., 65.]], dtype=torch.float64)
    hkls = torch.tensor([[[1, 0, 0], [1, 1, 1], [1, -2, 2]]], dtype=torch.float64).expand(2, -1, -1)
    values = torch.tensor([[140., 80., 55., 12.], [140., 80., 35., 42.]], dtype=torch.float64, requires_grad=True)
    symbols = ["P31m", "I4/mmm"]

    def render(angles, indices, coefficients, symbol):
        return model.get_cw_sample_profile_terms(
            angles, wavelength=model.wavelength, peak_hkls=indices, spacegroup_symbol=symbol,
            sample_profile={"mustrain_type": "generalized", "mustrain_coefficients": coefficients},
        )[1]

    together = render(positions, hkls, values, symbols)
    separate = torch.cat([render(positions[i:i + 1], hkls[i:i + 1], values[i], symbol)
                          for i, symbol in enumerate(symbols)])
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(together.sum(), values)[0],
                               torch.autograd.grad(separate.sum(), values)[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("kind", ["tch", "fcj"])
def test_cuda_profiles_and_derivatives_match_cpu(kind):
    results = []
    for device in ("cpu", "cuda"):
        model = ModularCWForwardModel(q_min=0.5, q_max=6., wavelength=1.5406, device=device)
        values = torch.tensor([32.0037, 4., 3.], dtype=torch.float64, device=device, requires_grad=True)
        center, variance, gamma = values.unbind()
        pattern = torch.stack((center.reshape(1, 1), torch.ones((1, 1), dtype=torch.float64, device=device)))
        domain = torch.linspace(31.7, 32.3, 301, dtype=torch.float64, device=device)
        result = getattr(model, f"compute_{kind}_profile_layer_from_pattern")(
            pattern, domain=domain, return_diagnostics=False,
            instrument={"U": 0., "V": 0., "W": variance, "X": 0., "Y": 0., "Z": gamma, "SH/L": 0.002},
        ).histogram
        weights = torch.linspace(0.3, 1.7, domain.numel(), dtype=torch.float64, device=device)
        gradient, = torch.autograd.grad((result * weights).sum(), values)
        results.append((result.detach().cpu(), gradient.detach().cpu()))
    for cpu, cuda in zip(*results, strict=True):
        torch.testing.assert_close(cpu, cuda, rtol=2e-7, atol=1e-7)
