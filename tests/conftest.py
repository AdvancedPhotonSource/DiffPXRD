"""Synthetic inputs shared by the public correctness tests."""

from pymatgen.core import Lattice, Structure
import pytest
import torch

from diffpxrd import ModularCWForwardModel


@pytest.fixture
def model():
    return ModularCWForwardModel(q_min=0.5, q_max=6.0, wavelength=1.5406, device="cpu")


@pytest.fixture
def nacl():
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def lattice_profiles(model, nacl):
    def prepare(frames):
        batch = model.structures_to_batch([nacl] * frames, dtype=torch.float64)
        reflections = model.compile_reflections(batch)
        grid = torch.linspace(15.0, 80.0, 1301, dtype=torch.float64)

        def render(expansion):
            current = dict(batch, lengths=batch["lengths"] * (1 + expansion.reshape(-1, 1)))
            layers = model.forward_with_layers_fixed_hkls(current, reflections)
            return model.compute_tch_profile_layer(
                layers["correction"], domain=grid, return_diagnostics=False,
                instrument={"U": 0., "V": 0., "W": 16., "X": 0., "Y": 0., "Z": 2.},
            ).histogram

        return render

    return prepare
