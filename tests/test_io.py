"""Structure precision, displacement parameters, and measured-profile conventions."""

from importlib.resources import files

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import torch

from diffpxrd import (
    build_profile_pattern_batch,
    build_structure_from_cif,
    compose_profile_output,
    load_profile_file,
    lookup_xray_dispersion_terms,
)


def test_installed_resources_and_dispersion_terms():
    assert files("diffpxrd").joinpath("spacegroup_lookup.json").is_file()
    assert files("diffpxrd").joinpath("data/Xsect.dat").is_file()
    assert np.isfinite(lookup_xray_dispersion_terms(14, 1.5406)).all()


def test_structure_ingestion_preserves_precision_occupancies_and_thermal_terms(model):
    structure = Structure(
        Lattice.from_parameters(4.123456789, 5.234567891, 6.345678912, 83.17, 101.23, 92.31),
        [{"Li": 0.712345678901, "Na": 0.287654321099}, "Si"],
        [[0.123456789012, 0.234567891234, 0.345678912345], [0.3, 0.4, 0.5]],
        site_properties={"u_iso": [0.004123456789, 0.005],
                         "u_aniso": [None, [0.006, 0.004, 0.002, 0.001, 0., 0.]]},
    )
    batch = model.structure_to_batch(structure, dtype=torch.float64)
    np.testing.assert_array_equal(batch["lengths"].numpy()[0], structure.lattice.abc)
    np.testing.assert_array_equal(batch["frac_coords"].numpy(), structure.frac_coords)
    np.testing.assert_array_equal(batch["site_species_occs"].numpy()[0], [0.712345678901, 0.287654321099])
    np.testing.assert_array_equal(batch["site_uaniso"].numpy()[1], [0.006, 0.004, 0.002, 0.001, 0., 0.])
    assert batch["site_uiso"].tolist() == [0.004123456789, 0.0]
    assert all(value.dtype == torch.float64 for value in batch.values() if value.is_floating_point())
    assert not torch.equal(batch["lengths"], batch["lengths"].float().double())
    assert model.structure_to_batch(structure)["lengths"].dtype == torch.float32
    with pytest.raises(ValueError, match="dtype must be"):
        model.structure_to_batch(structure, dtype=torch.float16)


def test_cif_loader_preserves_anisotropic_displacement(tmp_path, model):
    path = tmp_path / "silicon.cif"
    path.write_text("""data_silicon
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0 0 0 1 0.005
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
Si1 0.006 0.004 0.002 0 0 0
""")
    batch = model.structure_to_batch(build_structure_from_cif(path), dtype=torch.float64)
    np.testing.assert_array_equal(batch["site_uaniso"].numpy(), [[0.006, 0.004, 0.002, 0., 0., 0.]])
    assert batch["site_uiso"].item() == 0.0


def test_profile_loading_sorts_and_converts_uncertainties(tmp_path):
    path = tmp_path / "profile.xy"
    path.write_text("# two_theta intensity sigma\n30 25 5\n10 9 3\n20 16 4\n")
    profile = load_profile_file(path, sigma_col=2, min_x=15)
    np.testing.assert_array_equal(profile.x, [20, 30])
    np.testing.assert_array_equal(profile.observed, [16, 25])
    np.testing.assert_allclose(profile.weights, [1 / 16, 1 / 25])
    batch = build_profile_pattern_batch(profile, dtype=torch.float64)
    assert batch["observed"].shape == (1, 2)
    assert batch["x"].dtype == torch.float64


def test_signal_and_background_remain_explicit():
    signal = np.array([2., 4., 6.])
    np.testing.assert_array_equal(compose_profile_output(signal), signal)
    np.testing.assert_array_equal(
        compose_profile_output(signal, background=[0.5, 1., 1.5], output_mode="ycalc"),
        [2.5, 5., 7.5],
    )
    with pytest.raises(ValueError, match="background is required"):
        compose_profile_output(signal, output_mode="ycalc")
