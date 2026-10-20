"""Representative symmetry settings, absences, and reciprocal-coordinate conventions."""

from itertools import product
from types import SimpleNamespace

import gemmi
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
import pytest
import torch

from diffpxrd.cw import contracts
from diffpxrd.cw.profile import calculate_mustrain_terms, get_mustrain_names
from diffpxrd.spacegroup_contract import resolve_standard_setting_spacegroup, systematic_absence_mask


@pytest.mark.parametrize("number", [2, 14, 62, 141, 162, 194, 225])
def test_representative_groups_match_independent_operations_and_absences(number):
    group = gemmi.find_spacegroup_by_number(number)
    symbol = group.xhm()
    expected = {tuple(map(tuple, np.rint(np.linalg.inv(op.rotation_matrix)).astype(int)))
                for op in SpaceGroup.from_int_number(number, hexagonal=True).symmetry_ops}
    assert set(contracts.get_reciprocal_symmetry_rotations(symbol, False)) == expected
    hkls = np.asarray(list(product(range(-2, 3), repeat=3)))
    allowed = [not group.operations().is_systematically_absent(tuple(map(int, hkl))) for hkl in hkls]
    np.testing.assert_array_equal(systematic_absence_mask(number, hkls), allowed)


def test_aliases_do_not_silently_change_axis_settings():
    assert contracts.get_spacegroup_number("Cmca") == contracts.get_spacegroup_number("Cmce") == 64
    assert set(contracts.get_reciprocal_symmetry_rotations("Cmca")) == set(
        contracts.get_reciprocal_symmetry_rotations("Cmce"))
    for symbol in ("R -3 m:H", "F d -3 m:2"):
        group = resolve_standard_setting_spacegroup(symbol)
        assert group.basisop.rot == gemmi.find_spacegroup_by_number(group.number).basisop.rot
    for symbol in ("P 1 1 21", "R -3 m:R"):
        with pytest.raises(ValueError, match="Unsupported axis setting"):
            resolve_standard_setting_spacegroup(symbol)
    with pytest.raises(ValueError, match="Unknown space-group symbol"):
        resolve_standard_setting_spacegroup("unknown")


def test_family_contraction_sums_intensity_without_double_counting_multiplicity():
    orbit = contracts.get_reciprocal_orbit_cached("F m -3 m", (2, 0, 0))
    result = contracts.build_reflection_contract(
        SimpleNamespace(device="cpu"), torch.tensor(orbit),
        torch.full((len(orbit),), 40., dtype=torch.float64),
        torch.ones(len(orbit), dtype=torch.float64),
        spacegroup_symbol="F m -3 m", include_overlaps=False,
    )
    assert result[0].tolist() == [list(orbit[0])]
    assert result[2].tolist() == result[3].tolist() == result[4].tolist() == [6]


def test_31m_strain_basis_has_published_normalization_and_orbit_invariance():
    seeds = torch.tensor([[1, 2, 3], [-2, 0, 1], [2, 1, -1]], dtype=torch.float64)
    group = gemmi.find_spacegroup_by_number(162)
    rotations = [np.rint(np.linalg.inv(np.asarray(op.rot) / op.DEN)).astype(int)
                 for op in group.operations()]
    orbit = torch.stack([seeds @ seeds.new_tensor(rotation) for rotation in rotations], dim=1)
    h, k, ell = orbit.unbind(-1)
    basal = h * h + h * k + k * k
    expected = torch.stack((basal**2, ell**4, 3 * ell**2 * basal,
                            4 * (h * h * k * ell + h * k * k * ell)), dim=-1)
    actual = calculate_mustrain_terms(None, orbit, laue_group="31m")
    assert get_mustrain_names(laue_group="31m") == ("S400", "S004", "S202", "S211")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, actual[:, :1].expand_as(actual), rtol=0, atol=0)


def test_reciprocal_geometry_uses_nonorthogonal_cell_and_float64(model):
    cell = np.array([[5.123456789, 0., 0.], [0.3, 6.217891234, 0.], [-0.2, 0.4, 7.311234567]])
    hkls = np.array([[1, 2, 3], [2, 1, 0], [1, -1, 2]])
    reciprocal = np.linalg.inv(cell).T
    vectors, axis = hkls @ reciprocal, np.array([0, 0, 1]) @ reciprocal
    expected = (vectors @ axis) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(axis))
    cosine, sine_squared = model.calculate_reciprocal_axis_angles(
        hkls, torch.tensor(cell, dtype=torch.float64), axis_hkl=[0, 0, 1], return_sin_sq=True,
    )
    assert cosine.dtype == torch.float64
    np.testing.assert_allclose(cosine.numpy()[0], expected, rtol=2e-13, atol=5e-15)
    np.testing.assert_allclose(sine_squared.numpy()[0], 1 - expected**2, rtol=2e-13, atol=5e-15)
