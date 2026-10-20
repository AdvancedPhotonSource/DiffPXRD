"""Check all 230 standard space groups; not part of the quick test suite.

Run from the repository root:
    python reproduce/validate_symmetry.py --output results/symmetry.json
"""

import argparse
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import gemmi
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
import torch

from diffpxrd.cw import contracts, profile
from diffpxrd.spacegroup_contract import get_spacegroup_metadata, systematic_absence_mask


def check_group(number):
    gemmi_group = gemmi.find_spacegroup_by_number(number)
    pymatgen_group = SpaceGroup.from_int_number(number, hexagonal=True)
    rotations = {tuple(map(tuple, np.rint(np.linalg.inv(op.rotation_matrix)).astype(int)))
                 for op in pymatgen_group.symmetry_ops}
    hkls = np.asarray(list(product(range(-3, 4), repeat=3)))
    allowed = [not gemmi_group.operations().is_systematically_absent(tuple(map(int, hkl))) for hkl in hkls]
    np.testing.assert_array_equal(systematic_absence_mask(number, hkls), allowed)
    for symbol in (gemmi_group.xhm(), pymatgen_group.symbol):
        assert contracts.get_spacegroup_number(symbol) == number
        assert profile.get_spacegroup_metadata_for_symbol(symbol) == get_spacegroup_metadata(number)
        for friedel in (False, True):
            expected = set(rotations)
            if friedel:
                expected |= {tuple(tuple(-value for value in row) for row in matrix) for matrix in rotations}
            actual = contracts.get_reciprocal_symmetry_rotations(symbol, friedel)
            assert len(actual) == len(expected) and set(actual) == expected
            # Multiples of 12 avoid extinctions while testing orbit contraction.
            orbit = sorted({tuple(np.asarray([12, 24, 36]) @ matrix) for matrix in expected})
            assert contracts.get_reciprocal_orbit_cached(symbol, (12, 24, 36), friedel) == tuple(orbit)
            intensities = torch.arange(1, len(orbit) + 1, dtype=torch.float64)
            result = contracts.build_reflection_contract(
                SimpleNamespace(device="cpu"), torch.tensor(orbit, dtype=torch.float64),
                torch.full((len(orbit),), 40., dtype=torch.float64), intensities,
                spacegroup_symbol=symbol, include_friedel_pairs=friedel, include_overlaps=False,
            )
            assert result[0].tolist() == [list(orbit[0])]
            assert result[1].tolist() == [40.]
            assert result[2].tolist() == [float(intensities.sum())]
            assert result[3].tolist() == result[4].tolist() == [len(orbit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": False, "space_groups_checked": 0,
              "versions": {name: version(name) for name in ("diffpxrd", "gemmi", "pymatgen", "torch")}}
    try:
        for number in range(1, 231):
            check_group(number)
            report["space_groups_checked"] += 1
        report.update(passed=True, symbol_variants_checked=460, reciprocal_orbits_checked=920)
    except Exception as error:
        report.update(failed_space_group=number, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Validated all 230 standard space groups; report: {args.output}")


if __name__ == "__main__":
    main()
