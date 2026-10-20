from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import gemmi
import numpy as np


@dataclass(frozen=True)
class SpacegroupMetadata:
    number: int
    xhm: str
    hm: str
    laue: str
    lattice: str
    unique_axis: str
    crystal_system: str
    point_group_hm: str


@dataclass(frozen=True)
class HKLContract:
    hkls: np.ndarray
    positions_deg: np.ndarray
    reciprocal_lengths: np.ndarray
    representative_mask: np.ndarray
    systematic_absence_mask: np.ndarray


LOOKUP_PATH = Path(__file__).with_name("spacegroup_lookup.json")


@lru_cache(maxsize=1)
def load_spacegroup_lookup() -> dict[int, SpacegroupMetadata]:
    records = json.loads(LOOKUP_PATH.read_text())
    return {
        int(record["number"]): SpacegroupMetadata(
            number=int(record["number"]),
            xhm=str(record["xhm"]),
            hm=str(record["hm"]),
            laue=str(record["laue"]),
            lattice=str(record["lattice"]),
            unique_axis=str(record["unique_axis"]),
            crystal_system=str(record["crystal_system"]),
            point_group_hm=str(record["point_group_hm"]),
        )
        for record in records
    }


def get_spacegroup_metadata(spacegroup_number: int) -> SpacegroupMetadata:
    lookup = load_spacegroup_lookup()
    if spacegroup_number not in lookup:
        raise KeyError(f"Unsupported space-group number: {spacegroup_number}")
    return lookup[spacegroup_number]


@lru_cache(maxsize=None)
def resolve_standard_setting_spacegroup(spacegroup_symbol: str) -> gemmi.SpaceGroup:
    """Resolve aliases without mixing supplied axes with number-canonical metadata."""
    # Pymatgen writes screw-axis subscripts as underscores (for example P2_1/c).
    group = gemmi.find_spacegroup_by_name(spacegroup_symbol.replace("_", ""))
    if group is None:
        raise ValueError(f"Unknown space-group symbol: {spacegroup_symbol!r}")
    canonical = gemmi.find_spacegroup_by_number(group.number)
    # GEMMI tabulates exact basis changes from one shared reference setting.
    # Equal linear parts allow origin changes, which preserve reciprocal orbits
    # and systematic absences, but not alternate axes or lattice settings.
    if group.basisop.rot != canonical.basisop.rot:
        raise ValueError(
            f"Unsupported axis setting {group.xhm()!r}; expected the axes of "
            f"{canonical.xhm()!r} for canonical absence and profile metadata"
        )
    return group


def swap_indices(axis: int, hkls: np.ndarray) -> np.ndarray:
    if axis in (1, -1):
        return hkls
    if axis in (2, -3):
        return hkls[:, [1, 2, 0]]
    return hkls[:, [2, 0, 1]]


def hx_to_rh(hkls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hkls = np.asarray(hkls, dtype=int)
    itk = -hkls[:, 0] + hkls[:, 1] + hkls[:, 2]
    valid = (itk % 3) == 0
    rh = np.zeros_like(hkls)
    rh[valid, 1] = itk[valid] // 3
    rh[valid, 0] = rh[valid, 1] + hkls[valid, 0]
    rh[valid, 2] = rh[valid, 1] - hkls[valid, 1]
    negative_first = rh[:, 0] < 0
    rh[negative_first] *= -1
    return rh, valid


def centering_allowed_mask(lattice: str, hkls: np.ndarray) -> np.ndarray:
    h, k, l = hkls.T  # noqa: E741 - conventional Miller-index notation
    if lattice == "A":
        return ((k + l) % 2) == 0
    if lattice == "B":
        return ((h + l) % 2) == 0
    if lattice == "C":
        return ((h + k) % 2) == 0
    if lattice == "I":
        return ((h + k + l) % 2) == 0
    if lattice == "F":
        return (((h + k) % 2) == 0) & (((h + l) % 2) == 0) & (((k + l) % 2) == 0)
    if lattice == "R":
        return ((-h + k + l) % 3) == 0
    return np.ones(hkls.shape[0], dtype=bool)


def get_unique_axis_code(unique_axis: str) -> int:
    if unique_axis == "":
        return 1
    return 1 + ["a", "b", "c"].index(unique_axis)


def laue_representative_mask(laue: str, unique_axis: str, hkls: np.ndarray) -> np.ndarray:
    h, k, l = hkls.T  # noqa: E741 - conventional Miller-index notation

    if laue == "-1":
        return (h > 0) | ((h == 0) & ((k > 0) | ((k == 0) & (l >= 0))))

    if laue == "2/m":
        axisnum = get_unique_axis_code(unique_axis)
        transformed = swap_indices(-axisnum, hkls)
        ht, kt, lt = transformed.T
        return (ht >= 0) & (((kt < 0) & (lt > 0)) | ((kt >= 0) & (lt >= 0)))

    if laue == "mmm":
        return (h >= 0) & (k >= 0) & (l >= 0)

    if laue in ("4/m", "6/m"):
        return (h >= 0) & (k >= 0) & (l >= 0) & ((h == 0) | (k > 0))

    if laue in ("4/mmm", "6/mmm"):
        return (h >= 0) & (k >= 0) & (l >= 0) & (k <= h)

    if laue in ("3", "3R"):
        hmin = np.where(l < 0, 1, 0)
        kmin = -np.trunc((h - 1.0) / 2.0).astype(int)
        return (h >= hmin) & (k >= kmin) & (k <= h)

    if laue in ("3m1", "3mR", "31m"):
        hmin = np.where(l < 0, 1, 0)
        kmin = np.where((laue == "31m") & (l < 0), 1, 0)
        kmax = np.where(np.isin(laue, ("3m1", "3mR")) & (l < 0), h - 1, h)
        return (h >= hmin) & (k >= kmin) & (k <= kmax)

    if laue == "m3m":
        return (h >= 0) & (k >= 0) & (l >= 0) & (k <= h) & (l <= k)

    if laue == "m3":
        return (h >= 0) & (k >= 0) & (l >= 0) & (k <= h) & (
            ((h == k) & (l <= h)) | ((h > k) & (l < h))
        )

    raise ValueError(f"Unsupported Laue group: {laue}")


def systematic_absence_mask(spacegroup_number: int, hkls: np.ndarray) -> np.ndarray:
    operations = gemmi.find_spacegroup_by_number(spacegroup_number).operations()
    return np.asarray(
        [not operations.is_systematically_absent(tuple(int(value) for value in hkl)) for hkl in hkls],
        dtype=bool,
    )


def build_spacegroup_hkl_contract(
    hkls: np.ndarray,
    reciprocal_lengths: np.ndarray,
    positions_deg: np.ndarray,
    *,
    spacegroup_number: int,
) -> HKLContract:
    metadata = get_spacegroup_metadata(spacegroup_number)

    hkls = np.asarray(np.rint(hkls), dtype=int)
    reciprocal_lengths = np.asarray(reciprocal_lengths, dtype=float)
    positions_deg = np.asarray(positions_deg, dtype=float)

    base_mask = np.any(hkls != 0, axis=1) & np.isfinite(reciprocal_lengths) & np.isfinite(positions_deg)
    hkls = hkls[base_mask]
    reciprocal_lengths = reciprocal_lengths[base_mask]
    positions_deg = positions_deg[base_mask]

    centering_mask = centering_allowed_mask(metadata.lattice, hkls)
    hkls = hkls[centering_mask]
    reciprocal_lengths = reciprocal_lengths[centering_mask]
    positions_deg = positions_deg[centering_mask]

    representative_mask = laue_representative_mask(metadata.laue, metadata.unique_axis, hkls)
    hkls = hkls[representative_mask]
    reciprocal_lengths = reciprocal_lengths[representative_mask]
    positions_deg = positions_deg[representative_mask]

    if metadata.laue in ("3R", "3mR"):
        hkls, valid_rh = hx_to_rh(hkls)
        reciprocal_lengths = reciprocal_lengths[valid_rh]
        positions_deg = positions_deg[valid_rh]
        representative_mask = representative_mask[representative_mask][valid_rh]

    non_absent_mask = systematic_absence_mask(spacegroup_number, hkls)
    hkls = hkls[non_absent_mask]
    reciprocal_lengths = reciprocal_lengths[non_absent_mask]
    positions_deg = positions_deg[non_absent_mask]

    order = np.argsort(positions_deg, kind="stable")
    hkls = hkls[order]
    reciprocal_lengths = reciprocal_lengths[order]
    positions_deg = positions_deg[order]

    return HKLContract(
        hkls=hkls,
        positions_deg=positions_deg,
        reciprocal_lengths=reciprocal_lengths,
        representative_mask=representative_mask,
        systematic_absence_mask=non_absent_mask,
    )
