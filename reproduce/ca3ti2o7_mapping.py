"""Research structure expansion and mapping, with no structural data.

Function bodies copied unchanged from experiments/atomic_trajectories/:
casrtio7_x0_endpoint_adequacy, casrtio7_x0_endmember_mapping,
casrtio7_x0_structural_response, casrtio7_x0_neutron_structure_series,
and casrtio7_x0_neutron_window_attribution. No research imports or configs.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.optimize import linear_sum_assignment


def _wyckoff_multiplicity(symbol: str) -> int:
    match = re.fullmatch(r"(\d+)[A-Za-z]", symbol)
    if match is None:
        raise ValueError(f"invalid Wyckoff symbol {symbol!r}")
    return int(match.group(1))


def build_reported_template(
    phase: dict[str, Any],
) -> tuple[Structure, dict[str, Any]]:
    sites = phase["sites"]
    lattice = Lattice.orthorhombic(*map(float, phase["lattice_angstrom"]))
    labels = [str(site["label"]) for site in sites]
    elements = [str(site["element"]) for site in sites]
    coordinates = [site["fractional_coordinates"] for site in sites]
    u_iso = [float(site["u_iso_angstrom2"]) for site in sites]
    occupancies = [float(site["occupancy"]) for site in sites]
    if any(value != 1.0 for value in occupancies):
        raise ValueError("endpoint templates currently require full occupancy")
    structure = Structure.from_spacegroup(
        str(phase["space_group_symbol"]),
        lattice,
        elements,
        coordinates,
        site_properties={
            "u_iso": u_iso,
            "source_label": labels,
            "reported_wyckoff": [str(site["wyckoff"]) for site in sites],
        },
    )
    expected_counts = {
        str(site["label"]): _wyckoff_multiplicity(str(site["wyckoff"]))
        for site in sites
    }
    observed_counts = Counter(structure.site_properties["source_label"])
    if observed_counts != expected_counts:
        raise ValueError(
            "reported Wyckoff multiplicities do not match generated structure: "
            f"expected {expected_counts}, observed {dict(observed_counts)}"
        )
    if len(structure) != 48 or structure.composition.reduced_formula != "Ca3Ti2O7":
        raise ValueError("reported endpoint template is not a 48-site Ca3Ti2O7 cell")
    detected_symbol, detected_number = structure.get_space_group_info(symprec=1.0e-2)
    if detected_number != int(phase["space_group_number"]):
        raise ValueError(
            f"reported template resolves to space group {detected_number}, "
            f"expected {phase['space_group_number']}"
        )
    return structure, {
        "reported_temperature_k": float(phase["temperature_k"]),
        "reported_space_group_symbol": str(phase["space_group_symbol"]),
        "reported_space_group_number": int(phase["space_group_number"]),
        "detected_space_group_symbol": detected_symbol,
        "detected_space_group_number": detected_number,
        "site_count": len(structure),
        "reduced_formula": structure.composition.reduced_formula,
        "reported_lattice_angstrom": list(map(float, phase["lattice_angstrom"])),
        "u_iso_range_angstrom2": [
            float(np.min(structure.site_properties["u_iso"])),
            float(np.max(structure.site_properties["u_iso"])),
        ],
        "generated_label_multiplicities": dict(sorted(observed_counts.items())),
    }



def signed_axis_transforms(
    *, allow_ab_swap: bool, allow_sign_reversal: bool
) -> list[dict[str, Any]]:
    permutations = [(0, 1, 2)]
    if allow_ab_swap:
        permutations.append((1, 0, 2))
    signs = list(itertools.product((-1, 1), repeat=3)) if allow_sign_reversal else [(1, 1, 1)]
    return [
        {"permutation": permutation, "signs": sign}
        for permutation in permutations
        for sign in signs
    ]


def transform_fractional(
    fractional: np.ndarray, permutation: tuple[int, int, int], signs: tuple[int, int, int]
) -> np.ndarray:
    coordinates = np.asarray(fractional, dtype=np.float64)
    return np.mod(coordinates[:, permutation] * np.asarray(signs)[None, :], 1.0)


def _periodic_cartesian_delta(
    first: np.ndarray, second: np.ndarray, lattice: np.ndarray
) -> np.ndarray:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta -= np.round(delta)
    return delta @ np.asarray(lattice, dtype=np.float64)


def assign_by_element(
    target_fractional: np.ndarray,
    source_fractional: np.ndarray,
    target_elements: np.ndarray,
    source_elements: np.ndarray,
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_fractional, dtype=np.float64)
    source = np.asarray(source_fractional, dtype=np.float64)
    target_species = np.asarray(target_elements)
    source_species = np.asarray(source_elements)
    if target.shape != source.shape or target.shape[1] != 3:
        raise ValueError("mapping coordinates must share shape (atom, 3)")
    assignment = np.full(len(target), -1, dtype=np.int64)
    for element in sorted(set(target_species.tolist())):
        target_index = np.flatnonzero(target_species == element)
        source_index = np.flatnonzero(source_species == element)
        if len(target_index) != len(source_index):
            raise ValueError(f"element count mismatch for {element}")
        delta = target[target_index, None, :] - source[None, source_index, :]
        delta -= np.round(delta)
        cartesian = np.einsum("ijk,kl->ijl", delta, lattice)
        cost = np.sum(np.square(cartesian), axis=2)
        rows, columns = linear_sum_assignment(cost)
        assignment[target_index[rows]] = source_index[columns]
    if np.any(assignment < 0) or len(set(assignment.tolist())) != len(assignment):
        raise RuntimeError("atom assignment is incomplete or non-bijective")
    displacement = _periodic_cartesian_delta(
        target, source[assignment], lattice
    )
    return assignment, displacement


def refine_origin_and_assignment(
    target_fractional: np.ndarray,
    source_fractional: np.ndarray,
    target_elements: np.ndarray,
    source_elements: np.ndarray,
    lattice: np.ndarray,
    initial_shift: np.ndarray,
    *,
    iterations: int,
) -> dict[str, Any]:
    shift = np.asarray(initial_shift, dtype=np.float64).copy()
    source = np.asarray(source_fractional, dtype=np.float64)
    for _ in range(iterations):
        shifted = np.mod(source + shift[None, :], 1.0)
        assignment, displacement = assign_by_element(
            target_fractional,
            shifted,
            target_elements,
            source_elements,
            lattice,
        )
        correction_cartesian = np.mean(displacement, axis=0)
        correction_fractional = correction_cartesian @ np.linalg.inv(lattice)
        shift += correction_fractional
    shifted = np.mod(source + shift[None, :], 1.0)
    assignment, displacement = assign_by_element(
        target_fractional,
        shifted,
        target_elements,
        source_elements,
        lattice,
    )
    norms = np.linalg.norm(displacement, axis=1)
    return {
        "shift": np.mod(shift, 1.0),
        "assignment": assignment,
        "displacement": displacement,
        "rms_displacement_angstrom": float(np.sqrt(np.mean(np.square(norms)))),
        "maximum_displacement_angstrom": float(np.max(norms)),
        "mean_displacement_angstrom": np.mean(displacement, axis=0),
    }


def map_endmember_arrays(
    target_fractional: np.ndarray,
    source_fractional: np.ndarray,
    target_elements: np.ndarray,
    source_elements: np.ndarray,
    target_lengths: np.ndarray,
    source_lengths: np.ndarray,
    *,
    allow_ab_swap: bool,
    allow_sign_reversal: bool,
    origin_seed_element: str,
    origin_refinement_iterations: int,
) -> dict[str, Any]:
    target = np.asarray(target_fractional, dtype=np.float64)
    source = np.asarray(source_fractional, dtype=np.float64)
    target_species = np.asarray(target_elements)
    source_species = np.asarray(source_elements)
    transform_results = []
    for transform in signed_axis_transforms(
        allow_ab_swap=allow_ab_swap,
        allow_sign_reversal=allow_sign_reversal,
    ):
        permutation = tuple(transform["permutation"])
        signs = tuple(transform["signs"])
        transformed = transform_fractional(source, permutation, signs)
        transformed_lengths = np.asarray(source_lengths, dtype=np.float64)[
            list(permutation)
        ]
        common_lengths = 0.5 * (
            np.asarray(target_lengths, dtype=np.float64) + transformed_lengths
        )
        lattice = np.diag(common_lengths)
        target_seed = target[target_species == origin_seed_element]
        source_seed = transformed[source_species == origin_seed_element]
        if min(len(target_seed), len(source_seed)) == 0:
            raise ValueError(f"origin seed element {origin_seed_element} is absent")
        shifts = {
            tuple(np.round(np.mod(first - second, 1.0), 10))
            for first in target_seed
            for second in source_seed
        }
        best = None
        for shift in sorted(shifts):
            result = refine_origin_and_assignment(
                target,
                transformed,
                target_species,
                source_species,
                lattice,
                np.asarray(shift),
                iterations=origin_refinement_iterations,
            )
            if best is None or result["rms_displacement_angstrom"] < best[
                "rms_displacement_angstrom"
            ]:
                best = result
        if best is None:
            raise RuntimeError("origin search produced no mapping")
        transform_results.append(
            {
                **best,
                "permutation": permutation,
                "signs": signs,
                "common_lengths_angstrom": common_lengths,
                "transformed_source_fractional": transformed,
            }
        )
    ranked = sorted(
        transform_results, key=lambda item: item["rms_displacement_angstrom"]
    )
    return {"best": ranked[0], "ranked_transforms": ranked}


def build_mapped_coordinate(
    endpoint: dict[str, Any], mapping_config: dict[str, Any]
) -> dict[str, Any]:
    structures = {
        phase: build_reported_template(endpoint["phase_templates"][phase])[0]
        for phase in ("a21am", "acaa")
    }
    target = structures["a21am"]
    source = structures["acaa"]
    target_elements = np.asarray([site.specie.symbol for site in target])
    source_elements = np.asarray([site.specie.symbol for site in source])
    rules = mapping_config["mapping"]
    mapped = map_endmember_arrays(
        target.frac_coords,
        source.frac_coords,
        target_elements,
        source_elements,
        np.asarray(target.lattice.abc),
        np.asarray(source.lattice.abc),
        allow_ab_swap=bool(rules["allow_ab_axis_swap"]),
        allow_sign_reversal=bool(rules["allow_axis_sign_reversal"]),
        origin_seed_element=str(rules["origin_seed_element"]),
        origin_refinement_iterations=int(rules["origin_refinement_iterations"]),
    )
    best = mapped["best"]
    assignment = best["assignment"]
    source_fractional = np.mod(
        best["transformed_source_fractional"] + best["shift"][None, :], 1.0
    )[assignment]
    lattice = np.diag(best["common_lengths_angstrom"])
    delta_fractional = best["displacement"] @ np.linalg.inv(lattice)
    return {
        "target": target,
        "source": source,
        "source_fractional": source_fractional,
        "delta_fractional": delta_fractional,
        "common_lengths_angstrom": best["common_lengths_angstrom"],
        "endpoint_rms_displacement_angstrom": best["rms_displacement_angstrom"],
        "permutation": best["permutation"],
        "signs": best["signs"],
        "origin_shift_fractional": best["shift"],
    }



def _structure_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature_k": float(record["temperature_k"]),
        "space_group_symbol": "A21am",
        "space_group_number": 36,
        "lattice_angstrom": record["lattice_angstrom"],
        "sites": record["sites"],
    }



def _build_structures(
    config: dict[str, Any], endpoint: dict[str, Any]
) -> tuple[tuple[Structure, ...], list[dict[str, Any]]]:
    records = list(config["a21am_structures"])
    records.append(
        {
            "cycle": "heating",
            "temperature_k": 1073.0,
            "lattice_angstrom": endpoint["phase_templates"]["a21am"][
                "lattice_angstrom"
            ],
            "sites": endpoint["phase_templates"]["a21am"]["sites"],
        }
    )
    structures = tuple(
        build_reported_template(_structure_record(item))[0] for item in records
    ) + (build_reported_template(endpoint["phase_templates"]["acaa"])[0],)
    metadata = [
        {
            "cycle": str(item["cycle"]),
            "temperature_k": float(item["temperature_k"]),
            "phase": "a21am",
        }
        for item in records
    ] + [{"cycle": "heating", "temperature_k": 1173.0, "phase": "acaa"}]
    return structures, metadata
