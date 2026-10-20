from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .layers import CorrectionLayer
from .layers import MixtureProfileLayer
from .layers import PeakGroupLayer
from .layers import ProfileLayer
from .layers import ReflectionContractLayer
from .layers import ReflectionLayer

try:
    from ..spacegroup_contract import resolve_standard_setting_spacegroup
    from ..spacegroup_contract import systematic_absence_mask
except ImportError:
    from spacegroup_contract import resolve_standard_setting_spacegroup
    from spacegroup_contract import systematic_absence_mask


@dataclass(frozen=True)
class CompiledReflectionContract:
    """Cached HKL-family metadata for fixed-HKL profile contraction."""

    spacegroup_symbol: str
    include_friedel_pairs: bool
    valid_raw_indices: torch.Tensor
    raw_to_family: torch.Tensor
    family_hkls: torch.Tensor
    family_multiplicities: torch.Tensor
    source_hkl_shape: tuple[int, ...]
    static_full_families: bool = False

    @property
    def num_families(self) -> int:
        return int(self.family_hkls.shape[0])


def _to_hkl_tuple(
    hkl: torch.Tensor | np.ndarray | tuple[int, int, int] | list[int],
) -> tuple[int, int, int]:
    if isinstance(hkl, torch.Tensor):
        values = hkl.detach().cpu().tolist()
    elif isinstance(hkl, np.ndarray):
        values = hkl.tolist()
    else:
        values = hkl
    return tuple(int(value) for value in values)


@lru_cache(maxsize=None)
def get_spacegroup_number(spacegroup_symbol: str) -> int:
    return int(resolve_standard_setting_spacegroup(spacegroup_symbol).number)

def combine_profile_layers(
    self,
    component_profiles: list[ProfileLayer],
    phase_weights: torch.Tensor | np.ndarray | list[float] | tuple[float, ...],
) -> MixtureProfileLayer:
    if len(component_profiles) == 0:
        raise ValueError("at least one component profile is required")

    reference_profile = component_profiles[0]
    if isinstance(phase_weights, torch.Tensor):
        weight_tensor = phase_weights.to(
            device=reference_profile.histogram.device,
            dtype=reference_profile.histogram.dtype,
        )
    else:
        weight_tensor = torch.tensor(
            phase_weights,
            dtype=reference_profile.histogram.dtype,
            device=reference_profile.histogram.device,
        )
    if weight_tensor.ndim != 1:
        raise ValueError("phase_weights must be one-dimensional")
    if weight_tensor.numel() != len(component_profiles):
        raise ValueError("phase_weights must match the number of component profiles")

    combined_histogram = None
    for profile_layer, weight in zip(component_profiles, weight_tensor):
        if profile_layer.histogram.shape != reference_profile.histogram.shape:
            raise ValueError("all component histograms must share the same shape")
        if profile_layer.pattern_domain.shape != reference_profile.pattern_domain.shape or not torch.allclose(
            profile_layer.pattern_domain,
            reference_profile.pattern_domain,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("all component profile domains must match")
        weighted_histogram = weight * profile_layer.histogram
        if combined_histogram is None:
            combined_histogram = weighted_histogram
        else:
            combined_histogram = combined_histogram + weighted_histogram

    if combined_histogram is None:
        raise RuntimeError("failed to build mixture histogram")

    return MixtureProfileLayer(
        pattern_domain=reference_profile.pattern_domain,
        phase_weights=weight_tensor,
        component_profiles=component_profiles,
        histogram=combined_histogram,
    )

def group_peak_list(
    self,
    positions: torch.Tensor,
    intensities: torch.Tensor,
    *,
    tolerance_deg: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite_mask = torch.isfinite(positions) & torch.isfinite(intensities) & (intensities > 0)
    positions = positions[finite_mask]
    intensities = intensities[finite_mask]
    if positions.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=self.device)
        return empty, empty, empty.to(dtype=torch.long)

    order = torch.argsort(positions)
    positions = positions[order]
    intensities = intensities[order]

    grouped_positions = [positions[0]]
    grouped_intensities = [intensities[0]]
    grouped_counts = [1]

    for position, intensity in zip(positions[1:], intensities[1:]):
        current_average = grouped_positions[-1] / grouped_counts[-1]
        if torch.abs(position - current_average) <= tolerance_deg:
            grouped_positions[-1] = grouped_positions[-1] + position
            grouped_intensities[-1] = grouped_intensities[-1] + intensity
            grouped_counts[-1] += 1
        else:
            grouped_positions.append(position)
            grouped_intensities.append(intensity)
            grouped_counts.append(1)

    averaged_positions = torch.stack(
        [position_sum / count for position_sum, count in zip(grouped_positions, grouped_counts)]
    )
    summed_intensities = torch.stack(grouped_intensities)
    raw_counts = torch.tensor(grouped_counts, dtype=torch.long, device=self.device)
    return averaged_positions, summed_intensities, raw_counts

def compute_peak_group_layer(
    self,
    correction_layer: CorrectionLayer,
    *,
    tolerance_deg: float = 1.0e-5,
) -> PeakGroupLayer:
    pattern = correction_layer.pattern
    grouped_positions: list[torch.Tensor] = []
    grouped_intensities: list[torch.Tensor] = []
    grouped_counts: list[torch.Tensor] = []

    for batch_index in range(pattern.shape[1]):
        positions = pattern[0, batch_index, :]
        intensities = pattern[1, batch_index, :]
        merged_positions, merged_intensities, raw_counts = self.group_peak_list(
            positions,
            intensities,
            tolerance_deg=tolerance_deg,
        )
        grouped_positions.append(merged_positions)
        grouped_intensities.append(merged_intensities)
        grouped_counts.append(raw_counts)

    return PeakGroupLayer(
        tolerance_deg=float(tolerance_deg),
        positions=grouped_positions,
        intensities=grouped_intensities,
        raw_counts=grouped_counts,
    )

@lru_cache(maxsize=None)
def get_reciprocal_symmetry_rotations(
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    rotations: list[tuple[tuple[int, int, int], ...]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for operation in resolve_standard_setting_spacegroup(spacegroup_symbol).operations():
        # Reciprocal-space row HKLs transform with the inverse of the
        # direct-space symmetry operator: h' = h * R^{-1}.
        inverse = operation.inverse()
        numerator = np.asarray(inverse.rot, dtype=np.int64)
        if np.any(numerator % inverse.DEN):
            raise ValueError("space-group rotation does not preserve integer HKLs")
        rotation = numerator // inverse.DEN
        key = tuple(tuple(int(value) for value in row) for row in rotation.tolist())
        if key not in seen:
            rotations.append(key)
            seen.add(key)

    if include_friedel_pairs:
        inverted_rotations = []
        for rotation in rotations:
            inverted = tuple(tuple(-value for value in row) for row in rotation)
            if inverted not in seen:
                inverted_rotations.append(inverted)
                seen.add(inverted)
        rotations.extend(inverted_rotations)

    return tuple(rotations)


@lru_cache(maxsize=None)
def get_reciprocal_symmetry_rotation_array(
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
) -> np.ndarray:
    return np.asarray(
        get_reciprocal_symmetry_rotations(
            spacegroup_symbol,
            include_friedel_pairs=include_friedel_pairs,
        ),
        dtype=np.int64,
    )


@lru_cache(maxsize=None)
def get_reciprocal_orbit_cached(
    spacegroup_symbol: str,
    hkl_tuple: tuple[int, int, int],
    include_friedel_pairs: bool = True,
) -> tuple[tuple[int, int, int], ...]:
    orbit = []
    for rotation in get_reciprocal_symmetry_rotations(
        spacegroup_symbol,
        include_friedel_pairs=include_friedel_pairs,
    ):
        rotated = (
            hkl_tuple[0] * rotation[0][0]
            + hkl_tuple[1] * rotation[1][0]
            + hkl_tuple[2] * rotation[2][0],
            hkl_tuple[0] * rotation[0][1]
            + hkl_tuple[1] * rotation[1][1]
            + hkl_tuple[2] * rotation[2][1],
            hkl_tuple[0] * rotation[0][2]
            + hkl_tuple[1] * rotation[1][2]
            + hkl_tuple[2] * rotation[2][2],
        )
        orbit.append(rotated)
    return tuple(sorted(set(orbit)))


@lru_cache(maxsize=None)
def get_hkl_family_metadata(
    spacegroup_symbol: str,
    hkl_tuple: tuple[int, int, int],
    include_friedel_pairs: bool = True,
) -> tuple[bool, tuple[int, int, int], int]:
    if hkl_tuple == (0, 0, 0):
        return False, hkl_tuple, 0

    non_absent = bool(
        systematic_absence_mask(
            get_spacegroup_number(spacegroup_symbol),
            np.asarray([hkl_tuple], dtype=np.int64),
        )[0]
    )
    if not non_absent:
        return False, hkl_tuple, 0

    orbit = get_reciprocal_orbit_cached(
        spacegroup_symbol,
        hkl_tuple,
        include_friedel_pairs=include_friedel_pairs,
    )
    return True, min(orbit), len(orbit)


def build_overlap_groups(
    positions: torch.Tensor,
    *,
    tolerance_deg: float = 1.0e-5,
) -> tuple[list[int], list[int]]:
    if positions.numel() == 0:
        return [], []

    position_values = [float(value) for value in positions.detach().cpu().tolist()]
    overlap_position_sums = [position_values[0]]
    overlap_counts = [1]
    family_to_overlap = [0]
    for position in position_values[1:]:
        current_average = overlap_position_sums[-1] / overlap_counts[-1]
        if abs(position - current_average) <= tolerance_deg:
            overlap_position_sums[-1] += position
            overlap_counts[-1] += 1
            family_to_overlap.append(len(overlap_position_sums) - 1)
        else:
            overlap_position_sums.append(position)
            overlap_counts.append(1)
            family_to_overlap.append(len(overlap_position_sums) - 1)
    return family_to_overlap, overlap_counts


def _empty_reflection_contract_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    empty_float = torch.empty(0, dtype=torch.float32, device=device)
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_hkl = torch.empty((0, 3), dtype=torch.long, device=device)
    return (
        empty_hkl,
        empty_float,
        empty_float,
        empty_long,
        empty_long,
        empty_long,
        empty_float,
        empty_float,
        empty_long,
        empty_long,
        empty_long,
    )


def _encode_hkl_triplets(
    hkls: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if hkls.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=hkls.device), 0, 1

    min_value = int(hkls.min().item())
    max_value = int(hkls.max().item())
    shift = -min_value if min_value < 0 else 0
    base = max_value + shift + 1
    if base <= 0:
        base = 1
    h_vals = hkls[..., 0].to(dtype=torch.long)
    k_vals = hkls[..., 1].to(dtype=torch.long)
    l_vals = hkls[..., 2].to(dtype=torch.long)
    encoded = ((h_vals + shift) * base + (k_vals + shift)) * base + (l_vals + shift)
    return encoded, shift, base


def _decode_hkl_triplet_codes(
    codes: torch.Tensor,
    *,
    shift: int,
    base: int,
) -> torch.Tensor:
    if codes.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=codes.device)
    base_sq = base * base
    h_vals = torch.div(codes, base_sq, rounding_mode="floor") - shift
    remainder = torch.remainder(codes, base_sq)
    k_vals = torch.div(remainder, base, rounding_mode="floor") - shift
    l_vals = torch.remainder(remainder, base) - shift
    return torch.stack((h_vals, k_vals, l_vals), dim=-1)


def _compute_vectorized_family_assignments(
    hkls: torch.Tensor,
    *,
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hkls.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=hkls.device)
        empty_hkl = torch.empty((0, 3), dtype=torch.long, device=hkls.device)
        return empty_hkl, empty_long, empty_long

    rotations = torch.as_tensor(
        get_reciprocal_symmetry_rotation_array(
            spacegroup_symbol,
            include_friedel_pairs=include_friedel_pairs,
        ),
        dtype=torch.long,
        device=hkls.device,
    )
    rotated_hkls = (
        hkls.to(dtype=torch.long).unsqueeze(1).unsqueeze(-1) * rotations.unsqueeze(0)
    ).sum(dim=2)
    rotated_codes, shift, base = _encode_hkl_triplets(rotated_hkls)

    canonical_rotation_indices = torch.argmin(rotated_codes, dim=1)
    canonical_codes = rotated_codes.gather(1, canonical_rotation_indices.unsqueeze(1)).squeeze(1)
    sorted_codes = torch.sort(rotated_codes, dim=1).values
    multiplicities = torch.ones(
        sorted_codes.shape[0],
        dtype=torch.long,
        device=hkls.device,
    )
    if sorted_codes.shape[1] > 1:
        multiplicities = multiplicities + torch.sum(
            sorted_codes[:, 1:] != sorted_codes[:, :-1],
            dim=1,
        ).to(dtype=torch.long)

    family_codes, raw_to_family = torch.unique(
        canonical_codes,
        sorted=True,
        return_inverse=True,
    )
    family_hkls = _decode_hkl_triplet_codes(
        family_codes,
        shift=shift,
        base=base,
    )
    return family_hkls, multiplicities, raw_to_family


def _canonicalize_contract_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized in {"", "auto", "reference"}:
        return "reference"
    raise ValueError("contract backend must be one of: auto, reference")


def _build_reflection_contract_reference(
    self,
    filtered_hkls: torch.Tensor,
    filtered_positions: torch.Tensor,
    filtered_intensities: torch.Tensor,
    *,
    spacegroup_symbol: str,
    overlap_tolerance_deg: float,
    include_friedel_pairs: bool,
    include_overlaps: bool,
) -> tuple[torch.Tensor, ...]:
    family_ids: dict[tuple[int, int, int], int] = {}
    family_canonicals: list[tuple[int, int, int]] = []
    family_multiplicity_list: list[int] = []
    valid_raw_indices: list[int] = []
    raw_to_family: list[int] = []

    for raw_index, hkl_tuple in enumerate(filtered_hkls.detach().cpu().tolist()):
        is_non_absent, canonical_hkl, multiplicity = get_hkl_family_metadata(
            spacegroup_symbol,
            _to_hkl_tuple(hkl_tuple),
            include_friedel_pairs=include_friedel_pairs,
        )
        if not is_non_absent:
            continue
        family_index = family_ids.get(canonical_hkl)
        if family_index is None:
            family_index = len(family_canonicals)
            family_ids[canonical_hkl] = family_index
            family_canonicals.append(canonical_hkl)
            family_multiplicity_list.append(multiplicity)
        valid_raw_indices.append(raw_index)
        raw_to_family.append(family_index)

    if not valid_raw_indices:
        return _empty_reflection_contract_tensors(self.device)

    valid_index_tensor = torch.tensor(valid_raw_indices, dtype=torch.long, device=self.device)
    raw_to_family_tensor = torch.tensor(raw_to_family, dtype=torch.long, device=self.device)
    filtered_positions = filtered_positions.index_select(0, valid_index_tensor)
    filtered_intensities = filtered_intensities.index_select(0, valid_index_tensor)

    num_families = len(family_canonicals)
    family_hkls = torch.tensor(family_canonicals, dtype=torch.long, device=self.device)
    family_multiplicities = torch.tensor(family_multiplicity_list, dtype=torch.long, device=self.device)
    family_raw_counts = torch.bincount(raw_to_family_tensor, minlength=num_families)
    family_position_sums = torch.zeros(num_families, dtype=filtered_positions.dtype, device=self.device).index_add(
        0,
        raw_to_family_tensor,
        filtered_positions,
    )
    family_intensities = torch.zeros(num_families, dtype=filtered_intensities.dtype, device=self.device).index_add(
        0,
        raw_to_family_tensor,
        filtered_intensities,
    )
    family_positions = family_position_sums / family_raw_counts.to(dtype=filtered_positions.dtype)

    family_order = torch.argsort(family_positions)
    family_hkls = family_hkls[family_order]
    family_positions = family_positions[family_order]
    family_intensities = family_intensities[family_order]
    family_multiplicities = family_multiplicities[family_order]
    family_raw_counts = family_raw_counts[family_order]
    inverse_family_order = torch.empty_like(family_order)
    inverse_family_order[family_order] = torch.arange(len(family_order), device=self.device)
    raw_to_family_tensor = inverse_family_order.index_select(0, raw_to_family_tensor)

    if not bool(include_overlaps):
        return (
            family_hkls,
            family_positions,
            family_intensities,
            family_multiplicities,
            family_raw_counts,
            raw_to_family_tensor,
            torch.empty(0, dtype=family_positions.dtype, device=self.device),
            torch.empty(0, dtype=family_intensities.dtype, device=self.device),
            torch.empty(0, dtype=torch.long, device=self.device),
            torch.empty(0, dtype=torch.long, device=self.device),
            torch.empty(0, dtype=torch.long, device=self.device),
        )

    family_to_overlap, overlap_family_counts_list = build_overlap_groups(
        family_positions,
        tolerance_deg=overlap_tolerance_deg,
    )
    family_to_overlap_tensor = torch.tensor(family_to_overlap, dtype=torch.long, device=self.device)
    overlap_family_counts = torch.tensor(overlap_family_counts_list, dtype=torch.long, device=self.device)
    num_overlaps = len(overlap_family_counts_list)
    overlap_position_sums = torch.zeros(num_overlaps, dtype=family_positions.dtype, device=self.device).index_add(
        0,
        family_to_overlap_tensor,
        family_positions,
    )
    overlap_positions = overlap_position_sums / overlap_family_counts.to(dtype=family_positions.dtype)
    overlap_intensities = torch.zeros(
        num_overlaps,
        dtype=family_intensities.dtype,
        device=self.device,
    ).index_add(
        0,
        family_to_overlap_tensor,
        family_intensities,
    )
    overlap_multiplicities = torch.zeros(num_overlaps, dtype=torch.long, device=self.device).index_add(
        0,
        family_to_overlap_tensor,
        family_multiplicities,
    )

    return (
        family_hkls,
        family_positions,
        family_intensities,
        family_multiplicities,
        family_raw_counts,
        raw_to_family_tensor,
        overlap_positions,
        overlap_intensities,
        overlap_multiplicities,
        overlap_family_counts,
        family_to_overlap_tensor,
    )


def _as_single_hkl_tensor(
    self,
    hkl_pts: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]],
) -> torch.Tensor:
    if isinstance(hkl_pts, torch.Tensor):
        hkl_tensor = hkl_pts.to(device=self.device)
    else:
        hkl_tensor = torch.tensor(hkl_pts, device=self.device)
    if hkl_tensor.ndim == 3:
        if hkl_tensor.shape[0] < 1:
            raise ValueError("batched hkl_pts must contain at least one batch member")
        hkl_tensor = hkl_tensor[0]
    if hkl_tensor.ndim != 2 or hkl_tensor.shape[-1] != 3:
        raise ValueError("hkl_pts must have shape (n, 3) or (batch, n, 3)")
    return hkl_tensor


def compile_reflection_contract(
    self,
    hkl_pts: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]],
    *,
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
    static_full_families: bool = False,
) -> CompiledReflectionContract:
    hkl_tensor = _as_single_hkl_tensor(self, hkl_pts)
    rounded_hkls = torch.round(hkl_tensor).to(dtype=torch.long)

    family_ids: dict[tuple[int, int, int], int] = {}
    family_canonicals: list[tuple[int, int, int]] = []
    family_multiplicity_list: list[int] = []
    valid_raw_indices: list[int] = []
    raw_to_family: list[int] = []

    for raw_index, hkl_tuple in enumerate(rounded_hkls.detach().cpu().tolist()):
        is_non_absent, canonical_hkl, multiplicity = get_hkl_family_metadata(
            spacegroup_symbol,
            _to_hkl_tuple(hkl_tuple),
            include_friedel_pairs=include_friedel_pairs,
        )
        if not is_non_absent:
            continue
        family_index = family_ids.get(canonical_hkl)
        if family_index is None:
            family_index = len(family_canonicals)
            family_ids[canonical_hkl] = family_index
            family_canonicals.append(canonical_hkl)
            family_multiplicity_list.append(multiplicity)
        valid_raw_indices.append(raw_index)
        raw_to_family.append(family_index)

    return CompiledReflectionContract(
        spacegroup_symbol=str(spacegroup_symbol),
        include_friedel_pairs=bool(include_friedel_pairs),
        valid_raw_indices=torch.tensor(valid_raw_indices, dtype=torch.long, device=self.device),
        raw_to_family=torch.tensor(raw_to_family, dtype=torch.long, device=self.device),
        family_hkls=torch.tensor(family_canonicals, dtype=torch.long, device=self.device),
        family_multiplicities=torch.tensor(family_multiplicity_list, dtype=torch.long, device=self.device),
        source_hkl_shape=tuple(int(value) for value in hkl_tensor.shape),
        static_full_families=bool(static_full_families),
    )


def apply_compiled_reflection_contract_single(
    self,
    compiled_contract: CompiledReflectionContract,
    positions: torch.Tensor,
    intensities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if positions.ndim != 1 or intensities.ndim != 1:
        raise ValueError("positions and intensities must be one-dimensional for a single compiled contract")
    if positions.shape != intensities.shape:
        raise ValueError("positions and intensities must have matching shape")
    if compiled_contract.valid_raw_indices.numel() == 0:
        empty = _empty_reflection_contract_tensors(self.device)
        return empty[0], empty[1].to(dtype=positions.dtype), empty[2].to(dtype=intensities.dtype), empty[3], empty[4]

    valid_indices = compiled_contract.valid_raw_indices.to(device=positions.device)
    raw_positions = positions.index_select(0, valid_indices)
    raw_intensities = intensities.index_select(0, valid_indices)
    raw_to_family = compiled_contract.raw_to_family.to(device=positions.device)
    profile_mask = (
        torch.isfinite(raw_positions) & torch.isfinite(raw_intensities) & (raw_intensities >= 0)
        & ((raw_positions != 0) | (raw_intensities != 0))
    )
    num_families = int(compiled_contract.num_families)
    if bool(compiled_contract.static_full_families):
        profile_weight = profile_mask.to(dtype=positions.dtype)
        family_counts_float = torch.zeros(num_families, dtype=positions.dtype, device=positions.device).index_add(
            0,
            raw_to_family,
            profile_weight,
        )
        family_position_sums = torch.zeros(num_families, dtype=positions.dtype, device=positions.device).index_add(
            0,
            raw_to_family,
            torch.where(profile_mask, raw_positions, torch.zeros_like(raw_positions)),
        )
        family_intensities = torch.zeros(num_families, dtype=intensities.dtype, device=intensities.device).index_add(
            0,
            raw_to_family,
            torch.where(profile_mask, raw_intensities, torch.zeros_like(raw_intensities)),
        )
        family_positions = torch.where(
            family_counts_float > 0,
            family_position_sums / torch.clamp(family_counts_float, min=1.0),
            torch.zeros_like(family_position_sums),
        )
        return (
            compiled_contract.family_hkls.to(device=positions.device),
            family_positions,
            family_intensities,
            compiled_contract.family_multiplicities.to(device=positions.device),
            family_counts_float.to(dtype=torch.long),
        )
    filtered_positions = raw_positions[profile_mask]
    filtered_intensities = raw_intensities[profile_mask]
    filtered_to_family = raw_to_family[profile_mask]
    if filtered_positions.numel() == 0:
        empty = _empty_reflection_contract_tensors(self.device)
        return empty[0], empty[1].to(dtype=positions.dtype), empty[2].to(dtype=intensities.dtype), empty[3], empty[4]

    family_raw_counts = torch.bincount(filtered_to_family, minlength=num_families)
    active_family_mask = family_raw_counts > 0
    family_position_sums = torch.zeros(num_families, dtype=positions.dtype, device=positions.device).index_add(
        0,
        filtered_to_family,
        filtered_positions,
    )
    family_intensity_sums = torch.zeros(num_families, dtype=intensities.dtype, device=intensities.device).index_add(
        0,
        filtered_to_family,
        filtered_intensities,
    )
    active_counts = family_raw_counts[active_family_mask]
    family_positions = family_position_sums[active_family_mask] / active_counts.to(dtype=positions.dtype)
    family_intensities = family_intensity_sums[active_family_mask]
    family_hkls = compiled_contract.family_hkls.to(device=positions.device)[active_family_mask]
    family_multiplicities = compiled_contract.family_multiplicities.to(device=positions.device)[active_family_mask]

    family_order = torch.argsort(family_positions)
    return (
        family_hkls[family_order],
        family_positions[family_order],
        family_intensities[family_order],
        family_multiplicities[family_order],
        active_counts[family_order],
    )


def _apply_static_contract_to_batch(
    self,
    compiled_contract: CompiledReflectionContract,
    positions: torch.Tensor,
    intensities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract a batch through one fixed family map without a per-pattern loop.

    Every pattern keeps all families in compiled order, so the result equals the
    per-pattern static contraction and needs no padding.
    """
    valid_indices = compiled_contract.valid_raw_indices.to(device=positions.device)
    raw_to_family = compiled_contract.raw_to_family.to(device=positions.device)
    raw_positions = positions.index_select(1, valid_indices)
    raw_intensities = intensities.index_select(1, valid_indices)
    profile_mask = (
        torch.isfinite(raw_positions) & torch.isfinite(raw_intensities) & (raw_intensities >= 0)
        & ((raw_positions != 0) | (raw_intensities != 0))
    )
    shape = (positions.shape[0], int(compiled_contract.num_families))
    counts = torch.zeros(shape, dtype=positions.dtype, device=positions.device).index_add(
        1, raw_to_family, profile_mask.to(dtype=positions.dtype))
    position_sums = torch.zeros(shape, dtype=positions.dtype, device=positions.device).index_add(
        1, raw_to_family, torch.where(profile_mask, raw_positions, torch.zeros_like(raw_positions)))
    family_intensities = torch.zeros(shape, dtype=intensities.dtype, device=intensities.device).index_add(
        1, raw_to_family, torch.where(profile_mask, raw_intensities, torch.zeros_like(raw_intensities)))
    family_positions = torch.where(
        counts > 0, position_sums / torch.clamp(counts, min=1.0), torch.zeros_like(position_sums))
    family_hkls = compiled_contract.family_hkls.to(device=positions.device).unsqueeze(0).expand(shape[0], -1, -1)
    return self.build_pattern_tensor(family_positions, family_intensities), family_hkls


def apply_compiled_reflection_contract_to_pattern(
    self,
    compiled_contract: CompiledReflectionContract,
    positions: torch.Tensor,
    intensities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.shape != intensities.shape:
        raise ValueError("positions and intensities must have matching shape")
    if positions.ndim == 1:
        family_hkls, family_positions, family_intensities, _multiplicities, _counts = (
            self.apply_compiled_reflection_contract_single(compiled_contract, positions, intensities)
        )
        return self.build_pattern_tensor(family_positions, family_intensities), family_hkls.unsqueeze(0)
    if positions.ndim != 2:
        raise ValueError("positions and intensities must have shape (n,) or (batch, n)")
    if bool(compiled_contract.static_full_families) and compiled_contract.valid_raw_indices.numel() > 0:
        return _apply_static_contract_to_batch(self, compiled_contract, positions, intensities)

    contracted_hkls: list[torch.Tensor] = []
    contracted_positions: list[torch.Tensor] = []
    contracted_intensities: list[torch.Tensor] = []
    max_count = 0
    for batch_index in range(positions.shape[0]):
        family_hkls, family_positions, family_intensities, _multiplicities, _counts = (
            self.apply_compiled_reflection_contract_single(
                compiled_contract,
                positions[batch_index],
                intensities[batch_index],
            )
        )
        contracted_hkls.append(family_hkls)
        contracted_positions.append(family_positions)
        contracted_intensities.append(family_intensities)
        max_count = max(max_count, int(family_positions.numel()))

    if max_count == 0:
        empty_positions = torch.zeros((positions.shape[0], 0), dtype=positions.dtype, device=positions.device)
        empty_intensities = torch.zeros((positions.shape[0], 0), dtype=intensities.dtype, device=intensities.device)
        empty_hkls = torch.zeros((positions.shape[0], 0, 3), dtype=torch.long, device=positions.device)
        return self.build_pattern_tensor(empty_positions, empty_intensities), empty_hkls

    padded_positions = torch.zeros((positions.shape[0], max_count), dtype=positions.dtype, device=positions.device)
    padded_intensities = torch.zeros((positions.shape[0], max_count), dtype=intensities.dtype, device=intensities.device)
    padded_hkls = torch.zeros((positions.shape[0], max_count, 3), dtype=torch.long, device=positions.device)
    for batch_index, (family_hkls, family_positions, family_intensities) in enumerate(
        zip(contracted_hkls, contracted_positions, contracted_intensities, strict=True)
    ):
        count = int(family_positions.numel())
        if count == 0:
            continue
        padded_positions[batch_index, :count] = family_positions
        padded_intensities[batch_index, :count] = family_intensities
        padded_hkls[batch_index, :count] = family_hkls
    return self.build_pattern_tensor(padded_positions, padded_intensities), padded_hkls


def reciprocal_orbit(
    self,
    hkl: torch.Tensor | np.ndarray | tuple[int, int, int] | list[int],
    *,
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
) -> tuple[tuple[int, int, int], ...]:
    return get_reciprocal_orbit_cached(
        spacegroup_symbol,
        _to_hkl_tuple(hkl),
        include_friedel_pairs=include_friedel_pairs,
    )

def compute_reciprocal_multiplicities(
    self,
    hkl_pts: torch.Tensor,
    *,
    spacegroup_symbol: str,
    include_friedel_pairs: bool = True,
) -> torch.Tensor:
    rounded_hkls = torch.round(hkl_pts).to(dtype=torch.long)
    multiplicities = torch.zeros(
        rounded_hkls.shape[0],
        dtype=torch.long,
        device=self.device,
    )
    valid_hkl_mask = torch.any(rounded_hkls != 0, dim=1)
    if not torch.any(valid_hkl_mask):
        return multiplicities

    family_hkls, raw_multiplicities, raw_to_family = _compute_vectorized_family_assignments(
        rounded_hkls[valid_hkl_mask],
        spacegroup_symbol=spacegroup_symbol,
        include_friedel_pairs=include_friedel_pairs,
    )
    if family_hkls.numel() == 0:
        return multiplicities

    non_absent_family_mask = torch.from_numpy(
        systematic_absence_mask(
            get_spacegroup_number(spacegroup_symbol),
            np.asarray(family_hkls.detach().cpu().numpy(), dtype=np.int64),
        )
    ).to(device=self.device, dtype=torch.bool)
    valid_raw_mask = non_absent_family_mask.index_select(0, raw_to_family)
    valid_indices = torch.nonzero(valid_hkl_mask, as_tuple=False).squeeze(1)[valid_raw_mask]
    multiplicities[valid_indices] = raw_multiplicities[valid_raw_mask]
    return multiplicities

def build_reflection_contract(
    self,
    hkl_pts: torch.Tensor,
    two_thetas: torch.Tensor,
    intensities: torch.Tensor,
    *,
    spacegroup_symbol: str,
    overlap_tolerance_deg: float = 1.0e-5,
    include_friedel_pairs: bool = True,
    include_overlaps: bool = True,
    backend: str = "auto",
) -> tuple[torch.Tensor, ...]:
    rounded_hkls = torch.round(hkl_pts).to(dtype=torch.long)
    # An allowed family can have exactly zero intensity through motif cancellation.
    # Correction-layer invalid reflections instead use the joint (0, 0) sentinel.
    finite_mask = (
        torch.isfinite(two_thetas)
        & torch.isfinite(intensities)
        & (intensities >= 0)
        & ((two_thetas != 0) | (intensities != 0))
        & torch.any(rounded_hkls != 0, dim=1)
    )
    filtered_hkls = rounded_hkls[finite_mask]
    filtered_positions = two_thetas[finite_mask]
    filtered_intensities = intensities[finite_mask]

    if filtered_hkls.numel() == 0:
        return _empty_reflection_contract_tensors(self.device)
    _canonicalize_contract_backend(backend)
    return _build_reflection_contract_reference(
        self,
        filtered_hkls,
        filtered_positions,
        filtered_intensities,
        spacegroup_symbol=spacegroup_symbol,
        overlap_tolerance_deg=overlap_tolerance_deg,
        include_friedel_pairs=include_friedel_pairs,
        include_overlaps=bool(include_overlaps),
    )

def compute_reflection_contract_layer(
    self,
    reflection_layer: ReflectionLayer,
    correction_layer: CorrectionLayer,
    *,
    spacegroup_symbols: str | list[str] | tuple[str, ...],
    overlap_tolerance_deg: float = 1.0e-5,
    include_friedel_pairs: bool = True,
    include_overlaps: bool = True,
    backend: str = "auto",
) -> ReflectionContractLayer:
    _canonicalize_contract_backend(backend)
    if isinstance(spacegroup_symbols, str):
        symbols = [spacegroup_symbols] * correction_layer.pattern.shape[1]
    else:
        symbols = list(spacegroup_symbols)
    if len(symbols) != correction_layer.pattern.shape[1]:
        raise ValueError("spacegroup_symbols must match the batch size")

    family_hkls: list[torch.Tensor] = []
    family_positions: list[torch.Tensor] = []
    family_intensities: list[torch.Tensor] = []
    family_multiplicities: list[torch.Tensor] = []
    family_raw_counts: list[torch.Tensor] = []
    raw_to_family: list[torch.Tensor] = []
    overlap_positions: list[torch.Tensor] = []
    overlap_intensities: list[torch.Tensor] = []
    overlap_multiplicities: list[torch.Tensor] = []
    overlap_family_counts: list[torch.Tensor] = []
    family_to_overlap: list[torch.Tensor] = []
    for batch_index, symbol in enumerate(symbols):
        (
            case_family_hkls,
            case_family_positions,
            case_family_intensities,
            case_family_multiplicities,
            case_family_raw_counts,
            case_raw_to_family,
            case_overlap_positions,
            case_overlap_intensities,
            case_overlap_multiplicities,
            case_overlap_family_counts,
            case_family_to_overlap,
        ) = self.build_reflection_contract(
            reflection_layer.hkl_pts[batch_index],
            correction_layer.two_thetas[batch_index],
            correction_layer.corrected_intensities[batch_index],
            spacegroup_symbol=symbol,
            overlap_tolerance_deg=overlap_tolerance_deg,
            include_friedel_pairs=include_friedel_pairs,
            include_overlaps=bool(include_overlaps),
            backend="reference",
        )
        family_hkls.append(case_family_hkls)
        family_positions.append(case_family_positions)
        family_intensities.append(case_family_intensities)
        family_multiplicities.append(case_family_multiplicities)
        family_raw_counts.append(case_family_raw_counts)
        raw_to_family.append(case_raw_to_family)
        overlap_positions.append(case_overlap_positions)
        overlap_intensities.append(case_overlap_intensities)
        overlap_multiplicities.append(case_overlap_multiplicities)
        overlap_family_counts.append(case_overlap_family_counts)
        family_to_overlap.append(case_family_to_overlap)

    return ReflectionContractLayer(
        spacegroup_symbols=symbols,
        family_hkls=family_hkls,
        family_positions=family_positions,
        family_intensities=family_intensities,
        family_multiplicities=family_multiplicities,
        family_raw_counts=family_raw_counts,
        raw_to_family=raw_to_family,
        overlap_positions=overlap_positions,
        overlap_intensities=overlap_intensities,
        overlap_multiplicities=overlap_multiplicities,
        overlap_family_counts=overlap_family_counts,
        family_to_overlap=family_to_overlap,
    )

__all__ = [
    "CompiledReflectionContract",
    "apply_compiled_reflection_contract_single",
    "apply_compiled_reflection_contract_to_pattern",
    "combine_profile_layers",
    "compile_reflection_contract",
    "group_peak_list",
    "compute_peak_group_layer",
    "get_reciprocal_symmetry_rotations",
    "reciprocal_orbit",
    "compute_reciprocal_multiplicities",
    "build_reflection_contract",
    "compute_reflection_contract_layer",
]
