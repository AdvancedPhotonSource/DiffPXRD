from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from pymatgen.core import Element

from ..xray_atomic_data import lookup_xray_form_factor_coeffs

@dataclass
class ReflectionLayer:
    cell_matrix: torch.Tensor
    hkl_pts: torch.Tensor
    recip_lengths: torch.Tensor

@dataclass(frozen=True)
class FrozenReflectionSet:
    hkl_pts: torch.Tensor
    q_min: float
    q_max: float

@dataclass
class ScatteringLayer:
    frac_coords: torch.Tensor
    g_dot_r: torch.Tensor
    structure_factors: torch.Tensor
    base_intensities: torch.Tensor
    wavelength: float | None = None
    phase_layout: str = "legacy_global"
    num_atoms: torch.Tensor | None = None
    site_uiso: torch.Tensor | None = None
    site_uaniso: torch.Tensor | None = None
    site_species_occs: torch.Tensor | None = None
    site_species_zs: torch.Tensor | None = None
    site_species_ff_a: torch.Tensor | None = None
    site_species_ff_b: torch.Tensor | None = None
    site_species_ff_c: torch.Tensor | None = None
    site_species_fp: torch.Tensor | None = None
    site_species_fpp: torch.Tensor | None = None
    atomic_scattering_factors: torch.Tensor | None = None
    weighted_scattering_factors: torch.Tensor | None = None

@dataclass
class CorrectionLayer:
    wavelength: float
    lorentz_factor: torch.Tensor
    preferred_orientation_factor: torch.Tensor | None
    two_thetas: torch.Tensor
    corrected_intensities: torch.Tensor
    pattern: torch.Tensor
    cell_matrix: torch.Tensor | None = None
    hkl_pts: torch.Tensor | None = None
    polarization_factor: torch.Tensor | None = None
    absorption_factor: torch.Tensor | None = None
    extinction_factor: torch.Tensor | None = None
    position_shift_deg: torch.Tensor | None = None
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None

@dataclass
class ProfileLayer:
    pattern_domain: torch.Tensor
    fwhm: torch.Tensor
    gaussian_peak: torch.Tensor
    lorentz_peak: torch.Tensor
    histogram: torch.Tensor
    sigma_sq_centideg2: torch.Tensor | None = None
    gamma_centideg: torch.Tensor | None = None
    eta: torch.Tensor | None = None
    support_mask: torch.Tensor | None = None
    support_lower_deg: torch.Tensor | None = None
    support_upper_deg: torch.Tensor | None = None
    fcj_quadrature_terms: torch.Tensor | None = None

@dataclass
class WeightedProfileLayer:
    pattern_domain: torch.Tensor
    wavelengths: list[float]
    weights: torch.Tensor
    component_corrections: list[CorrectionLayer]
    component_profiles: list[ProfileLayer]
    histogram: torch.Tensor

@dataclass
class MixtureProfileLayer:
    pattern_domain: torch.Tensor
    phase_weights: torch.Tensor
    component_profiles: list[ProfileLayer]
    histogram: torch.Tensor

@dataclass
class PeakGroupLayer:
    tolerance_deg: float
    positions: list[torch.Tensor]
    intensities: list[torch.Tensor]
    raw_counts: list[torch.Tensor]

@dataclass
class ReflectionContractLayer:
    spacegroup_symbols: list[str]
    family_hkls: list[torch.Tensor]
    family_positions: list[torch.Tensor]
    family_intensities: list[torch.Tensor]
    family_multiplicities: list[torch.Tensor]
    family_raw_counts: list[torch.Tensor]
    raw_to_family: list[torch.Tensor]
    overlap_positions: list[torch.Tensor]
    overlap_intensities: list[torch.Tensor]
    overlap_multiplicities: list[torch.Tensor]
    overlap_family_counts: list[torch.Tensor]
    family_to_overlap: list[torch.Tensor]

@lru_cache(maxsize=None)
def default_atomic_form_factor_constants() -> list[list[list[float]]]:
    constants: list[list[list[float]]] = []
    for atomic_number in range(1, 99):
        element = Element.from_Z(atomic_number)
        fa, fb, _fc, _z, _label = lookup_xray_form_factor_coeffs(element)
        constants.append(
            [
                [float(fa[0]), float(fb[0])],
                [float(fa[1]), float(fb[1])],
                [float(fa[2]), float(fb[2])],
                [float(fa[3]), float(fb[3])],
            ]
        )
    return constants

CW_PROFILE_WIDTH_FLOOR_CENTIDEG = 1.0e-3

__all__ = [
    "CW_PROFILE_WIDTH_FLOOR_CENTIDEG",
    "CorrectionLayer",
    "FrozenReflectionSet",
    "MixtureProfileLayer",
    "PeakGroupLayer",
    "ProfileLayer",
    "ReflectionContractLayer",
    "ReflectionLayer",
    "ScatteringLayer",
    "WeightedProfileLayer",
    "default_atomic_form_factor_constants",
]
