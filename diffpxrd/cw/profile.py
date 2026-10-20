from __future__ import annotations

from functools import lru_cache
from math import pi
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .layers import CW_PROFILE_WIDTH_FLOOR_CENTIDEG
from .layers import CorrectionLayer
from .layers import ProfileLayer
from .layers import ReflectionLayer
from .layers import ScatteringLayer
from .layers import WeightedProfileLayer

from .fcj import compute_fcj_profile_histogram
from .fcj import compute_fcj_window_bounds
from .fcj import select_fcj_num_terms
from ..spacegroup_contract import SpacegroupMetadata
from ..spacegroup_contract import get_spacegroup_metadata
from ..spacegroup_contract import resolve_standard_setting_spacegroup

def _canonicalize_cw_profile_mode(
    self,
    value: Any,
    *,
    field_name: str,
    allow_generalized: bool = False,
) -> str:
    mode = str(value).strip().lower()
    if mode in {"", "none"}:
        return "none"
    if mode in {"isotropic", "iso", "i"}:
        return "isotropic"
    if mode in {"uniaxial", "uni", "u"}:
        return "uniaxial"
    if allow_generalized:
        generalized_aliases = {"generalized", "generalised"}
        if field_name == "size_type":
            generalized_aliases.update({"ellipsoidal", "ellipsoid", "ellip"})
        else:
            generalized_aliases.update({"stephens", "pw_stephens", "pw-stephens"})
        if mode in generalized_aliases:
            return "generalized"
    raise ValueError(f"unsupported CW sample {field_name}: {value}")

def _expand_spacegroup_symbols(
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None,
    *,
    batch_size: int,
) -> list[str | None]:
    if spacegroup_symbol is None:
        return [None] * batch_size
    if isinstance(spacegroup_symbol, str):
        return [spacegroup_symbol] * batch_size
    symbols = list(spacegroup_symbol)
    if len(symbols) != batch_size:
        raise ValueError("spacegroup_symbol batch dimension must match peak_hkls batch dimension")
    return symbols

@lru_cache(maxsize=None)
def get_spacegroup_metadata_for_symbol(spacegroup_symbol: str) -> SpacegroupMetadata:
    return get_spacegroup_metadata(resolve_standard_setting_spacegroup(spacegroup_symbol).number)

def get_mustrain_names(
    *,
    laue_group: str,
    unique_axis: str = "",
) -> tuple[str, ...]:
    if laue_group in {"m3", "m3m"}:
        return ("S400", "S220")
    if laue_group in {"6/m", "6/mmm", "3m1", "3"}:
        return ("S400", "S004", "S202")
    if laue_group in {"31m"}:
        return ("S400", "S004", "S202", "S211")
    if laue_group in {"3R", "3mR"}:
        return ("S400", "S220", "S310", "S211")
    if laue_group in {"4/m", "4/mmm"}:
        return ("S400", "S004", "S220", "S022")
    if laue_group in {"mmm"}:
        return ("S400", "S040", "S004", "S220", "S202", "S022")
    if laue_group in {"2/m"}:
        if unique_axis == "a":
            extra = ("S013", "S031", "S211")
        elif unique_axis == "b":
            extra = ("S301", "S103", "S121")
        else:
            extra = ("S130", "S310", "S112")
        return ("S400", "S040", "S004", "S220", "S202", "S022", *extra)
    return (
        "S400",
        "S040",
        "S004",
        "S220",
        "S202",
        "S022",
        "S310",
        "S103",
        "S031",
        "S130",
        "S301",
        "S013",
        "S211",
        "S121",
        "S112",
    )

def calculate_mustrain_terms(
    self,
    peak_hkls: torch.Tensor,
    *,
    laue_group: str,
    unique_axis: str = "",
) -> torch.Tensor:
    """Return the supported Stephens coefficient basis for the supplied HKLs.

    ``3R``/``3mR`` require rhombohedral axes; ``3``/``3m1`` use hexagonal
    axes. The three-term ``3``/``3m1`` and four-term ``3R`` models are
    restricted invariant subspaces, not complete general trigonal models.
    ``31m`` uses S211 in International Tables H, equation 3.3.42, with
    basis 4*h*k*l*(h+k); the obsolete S301 convention is not an alias.
    """
    h = peak_hkls[..., 0]
    k = peak_hkls[..., 1]
    ell = peak_hkls[..., 2]
    hk_sq = torch.square(h * k)
    hl_sq = torch.square(h * ell)
    kl_sq = torch.square(k * ell)

    if laue_group in {"m3", "m3m"}:
        return torch.stack(
            (
                h**4 + k**4 + ell**4,
                3.0 * (hk_sq + hl_sq + kl_sq),
            ),
            dim=-1,
        )
    if laue_group == "31m":
        return torch.stack((
            (h*h + h*k + k*k)**2,
            ell**4,
            3.0 * ell*ell * (h*h + h*k + k*k),
            4.0 * h * k * ell * (h + k),
        ), dim=-1)
    if laue_group in {"6/m", "6/mmm", "3m1", "3"}:
        return torch.stack(
            (
                h**4 + k**4 + 2.0 * k * h**3 + 2.0 * h * k**3 + 3.0 * hk_sq,
                ell**4,
                3.0 * (hl_sq + kl_sq + h * k * ell**2),
            ),
            dim=-1,
        )
    if laue_group in {"3R", "3mR"}:
        return torch.stack(
            (
                h**4 + k**4 + ell**4,
                3.0 * (hk_sq + hl_sq + kl_sq),
                2.0 * (h * ell**3 + ell * k**3 + k * h**3)
                + 2.0 * (ell * h**3 + k * ell**3 + h * k**3),
                4.0 * (k * ell * h**2 + h * ell * k**2 + h * k * ell**2),
            ),
            dim=-1,
        )
    if laue_group in {"4/m", "4/mmm"}:
        return torch.stack(
            (
                h**4 + k**4,
                ell**4,
                3.0 * hk_sq,
                3.0 * (hl_sq + kl_sq),
            ),
            dim=-1,
        )
    if laue_group in {"mmm"}:
        return torch.stack(
            (
                h**4,
                k**4,
                ell**4,
                3.0 * hk_sq,
                3.0 * hl_sq,
                3.0 * kl_sq,
            ),
            dim=-1,
        )
    if laue_group in {"2/m"}:
        base_terms = [
            h**4,
            k**4,
            ell**4,
            3.0 * hk_sq,
            3.0 * hl_sq,
            3.0 * kl_sq,
        ]
        if unique_axis == "a":
            base_terms.extend((2.0 * k * ell**3, 2.0 * ell * k**3, 4.0 * k * ell * h**2))
        elif unique_axis == "b":
            base_terms.extend((2.0 * ell * h**3, 2.0 * h * ell**3, 4.0 * h * ell * k**2))
        else:
            base_terms.extend((2.0 * h * k**3, 2.0 * k * h**3, 4.0 * h * k * ell**2))
        return torch.stack(tuple(base_terms), dim=-1)
    return torch.stack(
        (
            h**4,
            k**4,
            ell**4,
            3.0 * hk_sq,
            3.0 * hl_sq,
            3.0 * kl_sq,
            2.0 * k * h**3,
            2.0 * h * ell**3,
            2.0 * ell * k**3,
            2.0 * h * k**3,
            2.0 * ell * h**3,
            2.0 * k * ell**3,
            4.0 * k * ell * h**2,
            4.0 * h * ell * k**2,
            4.0 * k * h * ell**2,
        ),
        dim=-1,
    )

def _resolve_generalized_mustrain_metadata(
    self,
    sample_profile: dict[str, Any],
    *,
    batch_size: int,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, str, tuple[str, ...]]]:
    laue_group = sample_profile.get("mustrain_laue_group")
    unique_axis = str(sample_profile.get("mustrain_unique_axis", "")).strip().lower()
    if laue_group is not None:
        laue_str = str(laue_group).strip()
        return [
            (laue_str, unique_axis, self.get_mustrain_names(laue_group=laue_str, unique_axis=unique_axis))
        ] * batch_size

    explicit_spacegroup_number = sample_profile.get("mustrain_spacegroup_number")
    if explicit_spacegroup_number is not None:
        metadata = get_spacegroup_metadata(int(explicit_spacegroup_number))
        names = self.get_mustrain_names(
            laue_group=metadata.laue,
            unique_axis=metadata.unique_axis,
        )
        return [(metadata.laue, metadata.unique_axis, names)] * batch_size

    symbol_list = self._expand_spacegroup_symbols(spacegroup_symbol, batch_size=batch_size)
    if any(symbol is None for symbol in symbol_list):
        raise ValueError(
            "generalized CW sample mustrain_type requires spacegroup_symbol, "
            "mustrain_spacegroup_number, or mustrain_laue_group"
        )

    resolved: list[tuple[str, str, tuple[str, ...]]] = []
    for symbol in symbol_list:
        assert symbol is not None
        metadata = self.get_spacegroup_metadata_for_symbol(symbol)
        resolved.append(
            (
                metadata.laue,
                metadata.unique_axis,
                self.get_mustrain_names(
                    laue_group=metadata.laue,
                    unique_axis=metadata.unique_axis,
                ),
            )
        )
    return resolved

def _prepare_generalized_mustrain_coefficients(
    self,
    coefficient_values: Any,
    *,
    coefficient_names: list[tuple[str, ...]],
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(coefficient_names)
    if isinstance(coefficient_values, dict):
        if ("S400", "S004", "S202", "S211") in coefficient_names:
            # A shared mixed-batch dictionary may contain the other groups'
            # coefficients, but it must never silently reinterpret old S301.
            normalized = [str(key).upper() for key in coefficient_values]
            allowed = {name for names in coefficient_names for name in names}
            if "S301" in normalized and ("S301" not in allowed or "S211" not in normalized):
                raise ValueError("31m requires explicit S211; obsolete S301 is not an alias, even when zero")
            if len(set(normalized)) != len(normalized) or not set(normalized).issubset(allowed):
                raise ValueError("31m mustrain_coefficients contain unknown or ambiguous coefficient keys")
            if len({len(names) for names in coefficient_names}) != 1:
                raise ValueError("batched generalized mustrain requires a consistent coefficient count")
        rows = []
        for names in coefficient_names:
            rows.append(
                [
                    float(coefficient_values.get(name, coefficient_values.get(name.lower(), 0.0)))
                    for name in names
                ]
            )
        return torch.tensor(rows, dtype=reference.dtype, device=reference.device)

    if isinstance(coefficient_values, torch.Tensor):
        coeff_tensor = coefficient_values.to(device=reference.device, dtype=reference.dtype)
    else:
        coeff_tensor = torch.tensor(coefficient_values, dtype=reference.dtype, device=reference.device)

    if coeff_tensor.ndim == 1:
        coeff_tensor = coeff_tensor.unsqueeze(0)
    if coeff_tensor.ndim != 2:
        raise ValueError("generalized mustrain_coefficients must be one- or two-dimensional")
    if coeff_tensor.shape[0] == 1 and batch_size != 1:
        coeff_tensor = coeff_tensor.expand(batch_size, -1)
    if coeff_tensor.shape[0] != batch_size:
        raise ValueError("generalized mustrain_coefficients batch dimension must match peak_hkls batch dimension")

    expected_terms = {len(names) for names in coefficient_names}
    if len(expected_terms) != 1:
        raise ValueError("batched generalized mustrain requires a consistent coefficient count")
    expected_count = next(iter(expected_terms))
    if coeff_tensor.shape[1] != expected_count:
        raise ValueError(
            f"generalized mustrain_coefficients must have {expected_count} terms for "
            f"{coefficient_names[0]}"
        )
    return coeff_tensor

def _expand_six_term_symmetric_matrix(coefficients: torch.Tensor) -> torch.Tensor:
    c11 = coefficients[..., 0]
    c22 = coefficients[..., 1]
    c33 = coefficients[..., 2]
    c12 = coefficients[..., 3]
    c13 = coefficients[..., 4]
    c23 = coefficients[..., 5]
    row0 = torch.stack((c11, c12, c13), dim=-1)
    row1 = torch.stack((c12, c22, c23), dim=-1)
    row2 = torch.stack((c13, c23, c33), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)

def _prepare_generalized_size_coefficients(
    self,
    coefficient_values: Any,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size = reference.shape[0]
    if isinstance(coefficient_values, dict):
        name_map = (
            ("S11", "s11", "Size;0", "size;0", "0"),
            ("S22", "s22", "Size;1", "size;1", "1"),
            ("S33", "s33", "Size;2", "size;2", "2"),
            ("S12", "s12", "Size;3", "size;3", "3"),
            ("S13", "s13", "Size;4", "size;4", "4"),
            ("S23", "s23", "Size;5", "size;5", "5"),
        )
        row = []
        for aliases in name_map:
            value = 0.0
            for alias in aliases:
                if alias in coefficient_values:
                    value = float(coefficient_values[alias])
                    break
            row.append(value)
        rows = [row] * batch_size
        return torch.tensor(rows, dtype=reference.dtype, device=reference.device)

    if isinstance(coefficient_values, torch.Tensor):
        coeff_tensor = coefficient_values.to(device=reference.device, dtype=reference.dtype)
    else:
        coeff_tensor = torch.tensor(coefficient_values, dtype=reference.dtype, device=reference.device)

    if coeff_tensor.ndim == 1:
        coeff_tensor = coeff_tensor.unsqueeze(0)
    if coeff_tensor.ndim != 2:
        raise ValueError("generalized size_coefficients must be one- or two-dimensional")
    if coeff_tensor.shape[0] == 1 and batch_size != 1:
        coeff_tensor = coeff_tensor.expand(batch_size, -1)
    if coeff_tensor.shape[0] != batch_size:
        raise ValueError("generalized size_coefficients batch dimension must match peak_hkls batch dimension")
    if coeff_tensor.shape[1] != 6:
        raise ValueError("generalized size_coefficients must have 6 terms [S11,S22,S33,S12,S13,S23]")
    return coeff_tensor

def _prepare_profile_geometry_tensor(
    self,
    value: torch.Tensor | np.ndarray | list[float] | list[list[float]],
    *,
    name: str,
    expected_last_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=self.device, dtype=dtype)
    else:
        tensor = torch.tensor(value, dtype=dtype, device=self.device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] != expected_last_dim:
        raise ValueError(f"{name} must have shape (batch, n, {expected_last_dim}) or (n, {expected_last_dim})")
    return tensor

def _prepare_profile_axis_tensor(
    self,
    axis_hkl: torch.Tensor | np.ndarray | list[float] | list[list[float]],
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(axis_hkl, torch.Tensor):
        axis_tensor = axis_hkl.to(device=self.device, dtype=dtype)
    else:
        axis_tensor = torch.tensor(axis_hkl, dtype=dtype, device=self.device)
    if axis_tensor.ndim == 1:
        axis_tensor = axis_tensor.reshape(1, 1, 3)
    elif axis_tensor.ndim == 2 and axis_tensor.shape[-1] == 3:
        axis_tensor = axis_tensor.unsqueeze(1)
    if axis_tensor.ndim != 3 or axis_tensor.shape[-1] != 3:
        raise ValueError("axis_hkl must have shape (3,), (batch, 3), or (batch, 1, 3)")
    if axis_tensor.shape[0] == 1 and batch_size != 1:
        axis_tensor = axis_tensor.expand(batch_size, -1, -1)
    if axis_tensor.shape[0] != batch_size:
        raise ValueError("axis_hkl batch dimension must match peak_hkls batch dimension")
    if not torch.any(torch.abs(axis_tensor.reshape(axis_tensor.shape[0], -1)) > 0.0):
        raise ValueError("uniaxial sample broadening requires a non-zero axis_hkl")
    return axis_tensor

def calculate_reciprocal_axis_angles(
    self,
    peak_hkls: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]],
    cell_matrix: torch.Tensor | np.ndarray | list[list[float]],
    *,
    axis_hkl: torch.Tensor | np.ndarray | list[float] | list[list[float]],
    return_sin_sq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Tensor operands opt into higher precision; list/NumPy defaults stay float32.
    geometry_dtype = torch.float32
    for value in (peak_hkls, cell_matrix, axis_hkl):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            geometry_dtype = torch.promote_types(geometry_dtype, value.dtype)
    hkl_tensor = self._prepare_profile_geometry_tensor(
        peak_hkls,
        name="peak_hkls",
        expected_last_dim=3,
        dtype=geometry_dtype,
    )
    cell_tensor = self._prepare_profile_geometry_tensor(
        cell_matrix,
        name="cell_matrix",
        expected_last_dim=3,
        dtype=geometry_dtype,
    )
    if cell_tensor.shape[1] != 3:
        raise ValueError("cell_matrix must have shape (batch, 3, 3) or (3, 3)")
    if cell_tensor.shape[0] == 1 and hkl_tensor.shape[0] != 1:
        cell_tensor = cell_tensor.expand(hkl_tensor.shape[0], -1, -1)
    if hkl_tensor.shape[0] != cell_tensor.shape[0]:
        raise ValueError("peak_hkls and cell_matrix batch dimensions must match")

    axis_tensor = self._prepare_profile_axis_tensor(
        axis_hkl,
        batch_size=hkl_tensor.shape[0],
        dtype=hkl_tensor.dtype,
    )

    recip_latt = torch.transpose(torch.linalg.inv(cell_tensor), 1, 2)
    h_vecs = torch.matmul(hkl_tensor, recip_latt)
    axis_vecs = torch.matmul(axis_tensor, recip_latt)
    if axis_vecs.shape[1] == 1 and h_vecs.shape[1] != 1:
        axis_vecs = axis_vecs.expand(-1, h_vecs.shape[1], -1)
    if axis_vecs.shape[1] != h_vecs.shape[1]:
        raise ValueError("axis_hkl must provide either one axis per batch or one axis per peak")

    h_norm = torch.linalg.norm(h_vecs, dim=2)
    axis_norm = torch.linalg.norm(axis_vecs, dim=2)
    denom = torch.clamp(h_norm * axis_norm, min=1.0e-8)
    cos_p = torch.sum(h_vecs * axis_vecs, dim=2) / denom
    cos_p = torch.clamp(cos_p, min=-1.0, max=1.0)
    sin_sq = torch.clamp(1.0 - torch.square(cos_p), min=0.0)
    if return_sin_sq:
        return cos_p, sin_sq
    return cos_p, torch.sqrt(sin_sq)

def calculate_cw_polarization_factor(
    self,
    two_thetas: torch.Tensor,
    *,
    instrument: dict[str, float | int | str] | None = None,
) -> torch.Tensor:
    if not instrument or "Polariz." not in instrument:
        return 1.0 + torch.square(torch.cos(torch.deg2rad(2.0 * two_thetas)))
    polarization = self._as_tensor_scalar(instrument.get("Polariz.", 0.0), two_thetas)
    azimuth = self._as_tensor_scalar(instrument.get("Azimuth", 0.0), two_thetas)
    cos_azm_sq = torch.square(torch.cos(torch.deg2rad(azimuth)))
    sin_azm_sq = torch.square(torch.sin(torch.deg2rad(azimuth)))
    cos_tth_sq = torch.square(torch.cos(torch.deg2rad(two_thetas)))
    return (
        ((1.0 - polarization) * cos_azm_sq + polarization * sin_azm_sq) * cos_tth_sq
        + (1.0 - polarization) * sin_azm_sq
        + polarization * cos_azm_sq
    )

def calculate_cw_absorption_factor(
    self,
    two_thetas: torch.Tensor,
    *,
    instrument: dict[str, float | int | str] | None = None,
    sample_parameters: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not sample_parameters:
        return torch.ones_like(two_thetas)

    geometry = str(sample_parameters.get("geometry", "Bragg-Brentano"))
    if "Bragg" in geometry:
        surf_rough_a = self._as_tensor_scalar(sample_parameters.get("surf_rough_a", 0.0), two_thetas)
        surf_rough_b = self._as_tensor_scalar(sample_parameters.get("surf_rough_b", 0.0), two_thetas)
        sin_half = torch.clamp(torch.sin(torch.deg2rad(two_thetas / 2.0)), min=1.0e-6)
        t1 = torch.exp(-surf_rough_b / sin_half)
        t2 = surf_rough_a + (1.0 - surf_rough_a) * torch.exp(-surf_rough_b)
        return (surf_rough_a + (1.0 - surf_rough_a) * t1) / torch.clamp(t2, min=1.0e-6)

    absorption_value = sample_parameters.get("absorption")
    if absorption_value is None:
        absorption_value = sample_parameters.get("absorption_coeff")
    if absorption_value is None:
        absorption_value = sample_parameters.get("mu_r", 0.0)
    if not isinstance(absorption_value, torch.Tensor) and float(absorption_value) == 0.0:
        return torch.ones_like(two_thetas) + two_thetas * 0.0
    mu_r = torch.clamp(self._as_tensor_scalar(absorption_value, two_thetas), min=0.0)
    sth_sq = torch.square(torch.sin(torch.deg2rad(two_thetas / 2.0)))

    if "Cylinder" in geometry or "Debye" in geometry:
        under3 = torch.exp(
            -(16.0 / (3.0 * np.pi)) * mu_r
            - (
                (25.99978 - 0.01911 * torch.pow(sth_sq, 0.25)) * torch.exp(-0.024551 * sth_sq)
                + 0.109561 * torch.sqrt(sth_sq)
                - 26.04556
            )
            * torch.square(mu_r)
            - (
                -0.02489
                - 0.39499 * sth_sq
                + 1.219077 * torch.pow(sth_sq, 1.5)
                - 1.31268 * torch.square(sth_sq)
                + 0.871081 * torch.pow(sth_sq, 2.5)
                - 0.2327 * torch.pow(sth_sq, 3.0)
            )
            * torch.pow(mu_r, 3.0)
            - (0.003045 + 0.018167 * sth_sq - 0.03305 * torch.square(sth_sq)) * torch.pow(mu_r, 4.0)
        )
        over3_t1 = (
            1.433902
            + 11.07504 * sth_sq
            - 8.77629 * torch.square(sth_sq)
            + 10.02088 * torch.pow(sth_sq, 3.0)
            - 3.36778 * torch.pow(sth_sq, 4.0)
        )
        over3_t2 = (
            (0.013869 - 0.01249 * sth_sq) * torch.exp(3.27094 * sth_sq)
            + (0.337894 + 13.77317 * sth_sq) / torch.pow(1.0 + 11.53544 * sth_sq, 1.555039)
        )
        over3_t3 = (
            1.933433 / torch.pow(1.0 + 23.12967 * sth_sq, 1.686715)
            - 0.13576 * torch.sqrt(sth_sq)
            + 1.163198
        )
        over3_t4 = 0.044365 - 0.04259 / torch.pow(1.0 + 0.41051 * sth_sq, 148.4202)
        over3_base = torch.clamp(1.0 + over3_t2 * (mu_r - 3.0), min=1.0e-8)
        over3 = ((over3_t1 - over3_t4) / torch.pow(over3_base, over3_t3) + over3_t4) / 100.0
        return torch.where(mu_r <= 3.0, under3, over3)

    if "Fixed" in geometry:
        mu_t = 2.0 * mu_r
        exp_t1 = torch.exp(-mu_t)
        cos_tth = torch.clamp(torch.cos(torch.deg2rad(two_thetas)), min=1.0e-6)
        exp_t2 = torch.exp(-mu_t / cos_tth)
        tb = mu_t - mu_t / cos_tth
        return (exp_t2 - exp_t1) / torch.where(torch.abs(tb) > 1.0e-8, tb, torch.ones_like(tb))

    if "Tilting" in geometry:
        mu_t = 2.0 * mu_r
        cos_half = torch.clamp(torch.cos(torch.deg2rad(two_thetas / 2.0)), min=1.0e-6)
        return torch.exp(-mu_t / cos_half) / cos_half

    return torch.ones_like(two_thetas)

def calculate_cw_powder_extinction_factor(
    self,
    two_thetas: torch.Tensor,
    base_intensities: torch.Tensor,
    cell_matrix: torch.Tensor,
    *,
    wavelength: float,
    instrument: dict[str, float | int | str] | None = None,
    phase_parameters: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not phase_parameters:
        return torch.ones_like(two_thetas)
    extinction_value = phase_parameters.get("extinction")
    if extinction_value is None:
        extinction_value = phase_parameters.get("Extinction", 0.0)
    extinction = self._as_tensor_scalar(extinction_value, two_thetas)
    if torch.all(extinction == 0):
        return torch.ones_like(two_thetas)

    sth_sq = torch.square(torch.sin(torch.deg2rad(two_thetas / 2.0)))
    c2th = 1.0 - 2.0 * sth_sq
    cell_volume = torch.abs(torch.linalg.det(cell_matrix)).unsqueeze(1)
    flv2 = base_intensities * torch.square(wavelength / torch.clamp(cell_volume, min=1.0e-8))
    inst_type = "" if not instrument else str(instrument.get("Type", ""))
    is_xray = ("X" in inst_type.upper()) or (instrument is not None and "Polariz." in instrument) or not inst_type
    if is_xray:
        flv2 = flv2 * (0.079411 * (1.0 + torch.square(c2th)) / 2.0)
    xfac = flv2 * extinction
    exb = torch.where(
        xfac > -1.0,
        torch.rsqrt(torch.clamp(1.0 + xfac, min=1.0e-8)),
        torch.ones_like(xfac),
    )
    exl = torch.ones_like(xfac)
    series_coeffs = (-0.5, 0.25, -0.10416667, 0.036458333, -0.0109375, 2.8497409e-3)
    series_sum = torch.zeros_like(xfac)
    for power, coeff in enumerate(series_coeffs, start=1):
        series_sum = series_sum + coeff * torch.pow(torch.clamp(xfac, min=0.0), power)
    low_mask = (xfac > 0.0) & (xfac <= 1.0)
    exl = torch.where(low_mask, 1.0 + series_sum, exl)
    high_mask = xfac > 1.0
    high_value = np.sqrt(2.0 / np.pi) * (1.0 - 0.125 / torch.clamp(xfac, min=1.0e-8)) / torch.sqrt(
        torch.clamp(xfac, min=1.0e-8)
    )
    exl = torch.where(high_mask, high_value, exl)
    return exb * sth_sq + exl * (1.0 - sth_sq)

def compute_correction_layer(
    self,
    reflection_layer: ReflectionLayer,
    scattering_layer: ScatteringLayer,
    wavelength: float | None = None,
    sample_parameters: dict[str, Any] | None = None,
    instrument: dict[str, float | int | str] | None = None,
    phase_parameters: dict[str, Any] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> CorrectionLayer:
    if wavelength is None:
        wavelength = self.wavelength
    correction_wavelength = float(wavelength)
    can_reuse_exact_scattering = (
        scattering_layer.wavelength is not None
        and abs(float(scattering_layer.wavelength) - correction_wavelength) <= 1.0e-12
    )
    if (
        scattering_layer.num_atoms is not None
        and scattering_layer.site_species_occs is not None
        and scattering_layer.site_species_zs is not None
        and scattering_layer.site_species_ff_a is not None
        and scattering_layer.site_species_ff_b is not None
        and scattering_layer.site_species_ff_c is not None
        and not can_reuse_exact_scattering
    ):
        if scattering_layer.phase_layout != "legacy_global":
            _g_dot_r, structure_factors = self.compute_segment_reduce_structure_factors(
                reflection_layer,
                num_atoms=scattering_layer.num_atoms,
                frac_coords=scattering_layer.frac_coords,
                site_species_ff_a=scattering_layer.site_species_ff_a,
                site_species_ff_b=scattering_layer.site_species_ff_b,
                site_species_ff_c=scattering_layer.site_species_ff_c,
                site_species_occs=scattering_layer.site_species_occs,
                site_species_zs=scattering_layer.site_species_zs,
                site_uiso=scattering_layer.site_uiso,
                site_uaniso=scattering_layer.site_uaniso,
                wavelength=correction_wavelength,
            )
        else:
            num_atoms = scattering_layer.num_atoms
            cumulative_starts = torch.cat(
                (torch.tensor([0], device=self.device), num_atoms.cumsum(dim=0)[:-1])
            )
            complex_dtype = (
                torch.complex128
                if reflection_layer.hkl_pts.dtype == torch.float64
                else torch.complex64
            )
            structure_factors = torch.zeros(
                (num_atoms.size(0), reflection_layer.hkl_pts.size(1)),
                dtype=complex_dtype,
                device=self.device,
            )
            for i in range(num_atoms.size(0)):
                start_idx = int(cumulative_starts[i].item())
                end_idx = int((cumulative_starts[i] + num_atoms[i]).item())
                _, _, structure_factor = self.compute_exact_site_scattering_for_batch(
                    reflection_layer.recip_lengths[i],
                    reflection_layer.hkl_pts[i],
                    reflection_layer.cell_matrix[i],
                    scattering_layer.g_dot_r[i, :, start_idx:end_idx],
                    scattering_layer.site_species_ff_a[start_idx:end_idx],
                    scattering_layer.site_species_ff_b[start_idx:end_idx],
                    scattering_layer.site_species_ff_c[start_idx:end_idx],
                    scattering_layer.site_species_occs[start_idx:end_idx],
                    scattering_layer.site_species_zs[start_idx:end_idx],
                    scattering_layer.site_uiso[start_idx:end_idx]
                    if scattering_layer.site_uiso is not None
                    else None,
                    scattering_layer.site_uaniso[start_idx:end_idx]
                    if scattering_layer.site_uaniso is not None
                    else None,
                    wavelength=correction_wavelength,
                )
                structure_factors[i] = structure_factor
        base_intensities = (structure_factors * torch.conj(structure_factors)).real
    else:
        base_intensities = scattering_layer.base_intensities
    sin_theta = wavelength * reflection_layer.recip_lengths / 2
    valid_mask = torch.isfinite(sin_theta) & (torch.abs(sin_theta) <= 1.0)
    safe_sin_theta = torch.where(
        valid_mask,
        torch.clamp(sin_theta, min=-1.0, max=1.0),
        torch.zeros_like(sin_theta),
    )
    lorentz_factor = self.calculate_lorentz_factor(
        reflection_layer.recip_lengths,
        wavelength=correction_wavelength,
    )
    raw_two_thetas = (360 / pi) * torch.asin(safe_sin_theta)
    two_thetas = self.apply_cw_position_corrections(
        raw_two_thetas,
        instrument=instrument,
        sample_parameters=sample_parameters,
        phase_parameters=phase_parameters,
    )
    position_shift_deg = two_thetas - raw_two_thetas
    preferred_orientation_factor = self.calculate_preferred_orientation_factor(
        reflection_layer.hkl_pts,
        reflection_layer.cell_matrix,
        two_thetas,
        sample_parameters=sample_parameters,
        instrument=instrument,
        spacegroup_symbol=spacegroup_symbol,
    )
    polarization_factor = self.calculate_cw_polarization_factor(
        two_thetas,
        instrument=instrument,
    )
    absorption_factor = self.calculate_cw_absorption_factor(
        two_thetas,
        instrument=instrument,
        sample_parameters=sample_parameters,
    )
    extinction_factor = self.calculate_cw_powder_extinction_factor(
        two_thetas,
        base_intensities,
        reflection_layer.cell_matrix,
        wavelength=correction_wavelength,
        instrument=instrument,
        phase_parameters=phase_parameters,
    )
    reciprocal_volume = torch.reciprocal(
        torch.clamp(torch.abs(torch.linalg.det(reflection_layer.cell_matrix)), min=1.0e-8)
    ).unsqueeze(1)
    corrected_intensities = (
        base_intensities
        * reciprocal_volume
        * lorentz_factor
        * preferred_orientation_factor
        * polarization_factor
        * absorption_factor
        * extinction_factor
    )
    invalid_mask = (
        ~valid_mask
        | torch.isinf(corrected_intensities)
        | torch.isnan(corrected_intensities)
        | torch.isinf(two_thetas)
        | torch.isnan(two_thetas)
    )
    corrected_intensities = corrected_intensities.clone()
    two_thetas = two_thetas.clone()
    corrected_intensities[invalid_mask] = 0
    two_thetas[invalid_mask] = 0
    pattern = torch.stack((two_thetas, corrected_intensities))
    return CorrectionLayer(
        wavelength=correction_wavelength,
        lorentz_factor=lorentz_factor,
        preferred_orientation_factor=preferred_orientation_factor,
        two_thetas=two_thetas,
        corrected_intensities=corrected_intensities,
        pattern=pattern,
        cell_matrix=reflection_layer.cell_matrix,
        hkl_pts=reflection_layer.hkl_pts,
        polarization_factor=polarization_factor,
        absorption_factor=absorption_factor,
        extinction_factor=extinction_factor,
        position_shift_deg=position_shift_deg,
        spacegroup_symbol=spacegroup_symbol,
    )

def caglioti_fwhm(self, two_thetas: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
    rad_theta = torch.deg2rad(two_thetas / 2)
    return torch.sqrt(
        uvw[:, 0].unsqueeze(1) * torch.square(torch.tan(rad_theta))
        + uvw[:, 1].unsqueeze(1) * torch.tan(rad_theta)
        + uvw[:, 2].unsqueeze(1)
    )

def build_pattern_tensor(
    self,
    positions: torch.Tensor | np.ndarray | list[float],
    intensities: torch.Tensor | np.ndarray | list[float],
) -> torch.Tensor:
    if isinstance(positions, torch.Tensor):
        position_tensor = positions.to(device=self.device)
    else:
        position_tensor = torch.tensor(positions, device=self.device)
    if isinstance(intensities, torch.Tensor):
        intensity_tensor = intensities.to(device=self.device)
    else:
        intensity_tensor = torch.tensor(intensities, device=self.device)

    float_dtype = torch.promote_types(position_tensor.dtype, intensity_tensor.dtype)
    position_tensor = position_tensor.to(dtype=float_dtype)
    intensity_tensor = intensity_tensor.to(dtype=float_dtype)

    if position_tensor.ndim == 1:
        position_tensor = position_tensor.unsqueeze(0)
    if intensity_tensor.ndim == 1:
        intensity_tensor = intensity_tensor.unsqueeze(0)
    if position_tensor.shape != intensity_tensor.shape:
        raise ValueError("positions and intensities must have the same shape")
    return torch.stack((position_tensor, intensity_tensor))

def compute_profile_layer_from_pattern(
    self,
    pattern: torch.Tensor,
    *,
    uvw: torch.Tensor,
    num_steps: int = 256,
    fraction_gaussian: float = 0.9,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
) -> ProfileLayer:
    pattern_domain = self.build_pattern_domain(num_steps=num_steps, domain=domain, dtype=pattern.dtype)
    fwhm = self.caglioti_fwhm(pattern[0, :, :], uvw)
    ag = 0.9394372787 / fwhm
    bg = 2.77258872224 / (fwhm**2)
    peaks = (
        torch.ones(pattern.size()[1], 1, pattern.size()[2], device=self.device)
        * pattern_domain.unsqueeze(1)
    )
    gaussian_peak = fraction_gaussian * ag.unsqueeze(1) * torch.exp(
        -bg.unsqueeze(1) * torch.square(peaks - pattern[0, :, :].unsqueeze(1))
    )
    lorentz_peak = (1 - fraction_gaussian) * fwhm.unsqueeze(1) / (
        6.28318530718
        * (
            torch.square(peaks - pattern[0, :, :].unsqueeze(1))
            + torch.square(fwhm.unsqueeze(1) / 2)
        )
    )
    histogram = torch.nansum(
        pattern[1, :, :].unsqueeze(1) * (gaussian_peak + lorentz_peak),
        axis=2,
    )
    return ProfileLayer(
        pattern_domain=pattern_domain,
        fwhm=fwhm,
        gaussian_peak=gaussian_peak,
        lorentz_peak=lorentz_peak,
        histogram=histogram,
    )

def compute_profile_layer(
    self,
    pattern: torch.Tensor,
    *,
    uvw: torch.Tensor,
    num_steps: int = 256,
    fraction_gaussian: float = 0.9,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
) -> ProfileLayer:
    return self.compute_profile_layer_from_pattern(
        pattern,
        uvw=uvw,
        num_steps=num_steps,
        fraction_gaussian=fraction_gaussian,
        domain=domain,
    )

def get_cw_sig(
    self,
    two_thetas: torch.Tensor,
    instrument: dict[str, float | int],
) -> torch.Tensor:
    tan_theta = torch.tan(torch.deg2rad(two_thetas / 2.0))
    u_term = self._as_tensor_scalar(instrument["U"], two_thetas)
    v_term = self._as_tensor_scalar(instrument["V"], two_thetas)
    w_term = self._as_tensor_scalar(instrument["W"], two_thetas)
    return (
        u_term * torch.square(tan_theta)
        + v_term * tan_theta
        + w_term
    )

def get_cw_gam(
    self,
    two_thetas: torch.Tensor,
    instrument: dict[str, float | int],
) -> torch.Tensor:
    half_angle_rad = torch.deg2rad(two_thetas / 2.0)
    x_term = self._as_tensor_scalar(instrument["X"], two_thetas)
    y_term = self._as_tensor_scalar(instrument["Y"], two_thetas)
    z_term = self._as_tensor_scalar(instrument.get("Z", 0.0), two_thetas)
    return (
        x_term / torch.cos(half_angle_rad)
        + y_term * torch.tan(half_angle_rad)
        + z_term
    )

def get_cw_sample_profile_terms(
    self,
    two_thetas: torch.Tensor,
    *,
    wavelength: float | torch.Tensor,
    sample_profile: dict[str, Any] | None = None,
    peak_hkls: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]] | None = None,
    cell_matrix: torch.Tensor | np.ndarray | list[list[float]] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_sig = torch.zeros_like(two_thetas)
    sample_gam = torch.zeros_like(two_thetas)
    if not sample_profile:
        return sample_sig, sample_gam

    size_type = self._canonicalize_cw_profile_mode(
        sample_profile.get("size_type", "none"),
        field_name="size_type",
        allow_generalized=True,
    )
    mustrain_type = self._canonicalize_cw_profile_mode(
        sample_profile.get("mustrain_type", "none"),
        field_name="mustrain_type",
        allow_generalized=True,
    )

    ateln2 = 8.0 * np.log(2.0)
    half_angle_rad = torch.deg2rad(two_thetas / 2.0)
    safe_cos_theta = torch.clamp(torch.cos(half_angle_rad), min=1.0e-6)
    safe_sin_theta = torch.clamp(torch.sin(half_angle_rad), min=1.0e-6)
    tan_theta = torch.tan(half_angle_rad)
    wavelength_tensor = self._as_tensor_scalar(wavelength, two_thetas)
    if size_type in {"uniaxial", "generalized"}:
        if peak_hkls is None or cell_matrix is None:
            raise ValueError(
                "CW sample size broadening requires peak_hkls and cell_matrix"
            )
    if mustrain_type == "uniaxial":
        if peak_hkls is None or cell_matrix is None:
            raise ValueError(
                "uniaxial CW sample broadening requires peak_hkls and cell_matrix"
            )

    if size_type == "isotropic":
        size_i = torch.clamp(
            self._as_tensor_scalar(sample_profile.get("size_i", 0.0), two_thetas),
            min=1.0e-6,
        )
        size_mix = self._as_tensor_scalar(sample_profile.get("size_mix", 1.0), two_thetas)
        size_gamma = (1.8 * wavelength_tensor) / (np.pi * size_i * safe_cos_theta)
        sample_gam = sample_gam + size_gamma * size_mix
        sample_sig = sample_sig + torch.square(size_gamma * (1.0 - size_mix))
    elif size_type == "uniaxial":
        size_a_value = sample_profile.get("size_a")
        if size_a_value is None:
            raise ValueError('uniaxial CW sample size_type requires "size_a"')
        axis_cos, axis_sin_sq = self.calculate_reciprocal_axis_angles(
            peak_hkls,
            cell_matrix,
            axis_hkl=sample_profile.get("size_axis_hkl", [0, 0, 1]),
            return_sin_sq=True,
        )
        size_i = torch.clamp(
            self._as_tensor_scalar(sample_profile.get("size_i", 0.0), two_thetas),
            min=1.0e-6,
        )
        size_a = torch.clamp(
            self._as_tensor_scalar(size_a_value, two_thetas),
            min=1.0e-6,
        )
        size_mix = self._as_tensor_scalar(sample_profile.get("size_mix", 1.0), two_thetas)
        size_gamma = (1.8 * wavelength_tensor) / (np.pi * size_i * size_a * safe_cos_theta)
        # Avoid differentiating sqrt(sin_sq) at axis-parallel reflections.
        size_gamma = size_gamma * torch.sqrt(
            axis_sin_sq * torch.square(size_a) + torch.square(axis_cos * size_i)
        )
        sample_gam = sample_gam + size_gamma * size_mix
        sample_sig = sample_sig + torch.square(size_gamma * (1.0 - size_mix))
    elif size_type == "generalized":
        coefficient_values = sample_profile.get("size_coefficients")
        if coefficient_values is None:
            coefficient_values = sample_profile.get("size_coeffs")
        if coefficient_values is None:
            raise ValueError(
                'generalized CW sample size_type requires "size_coefficients" or "size_coeffs"'
            )

        hkl_tensor = self._prepare_profile_geometry_tensor(
            peak_hkls,
            name="peak_hkls",
            expected_last_dim=3,
            dtype=two_thetas.dtype,
        )
        cell_tensor = self._prepare_profile_geometry_tensor(
            cell_matrix,
            name="cell_matrix",
            expected_last_dim=3,
            dtype=two_thetas.dtype,
        )
        if cell_tensor.shape[1:] != (3, 3):
            raise ValueError("cell_matrix must have shape (batch, 3, 3) or (3, 3)")
        if hkl_tensor.shape[0] != two_thetas.shape[0]:
            raise ValueError("peak_hkls batch dimension must match two_thetas batch dimension")
        if cell_tensor.shape[0] == 1 and hkl_tensor.shape[0] != 1:
            cell_tensor = cell_tensor.expand(hkl_tensor.shape[0], -1, -1)
        if cell_tensor.shape[0] != hkl_tensor.shape[0]:
            raise ValueError("peak_hkls and cell_matrix batch dimensions must match")

        coeff_tensor = self._prepare_generalized_size_coefficients(
            coefficient_values,
            reference=two_thetas,
        )
        recip_latt = torch.transpose(torch.linalg.inv(cell_tensor), 1, 2)
        reciprocal_vectors = torch.matmul(hkl_tensor, recip_latt)
        reciprocal_norm = torch.clamp(
            torch.linalg.norm(reciprocal_vectors, dim=2, keepdim=True),
            min=1.0e-12,
        )
        reciprocal_dirs = reciprocal_vectors / reciprocal_norm
        size_tensor = self._expand_six_term_symmetric_matrix(coeff_tensor)
        directional_size = torch.einsum(
            "bni,bij,bnj->bn",
            reciprocal_dirs,
            size_tensor,
            reciprocal_dirs,
        )
        size_length = torch.rsqrt(torch.clamp(directional_size, min=1.0e-12))
        size_mix = self._as_tensor_scalar(sample_profile.get("size_mix", 1.0), two_thetas)
        size_gamma = (1.8 * wavelength_tensor) / (np.pi * safe_cos_theta * size_length)
        sample_gam = sample_gam + size_gamma * size_mix
        sample_sig = sample_sig + torch.square(size_gamma * (1.0 - size_mix))

    if mustrain_type == "isotropic":
        mustrain_i = torch.clamp(
            self._as_tensor_scalar(sample_profile.get("mustrain_i", 0.0), two_thetas),
            min=0.0,
        )
        mustrain_mix = self._as_tensor_scalar(sample_profile.get("mustrain_mix", 1.0), two_thetas)
        strain_gamma = 0.018 * mustrain_i * tan_theta / np.pi
        sample_gam = sample_gam + strain_gamma * mustrain_mix
        sample_sig = sample_sig + torch.square(strain_gamma * (1.0 - mustrain_mix))
    elif mustrain_type == "uniaxial":
        mustrain_a_value = sample_profile.get("mustrain_a")
        if mustrain_a_value is None:
            raise ValueError('uniaxial CW sample mustrain_type requires "mustrain_a"')
        axis_cos, axis_sin_sq = self.calculate_reciprocal_axis_angles(
            peak_hkls,
            cell_matrix,
            axis_hkl=sample_profile.get("mustrain_axis_hkl", [0, 0, 1]),
            return_sin_sq=True,
        )
        mustrain_i = torch.clamp(
            self._as_tensor_scalar(sample_profile.get("mustrain_i", 0.0), two_thetas),
            min=0.0,
        )
        mustrain_a = torch.clamp(
            self._as_tensor_scalar(mustrain_a_value, two_thetas),
            min=0.0,
        )
        mustrain_mix = self._as_tensor_scalar(sample_profile.get("mustrain_mix", 1.0), two_thetas)
        strain_gamma = 0.018 * mustrain_i * mustrain_a * tan_theta / np.pi
        # Keep the 1e-8 denominator floor without sqrt(0)'s singular backward.
        strain_gamma = strain_gamma / torch.sqrt(
            torch.clamp(
                torch.square(mustrain_i * axis_cos) + torch.square(mustrain_a) * axis_sin_sq,
                min=1.0e-16,
            )
        )
        sample_gam = sample_gam + strain_gamma * mustrain_mix
        sample_sig = sample_sig + torch.square(strain_gamma * (1.0 - mustrain_mix))
    elif mustrain_type == "generalized":
        if peak_hkls is None:
            raise ValueError("generalized CW sample broadening requires peak_hkls")
        coefficient_values = sample_profile.get("mustrain_coefficients")
        if coefficient_values is None:
            coefficient_values = sample_profile.get("mustrain_coeffs")
        if coefficient_values is None:
            raise ValueError(
                'generalized CW sample mustrain_type requires "mustrain_coefficients" '
                'or "mustrain_coeffs"'
            )

        hkl_tensor = self._prepare_profile_geometry_tensor(
            peak_hkls,
            name="peak_hkls",
            expected_last_dim=3,
            dtype=two_thetas.dtype,
        )
        if hkl_tensor.shape[0] != two_thetas.shape[0]:
            raise ValueError("peak_hkls batch dimension must match two_thetas batch dimension")

        metadata = self._resolve_generalized_mustrain_metadata(
            sample_profile,
            batch_size=hkl_tensor.shape[0],
            spacegroup_symbol=spacegroup_symbol,
        )
        coefficient_names = [names for _laue, _uniq, names in metadata]
        coeff_tensor = self._prepare_generalized_mustrain_coefficients(
            coefficient_values,
            coefficient_names=coefficient_names,
            reference=two_thetas,
        )

        term_batches = [
            self.calculate_mustrain_terms(
                hkl_tensor[batch_index],
                laue_group=laue_group,
                unique_axis=unique_axis,
            )
            for batch_index, (laue_group, unique_axis, _names) in enumerate(metadata)
        ]
        mustrain_terms = torch.stack(term_batches, dim=0)
        strain_sum = torch.sum(mustrain_terms * coeff_tensor.unsqueeze(1), dim=2)
        d_spacing = wavelength_tensor / (2.0 * safe_sin_theta)
        mustrain_mix = self._as_tensor_scalar(sample_profile.get("mustrain_mix", 1.0), two_thetas)
        strain_gamma = 0.018 * torch.square(d_spacing) * tan_theta * torch.sqrt(
            torch.clamp(strain_sum, min=0.0)
        ) / np.pi
        sample_gam = sample_gam + strain_gamma * mustrain_mix
        sample_sig = sample_sig + torch.square(strain_gamma * (1.0 - mustrain_mix))

    return sample_sig / ateln2, sample_gam

def get_cw_peak_window(
    self,
    two_thetas: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    shl: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaussian_sigma_deg = torch.sqrt(
        torch.clamp(sigma_sq_centideg2, min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG)
    ) / 100.0
    lorentz_gamma_deg = torch.clamp(
        gamma_centideg,
        min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG,
    ) / 100.0
    fwhm = torch.clamp(2.355 * gaussian_sigma_deg + lorentz_gamma_deg, min=1.0e-4)
    shl_tensor = torch.clamp(self._as_tensor_scalar(shl, two_thetas), min=0.002)
    fmin = 50.0 * (fwhm + shl_tensor * torch.abs(torch.cos(torch.deg2rad(two_thetas))))
    fmax = 75.0 * fwhm
    reverse_mask = two_thetas > 90.0
    low_extent = torch.where(reverse_mask, fmax, fmin)
    high_extent = torch.where(reverse_mask, fmin, fmax)
    return low_extent, high_extent

def apply_cw_position_corrections(
    self,
    two_thetas: torch.Tensor,
    *,
    instrument: dict[str, float | int] | None = None,
    sample_parameters: dict[str, float | int | str] | None = None,
    phase_parameters: dict[str, Any] | None = None,
) -> torch.Tensor:
    corrected = two_thetas
    if sample_parameters:
        geometry = str(sample_parameters.get("geometry", "Bragg-Brentano"))
        gonio_radius_mm = torch.clamp(
            self._as_tensor_scalar(sample_parameters.get("gonio_radius_mm", 200.0), two_thetas),
            min=1.0e-6,
        )
        const = 9.0e-2 / (np.pi * gonio_radius_mm)
        if "Bragg" in geometry:
            shift_microns = self._as_tensor_scalar(sample_parameters.get("shift_microns", 0.0), two_thetas)
            transparency_cm = self._as_tensor_scalar(sample_parameters.get("transparency_cm", 0.0), two_thetas)
            corrected = corrected - const * (
                4.0 * shift_microns * torch.cos(torch.deg2rad(two_thetas / 2.0))
                + transparency_cm * torch.sin(torch.deg2rad(two_thetas)) * 100.0
            )
        elif "Debye" in geometry:
            displace_x_microns = self._as_tensor_scalar(sample_parameters.get("displace_x_microns", 0.0), two_thetas)
            displace_y_microns = self._as_tensor_scalar(sample_parameters.get("displace_y_microns", 0.0), two_thetas)
            layer_disp_value = 0.0
            if phase_parameters:
                for key in ("layer_disp_microns", "layer_disp", "Layer Disp", "LayerDisp"):
                    if key in phase_parameters:
                        layer_disp_value = phase_parameters[key]
                        break
            layer_disp_microns = self._as_tensor_scalar(layer_disp_value, two_thetas)
            corrected = corrected - 2.0 * const * (
                displace_x_microns * torch.cos(torch.deg2rad(two_thetas))
                + (displace_y_microns + layer_disp_microns) * torch.sin(torch.deg2rad(two_thetas))
            )
    if instrument:
        zero_deg = self._as_tensor_scalar(instrument.get("Zero", 0.0), two_thetas)
        corrected = corrected + zero_deg
    return corrected

def apply_cw_position_corrections_to_pattern(
    self,
    pattern: torch.Tensor,
    *,
    instrument: dict[str, float | int] | None = None,
    sample_parameters: dict[str, float | int | str] | None = None,
    phase_parameters: dict[str, Any] | None = None,
) -> torch.Tensor:
    corrected_positions = self.apply_cw_position_corrections(
        pattern[0, :, :],
        instrument=instrument,
        sample_parameters=sample_parameters,
        phase_parameters=phase_parameters,
    )
    return self.build_pattern_tensor(corrected_positions, pattern[1, :, :])

def apply_cw_intensity_corrections_to_pattern(
    self,
    pattern: torch.Tensor,
    *,
    instrument: dict[str, float | int | str] | None = None,
    sample_parameters: dict[str, float | int | str] | None = None,
) -> torch.Tensor:
    if not sample_parameters:
        return pattern
    intensity_factor = self.calculate_cw_absorption_factor(
        pattern[0, :, :],
        instrument=instrument,
        sample_parameters=sample_parameters,
    )
    corrected_intensities = pattern[1, :, :] * intensity_factor
    return self.build_pattern_tensor(pattern[0, :, :], corrected_intensities)


def _tch_peak_chunk(
    pattern_domain: torch.Tensor,
    centers: torch.Tensor,
    intensities: torch.Tensor,
    gaussian_sigma_deg: torch.Tensor,
    lorentz_hwhm_deg: torch.Tensor,
    eta: torch.Tensor,
    low_extent: torch.Tensor,
    high_extent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    peak_grid = pattern_domain.reshape(1, -1, 1)
    center = centers.unsqueeze(1)
    gaussian_peak = (
        (1.0 - eta).unsqueeze(1)
        * (1.0 / (gaussian_sigma_deg.unsqueeze(1) * np.sqrt(2.0 * np.pi)))
        * torch.exp(
            -0.5
            * torch.square((peak_grid - center) / gaussian_sigma_deg.unsqueeze(1))
        )
    )
    lorentz_peak = (
        eta.unsqueeze(1)
        * (lorentz_hwhm_deg.unsqueeze(1) / np.pi)
        / (
            torch.square(peak_grid - center)
            + torch.square(lorentz_hwhm_deg.unsqueeze(1))
        )
    )
    support_mask = (
        (peak_grid >= (center - low_extent.unsqueeze(1)))
        & (peak_grid <= (center + high_extent.unsqueeze(1)))
    ).to(dtype=centers.dtype)
    gaussian_supported = gaussian_peak * support_mask
    lorentz_supported = lorentz_peak * support_mask
    histogram = torch.sum(
        intensities.unsqueeze(1) * (gaussian_supported + lorentz_supported),
        dim=2,
    )
    return histogram, gaussian_supported, lorentz_supported, support_mask


def _tch_peak_chunk_histogram(
    pattern_domain: torch.Tensor,
    centers: torch.Tensor,
    intensities: torch.Tensor,
    gaussian_sigma_deg: torch.Tensor,
    lorentz_hwhm_deg: torch.Tensor,
    eta: torch.Tensor,
    low_extent: torch.Tensor,
    high_extent: torch.Tensor,
) -> torch.Tensor:
    return _tch_peak_chunk(
        pattern_domain,
        centers,
        intensities,
        gaussian_sigma_deg,
        lorentz_hwhm_deg,
        eta,
        low_extent,
        high_extent,
    )[0]


@lru_cache(maxsize=None)
def _topas_circles_quadrature(num_terms: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(num_terms)
    transformed = 0.5 * (nodes + 1.0)
    transformed_weights = weights * (1.0 - transformed)
    transformed_weights = transformed_weights / np.sum(transformed_weights)
    return (
        transformed.astype(np.float64, copy=False),
        transformed_weights.astype(np.float64, copy=False),
    )


def _topas_circles_axial_histogram(
    pattern_domain: torch.Tensor,
    centers: torch.Tensor,
    intensities: torch.Tensor,
    gaussian_sigma_deg: torch.Tensor,
    lorentz_hwhm_deg: torch.Tensor,
    eta: torch.Tensor,
    extent: torch.Tensor,
    epsilon_m_deg: torch.Tensor,
    *,
    num_terms: int = 16,
) -> torch.Tensor:
    """Convolve pseudo-Voigt peaks with the TOPAS ``circles_conv`` kernel.

    The documented kernel is proportional to ``sqrt(|epsilon_m / epsilon|) - 1``
    on the interval from zero to ``epsilon_m``. Substituting
    ``epsilon = epsilon_m * t**2`` removes its integrable singularity and gives
    the normalized quadrature density ``2 * (1 - t)`` on ``t in [0, 1]``.
    """

    nodes_np, weights_np = _topas_circles_quadrature(num_terms)
    nodes = torch.as_tensor(nodes_np, dtype=centers.dtype, device=centers.device)
    weights = torch.as_tensor(weights_np, dtype=centers.dtype, device=centers.device)
    histogram = torch.zeros(
        (int(centers.shape[0]), int(pattern_domain.numel())),
        dtype=centers.dtype,
        device=centers.device,
    )
    for node, weight in zip(nodes.unbind(), weights.unbind()):
        shifted_centers = centers + epsilon_m_deg * torch.square(node)
        histogram = histogram + weight * _tch_peak_chunk_histogram(
            pattern_domain,
            shifted_centers,
            intensities,
            gaussian_sigma_deg,
            lorentz_hwhm_deg,
            eta,
            extent,
            extent,
        )
    return histogram


def _topas_simple_axial_epsilon_m(
    model: Any,
    two_thetas: torch.Tensor,
    instrument: dict[str, Any],
) -> torch.Tensor | None:
    axial_receiving_slit = instrument.get("topas_simple_axial_receiving_slit_mm")
    if axial_receiving_slit is None:
        return None
    secondary_radius = instrument.get("topas_secondary_radius_mm")
    if secondary_radius is None:
        raise ValueError("TOPAS simple axial model requires a secondary radius")
    receiving_slit_tensor = torch.clamp(
        model._as_tensor_scalar(axial_receiving_slit, two_thetas), min=0.0
    )
    secondary_radius_tensor = torch.clamp(
        model._as_tensor_scalar(secondary_radius, two_thetas), min=1.0e-12
    )
    tan_two_theta = torch.tan(torch.deg2rad(two_thetas))
    safe_tan_two_theta = torch.where(
        torch.abs(tan_two_theta) < 1.0e-12,
        torch.full_like(tan_two_theta, 1.0e-12),
        tan_two_theta,
    )
    return (
        # TOPAS Simple_Axial_Model: circles_conv =
        # -0.5 * Rad * (receiving_slit / Rs)^2 / tan(2 theta).
        -0.5
        * (180.0 / np.pi)
        * torch.square(receiving_slit_tensor / secondary_radius_tensor)
        / safe_tan_two_theta
    )


def compute_topas_pv_profile_layer_from_pattern(
    self,
    pattern: torch.Tensor,
    *,
    instrument: dict[str, float | int | torch.Tensor],
    wavelength: float | None = None,
    sample_profile: dict[str, Any] | None = None,
    num_steps: int = 256,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    peak_chunk_size: int | None = None,
    return_diagnostics: bool = True,
) -> ProfileLayer:
    """Render the six-parameter TOPAS ``PV_Peak_Type`` profile.

    TOPAS defines the common pseudo-Voigt FWHM and Lorentzian fraction as
    linear functions of tan(theta) and sec(theta). Optional crystallite-size
    terms use the documented Gaussian and Lorentzian FWHM expressions before
    the final pseudo-Voigt approximation.
    """

    two_thetas = pattern[0, :, :]
    pattern_domain = self.build_pattern_domain(num_steps=num_steps, domain=domain, dtype=two_thetas.dtype)
    theta = torch.deg2rad(two_thetas / 2.0)
    tan_theta = torch.tan(theta)
    sec_theta = torch.reciprocal(torch.clamp(torch.cos(theta), min=1.0e-8))

    def parameter(name: str) -> torch.Tensor:
        if name not in instrument:
            raise ValueError(f"TOPAS PV instrument is missing {name}")
        return self._as_tensor_scalar(instrument[name], two_thetas)

    base_fwhm = (
        parameter("PV_Ha")
        + parameter("PV_Hb") * tan_theta
        + parameter("PV_Hc") * sec_theta
    )
    base_eta = torch.clamp(
        parameter("PV_Lora")
        + parameter("PV_Lorb") * tan_theta
        + parameter("PV_Lorc") * sec_theta,
        min=0.0,
        max=1.0,
    )
    fwhm = torch.clamp(base_fwhm, min=1.0e-6)
    eta = base_eta

    if sample_profile:
        if wavelength is None:
            wavelength = self.wavelength
        wavelength_tensor = self._as_tensor_scalar(float(wavelength), two_thetas)
        cs_g = sample_profile.get("topas_cs_g_nm")
        cs_l = sample_profile.get("topas_cs_l_nm")
        gaussian_size_fwhm = torch.zeros_like(two_thetas)
        lorentzian_size_fwhm = torch.zeros_like(two_thetas)
        if cs_g is not None:
            cs_g_tensor = torch.clamp(
                self._as_tensor_scalar(cs_g, two_thetas), min=1.0e-6
            )
            gaussian_size_fwhm = (
                0.1
                * (180.0 / np.pi)
                * wavelength_tensor
                * sec_theta
                / cs_g_tensor
            )
        if cs_l is not None:
            cs_l_tensor = torch.clamp(
                self._as_tensor_scalar(cs_l, two_thetas), min=1.0e-6
            )
            lorentzian_size_fwhm = (
                0.1
                * (180.0 / np.pi)
                * wavelength_tensor
                * sec_theta
                / cs_l_tensor
            )
        gaussian_combined = torch.sqrt(
            torch.square(fwhm) + torch.square(gaussian_size_fwhm)
        )
        combined = torch.clamp(
            gaussian_combined + lorentzian_size_fwhm,
            min=1.0e-6,
        )
        eta = torch.clamp(
            (eta * fwhm + lorentzian_size_fwhm) / combined,
            min=0.0,
            max=1.0,
        )
        fwhm = combined

    gaussian_sigma_deg = fwhm / 2.35482004503
    lorentz_hwhm_deg = fwhm / 2.0
    extent = 75.0 * fwhm
    epsilon_m_deg = _topas_simple_axial_epsilon_m(
        self, two_thetas, instrument
    )
    peak_count = int(pattern.shape[2])
    chunk_size = max(peak_count, 1) if peak_chunk_size is None else int(peak_chunk_size)
    if chunk_size <= 0:
        raise ValueError("peak_chunk_size must be positive")

    histogram = torch.zeros(
        (int(pattern.shape[1]), int(pattern_domain.numel())),
        dtype=pattern.dtype,
        device=pattern.device,
    )
    gaussian_chunks: list[torch.Tensor] = []
    lorentz_chunks: list[torch.Tensor] = []
    support_chunks: list[torch.Tensor] = []
    for start in range(0, peak_count, chunk_size):
        stop = min(start + chunk_size, peak_count)
        symmetric_arguments = (
            pattern_domain,
            two_thetas[:, start:stop],
            pattern[1, :, start:stop],
            gaussian_sigma_deg[:, start:stop],
            lorentz_hwhm_deg[:, start:stop],
            eta[:, start:stop],
            extent[:, start:stop],
            extent[:, start:stop],
        )
        arguments = (
            symmetric_arguments
            if epsilon_m_deg is None
            else (
                pattern_domain,
                two_thetas[:, start:stop],
                pattern[1, :, start:stop],
                gaussian_sigma_deg[:, start:stop],
                lorentz_hwhm_deg[:, start:stop],
                eta[:, start:stop],
                extent[:, start:stop],
                epsilon_m_deg[:, start:stop],
            )
        )
        checkpoint_chunk = (
            not return_diagnostics
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in arguments)
        )
        if epsilon_m_deg is not None:
            axial_function = _topas_circles_axial_histogram
            chunk_histogram = (
                checkpoint(axial_function, *arguments, use_reentrant=False)
                if checkpoint_chunk
                else axial_function(*arguments)
            )
        elif checkpoint_chunk:
            chunk_histogram = checkpoint(
                _tch_peak_chunk_histogram,
                *arguments,
                use_reentrant=False,
            )
        else:
            chunk_histogram, gaussian_chunk, lorentz_chunk, support_chunk = (
                _tch_peak_chunk(*arguments)
            )
            if return_diagnostics:
                gaussian_chunks.append(gaussian_chunk)
                lorentz_chunks.append(lorentz_chunk)
                support_chunks.append(support_chunk)
        histogram = histogram + chunk_histogram

    if return_diagnostics and gaussian_chunks:
        gaussian_peak = torch.cat(gaussian_chunks, dim=2)
        lorentz_peak = torch.cat(lorentz_chunks, dim=2)
        support_mask = torch.cat(support_chunks, dim=2)
        axial_low = (
            torch.zeros_like(extent)
            if epsilon_m_deg is None
            else torch.clamp(-epsilon_m_deg, min=0.0)
        )
        axial_high = (
            torch.zeros_like(extent)
            if epsilon_m_deg is None
            else torch.clamp(epsilon_m_deg, min=0.0)
        )
        support_lower = two_thetas - extent - axial_low
        support_upper = two_thetas + extent + axial_high
    elif return_diagnostics:
        empty = torch.empty(
            (int(pattern.shape[1]), int(pattern_domain.numel()), 0),
            dtype=pattern.dtype,
            device=pattern.device,
        )
        gaussian_peak = empty
        lorentz_peak = empty.clone()
        support_mask = empty.clone()
        axial_low = (
            torch.zeros_like(extent)
            if epsilon_m_deg is None
            else torch.clamp(-epsilon_m_deg, min=0.0)
        )
        axial_high = (
            torch.zeros_like(extent)
            if epsilon_m_deg is None
            else torch.clamp(epsilon_m_deg, min=0.0)
        )
        support_lower = two_thetas - extent - axial_low
        support_upper = two_thetas + extent + axial_high
    else:
        gaussian_peak = torch.empty(0, dtype=pattern.dtype, device=pattern.device)
        lorentz_peak = torch.empty(0, dtype=pattern.dtype, device=pattern.device)
        support_mask = None
        support_lower = None
        support_upper = None

    return ProfileLayer(
        pattern_domain=pattern_domain,
        fwhm=fwhm,
        gaussian_peak=gaussian_peak,
        lorentz_peak=lorentz_peak,
        histogram=histogram,
        sigma_sq_centideg2=torch.square(gaussian_sigma_deg * 100.0),
        gamma_centideg=fwhm * 100.0,
        eta=eta,
        support_mask=support_mask,
        support_lower_deg=support_lower,
        support_upper_deg=support_upper,
    )


def compute_tch_profile_layer_from_pattern(
    self,
    pattern: torch.Tensor,
    *,
    instrument: dict[str, float | int | torch.Tensor],
    wavelength: float | None = None,
    sample_profile: dict[str, Any] | None = None,
    peak_hkls: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]] | None = None,
    cell_matrix: torch.Tensor | np.ndarray | list[list[float]] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
    num_steps: int = 256,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    peak_chunk_size: int | None = None,
    return_diagnostics: bool = True,
) -> ProfileLayer:
    two_thetas = pattern[0, :, :]
    pattern_domain = self.build_pattern_domain(num_steps=num_steps, domain=domain, dtype=two_thetas.dtype)
    if wavelength is None:
        wavelength = self.wavelength

    sample_sig, sample_gam = self.get_cw_sample_profile_terms(
        two_thetas,
        wavelength=float(wavelength),
        sample_profile=sample_profile,
        peak_hkls=peak_hkls,
        cell_matrix=cell_matrix,
        spacegroup_symbol=spacegroup_symbol,
    )
    sigma_sq_centideg2 = torch.clamp(
        self.get_cw_sig(two_thetas, instrument) + sample_sig,
        min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG,
    )
    gamma_centideg = torch.clamp(
        self.get_cw_gam(two_thetas, instrument) + sample_gam,
        min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG,
    )

    gaussian_sigma_deg = torch.sqrt(sigma_sq_centideg2) / 100.0
    gaussian_fwhm = 2.35482004503 * gaussian_sigma_deg
    lorentz_fwhm = gamma_centideg / 100.0
    fwhm = torch.pow(
        torch.pow(gaussian_fwhm, 5)
        + 2.69269 * torch.pow(gaussian_fwhm, 4) * lorentz_fwhm
        + 2.42843 * torch.pow(gaussian_fwhm, 3) * torch.pow(lorentz_fwhm, 2)
        + 4.47163 * torch.pow(gaussian_fwhm, 2) * torch.pow(lorentz_fwhm, 3)
        + 0.07842 * gaussian_fwhm * torch.pow(lorentz_fwhm, 4)
        + torch.pow(lorentz_fwhm, 5),
        0.2,
    )
    eta = torch.clamp(
        1.36603 * (lorentz_fwhm / fwhm)
        - 0.47719 * torch.square(lorentz_fwhm / fwhm)
        + 0.11116 * torch.pow(lorentz_fwhm / fwhm, 3),
        min=0.0,
        max=1.0,
    )

    # TCH mixes Gaussian and Lorentzian components with one common FWHM.
    gaussian_sigma_deg = fwhm / 2.35482004503
    lorentz_hwhm_deg = fwhm / 2.0

    epsilon_m_deg = _topas_simple_axial_epsilon_m(
        self, two_thetas, instrument
    )
    low_extent, high_extent = self.get_cw_peak_window(
        two_thetas,
        sigma_sq_centideg2,
        gamma_centideg,
        shl=0.0 if epsilon_m_deg is not None else instrument.get("SH/L", 0.0),
    )
    peak_count = int(pattern.shape[2])
    if peak_chunk_size is None:
        chunk_size = max(peak_count, 1)
    else:
        chunk_size = int(peak_chunk_size)
        if chunk_size <= 0:
            raise ValueError("peak_chunk_size must be positive")
    histogram = torch.zeros(
        (int(pattern.shape[1]), int(pattern_domain.numel())),
        dtype=pattern.dtype,
        device=pattern.device,
    )
    gaussian_chunks: list[torch.Tensor] = []
    lorentz_chunks: list[torch.Tensor] = []
    support_chunks: list[torch.Tensor] = []
    for start in range(0, peak_count, chunk_size):
        stop = min(start + chunk_size, peak_count)
        symmetric_arguments = (
            pattern_domain,
            two_thetas[:, start:stop],
            pattern[1, :, start:stop],
            gaussian_sigma_deg[:, start:stop],
            lorentz_hwhm_deg[:, start:stop],
            eta[:, start:stop],
            low_extent[:, start:stop],
            high_extent[:, start:stop],
        )
        arguments = (
            symmetric_arguments
            if epsilon_m_deg is None
            else (
                pattern_domain,
                two_thetas[:, start:stop],
                pattern[1, :, start:stop],
                gaussian_sigma_deg[:, start:stop],
                lorentz_hwhm_deg[:, start:stop],
                eta[:, start:stop],
                torch.maximum(
                    low_extent[:, start:stop], high_extent[:, start:stop]
                ),
                epsilon_m_deg[:, start:stop],
            )
        )
        checkpoint_chunk = (
            not return_diagnostics
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in arguments)
        )
        if epsilon_m_deg is not None:
            chunk_histogram = (
                checkpoint(
                    _topas_circles_axial_histogram,
                    *arguments,
                    use_reentrant=False,
                )
                if checkpoint_chunk
                else _topas_circles_axial_histogram(*arguments)
            )
        elif checkpoint_chunk:
            chunk_histogram = checkpoint(
                _tch_peak_chunk_histogram,
                *arguments,
                use_reentrant=False,
            )
        else:
            (
                chunk_histogram,
                gaussian_chunk,
                lorentz_chunk,
                support_chunk,
            ) = _tch_peak_chunk(*arguments)
            if return_diagnostics:
                gaussian_chunks.append(gaussian_chunk)
                lorentz_chunks.append(lorentz_chunk)
                support_chunks.append(support_chunk)
        histogram = histogram + chunk_histogram
    if return_diagnostics and gaussian_chunks:
        gaussian_peak = torch.cat(gaussian_chunks, dim=2)
        lorentz_peak = torch.cat(lorentz_chunks, dim=2)
        support_mask = torch.cat(support_chunks, dim=2)
        axial_low = (
            torch.zeros_like(low_extent)
            if epsilon_m_deg is None
            else torch.clamp(-epsilon_m_deg, min=0.0)
        )
        axial_high = (
            torch.zeros_like(high_extent)
            if epsilon_m_deg is None
            else torch.clamp(epsilon_m_deg, min=0.0)
        )
        support_lower = two_thetas - low_extent - axial_low
        support_upper = two_thetas + high_extent + axial_high
    elif return_diagnostics:
        empty_peaks = torch.empty(
            (int(pattern.shape[1]), int(pattern_domain.numel()), 0),
            dtype=pattern.dtype,
            device=pattern.device,
        )
        gaussian_peak = empty_peaks
        lorentz_peak = empty_peaks.clone()
        support_mask = empty_peaks.clone()
        axial_low = (
            torch.zeros_like(low_extent)
            if epsilon_m_deg is None
            else torch.clamp(-epsilon_m_deg, min=0.0)
        )
        axial_high = (
            torch.zeros_like(high_extent)
            if epsilon_m_deg is None
            else torch.clamp(epsilon_m_deg, min=0.0)
        )
        support_lower = two_thetas - low_extent - axial_low
        support_upper = two_thetas + high_extent + axial_high
    else:
        gaussian_peak = torch.empty(0, dtype=pattern.dtype, device=pattern.device)
        lorentz_peak = torch.empty(0, dtype=pattern.dtype, device=pattern.device)
        support_mask = None
        support_lower = None
        support_upper = None
    return ProfileLayer(
        pattern_domain=pattern_domain,
        fwhm=fwhm,
        gaussian_peak=gaussian_peak,
        lorentz_peak=lorentz_peak,
        histogram=histogram,
        sigma_sq_centideg2=sigma_sq_centideg2,
        gamma_centideg=gamma_centideg,
        eta=eta,
        support_mask=support_mask,
        support_lower_deg=support_lower,
        support_upper_deg=support_upper,
    )

def compute_tch_profile_layer(
    self,
    correction_layer: CorrectionLayer,
    *,
    instrument: dict[str, float | int],
    sample_profile: dict[str, Any] | None = None,
    num_steps: int = 256,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    peak_chunk_size: int | None = None,
    return_diagnostics: bool = True,
) -> ProfileLayer:
    return self.compute_tch_profile_layer_from_pattern(
        correction_layer.pattern,
        instrument=instrument,
        wavelength=correction_layer.wavelength,
        sample_profile=sample_profile,
        peak_hkls=correction_layer.hkl_pts,
        cell_matrix=correction_layer.cell_matrix,
        spacegroup_symbol=correction_layer.spacegroup_symbol,
        num_steps=num_steps,
        domain=domain,
        peak_chunk_size=peak_chunk_size,
        return_diagnostics=return_diagnostics,
    )

def compute_fcj_profile_layer_from_pattern(
    self,
    pattern: torch.Tensor,
    *,
    instrument: dict[str, float | int],
    wavelength: float | None = None,
    sample_profile: dict[str, Any] | None = None,
    peak_hkls: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]] | None = None,
    cell_matrix: torch.Tensor | np.ndarray | list[list[float]] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
    num_steps: int = 256,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    backend: str | None = None,
    return_diagnostics: bool = True,
) -> ProfileLayer:
    two_thetas = pattern[0, :, :]
    pattern_domain = self.build_pattern_domain(num_steps=num_steps, domain=domain, dtype=two_thetas.dtype)
    if wavelength is None:
        wavelength = self.wavelength

    sample_sig, sample_gam = self.get_cw_sample_profile_terms(
        two_thetas,
        wavelength=float(wavelength),
        sample_profile=sample_profile,
        peak_hkls=peak_hkls,
        cell_matrix=cell_matrix,
        spacegroup_symbol=spacegroup_symbol,
    )
    sigma_sq_centideg2 = torch.clamp(
        self.get_cw_sig(two_thetas, instrument) + sample_sig,
        min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG,
    )
    gamma_centideg = torch.clamp(
        self.get_cw_gam(two_thetas, instrument) + sample_gam,
        min=CW_PROFILE_WIDTH_FLOOR_CENTIDEG,
    )
    shl_scalar = torch.clamp(
        self._as_tensor_scalar(instrument.get("SH/L", 0.0), pattern[0, :, :]),
        min=0.002,
    )
    if shl_scalar.ndim == 0:
        batch_shl = shl_scalar.expand(pattern.shape[1])
        peak_shl = shl_scalar.expand_as(two_thetas)
    else:
        batch_shl = shl_scalar.reshape(pattern.shape[1], -1)[:, 0]
        peak_shl = shl_scalar.reshape(pattern.shape[1], -1)
        if peak_shl.shape[1] == 1:
            peak_shl = peak_shl.expand_as(two_thetas)
        elif peak_shl.shape != two_thetas.shape:
            peak_shl = peak_shl[:, :1].expand_as(two_thetas)
    histogram = compute_fcj_profile_histogram(
        two_thetas,
        pattern[1, :, :],
        sigma_sq_centideg2,
        gamma_centideg,
        pattern_domain,
        batch_shl,
        backend=backend,
    )
    fwhm = torch.clamp(2.355 * torch.sqrt(sigma_sq_centideg2) / 100.0 + gamma_centideg / 100.0, min=1.0e-4)
    if return_diagnostics:
        support_lower_deg, support_upper_deg = compute_fcj_window_bounds(
            two_thetas,
            sigma_sq_centideg2,
            gamma_centideg,
            peak_shl,
        )
        fcj_quadrature_terms = select_fcj_num_terms(
            two_thetas,
            sigma_sq_centideg2,
            gamma_centideg,
            peak_shl,
        )
    else:
        support_lower_deg = None
        support_upper_deg = None
        fcj_quadrature_terms = None
    empty = torch.empty(0, dtype=pattern.dtype, device=pattern.device)
    return ProfileLayer(
        pattern_domain=pattern_domain,
        fwhm=fwhm,
        gaussian_peak=empty,
        lorentz_peak=empty,
        histogram=histogram,
        sigma_sq_centideg2=sigma_sq_centideg2,
        gamma_centideg=gamma_centideg,
        support_lower_deg=support_lower_deg,
        support_upper_deg=support_upper_deg,
        fcj_quadrature_terms=fcj_quadrature_terms,
    )

def compute_fcj_profile_layer(
    self,
    correction_layer: CorrectionLayer,
    *,
    instrument: dict[str, float | int],
    sample_profile: dict[str, Any] | None = None,
    num_steps: int = 256,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    backend: str | None = None,
    return_diagnostics: bool = True,
) -> ProfileLayer:
    return self.compute_fcj_profile_layer_from_pattern(
        correction_layer.pattern,
        instrument=instrument,
        wavelength=correction_layer.wavelength,
        sample_profile=sample_profile,
        peak_hkls=correction_layer.hkl_pts,
        cell_matrix=correction_layer.cell_matrix,
        spacegroup_symbol=correction_layer.spacegroup_symbol,
        num_steps=num_steps,
        domain=domain,
        backend=backend,
        return_diagnostics=return_diagnostics,
    )

def compute_weighted_profile_layer(
    self,
    reflection_layer: ReflectionLayer,
    scattering_layer: ScatteringLayer,
    *,
    wavelengths: list[float] | tuple[float, ...],
    weights: torch.Tensor | np.ndarray | list[float] | tuple[float, ...],
    uvw: torch.Tensor | None = None,
    sample_parameters: dict[str, float | int | str | list[int] | tuple[int, int, int]] | None = None,
    instrument: dict[str, float | int | str] | None = None,
    sample_profile: dict[str, Any] | None = None,
    phase_parameters: dict[str, Any] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
    profile_model: str = "pseudo_voigt",
    num_steps: int = 256,
    fraction_gaussian: float = 0.9,
    domain: torch.Tensor | np.ndarray | list[float] | None = None,
    backend: str | None = None,
) -> WeightedProfileLayer:
    if len(wavelengths) == 0:
        raise ValueError("at least one wavelength is required")
    if isinstance(weights, torch.Tensor):
        weight_tensor = weights.to(device=self.device, dtype=torch.float32)
    else:
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
    if weight_tensor.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weight_tensor.numel() != len(wavelengths):
        raise ValueError("weights must match the number of wavelengths")
    profile_model_key = str(profile_model).strip().lower()
    if profile_model_key in {"pseudo_voigt", "pseudo-voigt", "pvoigt", "pv"}:
        profile_model_key = "pseudo_voigt"
    elif profile_model_key in {"tch", "thompson_cox_hastings", "thompson-cox-hastings"}:
        profile_model_key = "tch"
    elif profile_model_key in {"fcj", "fcj_exact", "exact_fcj", "exact-fcj"}:
        profile_model_key = "fcj_exact"
    else:
        raise ValueError(f"unsupported weighted profile_model: {profile_model}")
    if profile_model_key == "pseudo_voigt" and uvw is None:
        raise ValueError("uvw is required for pseudo_voigt weighted profiles")
    if profile_model_key in {"tch", "fcj_exact"} and instrument is None:
        raise ValueError(f"instrument is required for {profile_model_key} weighted profiles")

    component_corrections: list[CorrectionLayer] = []
    component_profiles: list[ProfileLayer] = []
    combined_histogram = None
    pattern_domain = None

    for wavelength, weight in zip(wavelengths, weight_tensor):
        correction_layer = self.compute_correction_layer(
            reflection_layer,
            scattering_layer,
            wavelength=float(wavelength),
            sample_parameters=sample_parameters,
            instrument=instrument,
            phase_parameters=phase_parameters,
            spacegroup_symbol=spacegroup_symbol,
        )
        if profile_model_key == "pseudo_voigt":
            assert uvw is not None
            profile_layer = self.compute_profile_layer(
                correction_layer.pattern,
                uvw=uvw,
                num_steps=num_steps,
                fraction_gaussian=fraction_gaussian,
                domain=domain,
            )
        elif profile_model_key == "tch":
            assert instrument is not None
            profile_layer = self.compute_tch_profile_layer(
                correction_layer,
                instrument=instrument,
                sample_profile=sample_profile,
                num_steps=num_steps,
                domain=domain,
            )
        else:
            assert instrument is not None
            profile_layer = self.compute_fcj_profile_layer(
                correction_layer,
                instrument=instrument,
                sample_profile=sample_profile,
                num_steps=num_steps,
                domain=domain,
                backend=backend,
            )
        component_corrections.append(correction_layer)
        component_profiles.append(profile_layer)
        weighted_histogram = weight * profile_layer.histogram
        if combined_histogram is None:
            combined_histogram = weighted_histogram
            pattern_domain = profile_layer.pattern_domain
        else:
            combined_histogram = combined_histogram + weighted_histogram

    if combined_histogram is None or pattern_domain is None:
        raise RuntimeError("failed to build weighted profile")

    return WeightedProfileLayer(
        pattern_domain=pattern_domain,
        wavelengths=[float(wavelength) for wavelength in wavelengths],
        weights=weight_tensor,
        component_corrections=component_corrections,
        component_profiles=component_profiles,
        histogram=combined_histogram,
    )

__all__ = [
    "_canonicalize_cw_profile_mode",
    "_expand_spacegroup_symbols",
    "get_spacegroup_metadata_for_symbol",
    "get_mustrain_names",
    "calculate_mustrain_terms",
    "_resolve_generalized_mustrain_metadata",
    "_prepare_generalized_mustrain_coefficients",
    "_expand_six_term_symmetric_matrix",
    "_prepare_generalized_size_coefficients",
    "_prepare_profile_geometry_tensor",
    "_prepare_profile_axis_tensor",
    "calculate_reciprocal_axis_angles",
    "calculate_cw_polarization_factor",
    "calculate_cw_absorption_factor",
    "calculate_cw_powder_extinction_factor",
    "compute_correction_layer",
    "caglioti_fwhm",
    "build_pattern_tensor",
    "compute_profile_layer_from_pattern",
    "compute_profile_layer",
    "get_cw_sig",
    "get_cw_gam",
    "get_cw_sample_profile_terms",
    "get_cw_peak_window",
    "apply_cw_position_corrections",
    "apply_cw_position_corrections_to_pattern",
    "apply_cw_intensity_corrections_to_pattern",
    "compute_topas_pv_profile_layer_from_pattern",
    "compute_tch_profile_layer_from_pattern",
    "compute_tch_profile_layer",
    "compute_fcj_profile_layer_from_pattern",
    "compute_fcj_profile_layer",
    "compute_weighted_profile_layer",
]
