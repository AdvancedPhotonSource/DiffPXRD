from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

try:
    from ..spacegroup_contract import get_spacegroup_metadata
except ImportError:
    from spacegroup_contract import get_spacegroup_metadata

def calculate_lorentz_factor(
    self,
    recip_lengths: torch.Tensor,
    wavelength: float | None = None,
) -> torch.Tensor:
    if wavelength is None:
        wavelength = self.wavelength
    sin_theta = wavelength * recip_lengths / 2
    valid_mask = torch.isfinite(sin_theta) & (torch.abs(sin_theta) <= 1.0)
    safe_sin_theta = torch.where(
        valid_mask,
        torch.clamp(sin_theta, min=-1.0, max=1.0),
        torch.zeros_like(sin_theta),
    )
    theta = torch.asin(safe_sin_theta)
    mask = valid_mask & (theta != 0)
    safe_theta = torch.where(mask, theta, torch.tensor(0.1, device=theta.device))
    # Use the CW powder Lorentz factor 1 / (2 sin^2(theta) cos(theta)).
    lorentz_factor = 0.5 / (torch.square(torch.sin(safe_theta)) * torch.cos(safe_theta))
    return torch.where(mask, lorentz_factor, torch.tensor(0.0, device=theta.device))

def calculate_march_dollase_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    *,
    sample_parameters: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not sample_parameters:
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)
    model = str(sample_parameters.get("preferred_orientation_model", "none")).lower()
    if model not in {"march_dollase", "md"}:
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)
    axis_hkl = sample_parameters.get("preferred_orientation_axis_hkl", [0, 0, 0])
    axis_tensor = torch.as_tensor(axis_hkl, dtype=torch.float32, device=self.device)
    valid_shape = (
        axis_tensor.shape == (3,)
        or (axis_tensor.ndim == 2 and axis_tensor.shape[-1] == 3)
        or (axis_tensor.ndim == 3 and axis_tensor.shape[-1] == 3)
    )
    if not valid_shape or not torch.any(torch.abs(axis_tensor.reshape(axis_tensor.shape[0] if axis_tensor.ndim > 1 else 1, -1)) > 0.0):
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)
    ratio = self._as_tensor_scalar(sample_parameters.get("preferred_orientation_ratio", 1.0), hkl_pts[..., 0])
    ratio = torch.clamp(ratio, min=1.0e-4)

    cos_p, sin_sq = self.calculate_reciprocal_axis_angles(
        hkl_pts,
        cell_matrix,
        axis_hkl=axis_hkl,
        return_sin_sq=True,
    )
    a_term = torch.reciprocal(
        torch.sqrt(torch.square(ratio * cos_p) + sin_sq / ratio)
    )
    return torch.pow(a_term, 3)

def _canonicalize_sample_symmetry(value: Any) -> str:
    sample_symmetry = str(value).strip().lower()
    if sample_symmetry in {"", "0", "cylindrical", "cylinder"}:
        return "0"
    if sample_symmetry in {"-1", "1", "none", "triclinic"}:
        return "-1"
    if sample_symmetry in {"2/m", "monoclinic", "shear - 2/m"}:
        return "2/m"
    if sample_symmetry in {"mmm", "orthorhombic", "rolling - mmm"}:
        return "mmm"
    if sample_symmetry in {"4/m", "tetragonal"}:
        return "4/m"
    if sample_symmetry in {"4/mmm", "tetragonal_high"}:
        return "4/mmm"
    if sample_symmetry in {"3", "3r", "trigonal"}:
        return "3"
    if sample_symmetry in {"3mr", "3mrh", "3mr_h", "3mrhombohedral", "3m_r", "3mrhex", "3m_rh"}:
        return "3mR"
    if sample_symmetry in {"3m1", "31m"}:
        return sample_symmetry
    if sample_symmetry in {"6/m", "hexagonal"}:
        return "6/m"
    if sample_symmetry in {"6/mmm", "hexagonal_high"}:
        return "6/mmm"
    if sample_symmetry in {"m3", "cubic"}:
        return "m3"
    if sample_symmetry in {"m3m", "cubic_high"}:
        return "m3m"
    raise ValueError(f"unsupported sample symmetry: {value}")

def _canonicalize_laue_group(value: str) -> str:
    laue_group = str(value).strip()
    if laue_group == "R3mR":
        return "3mR"
    if laue_group == "R3":
        return "3R"
    return laue_group

def _odf_check(laue_group: str, degree: int, order: int) -> bool:
    laue_group = _canonicalize_laue_group(laue_group)
    if degree % 2 != 0 or abs(order) > degree:
        return False
    if laue_group == "0":
        return order == 0
    if laue_group == "-1":
        return True
    if laue_group == "2/m":
        return abs(order) % 2 == 0
    if laue_group == "mmm":
        return abs(order) % 2 == 0 and order >= 0
    if laue_group == "4/m":
        return abs(order) % 4 == 0
    if laue_group == "4/mmm":
        return abs(order) % 4 == 0 and order >= 0
    if laue_group in {"3", "3R"}:
        return abs(order) % 3 == 0
    if laue_group in {"3mR", "3m1", "31m"}:
        return abs(order) % 3 == 0 and order >= 0
    if laue_group == "6/m":
        return abs(order) % 6 == 0
    if laue_group == "6/mmm":
        return abs(order) % 6 == 0 and order >= 0
    if laue_group in {"m3", "m3m"} and order > 0:
        if degree % 12 == 2:
            return order <= degree // 12
        return order <= degree // 12 + 1
    return False

def get_texture_coefficient_names(
    *,
    laue_group: str,
    sample_symmetry: str,
    order: int,
    include_sample_index: bool = True,
) -> tuple[str, ...]:
    laue_group = _canonicalize_laue_group(laue_group)
    sample_symmetry = _canonicalize_sample_symmetry(sample_symmetry)
    coefficient_names: list[str] = []
    for degree in range(2, int(order) + 1, 2):
        for sample_index in range(-degree, degree + 1):
            if not _odf_check(sample_symmetry, degree, sample_index):
                continue
            for crystal_index in range(-degree, degree + 1):
                if not _odf_check(laue_group, degree, crystal_index):
                    continue
                if include_sample_index:
                    coefficient_names.append(f"C({degree},{sample_index},{crystal_index})")
                else:
                    coefficient_names.append(f"C({degree},{crystal_index})")
    return tuple(coefficient_names)

def _prepare_texture_coefficients(
    self,
    coefficient_values: Any,
    *,
    coefficient_names: list[tuple[str, ...]],
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(coefficient_names)
    if isinstance(coefficient_values, dict):
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
        raise ValueError("texture coefficients must be one- or two-dimensional")
    if coeff_tensor.shape[0] == 1 and batch_size != 1:
        coeff_tensor = coeff_tensor.expand(batch_size, -1)
    if coeff_tensor.shape[0] != batch_size:
        raise ValueError("texture coefficients batch dimension must match hkl batch dimension")

    expected_terms = {len(names) for names in coefficient_names}
    if len(expected_terms) != 1:
        raise ValueError("batched texture coefficients require a consistent coefficient count")
    expected_count = next(iter(expected_terms))
    if coeff_tensor.shape[1] != expected_count:
        raise ValueError(
            f"texture coefficients must have {expected_count} terms for {coefficient_names[0]}"
        )
    return coeff_tensor

def _extract_cell_parameters(
    self,
    cell_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a_vec = cell_matrix[:, 0, :]
    b_vec = cell_matrix[:, 1, :]
    c_vec = cell_matrix[:, 2, :]
    a = torch.linalg.norm(a_vec, dim=1)
    b = torch.linalg.norm(b_vec, dim=1)
    c = torch.linalg.norm(c_vec, dim=1)
    alpha = torch.rad2deg(
        torch.acos(
            torch.clamp(
                torch.sum(b_vec * c_vec, dim=1) / torch.clamp(b * c, min=1.0e-8),
                min=-1.0,
                max=1.0,
            )
        )
    )
    beta = torch.rad2deg(
        torch.acos(
            torch.clamp(
                torch.sum(a_vec * c_vec, dim=1) / torch.clamp(a * c, min=1.0e-8),
                min=-1.0,
                max=1.0,
            )
        )
    )
    gamma = torch.rad2deg(
        torch.acos(
            torch.clamp(
                torch.sum(a_vec * b_vec, dim=1) / torch.clamp(a * b, min=1.0e-8),
                min=-1.0,
                max=1.0,
            )
        )
    )
    return a, b, c, alpha, beta, gamma

def calculate_crystal_angles(
    self,
    peak_hkls: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]],
    cell_matrix: torch.Tensor | np.ndarray | list[list[float]],
    *,
    laue_group: str,
    unique_axis: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    laue_group = self._canonicalize_laue_group(laue_group)
    hkl_tensor = self._prepare_profile_geometry_tensor(
        peak_hkls,
        name="peak_hkls",
        expected_last_dim=3,
        dtype=torch.float32,
    )
    cell_tensor = self._prepare_profile_geometry_tensor(
        cell_matrix,
        name="cell_matrix",
        expected_last_dim=3,
        dtype=torch.float32,
    )
    if cell_tensor.shape[1] != 3:
        raise ValueError("cell_matrix must have shape (batch, 3, 3) or (3, 3)")
    if cell_tensor.shape[0] == 1 and hkl_tensor.shape[0] != 1:
        cell_tensor = cell_tensor.expand(hkl_tensor.shape[0], -1, -1)
    if hkl_tensor.shape[0] != cell_tensor.shape[0]:
        raise ValueError("peak_hkls and cell_matrix batch dimensions must match")

    real_metric = torch.matmul(cell_tensor, torch.transpose(cell_tensor, 1, 2))
    recip_metric = torch.linalg.inv(real_metric)
    dh = torch.einsum("bni,bij,bnj->bn", hkl_tensor, recip_metric, hkl_tensor)
    dh = torch.clamp(dh, min=1.0e-12)

    if laue_group == "2/m":
        if unique_axis == "a":
            ref_axis = torch.tensor([1.0, 0.0, 0.0], dtype=hkl_tensor.dtype, device=hkl_tensor.device)
        elif unique_axis == "b":
            ref_axis = torch.tensor([0.0, 1.0, 0.0], dtype=hkl_tensor.dtype, device=hkl_tensor.device)
        else:
            ref_axis = torch.tensor([0.0, 0.0, 1.0], dtype=hkl_tensor.dtype, device=hkl_tensor.device)
    elif laue_group in {"3R", "3mR"}:
        ref_axis = torch.tensor([1.0, 1.0, 1.0], dtype=hkl_tensor.dtype, device=hkl_tensor.device)
    else:
        ref_axis = torch.tensor([0.0, 0.0, 1.0], dtype=hkl_tensor.dtype, device=hkl_tensor.device)
    ref_axis = ref_axis.reshape(1, 3).expand(hkl_tensor.shape[0], -1)
    dr = torch.einsum("bi,bij,bj->b", ref_axis, recip_metric, ref_axis).unsqueeze(1)
    dhr = torch.einsum("bni,bij,bj->bn", hkl_tensor, recip_metric, ref_axis)
    dhr = dhr / torch.sqrt(torch.clamp(dr * dh, min=1.0e-12))
    phi = torch.rad2deg(torch.acos(torch.clamp(dhr, min=-1.0, max=1.0)))

    a, b, c, alpha, beta, gamma = self._extract_cell_parameters(cell_tensor)
    h = hkl_tensor[..., 0]
    k = hkl_tensor[..., 1]
    ell = hkl_tensor[..., 2]
    sq3 = np.sqrt(3.0)

    if laue_group == "-1":
        ba = k * a.unsqueeze(1) / (
            b.unsqueeze(1) - h * torch.cos(torch.deg2rad(gamma)).unsqueeze(1)
        )
        bb = h * torch.square(torch.sin(torch.deg2rad(gamma))).unsqueeze(1)
    elif laue_group == "2/m":
        if unique_axis == "a":
            ba = ell * b.unsqueeze(1) / (
                c.unsqueeze(1) - k * torch.cos(torch.deg2rad(alpha)).unsqueeze(1)
            )
            bb = k * torch.square(torch.sin(torch.deg2rad(alpha))).unsqueeze(1)
        elif unique_axis == "b":
            ba = h * c.unsqueeze(1) / (
                a.unsqueeze(1) - ell * torch.cos(torch.deg2rad(beta)).unsqueeze(1)
            )
            bb = ell * torch.square(torch.sin(torch.deg2rad(beta))).unsqueeze(1)
        else:
            ba = k * a.unsqueeze(1) / (
                b.unsqueeze(1) - h * torch.cos(torch.deg2rad(gamma)).unsqueeze(1)
            )
            bb = h * torch.square(torch.sin(torch.deg2rad(gamma))).unsqueeze(1)
    elif laue_group in {"mmm", "4/m", "4/mmm"}:
        ba = k * a.unsqueeze(1)
        bb = h * b.unsqueeze(1)
    elif laue_group in {"3R", "3mR"}:
        ba = h + k - 2.0 * ell
        bb = sq3 * (h - k)
    elif laue_group in {"m3", "m3m"}:
        ba = k
        bb = h
    else:
        ba = h + 2.0 * k
        bb = sq3 * h
    beta_angle = torch.rad2deg(torch.atan2(ba, bb))
    return phi, beta_angle

def calculate_sample_angles(
    self,
    two_thetas: torch.Tensor,
    *,
    instrument: dict[str, float | int | str] | None = None,
    sample_parameters: dict[str, Any] | None = None,
    sample_orientation_angles: tuple[Any, Any, Any] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = "" if not sample_parameters else str(sample_parameters.get("geometry", "Bragg-Brentano"))
    if_coupled = "Bragg" in geometry
    half_tth = two_thetas / 2.0
    azimuth = 0.0
    if sample_parameters and "azimuth" in sample_parameters:
        azimuth = sample_parameters["azimuth"]
    elif instrument and "Azimuth" in instrument:
        azimuth = instrument["Azimuth"]
    if if_coupled:
        g_phi = self._as_tensor_scalar(0.0, two_thetas)
        g_chi = self._as_tensor_scalar(90.0, two_thetas)
        g_omega = self._as_tensor_scalar(0.0, two_thetas)
    else:
        g_phi = self._as_tensor_scalar(
            0.0 if not sample_parameters else sample_parameters.get("phi_deg", sample_parameters.get("phi", 0.0)),
            two_thetas,
        )
        g_chi = self._as_tensor_scalar(
            0.0 if not sample_parameters else sample_parameters.get("chi_deg", sample_parameters.get("chi", 0.0)),
            two_thetas,
        )
        g_omega = self._as_tensor_scalar(
            0.0 if not sample_parameters else sample_parameters.get("omega_deg", sample_parameters.get("omega", 0.0)),
            two_thetas,
        )
    g_azimuth = self._as_tensor_scalar(azimuth, two_thetas)
    s_omega = self._as_tensor_scalar(sample_orientation_angles[0], two_thetas)
    s_chi = self._as_tensor_scalar(sample_orientation_angles[1], two_thetas)
    s_phi = self._as_tensor_scalar(sample_orientation_angles[2], two_thetas)

    if if_coupled:
        gs_omega = torch.sin(torch.deg2rad(g_omega + half_tth))
        gc_omega = torch.cos(torch.deg2rad(g_omega + half_tth))
    else:
        gs_omega = torch.sin(torch.deg2rad(g_omega))
        gc_omega = torch.cos(torch.deg2rad(g_omega))
    gs_tth = torch.sin(torch.deg2rad(half_tth))
    gc_tth = torch.cos(torch.deg2rad(half_tth))
    gs_azm = torch.sin(torch.deg2rad(g_azimuth))
    gc_azm = torch.cos(torch.deg2rad(g_azimuth))
    gs_chi = torch.sin(torch.deg2rad(g_chi))
    gc_chi = torch.cos(torch.deg2rad(g_chi))
    gs_phi = torch.sin(torch.deg2rad(g_phi + s_phi))
    gc_phi = torch.cos(torch.deg2rad(g_phi + s_phi))
    ss_omega = torch.sin(torch.deg2rad(s_omega))
    sc_omega = torch.cos(torch.deg2rad(s_omega))
    ss_chi = torch.sin(torch.deg2rad(s_chi))
    sc_chi = torch.cos(torch.deg2rad(s_chi))

    at = -gs_tth * gc_omega + gc_tth * gc_azm * gs_omega
    bt = gs_tth * gs_omega + gc_tth * gc_azm * gc_omega
    ct = -gc_tth * gs_azm * gs_chi
    dt = -gc_tth * gs_azm * gc_chi

    bc1 = -at * gs_phi + (ct + bt * gc_chi) * gc_phi
    bc2 = dt - bt * gs_chi
    bc3 = at * gc_phi + (ct + bt * gc_chi) * gs_phi

    bc = bc1 * sc_omega * sc_chi + bc2 * sc_omega * ss_chi - bc3 * ss_omega
    psi = torch.rad2deg(torch.acos(torch.clamp(bc, min=-1.0, max=1.0)))

    ba = -bc1 * ss_chi + bc2 * sc_chi
    bb = bc1 * ss_omega * sc_chi + bc2 * ss_omega * ss_chi + bc3 * sc_omega
    gamma = torch.rad2deg(torch.atan2(bb, ba))
    return psi, gamma

def calculate_normalized_associated_legendre(
    self,
    degree: int,
    order: int,
    polar_angles_deg: torch.Tensor,
) -> torch.Tensor:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if order < 0 or order > degree:
        raise ValueError("order must satisfy 0 <= order <= degree")
    x = torch.clamp(torch.cos(torch.deg2rad(polar_angles_deg)), min=-1.0, max=1.0)
    p_mm = torch.ones_like(x)
    if order > 0:
        somx2 = torch.sqrt(torch.clamp(1.0 - torch.square(x), min=0.0))
        coeff = 1.0
        for _ in range(order):
            p_mm = -p_mm * coeff * somx2
            coeff += 2.0
    if degree == order:
        p_lm = p_mm
    else:
        p_m1m = x * (2 * order + 1) * p_mm
        if degree == order + 1:
            p_lm = p_m1m
        else:
            p_prev_prev = p_mm
            p_prev = p_m1m
            p_lm = p_prev
            for ell in range(order + 2, degree + 1):
                p_lm = (
                    (2 * ell - 1) * x * p_prev
                    - (ell + order - 1) * p_prev_prev
                ) / (ell - order)
                p_prev_prev = p_prev
                p_prev = p_lm
    log_norm = 0.5 * (
        np.log((2 * degree + 1) / 2.0)
        + torch.lgamma(torch.full_like(x, degree - order + 1.0))
        - torch.lgamma(torch.full_like(x, degree + order + 1.0))
    )
    return torch.exp(log_norm) * p_lm

def calculate_spherical_harmonic_angle(
    self,
    degree: int,
    order: int,
    parity_sign: float,
    azimuth_deg: torch.Tensor,
    polar_deg: torch.Tensor,
) -> torch.Tensor:
    abs_order = abs(int(order))
    base = self.calculate_normalized_associated_legendre(
        degree,
        abs_order,
        polar_deg,
    ) / np.sqrt(2.0 * np.pi)
    phase = -1.0 if (abs_order % 2) else 1.0
    azimuth_term = torch.deg2rad(abs_order * azimuth_deg)
    if order > 0:
        return phase * parity_sign * base * torch.cos(azimuth_term) * np.sqrt(2.0)
    if order == 0:
        return parity_sign * base
    # Negative orders use imag(Y_l^m) after converting to a real basis.
    return -parity_sign * base * torch.sin(azimuth_term) * np.sqrt(2.0)

def calculate_cubic_harmonic(
    self,
    degree: int,
    cubic_index: int,
    azimuth_deg: torch.Tensor,
    polar_deg: torch.Tensor,
) -> torch.Tensor:
    if degree == 0:
        return self.calculate_spherical_harmonic_angle(degree, cubic_index, 1.0, azimuth_deg, polar_deg)
    if degree == 4:
        return (
            0.5 * np.sqrt(7.0 / 3.0) * self.calculate_spherical_harmonic_angle(4, 0, 1.0, azimuth_deg, polar_deg)
            + 0.5 * np.sqrt(5.0 / 3.0) * self.calculate_spherical_harmonic_angle(4, 4, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 6:
        if cubic_index == 1:
            return (
                0.5 * np.sqrt(0.5) * self.calculate_spherical_harmonic_angle(6, 0, 1.0, azimuth_deg, polar_deg)
                - 0.5 * np.sqrt(7.0 / 2.0) * self.calculate_spherical_harmonic_angle(6, 4, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.25 * np.sqrt(11.0) * self.calculate_spherical_harmonic_angle(6, 2, 1.0, azimuth_deg, polar_deg)
            - 0.25 * np.sqrt(5.0) * self.calculate_spherical_harmonic_angle(6, 6, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 8:
        return (
            0.125 * np.sqrt(33.0) * self.calculate_spherical_harmonic_angle(8, 0, 1.0, azimuth_deg, polar_deg)
            + 0.25 * np.sqrt(7.0 / 3.0) * self.calculate_spherical_harmonic_angle(8, 4, 1.0, azimuth_deg, polar_deg)
            + 0.125 * np.sqrt(65.0 / 3.0) * self.calculate_spherical_harmonic_angle(8, 8, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 10:
        if cubic_index == 1:
            return (
                0.125 * np.sqrt(65.0 / 6.0) * self.calculate_spherical_harmonic_angle(10, 0, 1.0, azimuth_deg, polar_deg)
                - 0.25 * np.sqrt(11.0 / 2.0) * self.calculate_spherical_harmonic_angle(10, 4, 1.0, azimuth_deg, polar_deg)
                - 0.125 * np.sqrt(187.0 / 6.0) * self.calculate_spherical_harmonic_angle(10, 8, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.125 * np.sqrt(247.0 / 6.0) * self.calculate_spherical_harmonic_angle(10, 2, 1.0, azimuth_deg, polar_deg)
            + (1.0 / 16.0) * np.sqrt(19.0 / 3.0) * self.calculate_spherical_harmonic_angle(10, 6, 1.0, azimuth_deg, polar_deg)
            - (1.0 / 16.0) * np.sqrt(85.0) * self.calculate_spherical_harmonic_angle(10, 10, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 12:
        if cubic_index == 1:
            return (
                0.69550265 * self.calculate_spherical_harmonic_angle(12, 0, 1.0, azimuth_deg, polar_deg)
                + 0.31412555 * self.calculate_spherical_harmonic_angle(12, 4, 1.0, azimuth_deg, polar_deg)
                + 0.34844954 * self.calculate_spherical_harmonic_angle(12, 8, 1.0, azimuth_deg, polar_deg)
                + 0.54422797 * self.calculate_spherical_harmonic_angle(12, 12, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.55897937 * self.calculate_spherical_harmonic_angle(12, 4, 1.0, azimuth_deg, polar_deg)
            - 0.80626751 * self.calculate_spherical_harmonic_angle(12, 8, 1.0, azimuth_deg, polar_deg)
            + 0.19358400 * self.calculate_spherical_harmonic_angle(12, 12, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 14:
        return (
            0.44009645 * self.calculate_spherical_harmonic_angle(14, 0, 1.0, azimuth_deg, polar_deg)
            - 0.45768183 * self.calculate_spherical_harmonic_angle(14, 4, 1.0, azimuth_deg, polar_deg)
            - 0.49113230 * self.calculate_spherical_harmonic_angle(14, 8, 1.0, azimuth_deg, polar_deg)
            - 0.59634848 * self.calculate_spherical_harmonic_angle(14, 12, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 16:
        if cubic_index == 1:
            return (
                0.68136168 * self.calculate_spherical_harmonic_angle(16, 0, 1.0, azimuth_deg, polar_deg)
                + 0.27586801 * self.calculate_spherical_harmonic_angle(16, 4, 1.0, azimuth_deg, polar_deg)
                + 0.29048987 * self.calculate_spherical_harmonic_angle(16, 8, 1.0, azimuth_deg, polar_deg)
                + 0.32756975 * self.calculate_spherical_harmonic_angle(16, 12, 1.0, azimuth_deg, polar_deg)
                + 0.51764542 * self.calculate_spherical_harmonic_angle(16, 16, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.63704821 * self.calculate_spherical_harmonic_angle(16, 4, 1.0, azimuth_deg, polar_deg)
            - 0.32999033 * self.calculate_spherical_harmonic_angle(16, 8, 1.0, azimuth_deg, polar_deg)
            - 0.64798073 * self.calculate_spherical_harmonic_angle(16, 12, 1.0, azimuth_deg, polar_deg)
            + 0.25572816 * self.calculate_spherical_harmonic_angle(16, 16, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 18:
        if cubic_index == 1:
            return (
                0.45791513 * self.calculate_spherical_harmonic_angle(18, 0, 1.0, azimuth_deg, polar_deg)
                - 0.38645598 * self.calculate_spherical_harmonic_angle(18, 4, 1.0, azimuth_deg, polar_deg)
                - 0.40209462 * self.calculate_spherical_harmonic_angle(18, 8, 1.0, azimuth_deg, polar_deg)
                - 0.43746593 * self.calculate_spherical_harmonic_angle(18, 12, 1.0, azimuth_deg, polar_deg)
                - 0.53657149 * self.calculate_spherical_harmonic_angle(18, 16, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.14872751 * self.calculate_spherical_harmonic_angle(18, 4, 1.0, azimuth_deg, polar_deg)
            - 0.63774601 * self.calculate_spherical_harmonic_angle(18, 8, 1.0, azimuth_deg, polar_deg)
            + 0.72334167 * self.calculate_spherical_harmonic_angle(18, 12, 1.0, azimuth_deg, polar_deg)
            - 0.21894515 * self.calculate_spherical_harmonic_angle(18, 16, 1.0, azimuth_deg, polar_deg)
        )
    if degree == 20:
        if cubic_index == 1:
            return (
                0.67141495 * self.calculate_spherical_harmonic_angle(20, 0, 1.0, azimuth_deg, polar_deg)
                + 0.24982619 * self.calculate_spherical_harmonic_angle(20, 4, 1.0, azimuth_deg, polar_deg)
                + 0.25782846 * self.calculate_spherical_harmonic_angle(20, 8, 1.0, azimuth_deg, polar_deg)
                + 0.27469333 * self.calculate_spherical_harmonic_angle(20, 12, 1.0, azimuth_deg, polar_deg)
                + 0.31248919 * self.calculate_spherical_harmonic_angle(20, 16, 1.0, azimuth_deg, polar_deg)
                + 0.49719956 * self.calculate_spherical_harmonic_angle(20, 20, 1.0, azimuth_deg, polar_deg)
            )
        return (
            0.66299538 * self.calculate_spherical_harmonic_angle(20, 4, 1.0, azimuth_deg, polar_deg)
            - 0.11295259 * self.calculate_spherical_harmonic_angle(20, 8, 1.0, azimuth_deg, polar_deg)
            - 0.42738441 * self.calculate_spherical_harmonic_angle(20, 12, 1.0, azimuth_deg, polar_deg)
            - 0.52810433 * self.calculate_spherical_harmonic_angle(20, 16, 1.0, azimuth_deg, polar_deg)
            + 0.29347435 * self.calculate_spherical_harmonic_angle(20, 20, 1.0, azimuth_deg, polar_deg)
        )
    raise ValueError(f"cubic spherical harmonics currently support degrees <= 20, got L={degree}")

def calculate_kcl(
    self,
    degree: int,
    crystal_index: int,
    crystal_polar_deg: torch.Tensor,
    crystal_azimuth_deg: torch.Tensor,
    *,
    laue_group: str,
) -> torch.Tensor:
    laue_group = self._canonicalize_laue_group(laue_group)
    signed_order = int(crystal_index)
    if laue_group in {"m3", "m3m"}:
        return self.calculate_cubic_harmonic(
            degree,
            signed_order,
            crystal_azimuth_deg,
            crystal_polar_deg,
        )
    order = abs(signed_order)
    pcrs = self.calculate_normalized_associated_legendre(
        degree,
        order,
        crystal_polar_deg,
    ) / np.sqrt(2.0 * np.pi)
    if order:
        pcrs = pcrs * np.sqrt(2.0)
    if laue_group in {"mmm", "4/mmm", "6/mmm", "3mR", "3m1", "31m"}:
        if laue_group in {"3mR", "3m1", "31m"} and (order % 6 == 3):
            return pcrs * torch.sin(torch.deg2rad(signed_order * crystal_azimuth_deg))
        return pcrs * torch.cos(torch.deg2rad(signed_order * crystal_azimuth_deg))
    return pcrs * (
        torch.cos(torch.deg2rad(signed_order * crystal_azimuth_deg))
        + torch.sin(torch.deg2rad(signed_order * crystal_azimuth_deg))
    )

def calculate_ksl(
    self,
    degree: int,
    sample_index: int,
    sample_polar_deg: torch.Tensor,
    sample_azimuth_deg: torch.Tensor,
    *,
    sample_symmetry: str,
) -> torch.Tensor:
    sample_symmetry = self._canonicalize_sample_symmetry(sample_symmetry)
    signed_order = int(sample_index)
    order = abs(signed_order)
    psrs = self.calculate_normalized_associated_legendre(
        degree,
        order,
        sample_polar_deg,
    ) / np.sqrt(2.0 * np.pi)
    if order:
        psrs = psrs * np.sqrt(2.0)
    if sample_symmetry == "mmm":
        return psrs * torch.cos(torch.deg2rad(signed_order * sample_azimuth_deg))
    return psrs * (
        torch.cos(torch.deg2rad(signed_order * sample_azimuth_deg))
        + torch.sin(torch.deg2rad(signed_order * sample_azimuth_deg))
    )

def _resolve_texture_metadata(
    self,
    *,
    batch_size: int,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None,
    laue_group: Any = None,
    unique_axis: Any = "",
    spacegroup_number: Any = None,
) -> list[tuple[str, str]]:
    if laue_group is not None:
        laue_str = self._canonicalize_laue_group(str(laue_group))
        unique_str = str(unique_axis).strip().lower()
        return [(laue_str, unique_str)] * batch_size
    if spacegroup_number is not None:
        metadata = get_spacegroup_metadata(int(spacegroup_number))
        return [(self._canonicalize_laue_group(metadata.laue), metadata.unique_axis)] * batch_size
    symbol_list = self._expand_spacegroup_symbols(spacegroup_symbol, batch_size=batch_size)
    if any(symbol is None for symbol in symbol_list):
        raise ValueError("texture corrections require spacegroup_symbol or explicit Laue metadata")
    resolved: list[tuple[str, str]] = []
    for symbol in symbol_list:
        assert symbol is not None
        metadata = self.get_spacegroup_metadata_for_symbol(symbol)
        resolved.append((self._canonicalize_laue_group(metadata.laue), metadata.unique_axis))
    return resolved

def calculate_sh_preferred_orientation_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    two_thetas: torch.Tensor,
    *,
    sample_parameters: dict[str, Any],
    instrument: dict[str, float | int | str] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    order = int(sample_parameters.get("preferred_orientation_order", 0))
    coefficient_values = sample_parameters.get("preferred_orientation_coefficients")
    if coefficient_values is None:
        coefficient_values = sample_parameters.get("preferred_orientation_coeffs")
    if order <= 0 or coefficient_values is None:
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)

    metadata = self._resolve_texture_metadata(
        batch_size=hkl_pts.shape[0],
        spacegroup_symbol=spacegroup_symbol,
        laue_group=sample_parameters.get("preferred_orientation_laue_group"),
        unique_axis=sample_parameters.get("preferred_orientation_unique_axis", ""),
        spacegroup_number=sample_parameters.get("preferred_orientation_spacegroup_number"),
    )
    coefficient_names = [
        self.get_texture_coefficient_names(
            laue_group=laue_group,
            sample_symmetry="0",
            order=order,
            include_sample_index=False,
        )
        for laue_group, _unique_axis in metadata
    ]
    coeff_tensor = self._prepare_texture_coefficients(
        coefficient_values,
        coefficient_names=coefficient_names,
        reference=two_thetas,
    )
    crystal_angles = [
        self.calculate_crystal_angles(
            hkl_pts[batch_index],
            cell_matrix[batch_index],
            laue_group=laue_group,
            unique_axis=unique_axis,
        )
        for batch_index, (laue_group, unique_axis) in enumerate(metadata)
    ]
    psi, gamma = self.calculate_sample_angles(
        two_thetas,
        instrument=instrument,
        sample_parameters=sample_parameters,
        sample_orientation_angles=(0.0, 0.0, 0.0),
    )
    factor = torch.ones_like(two_thetas)
    for batch_index, names in enumerate(coefficient_names):
        if len(names) == 0:
            continue
        phi_deg, beta_deg = crystal_angles[batch_index]
        phi_deg = phi_deg.squeeze(0)
        beta_deg = beta_deg.squeeze(0)
        basis_terms = []
        for name in names:
            degree, crystal_index = eval(name.strip("C"))
            kcl = self.calculate_kcl(
                degree,
                crystal_index,
                phi_deg,
                beta_deg,
                laue_group=metadata[batch_index][0],
            )
            ksl = self.calculate_ksl(
                degree,
                0,
                psi[batch_index],
                gamma[batch_index],
                sample_symmetry="0",
            )
            basis_terms.append((4.0 * np.pi / (2.0 * degree + 1.0)) * kcl * ksl)
        factor[batch_index] = factor[batch_index] + torch.sum(
            coeff_tensor[batch_index].unsqueeze(1) * torch.stack(basis_terms, dim=0),
            dim=0,
        )
    return factor


_TOPAS_SH_NAME = re.compile(
    r"^y(?P<degree>[02468])(?P<order>[0-8])(?P<parity>[pm]?)$",
    flags=re.IGNORECASE,
)


def _parse_topas_sh_name(name: str, *, maximum_degree: int) -> tuple[int, int]:
    match = _TOPAS_SH_NAME.fullmatch(str(name).strip())
    if match is None:
        raise ValueError(f"unsupported TOPAS spherical-harmonic term: {name}")
    degree = int(match.group("degree"))
    order = int(match.group("order"))
    parity = match.group("parity").lower()
    if degree > maximum_degree or order > degree:
        raise ValueError(
            f"TOPAS spherical-harmonic term {name} exceeds configured order "
            f"{maximum_degree}"
        )
    if (order == 0 and parity) or (order > 0 and parity not in {"p", "m"}):
        raise ValueError(f"invalid TOPAS spherical-harmonic term: {name}")
    return degree, -order if parity == "m" else order


def _prepare_topas_sh_coefficients(
    coefficient_values: dict[str, Any],
    *,
    coefficient_names: tuple[str, ...],
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for name in coefficient_names:
        default = 1.0 if name == "y00" else 0.0
        value = coefficient_values.get(name, default)
        value_tensor = torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )
        if value_tensor.ndim == 0:
            value_tensor = value_tensor.expand(batch_size)
        elif value_tensor.shape == (1,):
            value_tensor = value_tensor.expand(batch_size)
        elif value_tensor.shape != (batch_size,):
            raise ValueError(
                f"TOPAS spherical-harmonic coefficient {name} must be scalar or "
                f"have shape ({batch_size},)"
            )
        rows.append(value_tensor)
    return torch.stack(rows, dim=1)


def calculate_topas_spherical_harmonic_component(
    self,
    degree: int,
    signed_order: int,
    polar_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a max-normalized TOPAS real spherical-harmonic component."""
    degree = int(degree)
    order = abs(int(signed_order))
    if degree not in {0, 2, 4, 6, 8} or order > degree:
        raise ValueError(
            "TOPAS spherical harmonics support even degrees from zero through eight"
        )
    c = torch.cos(torch.deg2rad(polar_deg))
    s = torch.sin(torch.deg2rad(polar_deg))
    c2 = torch.square(c)
    c4 = torch.square(c2)
    c6 = c4 * c2
    c8 = torch.square(c4)
    s2 = torch.square(s)
    s3 = s2 * s
    s4 = torch.square(s2)
    s5 = s4 * s
    s6 = s4 * s2
    s7 = s6 * s
    s8 = torch.square(s4)

    if degree == 0 and order == 0:
        radial = torch.ones_like(c)
    elif degree == 2:
        radial = {
            0: (3.0 * c2 - 1.0) * 0.5,
            1: 2.0 * c * s,
            2: s2,
        }[order]
    elif degree == 4:
        radial = {
            0: (3.0 - 30.0 * c2 + 35.0 * c4) * 0.125,
            1: c * (7.0 * c2 - 3.0) * s * 0.9469461818,
            2: (-1.0 + 7.0 * c2) * s2 * 0.7777777778,
            3: c * s3 * 3.0792014358,
            4: s4,
        }[order]
    elif degree == 6:
        radial = {
            0: (-5.0 + 105.0 * c2 - 315.0 * c4 + 231.0 * c6) * 0.0625,
            1: (-5.0 + 30.0 * c2 - 33.0 * c4)
            * s
            * c
            * 0.6913999628,
            2: (1.0 - 18.0 * c2 + 33.0 * c4) * s2 * 0.6454926483,
            3: (3.0 - 11.0 * c2) * c * s3 * 1.4168477165,
            4: (-1.0 + 11.0 * c2) * s4 * 0.81675,
            5: c * s5 * 3.8639254683,
            6: s6,
        }[order]
    elif degree == 8:
        radial = {
            0: (
                35.0
                - 1260.0 * c2
                + 6930.0 * c4
                - 12012.0 * c6
                + 6435.0 * c8
            )
            * 0.0078125,
            1: (35.0 * c - 385.0 * c * c2 + 1001.0 * c * c4 - 715.0 * c * c6)
            * s
            * 0.1134799545,
            2: (-1.0 + 33.0 * c2 - 143.0 * c4 + 143.0 * c6)
            * s2
            * 0.5637178511,
            3: (-3.0 * c + 26.0 * c * c2 - 39.0 * c * c4)
            * s3
            * 1.6913068375,
            4: (1.0 - 26.0 * c2 + 65.0 * c4) * s4 * 0.7011002983,
            5: (c - 5.0 * c * c2) * s5 * 5.2833000817,
            6: (-1.0 + 15.0 * c2) * s6 * 0.8329862557,
            7: c * s7 * 4.5135349314,
            8: s8,
        }[order]
    else:
        raise AssertionError("unreachable TOPAS spherical-harmonic degree")

    if order == 0:
        return radial
    azimuth = torch.deg2rad(order * azimuth_deg)
    angular = torch.sin(azimuth) if signed_order < 0 else torch.cos(azimuth)
    return radial * angular


def calculate_topas_sh_preferred_orientation_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    two_thetas: torch.Tensor,
    *,
    sample_parameters: dict[str, Any],
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Translate TOPAS ``spherical_harmonics_hkl`` for symmetric reflection."""
    order = int(sample_parameters.get("preferred_orientation_order", 0))
    if order not in {2, 4, 6, 8}:
        raise ValueError("TOPAS spherical-harmonic order must be 2, 4, 6, or 8")
    sh_alpha = float(sample_parameters.get("preferred_orientation_sh_alpha_degree", 0.0))
    if sh_alpha != 0.0:
        raise ValueError("nonzero TOPAS sh_alpha is not implemented")
    coefficient_values = sample_parameters.get("preferred_orientation_coefficients")
    if coefficient_values is None:
        coefficient_values = sample_parameters.get("preferred_orientation_coeffs")
    if not isinstance(coefficient_values, dict):
        raise ValueError("TOPAS spherical-harmonic coefficients must be a mapping")
    canonical_values = {
        str(name).strip().lower(): value for name, value in coefficient_values.items()
    }
    parsed = {
        name: _parse_topas_sh_name(name, maximum_degree=order)
        for name in canonical_values
    }
    parsed.setdefault("y00", (0, 0))
    coefficient_names = tuple(
        sorted(parsed, key=lambda name: (parsed[name][0], abs(parsed[name][1]), parsed[name][1]))
    )
    coefficients = _prepare_topas_sh_coefficients(
        canonical_values,
        coefficient_names=coefficient_names,
        batch_size=hkl_pts.shape[0],
        reference=two_thetas,
    )
    metadata = self._resolve_texture_metadata(
        batch_size=hkl_pts.shape[0],
        spacegroup_symbol=spacegroup_symbol,
        laue_group=sample_parameters.get("preferred_orientation_laue_group"),
        unique_axis=sample_parameters.get("preferred_orientation_unique_axis", ""),
        spacegroup_number=sample_parameters.get("preferred_orientation_spacegroup_number"),
    )
    factor = torch.zeros_like(two_thetas)
    for batch_index, (laue_group, unique_axis) in enumerate(metadata):
        polar_deg, azimuth_deg = self.calculate_crystal_angles(
            hkl_pts[batch_index],
            cell_matrix[batch_index],
            laue_group=laue_group,
            unique_axis=unique_axis,
        )
        basis = torch.stack(
            [
                self.calculate_topas_spherical_harmonic_component(
                    *parsed[name],
                    polar_deg.squeeze(0),
                    azimuth_deg.squeeze(0),
                )
                for name in coefficient_names
            ],
            dim=0,
        )
        factor[batch_index] = torch.sum(
            coefficients[batch_index].unsqueeze(1) * basis,
            dim=0,
        )
    return factor

def calculate_sh_texture_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    two_thetas: torch.Tensor,
    *,
    sample_parameters: dict[str, Any],
    instrument: dict[str, float | int | str] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    order = int(sample_parameters.get("texture_order", 0))
    coefficient_values = sample_parameters.get("texture_coefficients")
    if coefficient_values is None:
        coefficient_values = sample_parameters.get("texture_coeffs")
    if order <= 0 or coefficient_values is None:
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)

    sample_symmetry = self._canonicalize_sample_symmetry(
        sample_parameters.get("texture_sample_symmetry", "cylindrical")
    )
    metadata = self._resolve_texture_metadata(
        batch_size=hkl_pts.shape[0],
        spacegroup_symbol=spacegroup_symbol,
        laue_group=sample_parameters.get("texture_laue_group"),
        unique_axis=sample_parameters.get("texture_unique_axis", ""),
        spacegroup_number=sample_parameters.get("texture_spacegroup_number"),
    )
    coefficient_names = [
        self.get_texture_coefficient_names(
            laue_group=laue_group,
            sample_symmetry=sample_symmetry,
            order=order,
            include_sample_index=True,
        )
        for laue_group, _unique_axis in metadata
    ]
    coeff_tensor = self._prepare_texture_coefficients(
        coefficient_values,
        coefficient_names=coefficient_names,
        reference=two_thetas,
    )
    crystal_angles = [
        self.calculate_crystal_angles(
            hkl_pts[batch_index],
            cell_matrix[batch_index],
            laue_group=laue_group,
            unique_axis=unique_axis,
        )
        for batch_index, (laue_group, unique_axis) in enumerate(metadata)
    ]
    psi, gamma = self.calculate_sample_angles(
        two_thetas,
        instrument=instrument,
        sample_parameters=sample_parameters,
        sample_orientation_angles=(
            sample_parameters.get("texture_sample_omega_deg", sample_parameters.get("texture_sample_omega", 0.0)),
            sample_parameters.get("texture_sample_chi_deg", sample_parameters.get("texture_sample_chi", 0.0)),
            sample_parameters.get("texture_sample_phi_deg", sample_parameters.get("texture_sample_phi", 0.0)),
        ),
    )
    factor = torch.ones_like(two_thetas)
    for batch_index, names in enumerate(coefficient_names):
        if len(names) == 0:
            continue
        phi_deg, beta_deg = crystal_angles[batch_index]
        phi_deg = phi_deg.squeeze(0)
        beta_deg = beta_deg.squeeze(0)
        basis_terms = []
        for name in names:
            degree, sample_index, crystal_index = eval(name.strip("C"))
            kcl = self.calculate_kcl(
                degree,
                crystal_index,
                phi_deg,
                beta_deg,
                laue_group=metadata[batch_index][0],
            )
            ksl = self.calculate_ksl(
                degree,
                sample_index,
                psi[batch_index],
                gamma[batch_index],
                sample_symmetry=sample_symmetry,
            )
            basis_terms.append((4.0 * np.pi / (2.0 * degree + 1.0)) * kcl * ksl)
        factor[batch_index] = factor[batch_index] + torch.sum(
            coeff_tensor[batch_index].unsqueeze(1) * torch.stack(basis_terms, dim=0),
            dim=0,
        )
    return factor

def calculate_preferred_orientation_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    two_thetas: torch.Tensor,
    *,
    sample_parameters: dict[str, Any] | None = None,
    instrument: dict[str, float | int | str] | None = None,
    spacegroup_symbol: str | list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    if not sample_parameters:
        return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)
    model = str(sample_parameters.get("preferred_orientation_model", "none")).lower()
    if model in {"topas_sh", "topas_spherical_harmonics"}:
        return self.calculate_topas_sh_preferred_orientation_factor(
            hkl_pts,
            cell_matrix,
            two_thetas,
            sample_parameters=sample_parameters,
            spacegroup_symbol=spacegroup_symbol,
        )
    if int(sample_parameters.get("texture_order", 0)) > 0:
        return self.calculate_sh_texture_factor(
            hkl_pts,
            cell_matrix,
            two_thetas,
            sample_parameters=sample_parameters,
            instrument=instrument,
            spacegroup_symbol=spacegroup_symbol,
        )
    if model in {"march_dollase", "md"}:
        return self.calculate_march_dollase_factor(
            hkl_pts,
            cell_matrix,
            sample_parameters=sample_parameters,
        )
    if model in {"spherical_harmonics", "sh", "cylindrical_sh"}:
        return self.calculate_sh_preferred_orientation_factor(
            hkl_pts,
            cell_matrix,
            two_thetas,
            sample_parameters=sample_parameters,
            instrument=instrument,
            spacegroup_symbol=spacegroup_symbol,
        )
    return torch.ones(hkl_pts.shape[:2], dtype=torch.float32, device=self.device)

__all__ = [
    "calculate_lorentz_factor",
    "calculate_march_dollase_factor",
    "_canonicalize_sample_symmetry",
    "_canonicalize_laue_group",
    "_odf_check",
    "get_texture_coefficient_names",
    "_prepare_texture_coefficients",
    "_extract_cell_parameters",
    "calculate_crystal_angles",
    "calculate_sample_angles",
    "calculate_normalized_associated_legendre",
    "calculate_spherical_harmonic_angle",
    "calculate_cubic_harmonic",
    "calculate_kcl",
    "calculate_ksl",
    "_resolve_texture_metadata",
    "calculate_sh_preferred_orientation_factor",
    "_parse_topas_sh_name",
    "_prepare_topas_sh_coefficients",
    "calculate_topas_spherical_harmonic_component",
    "calculate_topas_sh_preferred_orientation_factor",
    "calculate_sh_texture_factor",
    "calculate_preferred_orientation_factor",
]
