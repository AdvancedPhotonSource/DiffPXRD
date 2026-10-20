from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import warnings

import numpy as np
import torch

_FCJ_QUADRATURE_NTERMS = (
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    400,
)
_STOFW = 2.35482005
_SQ2PI = 2.506628275
_PI = float(np.pi)
_FCJ_BUCKET_ELEMENT_BUDGET_FLOAT32 = 3_000_000
_FCJ_BUCKET_ELEMENT_BUDGET_FLOAT64 = 1_500_000
_FCJ_MAX_PEAK_CHUNK = 256
_COMPILED_FCJ_DISABLED_REASON: str | None = None


@dataclass(frozen=True)
class _PreparedFcjPeaks:
    batch_size: int
    domain_size: int
    domain: torch.Tensor
    peak_batch_ids: torch.Tensor
    peak_pos: torch.Tensor
    peak_intensities: torch.Tensor
    peak_sigma: torch.Tensor
    peak_gamma: torch.Tensor
    peak_shl_values: torch.Tensor
    i_beg: torch.Tensor
    i_fin: torch.Tensor
    selected_terms: torch.Tensor
    zero_shl_mask: torch.Tensor
    two_term_mask: torch.Tensor
    highq_mask: torch.Tensor


def _canonicalize_fcj_backend(backend: str | None) -> str:
    key = "auto" if backend is None else str(backend).strip().lower()
    if key in {"", "auto", "bucketed", "vectorized", "fast", "default"}:
        return "bucketed"
    if key in {"compiled", "compile", "torch_compile", "torch-compile", "bucketed_compile", "bucketed-compile"}:
        return "compiled"
    if key in {"triton", "cuda_triton", "cuda-triton", "fused"}:
        return "triton"
    if key in {"triton_highq", "highq", "fused_highq", "fcj_fused"}:
        return "triton_highq"
    if key in {"torch", "native", "autograd", "pure_torch", "pure-torch", "reference"}:
        return "torch"
    if key in {"exact", "legacy"}:
        raise ValueError("Legacy FCJ backends have been removed from diffpxrd.")
    raise ValueError(f"unsupported FCJ backend: {backend}")


def _validate_fcj_inputs(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> None:
    if positions.ndim != 2:
        raise ValueError("positions must have shape (batch, npeaks)")
    if intensities.shape != positions.shape:
        raise ValueError("intensities must match positions")
    if sigma_sq_centideg2.shape != positions.shape:
        raise ValueError("sigma_sq_centideg2 must match positions")
    if gamma_centideg.shape != positions.shape:
        raise ValueError("gamma_centideg must match positions")
    if domain.ndim != 1:
        raise ValueError("domain must be one-dimensional")
    if shl_values.ndim not in {0, 1}:
        raise ValueError("shl_values must be scalar or shape (batch,)")
    if shl_values.ndim == 1 and shl_values.shape[0] != positions.shape[0]:
        raise ValueError("batch shl_values must match the batch size")
    if positions.dtype not in {torch.float32, torch.float64}:
        raise ValueError("FCJ positions must use torch.float32 or torch.float64")
    _validate_fcj_shl(shl_values)


def _validate_fcj_shl(shl_values: torch.Tensor) -> None:
    if not bool(torch.isfinite(shl_values).all()) or bool((shl_values < 0).any()):
        raise ValueError("FCJ SH/L must be finite and nonnegative")


def _promote_asymmetric_fcj_rows(
    render: Callable[..., torch.Tensor],
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor | None:
    if positions.dtype != torch.float32 or positions.shape[0] == 0:
        return None
    batch_shl = shl_values.to(device=positions.device)
    if batch_shl.ndim == 0:
        batch_shl = batch_shl.expand(positions.shape[0])
    asymmetric = batch_shl > 0
    if not bool(asymmetric.any()):
        return None

    # The near-coincident trigonometric differences in FCJ need double precision.
    # Keep symmetric rows native, including when they share a batch with FCJ rows.
    peak_inputs = (positions, intensities, sigma_sq_centideg2, gamma_centideg)
    if bool(asymmetric.all()):
        return render(
            *(value.to(dtype=torch.float64) for value in peak_inputs),
            domain.to(dtype=torch.float64),
            shl_values.to(dtype=torch.float64),
        ).to(dtype=positions.dtype)
    histogram = positions.new_zeros((positions.shape[0], domain.numel()))
    for mask, dtype in ((asymmetric, torch.float64), (~asymmetric, torch.float32)):
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        values = tuple(value.index_select(0, rows).to(dtype=dtype) for value in peak_inputs)
        partial = render(*values, domain.to(dtype=dtype), batch_shl.index_select(0, rows).to(dtype=dtype))
        histogram = histogram.index_copy(0, rows, partial.to(dtype=positions.dtype))
    return histogram


def _get_widths_cw(
    pos_deg: float,
    sig_centideg2: float,
    gam_centideg: float,
    shl_value: float,
) -> tuple[tuple[float, float], float, float]:
    widths = (np.sqrt(sig_centideg2) / 100.0, gam_centideg / 100.0)
    fwhm = max(2.355 * widths[0] + widths[1], 0.0001)
    fmin = 50.0 * (fwhm + shl_value * abs(np.cos(np.deg2rad(pos_deg))))
    fmax = 75.0 * fwhm
    if pos_deg > 90.0:
        fmin, fmax = fmax, fmin
    return widths, fmin, fmax


@lru_cache(maxsize=None)
def _half_gauss_legendre_rule(num_terms: int) -> tuple[np.ndarray, np.ndarray]:
    xp, wp = np.polynomial.legendre.leggauss(num_terms)
    half_start = num_terms // 2
    return xp[half_start:].astype(np.float64, copy=False), wp[half_start:].astype(
        np.float64,
        copy=False,
    )


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".", maxsplit=1)[-1]


@lru_cache(maxsize=None)
def _half_gauss_legendre_rule_torch_cached(
    num_terms: int,
    device_type: str,
    device_index: int,
    dtype_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, dtype_name)
    device = torch.device(device_type if device_index < 0 else f"{device_type}:{device_index}")
    xp_np, wp_np = _half_gauss_legendre_rule(num_terms)
    xp = torch.as_tensor(xp_np, dtype=dtype, device=device)
    wp = torch.as_tensor(wp_np, dtype=dtype, device=device)
    return xp, wp


def _half_gauss_legendre_rule_torch(
    num_terms: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_obj = torch.device(device)
    return _half_gauss_legendre_rule_torch_cached(
        num_terms,
        device_obj.type,
        -1 if device_obj.index is None else int(device_obj.index),
        _torch_dtype_name(dtype),
    )


def _select_fcj_num_terms(peak_pos_deg: float, emin_deg: float, gam_centideg: float) -> int:
    peak_span_deg = abs(peak_pos_deg - emin_deg)
    if gam_centideg <= 0.0:
        target_terms = int(peak_span_deg * 200.0)
    else:
        target_terms = int(300.0 * peak_span_deg / gam_centideg)
    for num_terms in _FCJ_QUADRATURE_NTERMS:
        if target_terms <= num_terms:
            return num_terms
    return _FCJ_QUADRATURE_NTERMS[-1]


def _select_fcj_num_terms_tensor(
    peak_pos_deg: torch.Tensor,
    emin_deg: torch.Tensor,
    gam_centideg: torch.Tensor,
) -> torch.Tensor:
    peak_span_deg = torch.abs(peak_pos_deg - emin_deg)
    safe_gamma = torch.where(gam_centideg > 0.0, gam_centideg, torch.ones_like(gam_centideg))
    target_terms = torch.where(
        gam_centideg <= 0.0,
        peak_span_deg * 200.0,
        300.0 * peak_span_deg / safe_gamma,
    ).to(dtype=torch.int64)
    supported_terms = torch.as_tensor(_FCJ_QUADRATURE_NTERMS, dtype=torch.int64, device=peak_pos_deg.device)
    term_indices = torch.searchsorted(supported_terms, torch.clamp(target_terms, min=0), right=False)
    term_indices = torch.clamp(term_indices, max=supported_terms.shape[0] - 1)
    return supported_terms[term_indices]


def _pseudo_voigt_torch(
    dx_centideg: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
) -> torch.Tensor:
    sqrt_sig = torch.clamp(torch.sqrt(torch.clamp(sig_centideg2, min=0.0)), min=0.001)
    fwhg = _STOFW * sqrt_sig
    sumhm = (
        fwhg.pow(5)
        + 2.69269 * fwhg.pow(4) * gam_centideg
        + 2.42843 * fwhg.pow(3) * gam_centideg.pow(2)
        + 4.47163 * fwhg.pow(2) * gam_centideg.pow(3)
        + 0.07842 * fwhg * gam_centideg.pow(4)
        + gam_centideg.pow(5)
    )
    fwhm = torch.pow(torch.clamp(sumhm, min=1.0e-18), 0.2)
    frac = gam_centideg / torch.clamp(fwhm, min=1.0e-12)
    eta = 1.36603 * frac - 0.47719 * frac.pow(2) + 0.11116 * frac.pow(3)

    denom = fwhm.pow(2) + 4.0 * dx_centideg.pow(2)
    lorentz = (2.0 * fwhm / _PI) / torch.clamp(denom, min=1.0e-18)

    sigp = torch.clamp((fwhm / _STOFW).pow(2), min=1.0e-18)
    exponent = torch.clamp(-0.5 * dx_centideg.pow(2) / sigp, min=-20.0)
    gaussian = _STOFW * torch.exp(exponent) / (_SQ2PI * torch.clamp(fwhm, min=1.0e-12))
    return eta * lorentz + (1.0 - eta) * gaussian


class _PseudoVoigtCustomBackward(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        dx_centideg: torch.Tensor,
        sig_centideg2: torch.Tensor,
        gam_centideg: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_sig = torch.clamp(torch.sqrt(torch.clamp(sig_centideg2, min=0.0)), min=0.001)
        fwhg = _STOFW * sqrt_sig
        sumhm = (
            fwhg.pow(5)
            + 2.69269 * fwhg.pow(4) * gam_centideg
            + 2.42843 * fwhg.pow(3) * gam_centideg.pow(2)
            + 4.47163 * fwhg.pow(2) * gam_centideg.pow(3)
            + 0.07842 * fwhg * gam_centideg.pow(4)
            + gam_centideg.pow(5)
        )
        fwhm = torch.pow(torch.clamp(sumhm, min=1.0e-18), 0.2)
        frac = gam_centideg / torch.clamp(fwhm, min=1.0e-12)
        eta = 1.36603 * frac - 0.47719 * frac.pow(2) + 0.11116 * frac.pow(3)

        denom = fwhm.pow(2) + 4.0 * dx_centideg.pow(2)
        lorentz = (2.0 * fwhm / _PI) / torch.clamp(denom, min=1.0e-18)

        sigp = torch.clamp((fwhm / _STOFW).pow(2), min=1.0e-18)
        exponent_unclamped = -0.5 * dx_centideg.pow(2) / sigp
        exponent_active = exponent_unclamped > -20.0
        exponent = torch.clamp(exponent_unclamped, min=-20.0)
        gaussian = _STOFW * torch.exp(exponent) / (_SQ2PI * torch.clamp(fwhm, min=1.0e-12))
        ctx.save_for_backward(dx_centideg, sig_centideg2, gam_centideg, fwhg, fwhm, eta, lorentz, gaussian, exponent_active)
        return eta * lorentz + (1.0 - eta) * gaussian

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        dx_centideg, sig_centideg2, gam_centideg, fwhg, fwhm, eta, lorentz, gaussian, exponent_active = ctx.saved_tensors
        denom = fwhm.pow(2) + 4.0 * dx_centideg.pow(2)
        denom_sq = denom.pow(2)
        d_lorentz_dx = -(16.0 * fwhm * dx_centideg / _PI) / denom_sq
        d_lorentz_dfwhm = (2.0 / _PI) * (4.0 * dx_centideg.pow(2) - fwhm.pow(2)) / denom_sq

        exponent_mask = exponent_active.to(dtype=dx_centideg.dtype)
        d_exponent_dx = -_STOFW**2 * dx_centideg / fwhm.pow(2)
        d_exponent_dfwhm = _STOFW**2 * dx_centideg.pow(2) / fwhm.pow(3)
        d_gaussian_dx = gaussian * d_exponent_dx * exponent_mask
        d_gaussian_dfwhm = gaussian * (-1.0 / fwhm + d_exponent_dfwhm * exponent_mask)

        frac = gam_centideg / fwhm
        d_eta_dfrac = 1.36603 - 2.0 * 0.47719 * frac + 3.0 * 0.11116 * frac.pow(2)
        d_frac_dfwhm = -gam_centideg / fwhm.pow(2)
        d_frac_dgam = 1.0 / fwhm

        d_eta_dfwhm = d_eta_dfrac * d_frac_dfwhm
        d_eta_dgam_direct = d_eta_dfrac * d_frac_dgam
        common_dfwhm = (
            eta * d_lorentz_dfwhm
            + (1.0 - eta) * d_gaussian_dfwhm
            + d_eta_dfwhm * (lorentz - gaussian)
        )
        d_output_dx = eta * d_lorentz_dx + (1.0 - eta) * d_gaussian_dx

        d_sum_dfwhg = (
            5.0 * fwhg.pow(4)
            + 4.0 * 2.69269 * fwhg.pow(3) * gam_centideg
            + 3.0 * 2.42843 * fwhg.pow(2) * gam_centideg.pow(2)
            + 2.0 * 4.47163 * fwhg * gam_centideg.pow(3)
            + 0.07842 * gam_centideg.pow(4)
        )
        d_sum_dgam = (
            2.69269 * fwhg.pow(4)
            + 2.0 * 2.42843 * fwhg.pow(3) * gam_centideg
            + 3.0 * 4.47163 * fwhg.pow(2) * gam_centideg.pow(2)
            + 4.0 * 0.07842 * fwhg * gam_centideg.pow(3)
            + 5.0 * gam_centideg.pow(4)
        )
        d_fwhm_dsum = 0.2 * fwhm.pow(-4)
        active_sig = sig_centideg2 > 1.0e-6
        d_fwhg_dsig = torch.where(
            active_sig,
            fwhg.new_tensor(_STOFW * 0.5) / torch.clamp(torch.sqrt(sig_centideg2), min=1.0e-12),
            torch.zeros_like(sig_centideg2),
        )
        d_fwhm_dsig = d_fwhm_dsum * d_sum_dfwhg * d_fwhg_dsig
        d_fwhm_dgam = d_fwhm_dsum * d_sum_dgam

        d_output_dfwhm = common_dfwhm
        d_output_dsig = d_output_dfwhm * d_fwhm_dsig
        d_output_dgam = d_output_dfwhm * d_fwhm_dgam + d_eta_dgam_direct * (lorentz - gaussian)

        grad_dx = (grad_output * d_output_dx).sum_to_size(dx_centideg.shape)
        grad_sig = (grad_output * d_output_dsig).sum_to_size(sig_centideg2.shape)
        grad_gam = (grad_output * d_output_dgam).sum_to_size(gam_centideg.shape)
        return grad_dx, grad_sig, grad_gam


def _pseudo_voigt_custom_backward_torch(
    dx_centideg: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
) -> torch.Tensor:
    return _PseudoVoigtCustomBackward.apply(dx_centideg, sig_centideg2, gam_centideg)


def _fcj_peak_kernel_torch(
    peak_pos_deg: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
    shl_value: torch.Tensor,
    peak_domain_deg: torch.Tensor,
) -> torch.Tensor:
    if float(abs(shl_value.detach()).item()) <= 0.0:
        return _pseudo_voigt_torch((peak_domain_deg - peak_pos_deg) * 100.0, sig_centideg2, gam_centideg)

    theta_rad = torch.deg2rad(peak_pos_deg)
    sin_two_theta = torch.sin(theta_rad)
    cos_two_theta = torch.cos(theta_rad)
    cos_two_theta_sq = cos_two_theta.pow(2)
    sin_two_theta_sq = sin_two_theta.pow(2)

    emin_arg = torch.sqrt(1.0 + shl_value.pow(2)) * cos_two_theta
    emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))

    num_terms = _select_fcj_num_terms(
        float(peak_pos_deg.detach().item()),
        float(emin_deg.detach().item()),
        float(gam_centideg.detach().item()),
    )
    xp, wp = _half_gauss_legendre_rule_torch(
        num_terms,
        device=peak_domain_deg.device,
        dtype=peak_domain_deg.dtype,
    )

    delta_deg = emin_deg + (peak_pos_deg - emin_deg) * xp
    delta_rad = torch.deg2rad(delta_deg)
    sin_delta = torch.sin(delta_rad)
    cos_delta = torch.cos(delta_rad)
    safe_cos_delta = torch.where(
        cos_delta.abs() < 1.0e-12,
        torch.copysign(cos_delta.new_full((), 1.0e-12), cos_delta + (cos_delta == 0.0).to(cos_delta.dtype)),
        cos_delta,
    )
    reciprocal_cos_delta = 1.0 / safe_cos_delta

    low_angle_term = sin_two_theta_sq - sin_delta.pow(2)
    high_angle_term = cos_delta.pow(2) - cos_two_theta_sq
    integration_term = torch.where(peak_pos_deg > 45.0, high_angle_term, low_angle_term)
    positive_integration = integration_term > 0.0
    safe_sqrt_integration = torch.sqrt(
        torch.where(
            positive_integration,
            integration_term,
            torch.ones_like(integration_term),
        )
    )
    f_term = torch.where(
        positive_integration,
        torch.abs(cos_two_theta) / safe_sqrt_integration,
        torch.ones_like(integration_term),
    )
    g_term = (-1.0 + shl_value * f_term) * reciprocal_cos_delta
    weighted_g = wp * g_term
    weighted_sum = torch.sum(weighted_g)
    safe_weighted_sum = torch.where(
        weighted_sum.abs() > 0.0,
        weighted_sum,
        weighted_sum.new_tensor(1.0),
    )

    dx_centideg = (peak_domain_deg.unsqueeze(1) - delta_deg.unsqueeze(0)) * 100.0
    pseudo_voigt = _pseudo_voigt_torch(dx_centideg, sig_centideg2, gam_centideg)
    return torch.sum(pseudo_voigt * weighted_g.unsqueeze(0), dim=1) / safe_weighted_sum


def _compute_fcj_window_bounds(
    peak_pos_deg: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
    shl_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    widths_gauss = torch.sqrt(torch.clamp(sig_centideg2, min=0.0)) / 100.0
    widths_lorentz = gam_centideg / 100.0
    fwhm = torch.clamp(2.355 * widths_gauss + widths_lorentz, min=1.0e-4)
    fmin = 50.0 * (fwhm + shl_value * torch.abs(torch.cos(torch.deg2rad(peak_pos_deg))))
    fmax = 75.0 * fwhm
    lower_span = torch.where(peak_pos_deg > 90.0, fmax, fmin)
    upper_span = torch.where(peak_pos_deg > 90.0, fmin, fmax)
    return peak_pos_deg - lower_span, peak_pos_deg + upper_span


def _expand_fcj_diagnostic_shl(shl_values: torch.Tensor, peak_pos_deg: torch.Tensor) -> torch.Tensor:
    _validate_fcj_shl(shl_values)
    shl_tensor = shl_values.to(device=peak_pos_deg.device)
    if shl_tensor.ndim == 1 and peak_pos_deg.ndim == 2:
        if shl_tensor.shape[0] != peak_pos_deg.shape[0]:
            raise ValueError("one-dimensional diagnostic SH/L must match the batch size")
        shl_tensor = shl_tensor.unsqueeze(1)
    return shl_tensor.expand_as(peak_pos_deg)


def compute_fcj_window_bounds(
    peak_pos_deg: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    shl_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return internal-precision bounds; SH/L may be scalar, per row, or per peak."""
    shl_tensor = _expand_fcj_diagnostic_shl(shl_values, peak_pos_deg)
    bounds = _compute_fcj_window_bounds(
        peak_pos_deg,
        sigma_sq_centideg2.to(dtype=peak_pos_deg.dtype),
        gamma_centideg.to(dtype=peak_pos_deg.dtype),
        shl_tensor.to(dtype=peak_pos_deg.dtype),
    )
    asymmetric = shl_tensor > 0
    if peak_pos_deg.dtype == torch.float32 and bool(asymmetric.any()):
        promoted_bounds = _compute_fcj_window_bounds(
            peak_pos_deg.to(torch.float64), sigma_sq_centideg2.to(torch.float64),
            gamma_centideg.to(torch.float64), shl_tensor.to(torch.float64),
        )
        return tuple(torch.where(asymmetric, promoted, native.to(torch.float64))
                     for promoted, native in zip(promoted_bounds, bounds, strict=True))
    return bounds


def select_fcj_num_terms(
    peak_pos_deg: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    """Return the term-selection diagnostic for scalar, row, or per-peak SH/L."""
    _ = sigma_sq_centideg2
    expanded_shl = _expand_fcj_diagnostic_shl(shl_values, peak_pos_deg)
    shl_tensor = expanded_shl.to(dtype=peak_pos_deg.dtype)
    emin_arg = torch.sqrt(1.0 + shl_tensor.pow(2)) * torch.cos(torch.deg2rad(peak_pos_deg))
    emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))
    native = _select_fcj_num_terms_tensor(peak_pos_deg, emin_deg, gamma_centideg.to(dtype=peak_pos_deg.dtype))
    if peak_pos_deg.dtype == torch.float32 and bool((expanded_shl > 0).any()):
        promoted = select_fcj_num_terms(
            peak_pos_deg.to(torch.float64), sigma_sq_centideg2.to(torch.float64),
            gamma_centideg.to(torch.float64), shl_values.to(torch.float64),
        )
        return torch.where(expanded_shl > 0, promoted, native)
    return native


def _prepare_fcj_peaks(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
    *,
    validate_inputs: bool = True,
) -> _PreparedFcjPeaks:
    if validate_inputs:
        _validate_fcj_inputs(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    internal_dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
    positions_internal = positions.to(dtype=internal_dtype)
    intensities_internal = intensities.to(dtype=internal_dtype)
    sigma_internal = sigma_sq_centideg2.to(dtype=internal_dtype)
    gamma_internal = gamma_centideg.to(dtype=internal_dtype)
    domain_internal = domain.to(device=positions.device, dtype=internal_dtype)
    shl_internal = shl_values.to(device=positions.device, dtype=internal_dtype)

    batch_size, num_peaks = positions_internal.shape
    domain_size = int(domain_internal.shape[0])
    batch_ids = torch.arange(batch_size, device=positions.device, dtype=torch.long).unsqueeze(1).expand(batch_size, num_peaks)
    peak_shl = (
        shl_internal.reshape(1, 1).expand(batch_size, num_peaks)
        if shl_internal.ndim == 0
        else shl_internal.reshape(batch_size, 1).expand(batch_size, num_peaks)
    )
    clamped_sigma = torch.clamp(sigma_internal, min=1.0e-6)
    clamped_gamma = torch.clamp(gamma_internal, min=1.0e-6)
    valid_mask = (
        (intensities_internal > 0.0)
        & torch.isfinite(intensities_internal)
        & torch.isfinite(positions_internal)
        & torch.isfinite(clamped_sigma)
        & torch.isfinite(clamped_gamma)
    )

    peak_batch_ids = batch_ids[valid_mask]
    peak_pos = positions_internal[valid_mask]
    peak_intensities = intensities_internal[valid_mask]
    peak_sigma = clamped_sigma[valid_mask]
    peak_gamma = clamped_gamma[valid_mask]
    peak_shl_values = peak_shl[valid_mask]
    if peak_pos.numel() > 0:
        lower_bounds, upper_bounds = _compute_fcj_window_bounds(
            peak_pos,
            peak_sigma,
            peak_gamma,
            peak_shl_values,
        )
        i_beg = torch.searchsorted(domain_internal, lower_bounds, right=False).to(dtype=torch.long)
        i_fin = torch.searchsorted(domain_internal, upper_bounds, right=False).to(dtype=torch.long)
        window_mask = (i_fin - i_beg) >= 2
        peak_batch_ids = peak_batch_ids[window_mask]
        peak_pos = peak_pos[window_mask]
        peak_intensities = peak_intensities[window_mask]
        peak_sigma = peak_sigma[window_mask]
        peak_gamma = peak_gamma[window_mask]
        peak_shl_values = peak_shl_values[window_mask]
        i_beg = i_beg[window_mask]
        i_fin = i_fin[window_mask]
    else:
        i_beg = peak_batch_ids.new_empty(0)
        i_fin = peak_batch_ids.new_empty(0)

    selected_terms = peak_batch_ids.new_zeros(peak_pos.shape)
    zero_shl_mask = torch.abs(peak_shl_values) <= 0.0
    nonzero_shl_mask = ~zero_shl_mask
    if bool(nonzero_shl_mask.any()):
        nz_pos = peak_pos[nonzero_shl_mask]
        nz_shl = peak_shl_values[nonzero_shl_mask]
        nz_gamma = peak_gamma[nonzero_shl_mask]
        emin_arg = torch.sqrt(1.0 + nz_shl.pow(2)) * torch.cos(torch.deg2rad(nz_pos))
        emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))
        selected_terms[nonzero_shl_mask] = _select_fcj_num_terms_tensor(nz_pos, emin_deg, nz_gamma)
    two_term_mask = nonzero_shl_mask & (selected_terms == 2)
    highq_mask = nonzero_shl_mask & ~two_term_mask
    return _PreparedFcjPeaks(
        batch_size=int(batch_size),
        domain_size=int(domain_size),
        domain=domain_internal,
        peak_batch_ids=peak_batch_ids,
        peak_pos=peak_pos,
        peak_intensities=peak_intensities,
        peak_sigma=peak_sigma,
        peak_gamma=peak_gamma,
        peak_shl_values=peak_shl_values,
        i_beg=i_beg,
        i_fin=i_fin,
        selected_terms=selected_terms,
        zero_shl_mask=zero_shl_mask,
        two_term_mask=two_term_mask,
        highq_mask=highq_mask,
    )


def _choose_fcj_peak_chunk_size(
    max_window_size: int,
    num_terms: int | None,
    dtype: torch.dtype,
) -> int:
    quadrature_terms = 1 if num_terms is None else max(int(num_terms) // 2, 1)
    element_budget = (
        _FCJ_BUCKET_ELEMENT_BUDGET_FLOAT64
        if dtype == torch.float64
        else _FCJ_BUCKET_ELEMENT_BUDGET_FLOAT32
    )
    chunk_size = element_budget // max(max_window_size * quadrature_terms, 1)
    return max(1, min(_FCJ_MAX_PEAK_CHUNK, int(chunk_size)))


def _build_peak_domain_windows(
    domain: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    *,
    max_window_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_max_window_size = (
        int((i_fin - i_beg).max().item())
        if max_window_size is None
        else int(max_window_size)
    )
    offsets = torch.arange(resolved_max_window_size, device=domain.device, dtype=torch.long)
    domain_indices = i_beg.unsqueeze(1) + offsets.unsqueeze(0)
    valid_mask = domain_indices < i_fin.unsqueeze(1)
    safe_indices = torch.where(valid_mask, domain_indices, i_beg.unsqueeze(1))
    return safe_indices, domain[safe_indices], valid_mask


def _fcj_peak_kernel_bucketed(
    peak_pos_deg: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
    shl_values: torch.Tensor,
    peak_domain_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_terms: int,
) -> torch.Tensor:
    if peak_pos_deg.numel() == 0:
        return peak_domain_deg.new_zeros(peak_domain_deg.shape)

    peak_pos_column = peak_pos_deg.unsqueeze(1)
    shl_column = (
        shl_values.expand_as(peak_pos_deg).unsqueeze(1)
        if shl_values.ndim == 0
        else shl_values.reshape(-1).unsqueeze(1)
    )
    theta_rad = torch.deg2rad(peak_pos_column)
    sin_two_theta = torch.sin(theta_rad)
    cos_two_theta = torch.cos(theta_rad)
    cos_two_theta_sq = cos_two_theta.pow(2)
    sin_two_theta_sq = sin_two_theta.pow(2)

    emin_arg = torch.sqrt(1.0 + shl_column.pow(2)) * cos_two_theta
    emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))
    xp, wp = _half_gauss_legendre_rule_torch(
        num_terms,
        device=peak_domain_deg.device,
        dtype=peak_domain_deg.dtype,
    )
    xp = xp.unsqueeze(0)
    wp = wp.unsqueeze(0)

    delta_deg = emin_deg + (peak_pos_column - emin_deg) * xp
    if int(num_terms) == 2:
        dx_centideg = (peak_domain_deg - delta_deg) * 100.0
        kernel = _pseudo_voigt_custom_backward_torch(
            dx_centideg,
            sig_centideg2.reshape(-1, 1),
            gam_centideg.reshape(-1, 1),
        )
        return torch.where(valid_mask, kernel, torch.zeros_like(kernel))

    delta_rad = torch.deg2rad(delta_deg)
    sin_delta = torch.sin(delta_rad)
    cos_delta = torch.cos(delta_rad)
    safe_cos_delta = torch.where(
        cos_delta.abs() < 1.0e-12,
        torch.copysign(
            cos_delta.new_full(cos_delta.shape, 1.0e-12),
            cos_delta + (cos_delta == 0.0).to(cos_delta.dtype),
        ),
        cos_delta,
    )
    reciprocal_cos_delta = 1.0 / safe_cos_delta

    low_angle_term = sin_two_theta_sq - sin_delta.pow(2)
    high_angle_term = cos_delta.pow(2) - cos_two_theta_sq
    integration_term = torch.where(peak_pos_column > 45.0, high_angle_term, low_angle_term)
    positive_integration = integration_term > 0.0
    safe_sqrt_integration = torch.sqrt(
        torch.where(
            positive_integration,
            integration_term,
            torch.ones_like(integration_term),
        )
    )
    f_term = torch.where(
        positive_integration,
        torch.abs(cos_two_theta) / safe_sqrt_integration,
        torch.ones_like(integration_term),
    )
    g_term = (-1.0 + shl_column * f_term) * reciprocal_cos_delta
    weighted_g = wp * g_term
    weighted_sum = torch.sum(weighted_g, dim=1, keepdim=True)
    safe_weighted_sum = torch.where(
        weighted_sum.abs() > 0.0,
        weighted_sum,
        torch.ones_like(weighted_sum),
    )

    dx_centideg = (peak_domain_deg.unsqueeze(2) - delta_deg.unsqueeze(1)) * 100.0
    pseudo_voigt = _pseudo_voigt_custom_backward_torch(
        dx_centideg,
        sig_centideg2.reshape(-1, 1, 1),
        gam_centideg.reshape(-1, 1, 1),
    )
    kernel = torch.sum(pseudo_voigt * weighted_g.unsqueeze(1), dim=2) / safe_weighted_sum
    return torch.where(valid_mask, kernel, torch.zeros_like(kernel))


def _accumulate_fcj_bucket(
    histogram_flat: torch.Tensor,
    domain: torch.Tensor,
    domain_size: int,
    batch_indices: torch.Tensor,
    peak_pos_deg: torch.Tensor,
    intensities: torch.Tensor,
    sig_centideg2: torch.Tensor,
    gam_centideg: torch.Tensor,
    shl_values: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    *,
    num_terms: int | None,
    max_window_size: int | None = None,
    single_batch_offset: int | None = None,
) -> None:
    if peak_pos_deg.numel() == 0:
        return

    resolved_max_window_size = (
        int((i_fin - i_beg).max().item())
        if max_window_size is None
        else int(max_window_size)
    )
    if resolved_max_window_size <= 0:
        return
    peak_chunk_size = _choose_fcj_peak_chunk_size(resolved_max_window_size, num_terms, domain.dtype)
    for start_index in range(0, peak_pos_deg.shape[0], peak_chunk_size):
        stop_index = min(start_index + peak_chunk_size, peak_pos_deg.shape[0])
        chunk_pos = peak_pos_deg[start_index:stop_index]
        chunk_intensities = intensities[start_index:stop_index]
        chunk_sig = sig_centideg2[start_index:stop_index]
        chunk_gam = gam_centideg[start_index:stop_index]
        chunk_batch_indices = batch_indices[start_index:stop_index]
        chunk_i_beg = i_beg[start_index:stop_index]
        chunk_i_fin = i_fin[start_index:stop_index]
        chunk_shl = shl_values if shl_values.ndim == 0 else shl_values[start_index:stop_index]
        domain_indices, chunk_domain, valid_mask = _build_peak_domain_windows(
            domain,
            chunk_i_beg,
            chunk_i_fin,
            max_window_size=resolved_max_window_size,
        )
        if num_terms is None:
            kernel = _pseudo_voigt_custom_backward_torch(
                (chunk_domain - chunk_pos.unsqueeze(1)) * 100.0,
                chunk_sig.unsqueeze(1),
                chunk_gam.unsqueeze(1),
            )
            kernel = torch.where(valid_mask, kernel, torch.zeros_like(kernel))
        else:
            kernel = _fcj_peak_kernel_bucketed(
                chunk_pos,
                chunk_sig,
                chunk_gam,
                chunk_shl,
                chunk_domain,
                valid_mask,
                num_terms=num_terms,
            )
        contributions = chunk_intensities.unsqueeze(1) * kernel
        if single_batch_offset is None:
            global_indices = chunk_batch_indices.unsqueeze(1) * domain_size + domain_indices
        elif int(single_batch_offset) == 0:
            global_indices = domain_indices
        else:
            global_indices = domain_indices + int(single_batch_offset)
        histogram_flat.index_add_(
            0,
            global_indices.reshape(-1),
            contributions.reshape(-1),
        )


def compute_fcj_profile_histogram_torch(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    _validate_fcj_inputs(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
    )
    promoted = _promote_asymmetric_fcj_rows(
        compute_fcj_profile_histogram_torch,
        positions, intensities, sigma_sq_centideg2, gamma_centideg, domain, shl_values,
    )
    if promoted is not None:
        return promoted
    internal_dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
    positions64 = positions.to(dtype=internal_dtype)
    intensities64 = intensities.to(dtype=internal_dtype)
    sigma64 = sigma_sq_centideg2.to(dtype=internal_dtype)
    gamma64 = gamma_centideg.to(dtype=internal_dtype)
    domain64 = domain.to(device=positions.device, dtype=internal_dtype)
    shl64 = shl_values.to(device=positions.device, dtype=internal_dtype)

    histograms: list[torch.Tensor] = []
    batch_size, num_peaks = positions64.shape
    if batch_size == 0:
        return positions.new_zeros((0, domain.numel()))
    scalar_shl = shl64.ndim == 0
    for batch_index in range(batch_size):
        batch_histogram = domain64.new_zeros(domain64.shape[0])
        batch_shl = shl64 if scalar_shl else shl64[batch_index]
        batch_shl_scalar = float(batch_shl.detach().item())
        for peak_index in range(num_peaks):
            peak_intensity = intensities64[batch_index, peak_index]
            if float(peak_intensity.detach().item()) <= 0.0 or not bool(torch.isfinite(peak_intensity).item()):
                continue

            peak_pos = positions64[batch_index, peak_index]
            peak_sig = torch.clamp(sigma64[batch_index, peak_index], min=1.0e-6)
            peak_gam = torch.clamp(gamma64[batch_index, peak_index], min=1.0e-6)
            if not (
                bool(torch.isfinite(peak_pos).item())
                and bool(torch.isfinite(peak_sig).item())
                and bool(torch.isfinite(peak_gam).item())
            ):
                continue

            peak_pos_scalar = float(peak_pos.detach().item())
            peak_sig_scalar = float(peak_sig.detach().item())
            peak_gam_scalar = float(peak_gam.detach().item())
            _, fmin_deg, fmax_deg = _get_widths_cw(
                peak_pos_scalar,
                peak_sig_scalar,
                peak_gam_scalar,
                batch_shl_scalar,
            )
            lower_bound = domain64.new_tensor(peak_pos_scalar - fmin_deg)
            upper_bound = domain64.new_tensor(peak_pos_scalar + fmax_deg)
            i_beg = int(torch.searchsorted(domain64, lower_bound, right=False).item())
            i_fin = int(torch.searchsorted(domain64, upper_bound, right=False).item())
            if i_fin - i_beg < 2:
                continue

            peak_kernel = _fcj_peak_kernel_torch(
                peak_pos,
                peak_sig,
                peak_gam,
                batch_shl,
                domain64[i_beg:i_fin],
            )
            peak_indices = torch.arange(i_beg, i_fin, device=domain64.device, dtype=torch.long)
            batch_histogram = batch_histogram.index_add(0, peak_indices, peak_intensity * peak_kernel)
        histograms.append(batch_histogram)

    return torch.stack(histograms, dim=0).to(dtype=positions.dtype)


def compute_fcj_profile_histogram_bucketed(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    _validate_fcj_inputs(positions, intensities, sigma_sq_centideg2, gamma_centideg, domain, shl_values)
    promoted = _promote_asymmetric_fcj_rows(
        compute_fcj_profile_histogram_bucketed,
        positions, intensities, sigma_sq_centideg2, gamma_centideg, domain, shl_values,
    )
    if promoted is not None:
        return promoted
    prepared = _prepare_fcj_peaks(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
        validate_inputs=False,
    )
    if prepared.peak_pos.numel() == 0:
        return prepared.domain.new_zeros((prepared.batch_size, prepared.domain_size), dtype=positions.dtype)

    histogram_flat = prepared.domain.new_zeros(prepared.batch_size * prepared.domain_size)
    single_batch_offset = 0 if prepared.batch_size == 1 else None
    if bool(prepared.zero_shl_mask.any()):
        _accumulate_fcj_bucket(
            histogram_flat,
            prepared.domain,
            prepared.domain_size,
            prepared.peak_batch_ids[prepared.zero_shl_mask],
            prepared.peak_pos[prepared.zero_shl_mask],
            prepared.peak_intensities[prepared.zero_shl_mask],
            prepared.peak_sigma[prepared.zero_shl_mask],
            prepared.peak_gamma[prepared.zero_shl_mask],
            prepared.peak_shl_values[prepared.zero_shl_mask],
            prepared.i_beg[prepared.zero_shl_mask],
            prepared.i_fin[prepared.zero_shl_mask],
            num_terms=None,
            single_batch_offset=single_batch_offset,
        )

    nonzero_shl_mask = ~prepared.zero_shl_mask
    if bool(nonzero_shl_mask.any()):
        selected_terms = prepared.selected_terms[nonzero_shl_mask]
        used_terms = sorted(int(value) for value in torch.unique(selected_terms.detach()).cpu().tolist())
        for bucket_terms in used_terms:
            bucket_mask = nonzero_shl_mask & (prepared.selected_terms == int(bucket_terms))
            _accumulate_fcj_bucket(
                histogram_flat,
                prepared.domain,
                prepared.domain_size,
                prepared.peak_batch_ids[bucket_mask],
                prepared.peak_pos[bucket_mask],
                prepared.peak_intensities[bucket_mask],
                prepared.peak_sigma[bucket_mask],
                prepared.peak_gamma[bucket_mask],
                prepared.peak_shl_values[bucket_mask],
                prepared.i_beg[bucket_mask],
                prepared.i_fin[bucket_mask],
                num_terms=int(bucket_terms),
                single_batch_offset=single_batch_offset,
            )

    return histogram_flat.reshape(prepared.batch_size, prepared.domain_size).to(dtype=positions.dtype)


@lru_cache(maxsize=1)
def _compiled_bucketed_fcj_callable():
    if not hasattr(torch, "compile"):
        return None
    return torch.compile(
        compute_fcj_profile_histogram_bucketed,
        dynamic=True,
        fullgraph=False,
    )


def compute_fcj_profile_histogram_compiled(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    global _COMPILED_FCJ_DISABLED_REASON
    _validate_fcj_inputs(positions, intensities, sigma_sq_centideg2, gamma_centideg, domain, shl_values)
    promoted = _promote_asymmetric_fcj_rows(
        compute_fcj_profile_histogram_compiled,
        positions, intensities, sigma_sq_centideg2, gamma_centideg, domain, shl_values,
    )
    if promoted is not None:
        return promoted
    if _COMPILED_FCJ_DISABLED_REASON is not None:
        return compute_fcj_profile_histogram_bucketed(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    compiled = _compiled_bucketed_fcj_callable()
    if compiled is None:
        return compute_fcj_profile_histogram_bucketed(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    try:
        return compiled(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    except Exception as exc:
        _COMPILED_FCJ_DISABLED_REASON = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        warnings.warn(
            "torch.compile FCJ backend failed; falling back to bucketed FCJ backend. "
            f"Reason: {_COMPILED_FCJ_DISABLED_REASON}",
            RuntimeWarning,
            stacklevel=2,
        )
        return compute_fcj_profile_histogram_bucketed(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )


def compute_fcj_profile_histogram(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    """Render FCJ in double precision for positive SH/L, returning positions.dtype.

    Float32 rows with exactly zero SH/L retain the native symmetric path. Casts
    remain in the autograd graph; upstream scattering and loss dtypes are not
    changed. SH/L must be finite and nonnegative.
    """
    backend_key = _canonicalize_fcj_backend(backend)
    if backend_key == "bucketed":
        return compute_fcj_profile_histogram_bucketed(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    if backend_key == "compiled":
        return compute_fcj_profile_histogram_compiled(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    if backend_key == "triton":
        from .fcj_triton import compute_fcj_profile_histogram_triton

        return compute_fcj_profile_histogram_triton(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    if backend_key == "triton_highq":
        from .fcj_triton import compute_fcj_profile_histogram_triton_highq

        return compute_fcj_profile_histogram_triton_highq(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )
    return compute_fcj_profile_histogram_torch(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
    )


__all__ = [
    "compute_fcj_profile_histogram",
    "compute_fcj_profile_histogram_bucketed",
    "compute_fcj_profile_histogram_compiled",
    "compute_fcj_profile_histogram_torch",
    "compute_fcj_window_bounds",
    "select_fcj_num_terms",
]
