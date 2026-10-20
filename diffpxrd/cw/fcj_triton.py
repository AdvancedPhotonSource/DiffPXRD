from __future__ import annotations

from typing import Any

import torch

try:  # pragma: no cover - exercised only when Triton is installed in a CUDA env.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments keep the package importable.
    triton = None
    tl = None

from .fcj import _FCJ_QUADRATURE_NTERMS
from .fcj import _accumulate_fcj_bucket
from .fcj import _half_gauss_legendre_rule_torch
from .fcj import _prepare_fcj_peaks
from .fcj import _validate_fcj_inputs
from .fcj import compute_fcj_profile_histogram_bucketed


_TRITON_BLOCK_SIZE = 128
_HIGHQ_DENSE_GROUP_MAX_PEAKS = 2048
_HIGHQ_GROUPED_TERM_BUCKETS = (
    tuple(term for term in _FCJ_QUADRATURE_NTERMS if 2 < term <= 80),
    tuple(term for term in _FCJ_QUADRATURE_NTERMS if 80 < term <= 160),
    tuple(term for term in _FCJ_QUADRATURE_NTERMS if term > 160),
)
_HIGHQ_GROUPED_TERM_BUCKET_RANGES = tuple(
    (int(grouped_terms[0]), int(grouped_terms[-1]), grouped_terms)
    for grouped_terms in _HIGHQ_GROUPED_TERM_BUCKETS
    if grouped_terms
)


if triton is not None:

    @triton.jit
    def _fcj_pv_forward_kernel(
        positions,
        intensities,
        sigma_sq,
        gamma,
        i_beg,
        i_fin,
        batch_ids,
        domain,
        output,
        domain_size: tl.constexpr,
        max_window: tl.constexpr,
        block_size: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        begin = tl.load(i_beg + peak_id)
        end = tl.load(i_fin + peak_id)
        domain_index = begin + offsets
        valid = (offsets < max_window) & (domain_index < end)

        x_deg = tl.load(domain + domain_index, mask=valid, other=0.0)
        pos_deg = tl.load(positions + peak_id)
        intensity = tl.load(intensities + peak_id)
        sig = tl.maximum(tl.load(sigma_sq + peak_id), 1.0e-6)
        gam = tl.maximum(tl.load(gamma + peak_id), 1.0e-6)

        dx = (x_deg - pos_deg) * 100.0
        sqrt_sig = tl.maximum(tl.sqrt(tl.maximum(sig, 0.0)), 0.001)
        fwhg = 2.35482005 * sqrt_sig
        fwhg2 = fwhg * fwhg
        fwhg3 = fwhg2 * fwhg
        fwhg4 = fwhg2 * fwhg2
        fwhg5 = fwhg4 * fwhg
        gam2 = gam * gam
        gam3 = gam2 * gam
        gam4 = gam2 * gam2
        gam5 = gam4 * gam
        sumhm = (
            fwhg5
            + 2.69269 * fwhg4 * gam
            + 2.42843 * fwhg3 * gam2
            + 4.47163 * fwhg2 * gam3
            + 0.07842 * fwhg * gam4
            + gam5
        )
        fwhm = tl.exp(0.2 * tl.log(tl.maximum(sumhm, 1.0e-18)))
        frac = gam / tl.maximum(fwhm, 1.0e-12)
        eta = 1.36603 * frac - 0.47719 * frac * frac + 0.11116 * frac * frac * frac

        dx2 = dx * dx
        fwhm2 = fwhm * fwhm
        denom = tl.maximum(fwhm2 + 4.0 * dx2, 1.0e-18)
        lorentz = (2.0 * fwhm / 3.141592653589793) / denom

        sigp = tl.maximum((fwhm / 2.35482005) * (fwhm / 2.35482005), 1.0e-18)
        exponent = tl.maximum(-0.5 * dx2 / sigp, -20.0)
        gaussian = 2.35482005 * tl.exp(exponent) / (2.506628275 * tl.maximum(fwhm, 1.0e-12))
        value = intensity * (eta * lorentz + (1.0 - eta) * gaussian)

        batch_id = tl.load(batch_ids + peak_id)
        output_index = batch_id * domain_size + domain_index
        tl.atomic_add(output + output_index, value, sem="relaxed", mask=valid)

    @triton.jit
    def _fcj_pv_backward_kernel(
        positions,
        intensities,
        sigma_sq,
        gamma,
        i_beg,
        i_fin,
        batch_ids,
        domain,
        grad_output,
        grad_positions,
        grad_intensities,
        grad_sigma_sq,
        grad_gamma,
        domain_size: tl.constexpr,
        max_window: tl.constexpr,
        block_size: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        begin = tl.load(i_beg + peak_id)
        end = tl.load(i_fin + peak_id)
        domain_index = begin + offsets
        valid = (offsets < max_window) & (domain_index < end)

        batch_id = tl.load(batch_ids + peak_id)
        flat_index = batch_id * domain_size + domain_index
        upstream = tl.load(grad_output + flat_index, mask=valid, other=0.0)
        x_deg = tl.load(domain + domain_index, mask=valid, other=0.0)

        pos_deg = tl.load(positions + peak_id)
        intensity = tl.load(intensities + peak_id)
        sig = tl.maximum(tl.load(sigma_sq + peak_id), 1.0e-6)
        gam = tl.maximum(tl.load(gamma + peak_id), 1.0e-6)

        dx = (x_deg - pos_deg) * 100.0
        sqrt_sig = tl.maximum(tl.sqrt(tl.maximum(sig, 0.0)), 0.001)
        fwhg = 2.35482005 * sqrt_sig
        fwhg2 = fwhg * fwhg
        fwhg3 = fwhg2 * fwhg
        fwhg4 = fwhg2 * fwhg2
        fwhg5 = fwhg4 * fwhg
        gam2 = gam * gam
        gam3 = gam2 * gam
        gam4 = gam2 * gam2
        gam5 = gam4 * gam
        sumhm = (
            fwhg5
            + 2.69269 * fwhg4 * gam
            + 2.42843 * fwhg3 * gam2
            + 4.47163 * fwhg2 * gam3
            + 0.07842 * fwhg * gam4
            + gam5
        )
        fwhm = tl.exp(0.2 * tl.log(tl.maximum(sumhm, 1.0e-18)))
        frac = gam / tl.maximum(fwhm, 1.0e-12)
        eta = 1.36603 * frac - 0.47719 * frac * frac + 0.11116 * frac * frac * frac

        dx2 = dx * dx
        fwhm2 = fwhm * fwhm
        denom = tl.maximum(fwhm2 + 4.0 * dx2, 1.0e-18)
        denom_sq = denom * denom
        lorentz = (2.0 * fwhm / 3.141592653589793) / denom

        sigp = tl.maximum((fwhm / 2.35482005) * (fwhm / 2.35482005), 1.0e-18)
        exponent_unclamped = -0.5 * dx2 / sigp
        exponent_active = exponent_unclamped > -20.0
        exponent = tl.maximum(exponent_unclamped, -20.0)
        gaussian = 2.35482005 * tl.exp(exponent) / (2.506628275 * tl.maximum(fwhm, 1.0e-12))
        pseudo_voigt = eta * lorentz + (1.0 - eta) * gaussian

        d_lorentz_dx = -(16.0 * fwhm * dx / 3.141592653589793) / denom_sq
        d_lorentz_dfwhm = (2.0 / 3.141592653589793) * (4.0 * dx2 - fwhm2) / denom_sq
        exponent_mask = tl.where(exponent_active, 1.0, 0.0)
        d_exponent_dx = -(2.35482005 * 2.35482005) * dx / fwhm2
        d_exponent_dfwhm = (2.35482005 * 2.35482005) * dx2 / (fwhm2 * fwhm)
        d_gaussian_dx = gaussian * d_exponent_dx * exponent_mask
        d_gaussian_dfwhm = gaussian * (-1.0 / fwhm + d_exponent_dfwhm * exponent_mask)

        d_eta_dfrac = 1.36603 - 2.0 * 0.47719 * frac + 3.0 * 0.11116 * frac * frac
        d_frac_dfwhm = -gam / fwhm2
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
            5.0 * fwhg4
            + 4.0 * 2.69269 * fwhg3 * gam
            + 3.0 * 2.42843 * fwhg2 * gam2
            + 2.0 * 4.47163 * fwhg * gam3
            + 0.07842 * gam4
        )
        d_sum_dgam = (
            2.69269 * fwhg4
            + 2.0 * 2.42843 * fwhg3 * gam
            + 3.0 * 4.47163 * fwhg2 * gam2
            + 4.0 * 0.07842 * fwhg * gam3
            + 5.0 * gam4
        )
        fwhm4 = fwhm2 * fwhm2
        d_fwhm_dsum = 0.2 / fwhm4
        active_sig = sig > 1.0e-6
        d_fwhg_dsig = tl.where(
            active_sig,
            2.35482005 * 0.5 / tl.maximum(tl.sqrt(sig), 1.0e-12),
            0.0,
        )
        d_fwhm_dsig = d_fwhm_dsum * d_sum_dfwhg * d_fwhg_dsig
        d_fwhm_dgam = d_fwhm_dsum * d_sum_dgam
        d_output_dsig = common_dfwhm * d_fwhm_dsig
        d_output_dgam = common_dfwhm * d_fwhm_dgam + d_eta_dgam_direct * (lorentz - gaussian)

        masked_upstream = tl.where(valid, upstream, 0.0)
        grad_intensity_values = masked_upstream * pseudo_voigt
        common = masked_upstream * intensity
        grad_position_values = common * (-100.0 * d_output_dx)
        grad_sigma_values = common * d_output_dsig
        grad_gamma_values = common * d_output_dgam

        tl.atomic_add(grad_positions + peak_id, tl.sum(grad_position_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_intensities + peak_id, tl.sum(grad_intensity_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_sigma_sq + peak_id, tl.sum(grad_sigma_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_gamma + peak_id, tl.sum(grad_gamma_values, axis=0), sem="relaxed")

    @triton.jit
    def _fcj_highq_forward_kernel(
        intensities,
        sigma_sq,
        gamma,
        delta_deg_values,
        weighted_g_values,
        weighted_sum_values,
        i_beg,
        i_fin,
        batch_ids,
        domain,
        output,
        domain_size: tl.constexpr,
        max_window: tl.constexpr,
        half_terms: tl.constexpr,
        block_size: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        begin = tl.load(i_beg + peak_id)
        end = tl.load(i_fin + peak_id)
        domain_index = begin + offsets
        valid = (offsets < max_window) & (domain_index < end)

        x_deg = tl.load(domain + domain_index, mask=valid, other=0.0)
        intensity = tl.load(intensities + peak_id)
        sig = tl.maximum(tl.load(sigma_sq + peak_id), 1.0e-6)
        gam = tl.maximum(tl.load(gamma + peak_id), 1.0e-6)

        weighted_sum = tl.load(weighted_sum_values + peak_id)
        numerator = tl.zeros((block_size,), dtype=tl.float32)
        for term_index in range(0, half_terms):
            term_offset = peak_id * half_terms + term_index
            delta_deg = tl.load(delta_deg_values + term_offset)
            weighted_g = tl.load(weighted_g_values + term_offset)

            dx = (x_deg - delta_deg) * 100.0
            sqrt_sig = tl.maximum(tl.sqrt(tl.maximum(sig, 0.0)), 0.001)
            fwhg = 2.35482005 * sqrt_sig
            fwhg2 = fwhg * fwhg
            fwhg3 = fwhg2 * fwhg
            fwhg4 = fwhg2 * fwhg2
            fwhg5 = fwhg4 * fwhg
            gam2 = gam * gam
            gam3 = gam2 * gam
            gam4 = gam2 * gam2
            gam5 = gam4 * gam
            sumhm = (
                fwhg5
                + 2.69269 * fwhg4 * gam
                + 2.42843 * fwhg3 * gam2
                + 4.47163 * fwhg2 * gam3
                + 0.07842 * fwhg * gam4
                + gam5
            )
            fwhm = tl.exp(0.2 * tl.log(tl.maximum(sumhm, 1.0e-18)))
            frac = gam / tl.maximum(fwhm, 1.0e-12)
            eta = 1.36603 * frac - 0.47719 * frac * frac + 0.11116 * frac * frac * frac

            dx2 = dx * dx
            fwhm2 = fwhm * fwhm
            denom = tl.maximum(fwhm2 + 4.0 * dx2, 1.0e-18)
            lorentz = (2.0 * fwhm / 3.141592653589793) / denom

            sigp = tl.maximum((fwhm / 2.35482005) * (fwhm / 2.35482005), 1.0e-18)
            exponent = tl.maximum(-0.5 * dx2 / sigp, -20.0)
            gaussian = 2.35482005 * tl.exp(exponent) / (2.506628275 * tl.maximum(fwhm, 1.0e-12))
            pseudo_voigt = eta * lorentz + (1.0 - eta) * gaussian
            numerator += pseudo_voigt * weighted_g

        safe_weighted_sum = tl.where(tl.abs(weighted_sum) > 0.0, weighted_sum, 1.0)
        value = intensity * numerator / safe_weighted_sum
        batch_id = tl.load(batch_ids + peak_id)
        output_index = batch_id * domain_size + domain_index
        tl.atomic_add(output + output_index, value, sem="relaxed", mask=valid)

    @triton.jit
    def _fcj_highq_local_backward_kernel(
        positions,
        intensities,
        sigma_sq,
        gamma,
        delta_deg_values,
        weighted_g_values,
        weighted_sum_values,
        quadrature_nodes,
        quadrature_weights,
        shl_values,
        i_beg,
        i_fin,
        batch_ids,
        domain,
        grad_output,
        grad_positions,
        grad_intensities,
        grad_sigma_sq,
        grad_gamma,
        domain_size: tl.constexpr,
        max_window: tl.constexpr,
        half_terms: tl.constexpr,
        block_size: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        begin = tl.load(i_beg + peak_id)
        end = tl.load(i_fin + peak_id)
        domain_index = begin + offsets
        valid = (offsets < max_window) & (domain_index < end)

        batch_id = tl.load(batch_ids + peak_id)
        flat_index = batch_id * domain_size + domain_index
        upstream = tl.load(grad_output + flat_index, mask=valid, other=0.0)
        x_deg = tl.load(domain + domain_index, mask=valid, other=0.0)
        pos_deg = tl.load(positions + peak_id)
        intensity = tl.load(intensities + peak_id)
        sig = tl.maximum(tl.load(sigma_sq + peak_id), 1.0e-6)
        gam = tl.maximum(tl.load(gamma + peak_id), 1.0e-6)
        shl_value = tl.load(shl_values + peak_id)

        weighted_sum = tl.load(weighted_sum_values + peak_id)
        safe_weighted_sum = tl.where(tl.abs(weighted_sum) > 0.0, weighted_sum, 1.0)
        masked_upstream = tl.where(valid, upstream, 0.0)

        numerator = tl.zeros((block_size,), dtype=tl.float32)
        for term_index in range(0, half_terms):
            term_offset = peak_id * half_terms + term_index
            delta_deg = tl.load(delta_deg_values + term_offset)
            weighted_g = tl.load(weighted_g_values + term_offset)

            dx = (x_deg - delta_deg) * 100.0
            sqrt_sig = tl.maximum(tl.sqrt(tl.maximum(sig, 0.0)), 0.001)
            fwhg = 2.35482005 * sqrt_sig
            fwhg2 = fwhg * fwhg
            fwhg3 = fwhg2 * fwhg
            fwhg4 = fwhg2 * fwhg2
            fwhg5 = fwhg4 * fwhg
            gam2 = gam * gam
            gam3 = gam2 * gam
            gam4 = gam2 * gam2
            gam5 = gam4 * gam
            sumhm = (
                fwhg5
                + 2.69269 * fwhg4 * gam
                + 2.42843 * fwhg3 * gam2
                + 4.47163 * fwhg2 * gam3
                + 0.07842 * fwhg * gam4
                + gam5
            )
            fwhm = tl.exp(0.2 * tl.log(tl.maximum(sumhm, 1.0e-18)))
            frac = gam / tl.maximum(fwhm, 1.0e-12)
            eta = 1.36603 * frac - 0.47719 * frac * frac + 0.11116 * frac * frac * frac

            dx2 = dx * dx
            fwhm2 = fwhm * fwhm
            denom = tl.maximum(fwhm2 + 4.0 * dx2, 1.0e-18)
            lorentz = (2.0 * fwhm / 3.141592653589793) / denom

            sigp = tl.maximum((fwhm / 2.35482005) * (fwhm / 2.35482005), 1.0e-18)
            exponent_unclamped = -0.5 * dx2 / sigp
            exponent = tl.maximum(exponent_unclamped, -20.0)
            gaussian = 2.35482005 * tl.exp(exponent) / (2.506628275 * tl.maximum(fwhm, 1.0e-12))
            pseudo_voigt = eta * lorentz + (1.0 - eta) * gaussian
            numerator += pseudo_voigt * weighted_g

        kernel = numerator / safe_weighted_sum
        grad_intensity_values = masked_upstream * kernel
        grad_weighted_sum_values = masked_upstream * intensity * (-numerator / (safe_weighted_sum * safe_weighted_sum))

        deg_to_rad = 0.017453292519943295
        pos_rad = pos_deg * deg_to_rad
        sin_pos = tl.sin(pos_rad)
        cos_pos = tl.cos(pos_rad)
        abs_cos_pos = tl.abs(cos_pos)
        sign_cos_pos = tl.where(cos_pos > 0.0, 1.0, tl.where(cos_pos < 0.0, -1.0, 0.0))
        sqrt_one_shl_sq = tl.sqrt(1.0 + shl_value * shl_value)
        raw_emin_arg = sqrt_one_shl_sq * cos_pos
        emin_arg = tl.minimum(tl.maximum(raw_emin_arg, -1.0), 1.0)
        d_emin_denominator = tl.sqrt(tl.maximum(1.0 - emin_arg * emin_arg, 1.0e-18))
        d_emin_dpos = tl.where(
            tl.abs(raw_emin_arg) <= 1.0,
            sqrt_one_shl_sq * sin_pos / d_emin_denominator,
            0.0,
        )

        grad_sigma_values = tl.zeros((block_size,), dtype=tl.float32)
        grad_gamma_values = tl.zeros((block_size,), dtype=tl.float32)
        grad_position_values = tl.zeros((block_size,), dtype=tl.float32)
        d_weighted_sum_dpos = tl.full((), 0.0, dtype=tl.float32)
        for term_index in range(0, half_terms):
            term_offset = peak_id * half_terms + term_index
            delta_deg = tl.load(delta_deg_values + term_offset)
            weighted_g = tl.load(weighted_g_values + term_offset)
            quadrature_node = tl.load(quadrature_nodes + term_offset)
            quadrature_weight = tl.load(quadrature_weights + term_offset)

            dx = (x_deg - delta_deg) * 100.0
            sqrt_sig = tl.maximum(tl.sqrt(tl.maximum(sig, 0.0)), 0.001)
            fwhg = 2.35482005 * sqrt_sig
            fwhg2 = fwhg * fwhg
            fwhg3 = fwhg2 * fwhg
            fwhg4 = fwhg2 * fwhg2
            fwhg5 = fwhg4 * fwhg
            gam2 = gam * gam
            gam3 = gam2 * gam
            gam4 = gam2 * gam2
            gam5 = gam4 * gam
            sumhm = (
                fwhg5
                + 2.69269 * fwhg4 * gam
                + 2.42843 * fwhg3 * gam2
                + 4.47163 * fwhg2 * gam3
                + 0.07842 * fwhg * gam4
                + gam5
            )
            fwhm = tl.exp(0.2 * tl.log(tl.maximum(sumhm, 1.0e-18)))
            frac = gam / tl.maximum(fwhm, 1.0e-12)
            eta = 1.36603 * frac - 0.47719 * frac * frac + 0.11116 * frac * frac * frac

            dx2 = dx * dx
            fwhm2 = fwhm * fwhm
            denom = tl.maximum(fwhm2 + 4.0 * dx2, 1.0e-18)
            denom_sq = denom * denom
            lorentz = (2.0 * fwhm / 3.141592653589793) / denom

            sigp = tl.maximum((fwhm / 2.35482005) * (fwhm / 2.35482005), 1.0e-18)
            exponent_unclamped = -0.5 * dx2 / sigp
            exponent_active = exponent_unclamped > -20.0
            exponent = tl.maximum(exponent_unclamped, -20.0)
            gaussian = 2.35482005 * tl.exp(exponent) / (2.506628275 * tl.maximum(fwhm, 1.0e-12))
            pseudo_voigt = eta * lorentz + (1.0 - eta) * gaussian

            d_lorentz_dx = -(16.0 * fwhm * dx / 3.141592653589793) / denom_sq
            d_lorentz_dfwhm = (2.0 / 3.141592653589793) * (4.0 * dx2 - fwhm2) / denom_sq
            exponent_mask = tl.where(exponent_active, 1.0, 0.0)
            d_exponent_dx = -(2.35482005 * 2.35482005) * dx / fwhm2
            d_exponent_dfwhm = (2.35482005 * 2.35482005) * dx2 / (fwhm2 * fwhm)
            d_gaussian_dx = gaussian * d_exponent_dx * exponent_mask
            d_gaussian_dfwhm = gaussian * (-1.0 / fwhm + d_exponent_dfwhm * exponent_mask)

            d_eta_dfrac = 1.36603 - 2.0 * 0.47719 * frac + 3.0 * 0.11116 * frac * frac
            d_frac_dfwhm = -gam / fwhm2
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
                5.0 * fwhg4
                + 4.0 * 2.69269 * fwhg3 * gam
                + 3.0 * 2.42843 * fwhg2 * gam2
                + 2.0 * 4.47163 * fwhg * gam3
                + 0.07842 * gam4
            )
            d_sum_dgam = (
                2.69269 * fwhg4
                + 2.0 * 2.42843 * fwhg3 * gam
                + 3.0 * 4.47163 * fwhg2 * gam2
                + 4.0 * 0.07842 * fwhg * gam3
                + 5.0 * gam4
            )
            fwhm4 = fwhm2 * fwhm2
            d_fwhm_dsum = 0.2 / fwhm4
            active_sig = sig > 1.0e-6
            d_fwhg_dsig = tl.where(
                active_sig,
                2.35482005 * 0.5 / tl.maximum(tl.sqrt(sig), 1.0e-12),
                0.0,
            )
            d_fwhm_dsig = d_fwhm_dsum * d_sum_dfwhg * d_fwhg_dsig
            d_fwhm_dgam = d_fwhm_dsum * d_sum_dgam
            d_output_dsig = common_dfwhm * d_fwhm_dsig
            d_output_dgam = common_dfwhm * d_fwhm_dgam + d_eta_dgam_direct * (lorentz - gaussian)

            common = masked_upstream * intensity / safe_weighted_sum
            grad_delta_values = common * weighted_g * (-100.0 * d_output_dx)
            grad_weighted_g_values = common * pseudo_voigt
            grad_sigma_values += common * weighted_g * d_output_dsig
            grad_gamma_values += common * weighted_g * d_output_dgam

            d_delta_dpos = quadrature_node + (1.0 - quadrature_node) * d_emin_dpos
            delta_rad = delta_deg * deg_to_rad
            sin_delta = tl.sin(delta_rad)
            cos_delta = tl.cos(delta_rad)
            safe_cos_delta = tl.where(
                tl.abs(cos_delta) < 1.0e-12,
                tl.where(cos_delta < 0.0, -1.0e-12, 1.0e-12),
                cos_delta,
            )
            reciprocal_cos_delta = 1.0 / safe_cos_delta
            low_angle_term = sin_pos * sin_pos - sin_delta * sin_delta
            high_angle_term = cos_delta * cos_delta - cos_pos * cos_pos
            integration_term = tl.where(pos_deg > 45.0, high_angle_term, low_angle_term)
            positive_integration = integration_term > 0.0
            safe_sqrt_integration = tl.sqrt(tl.where(positive_integration, integration_term, 1.0))
            f_term = tl.where(
                positive_integration,
                abs_cos_pos / safe_sqrt_integration,
                1.0,
            )
            d_abs_cos_dpos = sign_cos_pos * (-sin_pos * deg_to_rad)
            d_integration_dpos = 2.0 * deg_to_rad * (
                sin_pos * cos_pos - sin_delta * cos_delta * d_delta_dpos
            )
            inv_sqrt = 1.0 / safe_sqrt_integration
            inv_sqrt_cubed = inv_sqrt * inv_sqrt * inv_sqrt
            d_f_dpos = tl.where(
                positive_integration,
                d_abs_cos_dpos * inv_sqrt - 0.5 * abs_cos_pos * d_integration_dpos * inv_sqrt_cubed,
                0.0,
            )
            base_g_numerator = -1.0 + shl_value * f_term
            d_inv_cos_dpos = tl.where(
                tl.abs(cos_delta) >= 1.0e-12,
                sin_delta * deg_to_rad * d_delta_dpos / (cos_delta * cos_delta),
                0.0,
            )
            d_g_dpos = shl_value * d_f_dpos * reciprocal_cos_delta + base_g_numerator * d_inv_cos_dpos
            d_weighted_g_dpos = quadrature_weight * d_g_dpos
            d_weighted_sum_dpos += d_weighted_g_dpos
            grad_position_values += grad_delta_values * d_delta_dpos + grad_weighted_g_values * d_weighted_g_dpos

        grad_position_values += grad_weighted_sum_values * d_weighted_sum_dpos
        tl.atomic_add(grad_positions + peak_id, tl.sum(grad_position_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_intensities + peak_id, tl.sum(grad_intensity_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_sigma_sq + peak_id, tl.sum(grad_sigma_values, axis=0), sem="relaxed")
        tl.atomic_add(grad_gamma + peak_id, tl.sum(grad_gamma_values, axis=0), sem="relaxed")

    @triton.jit
    def _fcj_highq_prepare_integration_kernel(
        positions,
        shl_values,
        emin_deg_values,
        source_nodes,
        source_weights,
        delta_deg_values,
        weighted_g_values,
        weighted_sum_values,
        quadrature_nodes,
        quadrature_weights,
        half_terms: tl.constexpr,
        block_terms: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        offsets = tl.arange(0, block_terms)
        valid = offsets < half_terms

        pos_deg = tl.load(positions + peak_id)
        shl_value = tl.load(shl_values + peak_id)
        emin_deg = tl.load(emin_deg_values + peak_id)
        xp = tl.load(source_nodes + offsets, mask=valid, other=0.0)
        wp = tl.load(source_weights + offsets, mask=valid, other=0.0)

        deg_to_rad = 0.017453292519943295
        pos_rad = pos_deg * deg_to_rad
        sin_pos = tl.sin(pos_rad)
        cos_pos = tl.cos(pos_rad)
        sin_pos_sq = sin_pos * sin_pos
        cos_pos_sq = cos_pos * cos_pos

        delta_deg = emin_deg + (pos_deg - emin_deg) * xp
        delta_rad = delta_deg * deg_to_rad
        sin_delta = tl.sin(delta_rad)
        cos_delta = tl.cos(delta_rad)
        safe_cos_delta = tl.where(
            tl.abs(cos_delta) < 1.0e-12,
            tl.where(cos_delta < 0.0, -1.0e-12, 1.0e-12),
            cos_delta,
        )
        low_angle_term = sin_pos_sq - sin_delta * sin_delta
        high_angle_term = cos_delta * cos_delta - cos_pos_sq
        integration_term = tl.where(pos_deg > 45.0, high_angle_term, low_angle_term)
        positive_integration = integration_term > 0.0
        safe_sqrt_integration = tl.sqrt(tl.where(positive_integration, integration_term, 1.0))
        f_term = tl.where(
            positive_integration,
            tl.abs(cos_pos) / safe_sqrt_integration,
            1.0,
        )
        g_term = (-1.0 + shl_value * f_term) / safe_cos_delta
        weighted_g = wp * g_term

        term_offset = peak_id * half_terms + offsets
        tl.store(delta_deg_values + term_offset, delta_deg, mask=valid)
        tl.store(weighted_g_values + term_offset, weighted_g, mask=valid)
        tl.store(quadrature_nodes + term_offset, xp, mask=valid)
        tl.store(quadrature_weights + term_offset, wp, mask=valid)
        tl.store(weighted_sum_values + peak_id, tl.sum(tl.where(valid, weighted_g, 0.0), axis=0))

    @triton.jit
    def _fcj_highq_prepare_group_integration_kernel(
        positions,
        shl_values,
        emin_deg_values,
        selected_terms,
        term_values,
        source_nodes,
        source_weights,
        delta_deg_values,
        weighted_g_values,
        weighted_sum_values,
        quadrature_nodes,
        quadrature_weights,
        half_max_terms: tl.constexpr,
        term_option_count: tl.constexpr,
        block_terms: tl.constexpr,
    ):
        peak_id = tl.program_id(0)
        offsets = tl.arange(0, block_terms)
        term_count = tl.load(selected_terms + peak_id)
        option_id = tl.full((), 0, dtype=tl.int32)
        matched = term_count == -1
        for candidate_id in range(0, term_option_count):
            candidate_terms = tl.load(term_values + candidate_id)
            is_match = term_count == candidate_terms
            option_id = tl.where(is_match, candidate_id, option_id)
            matched = matched | is_match
        half_terms = term_count // 2
        valid = matched & (offsets < half_terms) & (offsets < half_max_terms)
        table_offset = option_id * half_max_terms + offsets
        xp = tl.load(source_nodes + table_offset, mask=valid, other=0.0)
        wp = tl.load(source_weights + table_offset, mask=valid, other=0.0)

        pos_deg = tl.load(positions + peak_id)
        shl_value = tl.load(shl_values + peak_id)
        theta_rad = pos_deg * 0.017453292519943295
        sin_two_theta = tl.sin(theta_rad)
        cos_two_theta = tl.cos(theta_rad)
        cos_two_theta_sq = cos_two_theta * cos_two_theta
        sin_two_theta_sq = sin_two_theta * sin_two_theta

        emin_deg = tl.load(emin_deg_values + peak_id)
        delta_deg = emin_deg + (pos_deg - emin_deg) * xp
        delta_rad = delta_deg * 0.017453292519943295
        sin_delta = tl.sin(delta_rad)
        cos_delta = tl.cos(delta_rad)
        safe_cos_delta = tl.where(
            tl.abs(cos_delta) < 1.0e-12,
            tl.where(cos_delta < 0.0, -1.0e-12, 1.0e-12),
            cos_delta,
        )
        reciprocal_cos_delta = 1.0 / safe_cos_delta
        low_angle_term = sin_two_theta_sq - sin_delta * sin_delta
        high_angle_term = cos_delta * cos_delta - cos_two_theta_sq
        integration_term = tl.where(pos_deg > 45.0, high_angle_term, low_angle_term)
        positive_integration = integration_term > 0.0
        safe_sqrt_integration = tl.sqrt(tl.where(positive_integration, integration_term, 1.0))
        f_term = tl.where(
            positive_integration,
            tl.abs(cos_two_theta) / safe_sqrt_integration,
            1.0,
        )
        g_term = (-1.0 + shl_value * f_term) * reciprocal_cos_delta
        weighted_g = tl.where(valid, wp * g_term, 0.0)
        delta_deg = tl.where(valid, delta_deg, 0.0)
        xp = tl.where(valid, xp, 0.0)
        wp = tl.where(valid, wp, 0.0)
        term_offset = peak_id * half_max_terms + offsets
        store_mask = offsets < half_max_terms
        tl.store(delta_deg_values + term_offset, delta_deg, mask=store_mask)
        tl.store(weighted_g_values + term_offset, weighted_g, mask=store_mask)
        tl.store(quadrature_nodes + term_offset, xp, mask=store_mask)
        tl.store(quadrature_weights + term_offset, wp, mask=store_mask)
        tl.store(weighted_sum_values + peak_id, tl.sum(weighted_g, axis=0))


class _SinglePseudoVoigtFcjTriton(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        positions: torch.Tensor,
        intensities: torch.Tensor,
        sigma_sq_centideg2: torch.Tensor,
        gamma_centideg: torch.Tensor,
        batch_ids: torch.Tensor,
        i_beg: torch.Tensor,
        i_fin: torch.Tensor,
        domain: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is not available")
        domain_size = int(domain.shape[0])
        max_window = int((i_fin - i_beg).max().detach().cpu().item())
        output = domain.new_zeros(int(batch_size) * domain_size)
        grid = (int(positions.shape[0]), triton.cdiv(max_window, _TRITON_BLOCK_SIZE))
        _fcj_pv_forward_kernel[grid](
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            i_beg,
            i_fin,
            batch_ids,
            domain,
            output,
            domain_size,
            max_window,
            _TRITON_BLOCK_SIZE,
            num_warps=8,
        )
        ctx.save_for_backward(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            batch_ids,
            i_beg,
            i_fin,
            domain,
        )
        ctx.batch_size = int(batch_size)
        ctx.domain_size = int(domain_size)
        ctx.max_window = int(max_window)
        return output.reshape(int(batch_size), domain_size)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        if triton is None:
            raise RuntimeError("Triton is not available")
        (
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            batch_ids,
            i_beg,
            i_fin,
            domain,
        ) = ctx.saved_tensors
        grad_positions = torch.zeros_like(positions)
        grad_intensities = torch.zeros_like(intensities)
        grad_sigma = torch.zeros_like(sigma_sq_centideg2)
        grad_gamma = torch.zeros_like(gamma_centideg)
        grid = (int(positions.shape[0]), triton.cdiv(int(ctx.max_window), _TRITON_BLOCK_SIZE))
        _fcj_pv_backward_kernel[grid](
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            i_beg,
            i_fin,
            batch_ids,
            domain,
            grad_output.contiguous().reshape(-1),
            grad_positions,
            grad_intensities,
            grad_sigma,
            grad_gamma,
            int(ctx.domain_size),
            int(ctx.max_window),
            _TRITON_BLOCK_SIZE,
            num_warps=8,
        )
        return (
            grad_positions,
            grad_intensities,
            grad_sigma,
            grad_gamma,
            None,
            None,
            None,
            None,
            None,
        )


class _HighQuadratureFcjLocalTriton(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        positions: torch.Tensor,
        intensities: torch.Tensor,
        sigma_sq_centideg2: torch.Tensor,
        gamma_centideg: torch.Tensor,
        delta_deg: torch.Tensor,
        weighted_g: torch.Tensor,
        weighted_sum: torch.Tensor,
        quadrature_nodes: torch.Tensor,
        quadrature_weights: torch.Tensor,
        shl_values: torch.Tensor,
        batch_ids: torch.Tensor,
        i_beg: torch.Tensor,
        i_fin: torch.Tensor,
        domain: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is not available")
        domain_size = int(domain.shape[0])
        max_window = int((i_fin - i_beg).max().detach().cpu().item())
        output = domain.new_zeros(int(batch_size) * domain_size)
        grid = (int(intensities.shape[0]), triton.cdiv(max_window, _TRITON_BLOCK_SIZE))
        _fcj_highq_forward_kernel[grid](
            intensities.contiguous(),
            sigma_sq_centideg2.contiguous(),
            gamma_centideg.contiguous(),
            delta_deg.contiguous(),
            weighted_g.contiguous(),
            weighted_sum.contiguous(),
            i_beg.contiguous(),
            i_fin.contiguous(),
            batch_ids.contiguous(),
            domain.contiguous(),
            output,
            domain_size,
            max_window,
            int(delta_deg.shape[1]),
            _TRITON_BLOCK_SIZE,
            num_warps=8,
        )
        ctx.save_for_backward(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            shl_values,
            batch_ids,
            i_beg,
            i_fin,
            domain,
        )
        ctx.batch_size = int(batch_size)
        ctx.domain_size = int(domain_size)
        ctx.max_window = int(max_window)
        ctx.half_terms = int(delta_deg.shape[1])
        return output.reshape(int(batch_size), domain_size)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        if triton is None:
            raise RuntimeError("Triton is not available")
        (
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            shl_values,
            batch_ids,
            i_beg,
            i_fin,
            domain,
        ) = ctx.saved_tensors
        grad_positions = torch.zeros_like(positions)
        grad_intensities = torch.zeros_like(intensities)
        grad_sigma = torch.zeros_like(sigma_sq_centideg2)
        grad_gamma = torch.zeros_like(gamma_centideg)
        grid = (int(sigma_sq_centideg2.shape[0]), triton.cdiv(int(ctx.max_window), _TRITON_BLOCK_SIZE))
        _fcj_highq_local_backward_kernel[grid](
            positions.contiguous(),
            intensities.contiguous(),
            sigma_sq_centideg2.contiguous(),
            gamma_centideg.contiguous(),
            delta_deg.contiguous(),
            weighted_g.contiguous(),
            weighted_sum.contiguous(),
            quadrature_nodes.contiguous(),
            quadrature_weights.contiguous(),
            shl_values.contiguous(),
            i_beg.contiguous(),
            i_fin.contiguous(),
            batch_ids.contiguous(),
            domain.contiguous(),
            grad_output.contiguous().reshape(-1),
            grad_positions,
            grad_intensities,
            grad_sigma,
            grad_gamma,
            int(ctx.domain_size),
            int(ctx.max_window),
            int(ctx.half_terms),
            _TRITON_BLOCK_SIZE,
            num_warps=8,
        )
        return (
            grad_positions,
            grad_intensities,
            grad_sigma,
            grad_gamma,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _run_single_pv_triton(
    *,
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    batch_ids: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    domain: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if positions.numel() == 0:
        return domain.new_zeros((int(batch_size), int(domain.shape[0])))
    return _SinglePseudoVoigtFcjTriton.apply(
        positions.contiguous(),
        intensities.contiguous(),
        sigma_sq_centideg2.contiguous(),
        gamma_centideg.contiguous(),
        batch_ids.contiguous(),
        i_beg.contiguous(),
        i_fin.contiguous(),
        domain.contiguous(),
        int(batch_size),
    )


def _prepare_highq_bucket_integration(
    *,
    bucket_terms: int,
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    domain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xp, wp = _half_gauss_legendre_rule_torch(
        int(bucket_terms),
        device=domain.device,
        dtype=domain.dtype,
    )
    peak_pos_column = positions.unsqueeze(1)
    shl_column = shl_values.reshape(-1, 1)
    theta_rad = torch.deg2rad(peak_pos_column)
    sin_two_theta = torch.sin(theta_rad)
    cos_two_theta = torch.cos(theta_rad)
    cos_two_theta_sq = cos_two_theta.pow(2)
    sin_two_theta_sq = sin_two_theta.pow(2)
    emin_arg = torch.sqrt(1.0 + shl_column.pow(2)) * cos_two_theta
    emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))
    delta_deg = emin_deg + (peak_pos_column - emin_deg) * xp.unsqueeze(0)
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
    weighted_g = wp.unsqueeze(0) * g_term
    weighted_sum = torch.sum(weighted_g, dim=1)
    return delta_deg.contiguous(), weighted_g.contiguous(), weighted_sum.contiguous(), xp.contiguous(), wp.contiguous()


def _prepare_highq_bucket_integration_triton(
    *,
    bucket_terms: int,
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    domain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        return _prepare_highq_bucket_integration(
            bucket_terms=bucket_terms,
            positions=positions,
            shl_values=shl_values,
            domain=domain,
        )
    bucket_count = int(positions.shape[0])
    half_terms = int(bucket_terms) // 2
    delta_deg = positions.new_empty((bucket_count, half_terms))
    weighted_g = positions.new_empty((bucket_count, half_terms))
    quadrature_nodes = positions.new_empty((bucket_count, half_terms))
    quadrature_weights = positions.new_empty((bucket_count, half_terms))
    weighted_sum = positions.new_empty((bucket_count,))
    if bucket_count == 0:
        return (
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
        )

    xp, wp = _half_gauss_legendre_rule_torch(
        int(bucket_terms),
        device=domain.device,
        dtype=domain.dtype,
    )
    theta_rad = torch.deg2rad(positions)
    emin_arg = torch.sqrt(1.0 + shl_values.pow(2)) * torch.cos(theta_rad)
    emin_deg = torch.rad2deg(
        torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)),
    ).contiguous()
    block_terms = 1 << (half_terms - 1).bit_length()
    _fcj_highq_prepare_integration_kernel[(bucket_count,)](
        positions.contiguous(),
        shl_values.contiguous(),
        emin_deg,
        xp.contiguous(),
        wp.contiguous(),
        delta_deg,
        weighted_g,
        weighted_sum,
        quadrature_nodes,
        quadrature_weights,
        half_terms,
        block_terms,
        num_warps=8,
    )
    return delta_deg, weighted_g, weighted_sum, quadrature_nodes, quadrature_weights


def _highq_group_term_tables(
    grouped_terms: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    half_max_terms = int(max(grouped_terms)) // 2
    term_values = torch.as_tensor(grouped_terms, device=device, dtype=torch.long)
    node_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    for bucket_terms in grouped_terms:
        nodes, weights = _half_gauss_legendre_rule_torch(
            int(bucket_terms),
            device=device,
            dtype=dtype,
        )
        half_terms = int(bucket_terms) // 2
        node_row = torch.zeros(half_max_terms, device=device, dtype=dtype)
        weight_row = torch.zeros(half_max_terms, device=device, dtype=dtype)
        node_row[:half_terms] = nodes
        weight_row[:half_terms] = weights
        node_rows.append(node_row)
        weight_rows.append(weight_row)
    return term_values, torch.stack(node_rows, dim=0).contiguous(), torch.stack(weight_rows, dim=0).contiguous()


def _prepare_highq_group_integration_triton(
    *,
    grouped_terms: tuple[int, ...],
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    selected_terms: torch.Tensor,
    domain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    peak_count = int(positions.shape[0])
    half_max_terms = int(max(grouped_terms)) // 2
    delta_deg = positions.new_empty((peak_count, half_max_terms))
    weighted_g = positions.new_empty((peak_count, half_max_terms))
    quadrature_nodes = positions.new_empty((peak_count, half_max_terms))
    quadrature_weights = positions.new_empty((peak_count, half_max_terms))
    weighted_sum = positions.new_empty((peak_count,))
    if peak_count == 0:
        return (
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
        )

    term_values, nodes, weights = _highq_group_term_tables(
        tuple(int(value) for value in grouped_terms),
        device=domain.device,
        dtype=domain.dtype,
    )
    theta_rad = torch.deg2rad(positions)
    emin_arg = torch.sqrt(1.0 + shl_values.pow(2)) * torch.cos(theta_rad)
    emin_deg = torch.rad2deg(
        torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)),
    ).contiguous()
    block_terms = 1 << (half_max_terms - 1).bit_length()
    _fcj_highq_prepare_group_integration_kernel[(peak_count,)](
        positions.contiguous(),
        shl_values.contiguous(),
        emin_deg,
        selected_terms.contiguous(),
        term_values.contiguous(),
        nodes,
        weights,
        delta_deg,
        weighted_g,
        weighted_sum,
        quadrature_nodes,
        quadrature_weights,
        half_max_terms,
        int(term_values.shape[0]),
        block_terms,
        num_warps=8,
    )
    return delta_deg, weighted_g, weighted_sum, quadrature_nodes, quadrature_weights


def _iter_highq_grouped_integrations(
    *,
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    selected_terms: torch.Tensor,
    domain: torch.Tensor,
):
    for min_group_terms, max_group_terms, grouped_terms in _HIGHQ_GROUPED_TERM_BUCKET_RANGES:
        group_mask = (selected_terms >= min_group_terms) & (selected_terms <= max_group_terms)
        group_positions = positions[group_mask].contiguous()
        group_count = int(group_positions.shape[0])
        if group_count == 0:
            continue
        group_i_beg = i_beg[group_mask].contiguous()
        group_i_fin = i_fin[group_mask].contiguous()
        max_window = int((group_i_fin - group_i_beg).max().detach().cpu().item())
        if max_window <= 0:
            continue
        group_shl = shl_values[group_mask].contiguous()
        group_selected_terms = selected_terms[group_mask].contiguous()
        half_max_terms = max_group_terms // 2
        delta_deg = positions.new_zeros((group_count, half_max_terms))
        weighted_g = positions.new_zeros((group_count, half_max_terms))
        quadrature_nodes = positions.new_zeros((group_count, half_max_terms))
        quadrature_weights = positions.new_zeros((group_count, half_max_terms))
        weighted_sum = positions.new_zeros((group_count,))
        for bucket_terms in grouped_terms:
            bucket_mask = group_selected_terms == int(bucket_terms)
            bucket_positions = group_positions[bucket_mask].contiguous()
            bucket_count = int(bucket_positions.shape[0])
            if bucket_count == 0:
                continue
            (
                bucket_delta,
                bucket_weighted_g,
                bucket_weighted_sum,
                bucket_nodes,
                bucket_weights,
            ) = _prepare_highq_bucket_integration_triton(
                bucket_terms=int(bucket_terms),
                positions=bucket_positions,
                shl_values=group_shl[bucket_mask].contiguous(),
                domain=domain,
            )
            bucket_half_terms = int(bucket_terms) // 2
            delta_deg[bucket_mask, :bucket_half_terms] = bucket_delta
            weighted_g[bucket_mask, :bucket_half_terms] = bucket_weighted_g
            if bucket_nodes.ndim == 1:
                quadrature_nodes[bucket_mask, :bucket_half_terms] = bucket_nodes.unsqueeze(0).expand(bucket_count, -1)
                quadrature_weights[bucket_mask, :bucket_half_terms] = bucket_weights.unsqueeze(0).expand(bucket_count, -1)
            else:
                quadrature_nodes[bucket_mask, :bucket_half_terms] = bucket_nodes
                quadrature_weights[bucket_mask, :bucket_half_terms] = bucket_weights
            weighted_sum[bucket_mask] = bucket_weighted_sum
        yield (
            group_mask,
            group_i_beg,
            group_i_fin,
            int(max_window),
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            group_shl,
        )


def _iter_highq_dense_grouped_integrations(
    *,
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    selected_terms: torch.Tensor,
    domain: torch.Tensor,
):
    for min_group_terms, max_group_terms, grouped_terms in _HIGHQ_GROUPED_TERM_BUCKET_RANGES:
        group_active = (selected_terms >= min_group_terms) & (selected_terms <= max_group_terms)
        group_i_beg = i_beg.contiguous()
        group_i_fin = torch.where(group_active, i_fin, i_beg).contiguous()
        max_window = int((group_i_fin - group_i_beg).max().detach().cpu().item())
        if max_window <= 0:
            continue
        (
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
        ) = _prepare_highq_group_integration_triton(
            grouped_terms=tuple(int(value) for value in grouped_terms),
            positions=positions,
            shl_values=shl_values,
            selected_terms=selected_terms,
            domain=domain,
        )
        yield (
            group_i_beg,
            group_i_fin,
            int(max_window),
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            shl_values,
        )


def _run_highq_triton_forward(
    *,
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    shl_values: torch.Tensor,
    batch_ids: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    selected_terms: torch.Tensor,
    domain: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    domain_size = int(domain.shape[0])
    output = domain.new_zeros(int(batch_size) * domain_size)
    if int(positions.shape[0]) <= _HIGHQ_DENSE_GROUP_MAX_PEAKS:
        for (
            group_i_beg,
            group_i_fin,
            max_window,
            delta_deg,
            weighted_g,
            weighted_sum,
            _quadrature_nodes,
            _quadrature_weights,
            _group_shl,
        ) in (
            _iter_highq_dense_grouped_integrations(
                positions=positions,
                shl_values=shl_values,
                i_beg=i_beg,
                i_fin=i_fin,
                selected_terms=selected_terms,
                domain=domain,
            )
        ):
            grid = (int(positions.shape[0]), triton.cdiv(max_window, _TRITON_BLOCK_SIZE))
            _fcj_highq_forward_kernel[grid](
                intensities.contiguous(),
                sigma_sq_centideg2.contiguous(),
                gamma_centideg.contiguous(),
                delta_deg.contiguous(),
                weighted_g.contiguous(),
                weighted_sum.contiguous(),
                group_i_beg,
                group_i_fin,
                batch_ids.contiguous(),
                domain.contiguous(),
                output,
                domain_size,
                max_window,
                int(delta_deg.shape[1]),
                _TRITON_BLOCK_SIZE,
                num_warps=8,
            )
    else:
        for (
            group_mask,
            group_i_beg,
            group_i_fin,
            max_window,
            delta_deg,
            weighted_g,
            weighted_sum,
            _quadrature_nodes,
            _quadrature_weights,
            _group_shl,
        ) in (
            _iter_highq_grouped_integrations(
                positions=positions,
                shl_values=shl_values,
                i_beg=i_beg,
                i_fin=i_fin,
                selected_terms=selected_terms,
                domain=domain,
            )
        ):
            grid = (int(group_i_beg.shape[0]), triton.cdiv(max_window, _TRITON_BLOCK_SIZE))
            _fcj_highq_forward_kernel[grid](
                intensities[group_mask].contiguous(),
                sigma_sq_centideg2[group_mask].contiguous(),
                gamma_centideg[group_mask].contiguous(),
                delta_deg.contiguous(),
                weighted_g.contiguous(),
                weighted_sum.contiguous(),
                group_i_beg,
                group_i_fin,
                batch_ids[group_mask].contiguous(),
                domain.contiguous(),
                output,
                domain_size,
                max_window,
                int(delta_deg.shape[1]),
                _TRITON_BLOCK_SIZE,
                num_warps=8,
            )
    return output.reshape(int(batch_size), domain_size)


def _run_highq_triton_local_autograd(
    *,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    positions: torch.Tensor,
    shl_values: torch.Tensor,
    batch_ids: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    selected_terms: torch.Tensor,
    domain: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    outputs: list[torch.Tensor] = []
    if int(positions.shape[0]) <= _HIGHQ_DENSE_GROUP_MAX_PEAKS:
        with torch.no_grad():
            dense_grouped_integrations = list(
                _iter_highq_dense_grouped_integrations(
                    positions=positions.detach(),
                    shl_values=shl_values.detach(),
                    i_beg=i_beg,
                    i_fin=i_fin,
                    selected_terms=selected_terms,
                    domain=domain,
                )
            )
        for (
            group_i_beg,
            group_i_fin,
            _max_window,
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            group_shl,
        ) in dense_grouped_integrations:
            outputs.append(
                _HighQuadratureFcjLocalTriton.apply(
                    positions,
                    intensities,
                    sigma_sq_centideg2,
                    gamma_centideg,
                    delta_deg,
                    weighted_g,
                    weighted_sum,
                    quadrature_nodes,
                    quadrature_weights,
                    group_shl,
                    batch_ids,
                    group_i_beg,
                    group_i_fin,
                    domain,
                    int(batch_size),
                )
            )
    else:
        with torch.no_grad():
            grouped_integrations = list(
                _iter_highq_grouped_integrations(
                    positions=positions.detach(),
                    shl_values=shl_values.detach(),
                    i_beg=i_beg,
                    i_fin=i_fin,
                    selected_terms=selected_terms,
                    domain=domain,
                )
            )
        for (
            group_mask,
            group_i_beg,
            group_i_fin,
            _max_window,
            delta_deg,
            weighted_g,
            weighted_sum,
            quadrature_nodes,
            quadrature_weights,
            group_shl,
        ) in grouped_integrations:
            outputs.append(
                _HighQuadratureFcjLocalTriton.apply(
                    positions[group_mask],
                    intensities[group_mask],
                    sigma_sq_centideg2[group_mask],
                    gamma_centideg[group_mask],
                    delta_deg,
                    weighted_g,
                    weighted_sum,
                    quadrature_nodes,
                    quadrature_weights,
                    group_shl,
                    batch_ids[group_mask],
                    group_i_beg,
                    group_i_fin,
                    domain,
                    int(batch_size),
                )
            )
    if not outputs:
        return domain.new_zeros((int(batch_size), int(domain.shape[0])))
    histogram = outputs[0]
    for extra in outputs[1:]:
        histogram = histogram + extra
    return histogram


def _compute_fallback_histogram(
    *,
    batch_size: int,
    domain_size: int,
    domain: torch.Tensor,
    peak_batch_ids: torch.Tensor,
    peak_pos: torch.Tensor,
    peak_intensities: torch.Tensor,
    peak_sigma: torch.Tensor,
    peak_gamma: torch.Tensor,
    peak_shl_values: torch.Tensor,
    i_beg: torch.Tensor,
    i_fin: torch.Tensor,
    selected_terms: torch.Tensor,
) -> torch.Tensor:
    histogram_flat = domain.new_zeros(int(batch_size) * int(domain_size))
    used_terms = sorted(int(value) for value in torch.unique(selected_terms.detach()).cpu().tolist())
    single_batch_offset = 0 if int(batch_size) == 1 else None
    for bucket_terms in used_terms:
        bucket_mask = selected_terms == int(bucket_terms)
        _accumulate_fcj_bucket(
            histogram_flat,
            domain,
            int(domain_size),
            peak_batch_ids[bucket_mask],
            peak_pos[bucket_mask],
            peak_intensities[bucket_mask],
            peak_sigma[bucket_mask],
            peak_gamma[bucket_mask],
            peak_shl_values[bucket_mask],
            i_beg[bucket_mask],
            i_fin[bucket_mask],
            num_terms=int(bucket_terms),
            single_batch_offset=single_batch_offset,
        )
    return histogram_flat.reshape(int(batch_size), int(domain_size))


def _compute_fcj_profile_histogram_triton(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
    *,
    enable_highq_forward: bool = False,
) -> torch.Tensor:
    _validate_fcj_inputs(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
    )
    if bool((shl_values > 0).any()):
        raise NotImplementedError(
            "Triton FCJ kernels do not implement the required double-precision "
            "axial integration for positive SH/L; use backend='bucketed' or 'torch'."
        )
    if (
        triton is None
        or not positions.is_cuda
        or positions.dtype not in {torch.float32, torch.float64}
        or domain.requires_grad
        or shl_values.requires_grad
    ):
        return compute_fcj_profile_histogram_bucketed(
            positions,
            intensities,
            sigma_sq_centideg2,
            gamma_centideg,
            domain,
            shl_values,
        )

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
    local_parameter_requires_grad = (
        positions.requires_grad
        or intensities.requires_grad
        or sigma_sq_centideg2.requires_grad
        or gamma_centideg.requires_grad
    )
    can_use_highq_kernel = (
        bool(enable_highq_forward)
        and positions.dtype == torch.float32
    )
    use_highq_forward = can_use_highq_kernel and not local_parameter_requires_grad
    use_highq_local_autograd = can_use_highq_kernel and local_parameter_requires_grad

    peak_pos = prepared.peak_pos
    two_term_mask = prepared.two_term_mask
    if bool(two_term_mask.any()):
        two_pos = prepared.peak_pos[two_term_mask]
        two_shl = prepared.peak_shl_values[two_term_mask]
        emin_arg = torch.sqrt(1.0 + two_shl.pow(2)) * torch.cos(torch.deg2rad(two_pos))
        emin_deg = torch.rad2deg(torch.acos(torch.clamp(emin_arg, min=-1.0, max=1.0)))
        xp, _ = _half_gauss_legendre_rule_torch(
            2,
            device=prepared.domain.device,
            dtype=prepared.domain.dtype,
        )
        peak_pos = peak_pos.clone()
        peak_pos[two_term_mask] = emin_deg + (two_pos - emin_deg) * xp[0]

    supported_mask = prepared.zero_shl_mask | two_term_mask
    outputs: list[torch.Tensor] = []
    if bool(supported_mask.any()):
        outputs.append(
            _run_single_pv_triton(
                positions=peak_pos[supported_mask],
                intensities=prepared.peak_intensities[supported_mask],
                sigma_sq_centideg2=prepared.peak_sigma[supported_mask],
                gamma_centideg=prepared.peak_gamma[supported_mask],
                batch_ids=prepared.peak_batch_ids[supported_mask],
                i_beg=prepared.i_beg[supported_mask],
                i_fin=prepared.i_fin[supported_mask],
                domain=prepared.domain,
                batch_size=prepared.batch_size,
            )
        )

    unsupported_mask = prepared.highq_mask
    if bool(unsupported_mask.any()):
        if use_highq_local_autograd:
            outputs.append(
                _run_highq_triton_local_autograd(
                    positions=peak_pos[unsupported_mask],
                    intensities=prepared.peak_intensities[unsupported_mask],
                    sigma_sq_centideg2=prepared.peak_sigma[unsupported_mask],
                    gamma_centideg=prepared.peak_gamma[unsupported_mask],
                    shl_values=prepared.peak_shl_values[unsupported_mask],
                    batch_ids=prepared.peak_batch_ids[unsupported_mask],
                    i_beg=prepared.i_beg[unsupported_mask],
                    i_fin=prepared.i_fin[unsupported_mask],
                    selected_terms=prepared.selected_terms[unsupported_mask],
                    domain=prepared.domain,
                    batch_size=prepared.batch_size,
                )
            )
        elif use_highq_forward:
            outputs.append(
                _run_highq_triton_forward(
                    positions=peak_pos[unsupported_mask],
                    intensities=prepared.peak_intensities[unsupported_mask],
                    sigma_sq_centideg2=prepared.peak_sigma[unsupported_mask],
                    gamma_centideg=prepared.peak_gamma[unsupported_mask],
                    shl_values=prepared.peak_shl_values[unsupported_mask],
                    batch_ids=prepared.peak_batch_ids[unsupported_mask],
                    i_beg=prepared.i_beg[unsupported_mask],
                    i_fin=prepared.i_fin[unsupported_mask],
                    selected_terms=prepared.selected_terms[unsupported_mask],
                    domain=prepared.domain,
                    batch_size=prepared.batch_size,
                )
            )
        else:
            outputs.append(
                _compute_fallback_histogram(
                    batch_size=prepared.batch_size,
                    domain_size=prepared.domain_size,
                    domain=prepared.domain,
                    peak_batch_ids=prepared.peak_batch_ids[unsupported_mask],
                    peak_pos=peak_pos[unsupported_mask],
                    peak_intensities=prepared.peak_intensities[unsupported_mask],
                    peak_sigma=prepared.peak_sigma[unsupported_mask],
                    peak_gamma=prepared.peak_gamma[unsupported_mask],
                    peak_shl_values=prepared.peak_shl_values[unsupported_mask],
                    i_beg=prepared.i_beg[unsupported_mask],
                    i_fin=prepared.i_fin[unsupported_mask],
                    selected_terms=prepared.selected_terms[unsupported_mask],
                )
            )

    if not outputs:
        return prepared.domain.new_zeros((prepared.batch_size, prepared.domain_size), dtype=positions.dtype)
    histogram = outputs[0]
    for extra in outputs[1:]:
        histogram = histogram + extra
    return histogram.to(dtype=positions.dtype)


def compute_fcj_profile_histogram_triton(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    return _compute_fcj_profile_histogram_triton(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
        enable_highq_forward=False,
    )


def compute_fcj_profile_histogram_triton_highq(
    positions: torch.Tensor,
    intensities: torch.Tensor,
    sigma_sq_centideg2: torch.Tensor,
    gamma_centideg: torch.Tensor,
    domain: torch.Tensor,
    shl_values: torch.Tensor,
) -> torch.Tensor:
    return _compute_fcj_profile_histogram_triton(
        positions,
        intensities,
        sigma_sq_centideg2,
        gamma_centideg,
        domain,
        shl_values,
        enable_highq_forward=True,
    )


__all__ = ["compute_fcj_profile_histogram_triton", "compute_fcj_profile_histogram_triton_highq"]
