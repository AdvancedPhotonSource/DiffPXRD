from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class WeightedProfileFit:
    scale: float
    background: float
    background_coeffs: np.ndarray
    background_profile: np.ndarray
    fitted_profile: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    chi_square: float
    rwp: float
    rmse: float
    nrmse_vs_observed_max: float
    cosine_similarity: float


def _prepare_torch_weights(
    observed: torch.Tensor,
    weights: torch.Tensor | np.ndarray | None,
    *,
    eps: float,
) -> torch.Tensor:
    if weights is None:
        return torch.ones_like(observed)

    weight_tensor = torch.as_tensor(weights, dtype=observed.dtype, device=observed.device)
    if weight_tensor.shape != observed.shape:
        raise ValueError("weights must match the observed profile shape")

    valid_mask = torch.isfinite(weight_tensor) & (weight_tensor > 0)
    weight_tensor = torch.where(valid_mask, weight_tensor, torch.zeros_like(weight_tensor))
    zero_weight_mask = torch.sum(weight_tensor, dim=-1, keepdim=True) <= eps
    return torch.where(zero_weight_mask, torch.ones_like(weight_tensor), weight_tensor)


def _prepare_torch_x(
    x: torch.Tensor | np.ndarray | None,
    observed: torch.Tensor,
    *,
    background_degree: int,
) -> torch.Tensor | None:
    if background_degree <= 0:
        return None
    if x is None:
        raise ValueError("x must be provided when background_degree > 0")
    x_tensor = torch.as_tensor(x, dtype=observed.dtype, device=observed.device)
    if x_tensor.ndim == 1:
        if x_tensor.shape[0] != observed.shape[-1]:
            raise ValueError("x must match the observed profile length")
        x_tensor = x_tensor.unsqueeze(0).expand(observed.shape[0], -1)
    if x_tensor.shape != observed.shape:
        raise ValueError("x must match the observed profile shape")
    return x_tensor


def _normalized_chebyshev_basis_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D grids")
    batch_size, num_points = x.shape
    if degree == 0:
        return torch.ones((batch_size, num_points, 1), dtype=x.dtype, device=x.device)
    x_min = torch.min(x, dim=-1, keepdim=True).values
    x_max = torch.max(x, dim=-1, keepdim=True).values
    denom = torch.clamp(x_max - x_min, min=1.0e-12)
    t = 2.0 * (x - x_min) / denom - 1.0
    basis = torch.empty((batch_size, num_points, degree + 1), dtype=x.dtype, device=x.device)
    basis[..., 0] = 1.0
    basis[..., 1] = t
    for idx in range(2, degree + 1):
        basis[..., idx] = 2.0 * t * basis[..., idx - 1] - basis[..., idx - 2]
    return basis


def normalized_chebyshev_basis_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    return _normalized_chebyshev_basis_torch(x, degree)


def _prepare_background_basis(
    background_basis: torch.Tensor | None,
    x: torch.Tensor | None,
    observed: torch.Tensor,
    *,
    background_degree: int,
) -> torch.Tensor:
    if background_degree <= 0:
        raise ValueError("background_degree must be positive when preparing a polynomial background basis")
    if background_basis is None:
        if x is None:
            raise ValueError("x is required when fitting a polynomial background")
        return _normalized_chebyshev_basis_torch(x, background_degree)

    basis = torch.as_tensor(background_basis, dtype=observed.dtype, device=observed.device)
    if basis.ndim != 3 or basis.shape[1] != observed.shape[-1] or basis.shape[2] < 1:
        raise ValueError(
            "background_basis must have shape "
            f"(batch, {observed.shape[-1]}, basis_column) with at least one basis column"
        )
    if basis.shape[0] == 1 and observed.shape[0] != 1:
        basis = basis.expand(observed.shape[0], -1, -1)
    if basis.shape[0] != observed.shape[0]:
        raise ValueError("background_basis batch dimension must match observed profiles or be 1")
    return basis


def _solve_weighted_profile_torch(
    observed: torch.Tensor,
    calculated: torch.Tensor,
    *,
    weights: torch.Tensor,
    fit_scale: bool,
    fit_background: bool,
    x: torch.Tensor | None,
    background_basis: torch.Tensor | None,
    background_degree: int,
    eps: float,
    solve_method: str,
    constrain_scale_nonnegative: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]:
    batch_size = observed.shape[0]
    design_columns: list[torch.Tensor] = []
    resolved_background_basis: torch.Tensor | None = None
    if fit_scale:
        design_columns.append(calculated.unsqueeze(-1))
    if fit_background:
        if background_degree <= 0:
            design_columns.append(torch.ones((*observed.shape, 1), dtype=observed.dtype, device=observed.device))
        else:
            resolved_background_basis = _prepare_background_basis(
                background_basis,
                x,
                observed,
                background_degree=background_degree,
            )
            design_columns.append(resolved_background_basis)

    if not design_columns:
        diagnostics: dict[str, torch.Tensor | str] = {
            "solver_method": "none",
            "design_column_scale": torch.zeros((batch_size, 0), dtype=observed.dtype, device=observed.device),
            "design_singular_values": torch.zeros((batch_size, 0), dtype=observed.dtype, device=observed.device),
            "design_rank": torch.zeros(batch_size, dtype=torch.long, device=observed.device),
            "design_condition": torch.ones(batch_size, dtype=observed.dtype, device=observed.device),
            "ridge": torch.zeros(batch_size, dtype=observed.dtype, device=observed.device),
            "unconstrained_scale": torch.ones(batch_size, dtype=observed.dtype, device=observed.device),
            "scale_was_constrained": torch.zeros(batch_size, dtype=torch.bool, device=observed.device),
        }
        return (
            torch.ones(batch_size, dtype=observed.dtype, device=observed.device),
            torch.zeros_like(observed),
            torch.zeros((batch_size, 0), dtype=observed.dtype, device=observed.device),
            diagnostics,
        )

    design = torch.cat(design_columns, dim=-1)
    sqrt_weights = torch.sqrt(torch.clamp(weights, min=0.0))
    weighted_design = design * sqrt_weights.unsqueeze(-1)
    weighted_observed = observed * sqrt_weights
    column_scale = torch.linalg.norm(weighted_design, dim=-2)
    safe_column_scale = torch.where(column_scale > eps, column_scale, torch.ones_like(column_scale))
    scaled_design = weighted_design / safe_column_scale.unsqueeze(-2)
    scaled_gram = torch.matmul(scaled_design.transpose(-1, -2), scaled_design)
    eigvals = torch.linalg.eigvalsh(scaled_gram.detach())
    singular_values = torch.sqrt(torch.clamp(eigvals, min=0.0)).flip(-1)
    max_singular = singular_values[..., :1]
    rank = torch.sum(singular_values > torch.clamp(max_singular * eps, min=eps), dim=-1)
    min_retained = torch.where(rank > 0, singular_values.gather(-1, torch.clamp(rank - 1, min=0).unsqueeze(-1)).squeeze(-1), torch.zeros_like(rank, dtype=observed.dtype))
    full_rank = rank == singular_values.shape[-1]
    condition = torch.where(
        full_rank & (min_retained > eps),
        singular_values[..., 0] / min_retained,
        torch.full((batch_size,), float("inf"), dtype=observed.dtype, device=observed.device),
    )
    method = str(solve_method).strip().lower()
    if method in {"lstsq", "qr", "svd"}:
        try:
            scaled_coeffs = torch.linalg.lstsq(scaled_design, weighted_observed.unsqueeze(-1)).solution.squeeze(-1)
            solver_method = "lstsq"
            ridge = torch.zeros(batch_size, dtype=observed.dtype, device=observed.device)
        except RuntimeError:
            scaled_coeffs = torch.matmul(torch.linalg.pinv(scaled_design), weighted_observed.unsqueeze(-1)).squeeze(-1)
            solver_method = "pinv_fallback"
            ridge = torch.zeros(batch_size, dtype=observed.dtype, device=observed.device)
    elif method in {"normal", "normal_equations"}:
        eye = torch.eye(
            scaled_gram.shape[-1],
            dtype=scaled_gram.dtype,
            device=scaled_gram.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        ridge = torch.clamp(eps * torch.mean(torch.diagonal(scaled_gram, dim1=-2, dim2=-1), dim=-1), min=eps)
        aty = torch.matmul(scaled_design.transpose(-1, -2), weighted_observed.unsqueeze(-1))
        scaled_coeffs = torch.linalg.solve(scaled_gram + ridge[:, None, None] * eye, aty).squeeze(-1)
        solver_method = "normal_equations"
    else:
        raise ValueError(f"unsupported weighted profile solve_method: {solve_method}")
    coeffs = scaled_coeffs / safe_column_scale
    unconstrained_scale = (
        coeffs[:, 0].clone()
        if fit_scale
        else torch.ones(batch_size, dtype=observed.dtype, device=observed.device)
    )
    scale_was_constrained = torch.zeros(batch_size, dtype=torch.bool, device=observed.device)
    if fit_scale and bool(constrain_scale_nonnegative):
        scale_was_constrained = unconstrained_scale < 0.0
        constrained_scale = torch.where(scale_was_constrained, torch.zeros_like(unconstrained_scale), unconstrained_scale)
        if fit_background and coeffs.shape[-1] > 1:
            background_design = design[..., 1:]
            weighted_background_design = background_design * sqrt_weights.unsqueeze(-1)
            background_column_scale = torch.linalg.norm(weighted_background_design, dim=-2)
            background_safe_column_scale = torch.where(
                background_column_scale > eps,
                background_column_scale,
                torch.ones_like(background_column_scale),
            )
            scaled_background_design = weighted_background_design / background_safe_column_scale.unsqueeze(-2)
            if method in {"lstsq", "qr", "svd"}:
                try:
                    scaled_background_coeffs = torch.linalg.lstsq(
                        scaled_background_design,
                        weighted_observed.unsqueeze(-1),
                    ).solution.squeeze(-1)
                except RuntimeError:
                    scaled_background_coeffs = torch.matmul(
                        torch.linalg.pinv(scaled_background_design),
                        weighted_observed.unsqueeze(-1),
                    ).squeeze(-1)
            elif method in {"normal", "normal_equations"}:
                background_gram = torch.matmul(
                    scaled_background_design.transpose(-1, -2),
                    scaled_background_design,
                )
                background_eye = torch.eye(
                    background_gram.shape[-1],
                    dtype=background_gram.dtype,
                    device=background_gram.device,
                ).unsqueeze(0).expand(batch_size, -1, -1)
                background_ridge = torch.clamp(
                    eps * torch.mean(torch.diagonal(background_gram, dim1=-2, dim2=-1), dim=-1),
                    min=eps,
                )
                background_aty = torch.matmul(
                    scaled_background_design.transpose(-1, -2),
                    weighted_observed.unsqueeze(-1),
                )
                scaled_background_coeffs = torch.linalg.solve(
                    background_gram + background_ridge[:, None, None] * background_eye,
                    background_aty,
                ).squeeze(-1)
            else:
                raise ValueError(f"unsupported weighted profile solve_method: {solve_method}")
            boundary_background_coeffs = scaled_background_coeffs / background_safe_column_scale
            background_coeffs = torch.where(
                scale_was_constrained.unsqueeze(-1),
                boundary_background_coeffs,
                coeffs[:, 1:],
            )
            coeffs = torch.cat([constrained_scale.unsqueeze(-1), background_coeffs], dim=-1)
        else:
            coeffs = torch.cat([constrained_scale.unsqueeze(-1), coeffs[:, 1:]], dim=-1)
    diagnostics = {
        "solver_method": solver_method,
        "design_column_scale": safe_column_scale,
        "design_singular_values": singular_values,
        "design_rank": rank,
        "design_condition": condition,
        "ridge": ridge,
        "unconstrained_scale": unconstrained_scale,
        "scale_was_constrained": scale_was_constrained,
    }

    coeff_index = 0
    if fit_scale:
        scale = coeffs[:, coeff_index]
        coeff_index += 1
    else:
        scale = torch.ones(batch_size, dtype=observed.dtype, device=observed.device)

    if fit_background:
        background_coeffs = coeffs[:, coeff_index:]
        if background_degree <= 0:
            background_profile = background_coeffs[:, :1].expand_as(observed)
        else:
            if resolved_background_basis is None:
                raise RuntimeError("polynomial background basis was not prepared")
            background_profile = torch.matmul(resolved_background_basis, background_coeffs.unsqueeze(-1)).squeeze(-1)
    else:
        background_coeffs = torch.zeros((batch_size, 0), dtype=observed.dtype, device=observed.device)
        background_profile = torch.zeros_like(observed)
    return scale, background_profile, background_coeffs, diagnostics


def solve_weighted_scale_background_torch(
    observed: torch.Tensor,
    calculated: torch.Tensor,
    *,
    weights: torch.Tensor | np.ndarray | None = None,
    x: torch.Tensor | np.ndarray | None = None,
    background_basis: torch.Tensor | np.ndarray | None = None,
    fit_scale: bool = True,
    fit_background: bool = True,
    background_degree: int = 0,
    eps: float = 1.0e-12,
    solve_method: str = "lstsq",
    constrain_scale_nonnegative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observed.shape != calculated.shape:
        raise ValueError("observed and calculated profiles must have the same shape")

    observed_tensor = observed
    calculated_tensor = calculated
    weight_tensor = _prepare_torch_weights(observed_tensor, weights, eps=eps)
    x_tensor = (
        None
        if background_basis is not None and int(background_degree) > 0 and x is None
        else _prepare_torch_x(x, observed_tensor, background_degree=background_degree)
    )
    scale, background_profile, _, _ = _solve_weighted_profile_torch(
        observed_tensor,
        calculated_tensor,
        weights=weight_tensor,
        fit_scale=fit_scale,
        fit_background=fit_background,
        x=x_tensor,
        background_basis=None if background_basis is None else torch.as_tensor(background_basis, dtype=observed_tensor.dtype, device=observed_tensor.device),
        background_degree=int(background_degree),
        eps=eps,
        solve_method=solve_method,
        constrain_scale_nonnegative=bool(constrain_scale_nonnegative),
    )
    return scale, torch.mean(background_profile, dim=-1)


def solved_profile_wls_residuals_torch(
    observed: torch.Tensor,
    calculated: torch.Tensor,
    *,
    weights: torch.Tensor | np.ndarray | None = None,
    x: torch.Tensor | np.ndarray | None = None,
    background_basis: torch.Tensor | np.ndarray | None = None,
    fit_scale: bool = True,
    fit_background: bool = True,
    background_degree: int = 0,
    eps: float = 1.0e-12,
    solve_method: str = "lstsq",
    constrain_scale_nonnegative: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if observed.shape != calculated.shape:
        raise ValueError("observed and calculated profiles must have the same shape")

    weight_tensor = _prepare_torch_weights(observed, weights, eps=eps)
    x_tensor = (
        None
        if background_basis is not None and int(background_degree) > 0 and x is None
        else _prepare_torch_x(x, observed, background_degree=background_degree)
    )
    scale, background_profile, background_coeffs, solver_diagnostics = _solve_weighted_profile_torch(
        observed,
        calculated,
        weights=weight_tensor,
        fit_scale=fit_scale,
        fit_background=fit_background,
        x=x_tensor,
        background_basis=None if background_basis is None else torch.as_tensor(background_basis, dtype=observed.dtype, device=observed.device),
        background_degree=int(background_degree),
        eps=eps,
        solve_method=solve_method,
        constrain_scale_nonnegative=bool(constrain_scale_nonnegative),
    )
    fitted_profile = scale.unsqueeze(-1) * calculated + background_profile
    raw_residual = fitted_profile - observed
    sqrt_weights = torch.sqrt(torch.clamp(weight_tensor, min=0.0))
    weighted_residual = sqrt_weights * raw_residual
    diagnostics = {
        "scale": scale,
        "background": torch.mean(background_profile, dim=-1),
        "background_coeffs": background_coeffs,
        "background_profile": background_profile,
        "weights": weight_tensor,
        "sqrt_weights": sqrt_weights,
        "residual": raw_residual,
        "fitted_profile": fitted_profile,
    }
    diagnostics.update(solver_diagnostics)
    return weighted_residual, diagnostics


def solved_profile_wls_loss_torch(
    observed: torch.Tensor,
    calculated: torch.Tensor,
    *,
    weights: torch.Tensor | np.ndarray | None = None,
    x: torch.Tensor | np.ndarray | None = None,
    background_basis: torch.Tensor | np.ndarray | None = None,
    fit_scale: bool = True,
    fit_background: bool = True,
    background_degree: int = 0,
    reduction: str = "mean",
    eps: float = 1.0e-12,
    solve_method: str = "lstsq",
    constrain_scale_nonnegative: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if observed.shape != calculated.shape:
        raise ValueError("observed and calculated profiles must have the same shape")

    weighted_residual, diagnostics = solved_profile_wls_residuals_torch(
        observed,
        calculated,
        weights=weights,
        x=x,
        background_basis=background_basis,
        fit_scale=fit_scale,
        fit_background=fit_background,
        background_degree=background_degree,
        eps=eps,
        solve_method=solve_method,
        constrain_scale_nonnegative=bool(constrain_scale_nonnegative),
    )
    weight_tensor = diagnostics["weights"]
    weighted_sq_residual = torch.square(weighted_residual)

    if reduction == "sum":
        loss = torch.sum(weighted_sq_residual)
    elif reduction == "none":
        loss = torch.sum(weighted_sq_residual, dim=-1)
    elif reduction == "mean":
        loss = torch.sum(weighted_sq_residual, dim=-1) / torch.clamp(torch.sum(weight_tensor, dim=-1), min=eps)
    else:
        raise ValueError(f"unsupported reduction: {reduction}")

    return loss, diagnostics


def fit_weighted_profile_numpy(
    observed: np.ndarray,
    calculated: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    x: np.ndarray | None = None,
    fit_scale: bool = True,
    fit_background: bool = True,
    background_degree: int = 0,
    eps: float = 1.0e-12,
    solve_method: str = "lstsq",
    constrain_scale_nonnegative: bool = False,
) -> WeightedProfileFit:
    observed_array = np.asarray(observed, dtype=np.float64)
    calculated_array = np.asarray(calculated, dtype=np.float64)
    if observed_array.ndim != 1 or calculated_array.ndim != 1:
        raise ValueError("fit_weighted_profile_numpy expects 1D observed and calculated profiles")
    if observed_array.shape != calculated_array.shape:
        raise ValueError("observed and calculated profiles must have the same shape")

    weights_array = None if weights is None else np.asarray(weights, dtype=np.float64)
    x_array = None if x is None else np.asarray(x, dtype=np.float64)
    observed_tensor = torch.as_tensor(observed_array, dtype=torch.float64).unsqueeze(0)
    calculated_tensor = torch.as_tensor(calculated_array, dtype=torch.float64).unsqueeze(0)
    weights_tensor = None if weights_array is None else torch.as_tensor(weights_array, dtype=torch.float64).unsqueeze(0)
    x_tensor = None if x_array is None else torch.as_tensor(x_array, dtype=torch.float64).unsqueeze(0)

    weighted_residual, diagnostics = solved_profile_wls_residuals_torch(
        observed_tensor,
        calculated_tensor,
        weights=weights_tensor,
        x=x_tensor,
        fit_scale=fit_scale,
        fit_background=fit_background,
        background_degree=background_degree,
        eps=eps,
        solve_method=solve_method,
        constrain_scale_nonnegative=bool(constrain_scale_nonnegative),
    )
    raw_residual = diagnostics["residual"][0]
    fitted_profile = diagnostics["fitted_profile"][0]
    weight_tensor = diagnostics["weights"][0]
    chi_square = float(torch.sum(torch.square(weighted_residual[0])).item())
    observed_weighted_norm = max(float(torch.sum(weight_tensor * torch.square(observed_tensor[0])).item()), eps)
    rmse = float(torch.sqrt(torch.mean(torch.square(raw_residual))).item())
    observed_max = max(float(torch.max(torch.abs(observed_tensor[0])).item()), eps)
    cosine_denom = max(
        float(torch.linalg.norm(observed_tensor[0]).item() * torch.linalg.norm(fitted_profile).item()),
        eps,
    )
    cosine_similarity = float(torch.dot(observed_tensor[0], fitted_profile).item() / cosine_denom)
    return WeightedProfileFit(
        scale=float(diagnostics["scale"][0].item()),
        background=float(diagnostics["background"][0].item()),
        background_coeffs=diagnostics["background_coeffs"][0].detach().cpu().numpy(),
        background_profile=diagnostics["background_profile"][0].detach().cpu().numpy(),
        fitted_profile=fitted_profile.detach().cpu().numpy(),
        residual=raw_residual.detach().cpu().numpy(),
        weights=weight_tensor.detach().cpu().numpy(),
        chi_square=chi_square,
        rwp=float(np.sqrt(chi_square / observed_weighted_norm)),
        rmse=rmse,
        nrmse_vs_observed_max=float(rmse / observed_max),
        cosine_similarity=cosine_similarity,
    )


# Backward-compatible internal aliases. These names are intentionally no longer
# exported from diffpxrd.__init__ or documented in the README.
weighted_profile_residuals_torch = solved_profile_wls_residuals_torch
weighted_profile_loss_torch = solved_profile_wls_loss_torch
