from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


COMMENT_PREFIXES = ("#", "!", ";", "//")


@dataclass
class LoadedProfile:
    x: np.ndarray
    observed: np.ndarray
    weights: np.ndarray
    background: np.ndarray
    source_path: str


def _extract_numeric_tokens(line: str) -> list[float]:
    stripped = line.strip()
    if not stripped:
        return []
    if any(stripped.startswith(prefix) for prefix in COMMENT_PREFIXES):
        return []

    numeric_tokens: list[float] = []
    for token in stripped.replace(",", " ").split():
        try:
            numeric_tokens.append(float(token))
        except ValueError:
            continue
    return numeric_tokens


def infer_profile_weights(
    observed: Sequence[float] | np.ndarray,
    *,
    mode: str = "poisson",
    eps: float = 1.0,
) -> np.ndarray:
    observed_vec = np.asarray(observed, dtype=float).reshape(-1)
    if observed_vec.size == 0:
        raise ValueError("observed profile must be non-empty")

    if mode == "uniform":
        return np.ones_like(observed_vec, dtype=float)
    if mode == "poisson":
        return 1.0 / np.clip(np.abs(observed_vec), eps, None)
    raise ValueError(f"unsupported weight inference mode: {mode}")


def _as_background_profile(
    background: float | Sequence[float] | np.ndarray,
    *,
    reference: np.ndarray,
) -> np.ndarray:
    background_array = np.asarray(background, dtype=float)
    if background_array.ndim == 0:
        return np.full_like(reference, float(background_array), dtype=float)
    if background_array.shape != reference.shape:
        raise ValueError("background profile must match the signal shape")
    return background_array


def compose_profile_output(
    signal: Sequence[float] | np.ndarray,
    *,
    background: float | Sequence[float] | np.ndarray | None = None,
    output_mode: str = "signal",
) -> np.ndarray:
    signal_array = np.asarray(signal, dtype=float).reshape(-1)
    mode = str(output_mode).strip().lower()
    if mode == "signal":
        return signal_array
    if mode == "ycalc":
        if background is None:
            raise ValueError("background is required when output_mode='ycalc'")
        return signal_array + _as_background_profile(background, reference=signal_array)
    if mode == "flat_background_ycalc":
        background_value = 1.0 if background is None else background
        return signal_array + _as_background_profile(background_value, reference=signal_array)
    raise ValueError(f"unsupported profile output_mode: {output_mode}")


def load_profile_file(
    path: str | Path,
    *,
    x_col: int = 0,
    y_col: int = 1,
    weight_col: int | None = None,
    sigma_col: int | None = None,
    min_x: float | None = None,
    max_x: float | None = None,
    inferred_weight_mode: str = "poisson",
    eps: float = 1.0e-12,
) -> LoadedProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile file not found: {profile_path}")

    max_required_col = max(
        x_col,
        y_col,
        -1 if weight_col is None else weight_col,
        -1 if sigma_col is None else sigma_col,
    )

    x_values: list[float] = []
    y_values: list[float] = []
    aux_values: list[float] = []

    for raw_line in profile_path.read_text().splitlines():
        numeric_tokens = _extract_numeric_tokens(raw_line)
        if len(numeric_tokens) <= max_required_col:
            continue
        x_values.append(float(numeric_tokens[x_col]))
        y_values.append(float(numeric_tokens[y_col]))
        if weight_col is not None:
            aux_values.append(float(numeric_tokens[weight_col]))
        elif sigma_col is not None:
            aux_values.append(float(numeric_tokens[sigma_col]))

    if not x_values:
        raise ValueError(f"no numeric profile rows found in {profile_path}")

    x_array = np.asarray(x_values, dtype=float)
    observed_array = np.asarray(y_values, dtype=float)
    aux_array = np.asarray(aux_values, dtype=float) if aux_values else None

    valid_mask = np.isfinite(x_array) & np.isfinite(observed_array)
    if aux_array is not None:
        valid_mask = valid_mask & np.isfinite(aux_array)
    x_array = x_array[valid_mask]
    observed_array = observed_array[valid_mask]
    if aux_array is not None:
        aux_array = aux_array[valid_mask]

    if min_x is not None:
        range_mask = x_array >= float(min_x)
        x_array = x_array[range_mask]
        observed_array = observed_array[range_mask]
        if aux_array is not None:
            aux_array = aux_array[range_mask]
    if max_x is not None:
        range_mask = x_array <= float(max_x)
        x_array = x_array[range_mask]
        observed_array = observed_array[range_mask]
        if aux_array is not None:
            aux_array = aux_array[range_mask]

    if x_array.size == 0:
        raise ValueError(f"profile range filter removed all rows from {profile_path}")

    order = np.argsort(x_array)
    x_array = x_array[order]
    observed_array = observed_array[order]
    if aux_array is not None:
        aux_array = aux_array[order]

    if weight_col is not None:
        weights = np.where(aux_array is not None, aux_array, np.ones_like(observed_array))
        weights = np.where(np.isfinite(weights) & (weights > eps), weights, 0.0)
    elif sigma_col is not None:
        sigma = np.where(aux_array is not None, np.abs(aux_array), np.ones_like(observed_array))
        weights = 1.0 / np.clip(np.square(sigma), eps, None)
    else:
        weights = infer_profile_weights(observed_array, mode=inferred_weight_mode)

    if float(np.sum(weights)) <= eps:
        weights = np.ones_like(observed_array, dtype=float)

    return LoadedProfile(
        x=x_array,
        observed=observed_array,
        weights=weights,
        background=np.zeros_like(observed_array, dtype=float),
        source_path=str(profile_path),
    )


def build_profile_pattern_batch(
    profile: LoadedProfile,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return {
        "observed": torch.tensor(profile.observed, dtype=dtype, device=target_device).unsqueeze(0),
        "weights": torch.tensor(profile.weights, dtype=dtype, device=target_device).unsqueeze(0),
        "background": torch.tensor(profile.background, dtype=dtype, device=target_device).unsqueeze(0),
        "x": torch.tensor(profile.x, dtype=dtype, device=target_device),
    }
