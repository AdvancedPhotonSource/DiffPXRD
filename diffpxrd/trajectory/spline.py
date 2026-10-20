"""Clamped cubic B-splines for shared structural evolution models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class CubicSplineTrajectory:
    """A fixed, clamped cubic B-spline design for ordered trajectory states."""

    state_values: np.ndarray
    normalized_state: np.ndarray
    full_knots: np.ndarray
    design_matrix: np.ndarray
    input_min: float
    input_max: float
    degree: int = 3

    @classmethod
    def from_state_values(
        cls,
        state_values: np.ndarray | list[float] | tuple[float, ...],
        *,
        full_knots: np.ndarray | list[float] | tuple[float, ...],
        input_min: float | None = None,
        input_max: float | None = None,
    ) -> CubicSplineTrajectory:
        values = np.asarray(state_values, dtype=np.float64)
        knots = np.asarray(full_knots, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("state_values must be a nonempty finite one-dimensional array")
        if knots.ndim != 1 or not np.isfinite(knots).all():
            raise ValueError("full_knots must be a finite one-dimensional array")
        if np.any(np.diff(knots) < 0.0):
            raise ValueError("full_knots must be nondecreasing")

        lower = float(np.min(values) if input_min is None else input_min)
        upper = float(np.max(values) if input_max is None else input_max)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("input_max must be finite and larger than input_min")
        normalized = (values - lower) / (upper - lower)
        tolerance = 1.0e-12
        if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
            raise ValueError("state_values fall outside the locked normalization interval")
        normalized = np.clip(normalized, 0.0, 1.0)

        degree = 3
        coefficient_count = int(knots.size - degree - 1)
        if coefficient_count < degree + 1:
            raise ValueError("cubic B-spline requires at least four coefficients")
        if not np.allclose(knots[: degree + 1], 0.0, atol=tolerance, rtol=0.0):
            raise ValueError("cubic B-spline must be clamped at normalized state zero")
        if not np.allclose(knots[-(degree + 1) :], 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("cubic B-spline must be clamped at normalized state one")

        coefficients = np.eye(coefficient_count, dtype=np.float64)
        design = np.asarray(
            BSpline(knots, coefficients, degree, extrapolate=False)(normalized),
            dtype=np.float64,
        )
        if design.shape != (values.size, coefficient_count):
            raise RuntimeError("unexpected cubic B-spline design shape")
        if not np.isfinite(design).all():
            raise ValueError("cubic B-spline design contains non-finite values")
        if not np.allclose(design.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("cubic B-spline basis does not form a partition of unity")

        return cls(
            state_values=values.copy(),
            normalized_state=normalized,
            full_knots=knots.copy(),
            design_matrix=design,
            input_min=lower,
            input_max=upper,
            degree=degree,
        )

    @property
    def coefficient_count(self) -> int:
        return int(self.design_matrix.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.design_matrix.shape[0])

    def amplitudes(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Map mode-by-coefficient parameters to frame-by-mode amplitudes."""
        if (
            coefficients.ndim != 2
            or int(coefficients.shape[1]) != self.coefficient_count
        ):
            raise ValueError(
                "coefficients must have shape "
                f"(mode_count, {self.coefficient_count}), got {tuple(coefficients.shape)}"
            )
        design = torch.as_tensor(
            self.design_matrix,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        return design @ coefficients.transpose(0, 1)
