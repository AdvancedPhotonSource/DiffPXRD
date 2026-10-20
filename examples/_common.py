"""Small synthetic PXRD problem shared by the installation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
import torch

from diffpxrd import ModularCWForwardModel, solved_profile_wls_loss_torch


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; no CPU fallback is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


class SyntheticNaCl:
    def __init__(self, device: str, frames: int = 1):
        structure = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        self.model = ModularCWForwardModel(
            q_min=0.5, q_max=6.0, wavelength=1.5406, device=device,
        )
        self.batch = self.model.structures_to_batch([structure] * frames, dtype=torch.float64)
        self.reflections = self.model.compile_reflections(self.batch)
        self.grid = torch.linspace(15.0, 80.0, 2001, dtype=torch.float64, device=device)

    def profile(self, relative_expansion: torch.Tensor) -> torch.Tensor:
        batch = dict(self.batch)
        batch["lengths"] = self.batch["lengths"] * (1.0 + relative_expansion.reshape(-1, 1))
        layers = self.model.forward_with_layers_fixed_hkls(batch, self.reflections)
        return self.model.compute_tch_profile_layer(
            layers["correction"],
            instrument={"U": 0.0, "V": 0.0, "W": 16.0, "X": 0.0, "Y": 0.0, "Z": 2.0},
            domain=self.grid,
            return_diagnostics=False,
        ).histogram


def fit(parameters, render, observed):
    optimizer = torch.optim.LBFGS(
        [parameters], max_iter=60, tolerance_grad=1e-11,
        tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss, _ = solved_profile_wls_loss_torch(
            observed, render(), weights=torch.ones_like(observed),
            fit_scale=True, fit_background=True,
        )
        loss = loss.mean()
        loss.backward()
        return loss

    initial = float(closure().detach())
    optimizer.step(closure)
    final = float(closure().detach())
    if not np.isfinite(final) or final >= initial:
        raise RuntimeError(f"Synthetic refinement did not reduce the loss: {initial} -> {final}")
    return {"initial_loss": initial, "final_loss": final}


def write_summary(output: Path, summary: dict):
    text = json.dumps(summary, indent=2, allow_nan=False)
    (output / "summary.json").write_text(text + "\n")
    print(text)
