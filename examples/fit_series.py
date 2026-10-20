"""Fit a shared linear lattice-evolution model directly to synthetic profiles."""

import numpy as np
import torch

from _common import SyntheticNaCl, arguments, fit, write_summary


def main():
    args = arguments(__doc__)
    problem = SyntheticNaCl(args.device, frames=7)
    state = torch.linspace(0.0, 1.0, 7, dtype=torch.float64, device=args.device)
    design = torch.stack((torch.ones_like(state), state), dim=1)
    true_coefficients = torch.tensor([0.0005, 0.001], dtype=torch.float64, device=args.device)
    target = design @ true_coefficients
    observed = problem.profile(target).detach()
    observed = observed / observed.amax(dim=1, keepdim=True)
    coefficients = torch.nn.Parameter(torch.zeros_like(true_coefficients))
    summary = fit(coefficients, lambda: problem.profile(design @ coefficients), observed)
    refined = (design @ coefficients).detach()
    error = float(torch.max(torch.abs(refined - target)))
    if error > 1e-6:
        raise RuntimeError(f"Synthetic trajectory recovery error is too large: {error}")
    np.savetxt(
        args.output_dir / "trajectory.csv",
        np.column_stack((state.cpu().numpy(), (5.64 * (1.0 + target)).cpu().numpy(),
                         (5.64 * (1.0 + refined)).cpu().numpy())),
        delimiter=",", header="state,target_cell_angstrom,refined_cell_angstrom", comments="",
    )
    summary.update(example="synthetic_joint_series", device=args.device,
                   frames=state.numel(), shared_parameters=coefficients.numel(),
                   maximum_relative_cell_error=error)
    write_summary(args.output_dir, summary)


if __name__ == "__main__":
    main()
