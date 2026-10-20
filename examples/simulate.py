"""Calculate a synthetic NaCl profile and its lattice derivative."""

import numpy as np
import torch

from _common import SyntheticNaCl, arguments, write_summary


def main():
    args = arguments(__doc__)
    problem = SyntheticNaCl(args.device)
    expansion = torch.zeros(1, dtype=torch.float64, device=args.device, requires_grad=True)
    profile = problem.profile(expansion)
    gradient = torch.autograd.grad(profile.square().mean(), expansion)[0]
    if not torch.isfinite(profile).all() or not torch.isfinite(gradient).all():
        raise RuntimeError("Nonfinite profile or lattice derivative")
    np.savetxt(
        args.output_dir / "profile.csv",
        np.column_stack((problem.grid.cpu().numpy(), profile.detach().cpu().numpy()[0])),
        delimiter=",", header="two_theta_deg,signal", comments="",
    )
    write_summary(args.output_dir, {
        "example": "synthetic_simulation", "device": args.device, "dtype": "float64",
        "grid_points": problem.grid.numel(), "sites": int(problem.batch["num_atoms"][0]),
        "lattice_derivative": float(gradient[0]),
    })


if __name__ == "__main__":
    main()
