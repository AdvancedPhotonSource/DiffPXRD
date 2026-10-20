"""Refine the cubic cell of one synthetic NaCl diffraction pattern."""

import numpy as np
import torch

from _common import SyntheticNaCl, arguments, fit, write_summary


def main():
    args = arguments(__doc__)
    problem = SyntheticNaCl(args.device)
    target = torch.tensor([0.001], dtype=torch.float64, device=args.device)
    observed = problem.profile(target).detach()
    observed = observed / observed.amax()
    expansion = torch.nn.Parameter(torch.zeros_like(target))
    summary = fit(expansion, lambda: problem.profile(expansion), observed)
    error = float(torch.max(torch.abs(expansion.detach() - target)))
    if error > 1e-6:
        raise RuntimeError(f"Synthetic lattice recovery error is too large: {error}")
    calculated = problem.profile(expansion).detach()
    calculated = calculated / calculated.amax()
    np.savetxt(
        args.output_dir / "profile.csv",
        np.column_stack((problem.grid.cpu().numpy(), observed.cpu().numpy()[0], calculated.cpu().numpy()[0])),
        delimiter=",", header="two_theta_deg,observed_synthetic,calculated", comments="",
    )
    summary.update(example="synthetic_single_pattern", device=args.device,
                   target_cell_angstrom=5.64 * (1.0 + float(target[0])),
                   refined_cell_angstrom=5.64 * (1.0 + float(expansion.detach()[0])))
    write_summary(args.output_dir, summary)


if __name__ == "__main__":
    main()
