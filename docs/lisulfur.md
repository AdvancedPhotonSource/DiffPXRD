# Li2S Intensity Evolution

Download `data.zip` from [Zenodo record 3514967](https://doi.org/10.5281/zenodo.3514967)
and leave it unmodified. The command checks the archive SHA-256 before reading
the 131 diffraction patterns, electrochemistry, and published peak-area table.

```sh
python reproduce/lisulfur.py --source-zip data/lisulfur/data.zip \
  --output-dir results/lisulfur --device cpu
```

The default calculation fits the original counts and twenty paired realizations
retaining 10% of the counts, using seed 20260804. It compares independent
amplitudes, a shared monotone spline, and a shared kinetic-shape model. Each
output includes metrics, trajectories, fitted shape parameters, and the exact
settings. `--replicates 2 --max-steps 2` is a diagnostic run, not a reproduction
of the paper's uncertainty or stability results.

## Model and Evaluation

The Li2S template is calculated from its cubic crystal structure. A lattice
search and Gaussian width calibration use only cycles 1--4. Thereafter the
atomic coordinates, lattice parameter, line shape, and relative reflection
intensities are fixed: the evolving quantity is crystalline-phase intensity,
not a refined atomic structure.

The shared shape parameters are fitted to the 200 window at 30.5--32.5 degrees
in cycles 1--4. Progress is the normalized cumulative capacity change within
each half-cycle. Growth follows a bounded onset and power exponent; decay is
a normalized stretched-exponential function. The comparison spline has six
monotone knots on each branch. Both models fit a separate nonnegative cycle
amplitude. Linear backgrounds are eliminated analytically.

In cycles 5--8, the shape is held fixed and only a cycle amplitude is fitted
using the 200 window. The 111 window at 25.5--28.0 degrees is evaluated without
refitting its intensity scale. Its residuals and agreement with the published
peak-area measurements are distinct evaluation quantities.

The published template calculation uses float32, followed by float64 shape
optimization with L-BFGS and strong-Wolfe line search. This precision sequence
is retained for replay. The full workflow uses 120 optimizer iterations per fit.
Changing the template precision, initialization, or optimizer settings defines
a different calculation, even when the visible curves are similar.
