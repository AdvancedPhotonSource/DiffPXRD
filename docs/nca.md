# NCA Lattice Refinement

`reproduce/nca.py` fits the 003 reflection of the *operando* nickel-cobalt-aluminum
oxide series used in Figure 3. It compares one lattice parameter per pattern
with separate cubic capacity splines for charge and discharge. Both models fit
the same two effective peak widths, scale, and linear background per pattern.

## Prepare the Data

Download the original data from [Zenodo record 7049855](https://doi.org/10.5281/zenodo.7049855)
and extract the directory named
`Resolving high potential structural deterioration in Ni-rich layered cathode materials for lithium-ion batteries operando`.
Keep the published data unchanged. The preparation command needs only
`XRD/Operando files/*_back2.*.xy` and `XRD/icixrd.csv`; published TOPAS results
are not read or used as fitted targets or initial values.

```sh
python reproduce/nca.py --task prepare \
  --data-dir "/path/to/extracted/dataset" --output-dir data/nca
```

The command selects all 239 patterns, the first 239 matching electrochemistry
rows, and the inclusive 18.1--19.8 degree window. It retains the original 114
angular bins without interpolation or rebinning. The output is a compressed
`nca.npz` plus `input.json`, which records hashes of every consumed source file.
Data access and redistribution remain subject to the source deposit's terms.

The prepared file contains six arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `scan_numbers` | `(239,)` | Integers 1 through 239 |
| `time_sec` | `(239,)` | Published elapsed time in seconds |
| `capacity` | `(239,)` | Published `Q` column, unchanged |
| `state` | `(239,)` | `C` for charge, `D` for discharge |
| `two_theta_deg` | `(114,)` | Increasing measured angles in degrees |
| `raw_counts` | `(239, 114)` | Finite, nonnegative integer-valued counts |

For archive migration, `--task prepare --input-format archive` reads these same
six measured arrays from `science_trajectories.npz` in `--data-dir`. It ignores
every saved fitted trajectory. Both input routes use one numerical fitting
implementation. An archive extraction is not by itself proof that the raw-data
preparation is equivalent; compare all six arrays before releasing a prepared
packet.

## Run the Comparisons

```sh
python reproduce/nca.py --task noise --data-dir data/nca \
  --output-dir results/nca-noise --device cuda:0
python reproduce/nca.py --task holdout --data-dir data/nca \
  --output-dir results/nca-holdout --device cuda:0
python reproduce/nca.py --task timing --data-dir data/nca \
  --output-dir results/nca-timing --device cpu
```

Use `--device cpu` or `--device cuda:0` explicitly to select computation. CUDA
requests fail if CUDA is unavailable; they never switch to the CPU. The script
uses four CPU threads. Output directories must not already exist.

`noise` fits the original data and all twenty paired 10% count realizations.
Each independent pattern has a separate optimizer and observed-peak initial
value. The joint fit has 12 spline coefficients per branch. `check` runs the
original data and only the first noisy realization, still on all 239 patterns;
its output is explicitly diagnostic and cannot establish replicate stability.
It is not the lightweight installation example.

`holdout` omits every fifth pattern within each branch, leaving 191 training
patterns and 48 evaluation patterns. It reproduces the original holdout
comparison's batched independent training fit, rather than silently replacing
that analysis with the later separate-optimizer noise comparator. At predicted
`c`, both methods fit fresh widths, scale, and background to each omitted
profile. This is conditional profile agreement, not a blind profile prediction.

`timing` compares serial independent, batched independent, and joint fitting,
with one warm-up and three measured repetitions on the requested device. Input
reading, model setup, fitting, parameter extraction, and compressed output
writing are included. This portable version reads the minimal prepared packet;
the paper timings also read and validate a larger archival packet. Consequently
these measurements must not replace the published timing numbers without a
new explicitly identified benchmark. Hardware also affects absolute times.

## Numerical Recipe

The fixed structure is hexagonal `R-3mH`, with `a = 2.85756 A`, reference
`c = 14.2 A`, oxygen `z = 0.25383`, and Ni/Co/Al occupancies 0.90/0.05/0.05.
The wavelength is 1.5406 A. Lattice values are bounded by
`c = 14.2 exp(0.05 tanh(B u))`, where `B` is the identity matrix or the branch
spline design. Cubic splines use equally spaced interior knots and four repeated
knots at each endpoint of each branch's capacity range.

The Gaussian sigma is bounded to `[0.001, 0.3]` degrees and the Lorentzian FWHM
to `[0.001, 0.5]` degrees. They start at 0.06 and 0.10 degrees for every supplied
observation, including thinned data. The fixed FCJ axial-divergence parameter
is `SH/L = 0.02`; zero correction is 0.01813 degrees and polarization is 0.5.
These are effective widths of one reflection, not separate instrumental and
sample broadening estimates.

Each observed profile is normalized by its 0.02 and 0.995 quantiles, with
negative normalized bins clipped to zero. The objective is the mean squared
normalized-profile residual with uniform bin weights. Scale and linear
background are solved analytically at each evaluation. L-BFGS uses learning
rate 1, strong-Wolfe line search, 200 iterations, 500 nominal evaluations,
history size 50, gradient tolerance `1e-7`, and change tolerance `1e-12`.
The native line search can finish beyond the nominal evaluation limit.

For numerical continuity with the paper, structure construction and domain
rounding use float32, followed by float64 forward evaluation and optimization.
The least-squares coefficient initializer also rounds to float32 before
promotion. These choices are explicit rather than silently replaced with a
different all-float64 initialization. Reflection membership is fixed during
fitting and saved in `reflections.json`.

Thinning uses NumPy's default random generator and seed
`20260724 + 100 + 10007 * replicate` for replicates 0--19. Counts are thinned
before normalization. No fitted coordinates or widths are reused between
realizations. A nonpositive scale triggers the original prescribed retry;
initial and rejected outputs remain in the saved diagnostic arrays. Any
remaining nonpositive scale is an error, not an excluded observation.

The fixed seeds reproduce the count draws, but nondeterministic CUDA accumulation
can change refinement endpoints within the fixed L-BFGS iteration budget.
For reproduction of the recorded figures, use the frozen plotting packet with
`reproduce/figures.py`.

## Outputs and Verification

`summary.json` contains profile losses, bound/rank checks, and the RMS
across-pattern standard deviation across the twenty realizations (`ddof=0`).
Each scenario also saves fitted profiles, lattice parameters, effective widths,
and optimizer histories. Original-profile scoring holds the noisy fit's `c`
and widths fixed and refits only scale and background to the original counts.
Those counts overlap the thinned observations, so this measures agreement,
not independent predictive accuracy.

The manuscript's archived full comparison reports RMS standard deviations
0.00339737 A (independent) and 0.00107437 A (joint), and a median paired
original-profile loss ratio of 0.940928. These are archived reference values,
not newly generated results. A bounded deterministic check verified the
extracted command against the final research implementation; it did not rerun
the full twenty-realization comparison.
