# Ca3Ti2O7

This example refines collective mean atomic positions from the temperature-dependent
Ca3Ti2O7 powder diffraction series. It starts from prepared measured profiles and
reader-supplied neutron structures, not raw detector images. It does not assume that the
X-ray data uniquely determine every atomic coordinate.

## Run

```sh
python reproduce/ca3ti2o7.py --data-dir data/ca3ti2o7 --neutron-tables /path/to/my_neutron_tables.npz --output-dir results/ca3ti2o7 --device cuda:0
```

The default calculation reproduces the final profile prescription with separate
Gaussian and Lorentzian widths and the measured 213/117 integrated-area ratio.
It fits independent heating and cooling splines, performs independently
initialized framewise fits, fits the undistorted Acaa reference, and predicts
ratios at the other eligible measured temperatures. The full neutron start is
the displayed paper result; the half-neutron start is a sensitivity check, not
a replacement selected for a more favorable trajectory.

```sh
python reproduce/ca3ti2o7.py --data-dir data/ca3ti2o7 --neutron-tables /path/to/my_neutron_tables.npz --output-dir results/ca_sensitivity --device cuda:0 --case all --structural-check
```

`--case profile_only` removes the ratio term. `--case omit_doublet` removes both
the entire doublet window and the ratio term. The three `common_*` cases retain
the earlier common-width prescription for comparison. `--structural-check`
adds 100 bounded optimization iterations for the structural coefficients while
holding nonlinear nuisance parameters fixed. Linear scale and background are
still solved at each evaluation.

`--smoke` keeps all 42 profiles and the same windows but permits only two
optimization iterations and one start. Its outputs are labeled as a smoke
check, not fitted scientific results. CPU execution is selected explicitly with
`--device cpu`; a CUDA request never falls back silently.

## Inputs

The prepared packet has just two files: `metadata.json` and `measurements.npz`.
The JSON records the array-file SHA-256 checksum, physical configuration, original
source links and checksums, selected frame identities, and window definitions.
The NPZ contains only numerical arrays and Unicode strings; loading never permits
Python pickles. It contains no saved fitted structural trajectory or optimizer
state. Input arrays and original measurements are not modified.

The X-ray profiles and lattice parameters come from the experimental deposit
[10.6084/m9.figshare.12446306](https://doi.org/10.6084/m9.figshare.12446306).
The archive SHA-256 is
`4786813feccc228e623b4646e151d03096c1e95cd9919b03e8140e86eeed8b51`.
The neutron structures are not distributed. Transcribe the supplemental structural tables
of [Kong et al., Phys. Rev. B 107, 224103](https://doi.org/10.1103/PhysRevB.107.224103).
following [the CSV preparation instructions](ca3ti2o7_neutron_tables.md).
Pass the resulting NPZ separately as `--neutron-tables`; it is never loaded
implicitly from the measurement packet.

Preparation expands and matches the neutron structures to one ordered 48-site
cell and maps the Acaa reference into that convention. It selects 21 temperatures
per branch, nearest to evenly spaced targets from 313 to 1073 K on heating and
313 to 1055 K on cooling. It resamples measured intensities and standard deviations
linearly onto the median selected angular grid; the stored variance is the square
of that resampled standard deviation. Three early-cooling transition frames are
excluded from the single-phase refinement interval.

Windows are selected using reference-structure screening, reflection-content
ranking, tracked centers, and angular-sampling support. The prepared packet
freezes that selection rather than reranking windows after refinement. It retains
all tracked centers, not only the ten fitted windows, because nearest-center
ownership determines the pixels assigned to overlapping windows. The doublet
and 111 centers are tracked analytically from lattice parameters.

The combined input contract is below. `species` and `site_labels` remain in the
measurement packet; all `acaa_*`, `neutron_*` and `neighbor_reference_*` arrays
come only from your separately prepared file, not the deposit.

| Arrays | Shape and meaning |
| --- | --- |
| `two_theta`, `observed`, `variance` | `(P,)`, `(42,P)`, `(42,P)`; degree-valued angular grid, measured intensities and positive variances |
| `cycle`, `temperature`, `scan`, `source_index` | `(42,)`; 21 heating then 21 cooling nodes, increasing temperature within each branch |
| `lattice` | `(42,3)`; deposited orthorhombic lengths in angstrom |
| `window_centers`, `hkls` | `(42,W)`, `(1,H,3)`; tracked window centers in degrees and frozen signed reflections |
| `ratio`, `ratio_sigma` | `(42,)`; measured 213/117 area ratios with the 2% relative uncertainty floor used for fitting |
| `species`, `site_labels`, `acaa_fractional`, `acaa_lattice` | `(48,)`, `(48,)`, `(48,3)`, `(3,)`; species/site order and mapped undistorted reference |
| `neutron_fractional`, `neutron_lattice`, `neutron_u_iso` | `(N,48,3)`, `(N,3)`, `(N,48)`; matched neutron structures, lengths and site-specific isotropic displacement parameters in angstrom squared |
| `neutron_temperature`, `neutron_cycle` | `(N,)`; neutron temperature and branch identities |
| `neighbor_reference_fractional`, `neighbor_reference_lattice` | `(48,3)`, `(3,3)`; fixed structural reference used to select TiO6 neighbor identities |
| `series_temperature`, `series_cycle`, `series_scan`, `series_lattice` | 346 single-phase measurement identities and corresponding lattice lengths |
| `series_ratio`, `series_ratio_sigma`, `series_eligible` | 346 measured ratios, propagated measurement uncertainties without the fitting floor, and the mask retaining 343 eligible temperatures |
| `doublet_parameters` | `(42,6)`; measured areas 213/117, centers 213/117, common FWHM and Lorentzian fraction of the independent two-peak decomposition |

The ratio measurements and fixed doublet peak parameters come from the same
model-free two-peak decomposition. They are not structural fit outputs. Only a
new local linear background is fitted when rendering the saved decomposition for
profile-shape comparison. This is an in-sample diagnostic, not prediction.

## Calculation

The neutron basis is recomputed from the supplied coordinates. Fractional
differences from Acaa are wrapped to the nearest image, converted to Cartesian
displacements using mean neutron/Acaa lattice lengths, and stripped of rigid
translation. The first three right singular vectors of the uncentered heating
series are sign-canonicalized and normalized to one angstrom root-mean-square
displacement per unit amplitude over all 48 sites. The neutron structures specify
allowed collective movements; the X-ray profiles refine their amplitudes.

Heating and cooling each use three seven-coefficient cubic splines. A radial
transformation keeps their combined displacement amplitude below 0.4 angstrom.
Neutron amplitudes and site-specific displacement parameters are linearly
interpolated within each branch, with endpoint values held outside its neutron
temperature range. The displacement parameters remain fixed during refinement.
Framewise fits start directly from the neutron amplitudes, never from a joint fit.

Profiles use TCH broadening with one Gaussian width and an independent Lorentzian
multiplier per frame, plus bounded lattice and zero-shift corrections. One
nonnegative intensity scale is shared across the fitted windows in each frame;
each window has its own linear background. These linear parameters are eliminated
by weighted least squares. The objective is the mean, over frames, of the mean
standardized squared profile residual, plus the weighted sum of squared
standardized ratio residuals.

The original paper recipe converted structures through the engine's float32
input default before casting the resulting batch to float64. This conversion is
preserved for numerical replay. The neutron basis, variable coordinates,
objective, gradients, and optimizer operate in float64; preparation preserves
the precision of the reader's transcribed coordinates and lattice data.

Optimization uses the recorded L-BFGS block size, line search, two independent
starts, and termination checks. Stable objective/parameter changes are recorded
separately from strict gradient convergence. For an omitted window, its
diffraction scale comes from the training windows; only its local background is
fitted. At the 301 unfitted temperatures, spline amplitudes are evaluated directly
and fitted lattice scale factors are interpolated within the branch. This tests
interpolation of the intensity ratio, not independent atomic-structure validation.

## Outputs

Each fit writes `trajectory.csv`, `parameters.npz`, `profiles.npz`,
`window_scores.csv`, `history.json`, and `summary.json`. The reference case also
writes `temperature_predictions.csv` and `displacement_ratio_scan.csv`.
The trajectory table reports collective displacement, TiO6 tilt, and the coherent
c-axis rotation projection defined by the low-temperature neutron structure.
The three output quantities describe mean positions within the supplied basis;
they are not observations of individual atomic dynamics.

This command intentionally does not prepare raw detector data, reinterpret the
neutron site mapping, select new windows, or create publication figures. Those
preparation choices are explicit inputs rather than hidden dependencies on a
research checkout.
