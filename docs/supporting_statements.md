# Supporting Statements

Run `python reproduce/supporting_statements.py --data-dir source_data --output
results/supporting_checks.json` after extracting the source-data ZIP. This is
saved-record arithmetic, not a refit, numerical engine replay, or benchmark.
It runs on a CPU with NumPy and pandas. It checks the nine statements identified
in the paper 1 deposit audit at their displayed precision. Inequalities use the
stated bounds without relaxed numerical tolerances. The Ca start-distance bound
is compared at four decimal places, as displayed in the manuscript.

`source_data/supporting/index.json` lists the source run and member for each
export. An unresolved source is listed in `missing`, never replaced by a
successful summary value copied from the manuscript. The checker reports each
missing statement and exits unsuccessfully. Staging a different or incomplete
run is not a supported fallback.

| Record | Recomputed check |
| --- | --- |
| `cu_benchmark.json` | All four arms, six saved repetitions including warm-up, and 17 patterns. GSAS-II status/finite-covariance criterion and L-BFGS terminal gradient/change criterion. Batched termination is batch-level. |
| `cu_starts.csv`, `cu_starts.json` | All displaced starts and both programs; 251/255 cross-program Cartesian oxygen distances at most 0.005 angstrom. The cell and start coordinates are experimental CuCr2O4 records, not licensed ICSD structures. Saved convergence flags are retained; the start writer did not save full terminal gradients. |
| `merged_equivalence.csv`, `.json` | Maximum saved profile, loss and lattice-gradient relative discrepancies at reference, isotropic and mixed strains. Mixed strain breaks cubic symmetry and is excluded only from that state's symmetry-preserving claim, not from the record. |
| `device_parity.csv` | 2,400 unique structure/setting pairs, complete per-observable decisions and exact lattice-input comparisons. All numerical rows use relative tolerance 5e-9 and normalized absolute tolerance 5e-11. Maximum-difference witnesses are checked. The full elementwise acceptance test cannot be rerun from summary errors alone; these are the original full-array decisions, not decisions reconstructed from sampled profiles. |
| `nca_quality.json` | Per-pattern losses and c parameters for serial/batched CPU/A100 fits, all three timed repetitions and warm-up. Ratio of mean profile losses and RMS paired lattice difference, also checked against the saved quality summary. |
| `ca_starts.csv`, `.json` | Additional starts compared with the displayed external-neutron-start trajectory and objective, not merely with the sensitivity run's repeated reference. RMS is over differences in scalar displacement magnitudes at 42 fitted nodes, not atomic-coordinate distances. Termination records are retained without calling stable trajectories converged. |
| `ca_profiles.json` | Heating endpoint doublet-window NRMSE: square root of mean squared observed-minus-predicted residual, divided by maximum observed intensity in the saved window. The JSON retains every selected angular bin and both profiles. |
| `ca_ratio_scan.csv` | The saved 314 K scan at fractions 0, 0.25 and 1 of the fitted displacement. No new forward evaluation. |
| `lisulfur_lines.csv`, `.json`, `lisulfur_frames.csv`, `lisulfur_reflections.csv` | Per-pattern sharp-peak assignments inside 30.5-32.5 degrees, all 131 frames, first six patterns and evaluation cycles 5-8. Absence is conditional on the original 400-count excess threshold above a 0.6-degree running median. Nearest-line assignment is not independent phase identification. |
| `validation_settings.json` | Parsed original standardization constants (0.01 angstrom, 5 degrees) and 11-ID instrument wavelength (0.413263 angstrom); no simulation imports. |

The research exporter runs on every smoke and full deposit build. The public
checker reads the extracted ZIP, not private inputs. Its JSON report is included
in the build comparison decision. Inspect `storage_notes` before publication:
a saved scratch copy is not evidence of permanent-storage finalization.
