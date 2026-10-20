# Paper Calculations

The commands here are independent of the research workspace. Install the package
with `python -m pip install '.[reproduce]'`, then run them from the repository
root. Inputs remain separate from source code, and results go to a new output
directory. CUDA must be requested explicitly; it never falls back to CPU.

| Calculation | Command | Inputs and procedure |
| --- | --- | --- |
| NCA lattice evolution | `reproduce/nca.py` | [NCA](nca.md): measured patterns and electrochemistry from Zenodo |
| Li2S intensity evolution | `reproduce/lisulfur.py` | [Li-S](lisulfur.md): the unmodified Zenodo data archive |
| Ca3Ti2O7 collective displacement | `reproduce/ca3ti2o7.py` | [Ca3Ti2O7](ca3ti2o7.md): prepared measurements and [reader-transcribed neutron tables](ca3ti2o7_neutron_tables.md) |
| Diffraction reference comparison | `reproduce/validate.py` | [Validation](validation.md): structures and frozen GSAS-II reference arrays |
| Repeated CPU/GPU evaluation | `reproduce/benchmark.py` | [Benchmark](benchmark.md): structural input batches |
| Exhaustive symmetry checks | `reproduce/validate_symmetry.py` | No experimental data required |
| Final figures and SI tables | `reproduce/figures.py`, `reproduce/tables.py` | [Figures and tables](figures.md): frozen plotting-data packet; no fits rerun |

Each command has `--help`. The application documentation distinguishes measured
inputs, fixed model assumptions, refined parameters, and evaluation data. Small
diagnostic runs do not reproduce the full statistical comparisons.

NCA and Li-S can be prepared from their public source deposits. Ca3Ti2O7 starts
from a prepared packet; detector preprocessing and neutron-table transcription
are not part of its fitting command. Validation requires a separate reference
packet. Neither packet is bundled with the source tree. Access to ICSD-derived
structural inputs requires permission from their provider.

The repeated-evaluation benchmark measures the diffraction workload, not data
ingestion or full refinement. It is not the paper's GSAS-II sequential-refinement
benchmark. Absolute timings depend on hardware, thread count, precision, and
software versions; preserve these alongside the results.
