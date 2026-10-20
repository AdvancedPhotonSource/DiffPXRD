# diffPXRD

diffPXRD calculates constant-wavelength powder X-ray diffraction profiles in
PyTorch. It supports differentiation through crystal and profile parameters
and batched calculations for fitting individual patterns or diffraction series.

## Install

Use Python 3.11. From this directory:

```bash
python -m pip install .
```

For CUDA, install a CUDA-enabled PyTorch build appropriate for your system
before installing diffPXRD. CUDA and GSAS-II are not required for CPU use.

For the pinned Linux CPU environment, install `torch==2.5.1+cpu` from
`https://download.pytorch.org/whl/cpu`, then use
`python -m pip install -c requirements-cpu.txt '.[test,reproduce]'`.

## Run an example

```bash
python examples/simulate.py --output-dir outputs/simulate
python examples/refine.py --output-dir outputs/refine
python examples/fit_series.py --output-dir outputs/series
```

These examples generate synthetic NaCl patterns and need no downloaded data.
They demonstrate forward calculation, lattice refinement, and fitting a shared
lattice-evolution model to several profiles. They are installation examples,
not reproductions of the experimental results in the paper. Each writes CSV data
and a JSON summary. Add `--device cuda` to request GPU execution explicitly.

## Use the library

```python
import torch
from pymatgen.core import Lattice, Structure
from diffpxrd import ModularCWForwardModel

structure = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
model = ModularCWForwardModel(q_min=0.5, q_max=6.0, wavelength=1.5406, device="cpu")
batch = model.structure_to_batch(structure, dtype=torch.float64)
layers = model.forward_with_layers(batch)
profile = model.compute_tch_profile_layer(
    layers["correction"],
    instrument={"U": 0.0, "V": 0.0, "W": 16.0, "X": 0.0, "Y": 0.0, "Z": 2.0},
    domain=torch.linspace(15.0, 80.0, 2001, dtype=torch.float64),
    return_diagnostics=False,
)
```

See [conventions](docs/conventions.md) for units, precision, and differentiation
boundaries. The `examples/` scripts show the full calculation and optimization
without requiring a separate fitting framework.

## Repository guide

| Directory | Contents |
| --- | --- |
| `diffpxrd/` | Diffraction engine, profile objectives, input readers, and cubic splines. |
| `examples/` | Three self-contained examples with generated data. |
| `reproduce/` | Paper validation and application commands, separate from library use. |
| `tests/` | Six focused modules covering the numerical API. |
| `docs/` | Conventions, data preparation, reproduction, and third-party attribution. |

## Test

```bash
python -m pip install '.[test]'
python -m pytest -q
```

GPU checks are skipped when CUDA is unavailable. Skipped GPU tests do not
establish GPU correctness.

The exhaustive space-group check is separate:

```bash
python reproduce/validate_symmetry.py --output results/symmetry.json
```

## Reproduce the paper

The `reproduce/` commands regenerate the analyses, figures, and tables of the paper
"Differentiable powder X-ray diffraction enables analysis of evolving structures".

Install the analysis dependencies with `python -m pip install '.[reproduce]'`.
The figures and supplementary tables are rebuilt from the paper's data deposit
without refitting: unpack its `source_data.zip` and run `reproduce/figures.py` and
`reproduce/tables.py` as described in [figures and tables](docs/figures.md).
See [reproduction](docs/reproduction.md) for every analysis, its inputs, and its
command. Raw data, generated results, and large reference arrays are not part of
this source tree.

## Cite

If you use diffPXRD, please cite the paper; see [CITATION.cff](CITATION.cff).

## Attribution

diffPXRD is distributed under the [BSD license](LICENSE.txt).
Original diffPXRD outputs in the paper's data deposit use the same license.
Third-party measurements retain the terms credited in the deposit's `SOURCES.md`.
Kong et al. neutron coordinates are not redistributed; the Ca3Ti2O7 recipe and
Figure 5 require [reader-transcribed Tables S1-S2](docs/ca3ti2o7_neutron_tables.md).

This product includes software produced by UChicago Argonne, LLC under Contract
No. DE-AC02-06CH11357 with the Department of Energy.

See [third-party attribution](docs/third_party.md) for the bundled atomic-data
sources and license notices.
