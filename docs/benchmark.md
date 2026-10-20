# Repeated-Evaluation Benchmark

`reproduce/benchmark.py` measures three repeated numerical workloads: a forward
FCJ profile calculation, a weighted profile loss, and that loss plus gradients
with respect to two lattice-scale parameters. It does **not** run refinement,
GSAS-II sequential analysis, or an end-to-end structure-to-profile comparison.

```sh
python reproduce/benchmark.py --structures data/structures.json \
  --output-dir results/benchmark --device both
```

`both` measures a four-thread CPU and one explicitly selected CUDA device.
`cpu` or `cuda:0` measures only that device. CUDA requests fail if unavailable.
GPU work is always checked against CPU calculations before timing; CPU-only
output explicitly records that CPU/GPU agreement was not checked.

## Input

Provide the same 200 target structures used for the paper, not the Materials
Project starting structures. Licensed structure files are not bundled here.
The input file contains `{"structures": [...]}`; each list member has
`case_id` and `structure` fields:

```json
{
  "case_id": "unique-structure-name",
  "structure": {
    "lattice_matrix": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
    "species": ["Si"],
    "frac_coords": [[0, 0, 0]],
    "site_properties": {}
  }
}
```

This snippet illustrates the schema, not a benchmark input. Alternatively,
`--structures` can name a directory of individual JSON files with the same
`case_id`/`structure` schema. Lattice vectors are in angstroms, coordinates are
fractional, and site properties must be preserved from the frozen targets.
The command requires exactly 200 unique case identifiers unless `--limit N`
explicitly requests a diagnostic subset. Having 200 entries does not establish
that they are the paper cohort: retain the cohort manifest and input hashes.

## Fixed Workload

- Float64 structure conversion, calculation, loss, and gradients.
- 4,096 points from 5 to 35 degrees at wavelength 0.49593679373280103 A.
- FCJ profile with `U=1.163`, `V=-0.126`, `W=0.063`, `X=Y=Z=0`,
  `SH/L=0.002`, zero correction 0, and polarization 0.99.
- Batch sizes 1, 2, 4, and 8 for every structure.
- Shared CPU-built structures, fixed reflection lists, target profiles, weights,
  background basis, and candidate scales, transferred without regeneration.
- Equally spaced `ab` scales from 0.99 to 1.01 and `c` scales from 1.01 to 0.99.
- Profile weights `1/max(target, max(max(target)*1e-4, 1))`.
- Analytically fitted scale and degree-3 Chebyshev background, with the weighted
  loss summed across the batch.
- Three warm-ups followed by ten measured repetitions, with explicit CUDA
  synchronization. Candidate construction and scalar result transfer are
  included, as in the paper's repeated-evaluation workload.

Complete profile, loss, and gradient arrays must agree before timing. The
per-element tolerance is `5e-11 * max(abs(reference)) + 5e-9 * abs(reference)`;
candidate scale inputs must be exactly equal. A numerical failure or memory
limit stops execution. The command does not shrink a batch, drop a structure,
change precision, or substitute another device to complete a table.

## Output

`summary.json` records the real hardware, package versions, input file hashes,
reflection hashes, numerical discrepancies, all ten time samples, medians, and
profiles per second. `progress.json` preserves completed cells if execution
stops. A `--limit` run is labeled diagnostic throughout.

The numerical workload is extracted from the final paper benchmark. Absolute
times depend on hardware and software versions; a different GPU is not a
reproduction of the paper's A100 timing platform. The private release checks
compare this port's prepared tensors and complete numerical results against
the original implementation without making timing assertions.
