# Forward Validation

`reproduce/validate.py` recalculates diffraction from structural inputs and
compares it with prepared GSAS-II reference profiles and reflection arrays.
It does not compare two stored sets of final metrics or use stored diffPXRD
predictions as the reference.

The data deposit supplies the 200 ICSD identifiers and archived benchmark
results, not the structural tensors and GSAS-II reference arrays required by
this command. The command is for users with a separately prepared reference
packet and authorized access to the structures; no public packet is provided.
Identifiers alone cannot be used as its input.

## Run

```bash
python reproduce/validate.py --inputs /path/to/validation-inputs --output results/validation
```

The default command requires all 200 structures and nine configurations. For
an explicitly selected diagnostic subset:

```bash
python reproduce/validate.py --inputs /path/to/validation-inputs \
  --case STRUCTURE_ID --config laboratory_fcj --output results/validation-subset
```

The packet manifest lists available structural identifiers and configuration
names. A partial packet cannot silently stand in for the complete benchmark.
The CPU/float64 calculation matches the manuscript execution mode. `--device
cuda` is available for a separately identified GPU correctness check, not a
replacement for the manuscript's CPU validation.

## What Is Compared

The engine recomputes reciprocal vectors, structure factors, corrected
intensities, reflection families, and complete profiles. Instrument parameters,
sample broadening, preferred orientation, wavelength weights, and the angular
grid are preserved from the final benchmark recipe. Reflection enumeration uses
the union of the wavelength ranges; layer comparisons use the open angular
interval of the primary wavelength. Full-profile comparisons include all
wavelength components.

Profiles are signal-only: the GSAS-II calculated background is removed before
comparison. One least-squares scale factor is fitted, with uniform weights and
no fitted background. Matched reflection intensities have their own single
least-squares scale factor. Reflections are matched by reciprocal symmetry
orbits, including Friedel inversion.

The six criteria are unchanged:

| Quantity | Criterion |
| --- | --- |
| Missing reflection families | 0 |
| Extra reflection families | 0 |
| Profile normalized root-mean-square error | at most 0.005 |
| Profile cosine similarity | at least 0.995 |
| Maximum matched peak-position error | at most 0.00005 degrees |
| Matched-intensity normalized root-mean-square error | at most 0.0000002 |

Normalized errors use the maximum absolute reference value. Width differences
are reported as diagnostics; they do not acquire new pass/fail tolerances.
Nonfinite calculations and execution errors are failures, not omitted cases.
Results are written to `results.jsonl` and `summary.json`, including the input
manifest checksum and software versions.

## Reference Preparation

The original finalized evidence retained complete GSAS-II profiles but only
summary reflection metrics. The portable input exporter therefore has to
regenerate the missing reflection arrays with the original certified structure
import, pinned GSAS-II source, and documented ellipsoidal-size and trigonal
microstrain reference corrections. It checks the regenerated profiles against
the archived profiles and verifies the portable evaluator against the archived
six metrics before admitting a comparison to the packet. No structure is
re-standardized during this export.

Running the portable validator does not rerun GSAS-II or revalidate those
reference-generation adapters. It tests the installed diffPXRD engine against
the resulting frozen reference. The packet records that distinction, source
hashes, and reference provenance. The portable workflow has been checked on
one structure/configuration, not replayed over the full 200-structure benchmark.
It is separate from reconstructing the paper's figures and tables from the
archived results.

For a separate exhaustive check of standard-setting symmetry:

```bash
python reproduce/validate_symmetry.py --output results/symmetry.json
```
