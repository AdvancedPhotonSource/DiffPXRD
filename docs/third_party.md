# Third-party attribution

The project uses the author-supplied [BSD license](../LICENSE.txt).
Third-party components retain their own notices and terms; their source
notices and licenses accompany the distribution.

## GSAS-II components

`diffpxrd/_vendored_atmdata.py` contains the `XrayFF` coefficients from
[GSAS-II](https://github.com/AdvancedPhotonSource/GSAS-II). These supply the
X-ray form factors used by diffPXRD and are an unchanged extraction of the
[upstream table at commit dbc628f25d568e99a9c4a126319dd0cb308279ed](https://github.com/AdvancedPhotonSource/GSAS-II/blob/dbc628f25d568e99a9c4a126319dd0cb308279ed/GSASII/atmdata.py).
The accompanying `diffpxrd/data/Xsect.dat` anomalous-scattering data are
byte-identical to [the file at that revision](https://github.com/AdvancedPhotonSource/GSAS-II/blob/dbc628f25d568e99a9c4a126319dd0cb308279ed/GSASII/inputs/Xsect.dat).
The extraction omits unused orbital, electron, atomic-size, and neutron tables.
The X-ray table's scientific references are retained in the module.

The anomalous-scattering routines in `diffpxrd/xray_atomic_data.py` adapt
`GetXsectionCoeff` and `FPcalc`, including its interpolation and integration
helpers, from [GSASIIElem](https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/GSASIIElem.html).
The `GetElInfo` adapter retains only the X-ray dispersion and absorption
outputs. Local changes include package-relative data loading, caching, and
separate helper functions. The upstream module carries the notice:
Copyright 2008, Robert B. Von Dreele & Brian H. Toby (Argonne National
Laboratory).

Copyright 2010, UChicago Argonne, LLC, Operator of Argonne National Laboratory.
All rights reserved. The full [GSAS-II license](licenses/gsas-ii.txt), including
its conditions and disclaimers, applies to these components.

This product includes software produced by UChicago Argonne, LLC under Contract
No. DE-AC02-06CH11357 with the Department of Energy.

## Crystallographic software

diffPXRD uses [Gemmi](https://github.com/project-gemmi/gemmi) for space-group
operations and systematic absences, and
[pymatgen](https://github.com/materialsproject/pymatgen) for structure handling.
These are installed dependencies, not vendored copies. The bundled
`spacegroup_lookup.json` records metadata for the 230 space groups in the
canonical settings used by the engine; its Laue labels follow the engine's
reflection-selection conventions.

All eight fields of all 230 records were independently reproduced using
Gemmi 0.7.5. The recipe uses `find_spacegroup_by_number(1..230)`, reading
`number`, `xhm()`, `hm`, `laue_str()`, `centring_type()`,
`monoclinic_unique_axis()`, `crystal_system_str()`, and `point_group_hm()`.
The NUL unique-axis value becomes an empty string; Laue labels `-3`, `m-3`,
and `m-3m` become `3`, `m3`, and `m3m`. For `-3m`, reciprocal symmetry
operations with Friedel inversion distinguish the orientations: opposite-l
reflections are equivalent at k=0 for `31m`, or at h=k for `3m1`. The result
was also checked against the standard Hermann-Mauguin axial fields. This
establishes an equivalent generation procedure, not the table's historical
source.

## Data and examples

The synthetic examples build a conventional NaCl cell in code. This repository
does not bundle ICSD structures or external experimental datasets. Reproduction
inputs have their own source identifiers and access terms, described in the
[reproduction guide](reproduction.md). The code license does not grant rights
over those data.
