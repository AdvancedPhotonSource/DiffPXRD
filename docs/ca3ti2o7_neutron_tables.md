# Reader-Supplied Ca3Ti2O7 Neutron Tables

The deposit does not redistribute neutron coordinates, displacement parameters,
cell lengths, or mapped versions of the structural tables from Kong et al.,
*Physical Review B* **107**, 224103 (2023). Obtain the supplement through the
[article DOI](https://doi.org/10.1103/PhysRevB.107.224103) or the publisher's
[supplemental PDF](https://journals.aps.org/prb/supplemental/10.1103/PhysRevB.107.224103/RP_phase_HIF_SI_May27.pdf),
`RP_phase_HIF_SI_May27.pdf`, and respect its terms. Do not substitute the
Figshare X-ray structures or another composition for the neutron refinements.

## Transcribe

Use the **x = 0, Ca3Ti2O7** refinements in Tables S1 (heating) and S2 (cooling).
Transcribe every independent site's atom label, element, Wyckoff position,
fractional x/y/z coordinates, isotropic displacement parameter Uiso, and
occupancy, together with the temperature and conventional orthorhombic a/b/c
cell lengths. Retain the published central values at their printed precision;
omit parenthesized standard uncertainties. Preserve fixed coordinates too.
Use Uiso in angstrom squared, not Biso; apply any scale factor stated in the
table heading. All sites in this recipe have full occupancy (1).

Include all eleven structures: heating A21am at 313, 373, 423, 473, 673, 873,
and 1073 K; cooling A21am at 573, 773, and 973 K; and heating Acaa at 1173 K.
Use lower-case `a21am` and `acaa` in the CSV's `phase` column. Use `heating`
and `cooling` for `cycle`. The A21am site labels are `Ca1,Ca2,Ti,O1,O2,O3,O4`;
Acaa has `Ca1,Ca2,Ti,O1,O2,O3`. Use `Ti` rather than `Ti1`. Repeat the cell
lengths and structure identity on every site row. This gives 76 site rows.
Do not expand symmetry, swap axes, wrap coordinates, or match atoms yourself.

One UTF-8, comma-separated file must have exactly this header, in this order:

```csv
cycle,temperature_k,phase,label,element,wyckoff,a_angstrom,b_angstrom,c_angstrom,x,y,z,u_iso_angstrom2,occupancy
```

`temperature_k` is in kelvin; `a_angstrom,b_angstrom,c_angstrom` are in angstrom;
`x,y,z` are dimensionless fractional coordinates; `u_iso_angstrom2` is in
angstrom squared; `occupancy` is dimensionless. Wyckoff positions are strings
such as `4a`. Write ordinary decimal numbers, not uncertainty notation or units
inside cells. Row order is immaterial: the preparer restores the research
structure order and the site-label order above before expanding symmetry.

**Format examples only. These two rows contain made-up placeholder values,
not neutron data, and are not a complete or scientifically valid input.**

```csv
cycle,temperature_k,phase,label,element,wyckoff,a_angstrom,b_angstrom,c_angstrom,x,y,z,u_iso_angstrom2,occupancy
heating,313,a21am,Ca1,Ca,4a,6.1,6.2,21.3,0.123,0.234,0,0.0123,1
heating,313,a21am,Ca2,Ca,8b,6.1,6.2,21.3,0.345,0.456,0.167,0.0234,1
```

## Prepare And Use

Install the matching code release with its reproduction dependencies. From its
root, run the public preparer on your own transcription:

```sh
python reproduce/prepare_ca3ti2o7_neutron.py --csv /path/to/my_tables.csv \
  --output /path/to/my_neutron_tables.npz
python reproduce/figures.py --data-dir /path/to/source_data --figures 5 \
  --neutron-tables /path/to/my_neutron_tables.npz --output-dir results/figure5
python reproduce/ca3ti2o7.py --data-dir /path/to/reproduction_data/ca3ti2o7/inputs \
  --neutron-tables /path/to/my_neutron_tables.npz --output-dir results/ca3ti2o7 \
  --device cpu
```

Preparation reuses the research expansion and mapping function bodies in
`reproduce/ca3ti2o7_mapping.py`: pymatgen space-group expansion, element-wise
assignment, signed-axis/a-b permutation search, and five origin-refinement
iterations with Ti seeds. It writes the nine reader-supplied arrays consumed
by the recipe, including the common 48-site ordering, Acaa reference and the
neighbor-selection reference. It does not fit diffraction data. The displacement
basis is rebuilt by the recipe, not distributed in the deposit. Figure 5a uses
the prepared Acaa reference and cell together with the deposited fitted positions.
The NPZ and your CSV retain the source's terms; preparing them does not relicense
the structural information. Keep them outside files intended for redistribution.

The program rejects incomplete structure/site rosters, repeated labels,
inconsistent cells, nonfinite numbers, and invalid symmetry multiplicities.
These checks cannot detect every transcription error: compare every entry
against the supplement. Use the matching release environment because symmetry
expansion and tie-breaking must preserve site order for the stored fitted results.

Figures 1-4, Supplementary Figure S1, all supplementary tables, and the saved
supporting-statement checks do not need this file. Figure 2 separately needs
the authorized ICSD structure described in `figures.md`.
