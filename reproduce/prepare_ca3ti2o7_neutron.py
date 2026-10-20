"""Prepare a reader's transcription of Kong et al. Tables S1-S2, not supplied data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ("cycle", "temperature_k", "phase", "label", "element", "wyckoff",
          "a_angstrom", "b_angstrom", "c_angstrom", "x", "y", "z",
          "u_iso_angstrom2", "occupancy")
NEUTRON_KEYS = frozenset({
    "acaa_fractional", "acaa_lattice", "neutron_fractional", "neutron_lattice",
    "neutron_u_iso", "neutron_temperature", "neutron_cycle",
    "neighbor_reference_fractional", "neighbor_reference_lattice",
})
INSTRUCTIONS = "See docs/ca3ti2o7_neutron_tables.md; run reproduce/prepare_ca3ti2o7_neutron.py on your transcription."
SERIES = [("heating", t, "a21am") for t in (313., 373., 423., 473., 673., 873.)] + [
    ("cooling", t, "a21am") for t in (573., 773., 973.)]
ENDPOINTS = [("heating", 1073., "a21am"), ("heating", 1173., "acaa")]
LABELS = ("Ca1", "Ca2", "Ti", "O1", "O2", "O3", "O4")
MAPPING = {"mapping": {"allow_ab_axis_swap": True, "allow_axis_sign_reversal": True,
                       "origin_seed_element": "Ti", "origin_refinement_iterations": 5}}


def load_prepared(path):
    if path is None or not Path(path).is_file():
        raise ValueError("Ca3Ti2O7 requires --neutron-tables with a prepared NPZ. " + INSTRUCTIONS)
    with np.load(path, allow_pickle=False) as packet:
        if set(packet.files) != NEUTRON_KEYS:
            raise ValueError("Unexpected prepared neutron array roster. " + INSTRUCTIONS)
        arrays = {key: packet[key] for key in packet.files}
    shapes = {"acaa_fractional": (48, 3), "acaa_lattice": (3,),
              "neutron_fractional": (10, 48, 3), "neutron_lattice": (10, 3),
              "neutron_u_iso": (10, 48), "neutron_temperature": (10,),
              "neutron_cycle": (10,), "neighbor_reference_fractional": (48, 3),
              "neighbor_reference_lattice": (3, 3)}
    for key, value in arrays.items():
        if value.shape != shapes[key] or (key != "neutron_cycle" and
                (value.dtype != np.float64 or not np.isfinite(value).all())):
            raise ValueError(f"Invalid prepared neutron array: {key}. " + INSTRUCTIONS)
    return arrays


def read_transcription(path):
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("CSV header must exactly match the documented schema. " + INSTRUCTIONS)
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError(f"Incomplete CSV row {line}")
            values = {key: float(row[key]) for key in ("temperature_k", *FIELDS[6:])}
            if not np.isfinite(list(values.values())).all():
                raise ValueError(f"Nonfinite CSV row {line}")
            if (min(values[key] for key in FIELDS[6:9]) <= 0 or
                    values["u_iso_angstrom2"] < 0 or values["occupancy"] != 1):
                raise ValueError(f"Invalid lattice, displacement parameter or occupancy on row {line}")
            key = (row["cycle"], values["temperature_k"], row["phase"])
            groups.setdefault(key, []).append((row, values))
    if set(groups) != set(SERIES + ENDPOINTS):
        raise ValueError("Expected all eleven structures specified in the instructions")
    records = {}
    for key, rows in groups.items():
        labels = LABELS if key[2] == "a21am" else LABELS[:-1]
        by_label = {row["label"]: (row, values) for row, values in rows}
        if len(rows) != len(labels) or set(by_label) != set(labels):
            raise ValueError(f"Missing or repeated sites for {key}")
        cells = {tuple(values[k] for k in FIELDS[6:9]) for _, values in rows}
        if len(cells) != 1:
            raise ValueError(f"Inconsistent cell lengths for {key}")
        sites = []
        for label in labels:
            row, values = by_label[label]
            if row["element"] != ("Ca" if label.startswith("Ca") else "Ti" if label == "Ti" else "O"):
                raise ValueError(f"Wrong element for {key}, {label}")
            sites.append({"label": label, "element": row["element"], "wyckoff": row["wyckoff"],
                              "fractional_coordinates": [values[k] for k in ("x", "y", "z")],
                              "u_iso_angstrom2": values["u_iso_angstrom2"], "occupancy": values["occupancy"]})
        records[key] = {"cycle": key[0], "temperature_k": key[1], "sites": sites,
                            "lattice_angstrom": list(cells.pop()),
                            "space_group_symbol": "A21am" if key[2] == "a21am" else "Acaa",
                            "space_group_number": 36 if key[2] == "a21am" else 68}
    return ({"a21am_structures": [records[key] for key in SERIES]},
            {"phase_templates": {key[2]: records[key] for key in ENDPOINTS}})


def prepare(path):
    from ca3ti2o7_mapping import _build_structures, build_mapped_coordinate

    neutron, endpoint = read_transcription(path)
    structures, metadata = _build_structures(neutron, endpoint)
    structures, metadata = structures[:-1], metadata[:-1]
    coordinate = build_mapped_coordinate(endpoint, MAPPING)
    reference = coordinate["target"]
    for structure in structures:
        if (structure.species != reference.species or
                structure.site_properties["source_label"] != reference.site_properties["source_label"]):
            raise ValueError("Neutron structures do not share the reference site order")
    return {
        "acaa_fractional": coordinate["source_fractional"],
        "acaa_lattice": np.asarray(coordinate["source"].lattice.abc)[list(coordinate["permutation"])],
        "neutron_fractional": np.stack([structure.frac_coords for structure in structures]),
        "neutron_lattice": np.asarray([structure.lattice.abc for structure in structures]),
        "neutron_u_iso": np.asarray([structure.site_properties["u_iso"] for structure in structures]),
        "neutron_temperature": np.asarray([item["temperature_k"] for item in metadata]),
        "neutron_cycle": np.asarray([item["cycle"] for item in metadata]),
        "neighbor_reference_fractional": reference.frac_coords,
        "neighbor_reference_lattice": reference.lattice.matrix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = prepare(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    load_prepared(args.output)


if __name__ == "__main__":
    main()
