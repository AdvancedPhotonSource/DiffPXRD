"""Regenerate the five current supplementary tables from deposited source data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from timing_tables import timing_tables

CONFIGURATIONS = {
    "lab_cuka_tch": ("Cu K-alpha, TCH", r"Cu K$\alpha$, TCH"),
    "lab_cuka_fcj": ("Cu K-alpha, FCJ", r"Cu K$\alpha$, FCJ"),
    "alphadiffract_worker_uniaxial": ("11-ID, uniaxial broadening",) * 2,
    "alphadiffract_worker_texture": ("11-ID, preferred orientation",) * 2,
    "alphadiffract_stephens": ("11-ID, anisotropic microstrain",) * 2,
    "alphadiffract_ellipsoidal_size": ("11-ID, ellipsoidal size",) * 2,
    "synchrotron_20kev_fcj": ("20 keV, FCJ",) * 2,
    "synchrotron_20kev_worker_uniaxial": ("20 keV, uniaxial broadening",) * 2,
    "synchrotron_20kev_worker_texture": ("20 keV, preferred orientation",) * 2,
}
PARAMETERS = {
    "lattice_lengths": "Cell lengths", "lattice_angles": "Cell angles",
    "fractional_coordinates": "Atomic coordinates", "site_occupancies": "Occupancies",
    "isotropic_displacement": "Isotropic displacement parameters",
    "anisotropic_displacement": "Anisotropic displacement parameters",
    "simple_profile_uvw": "Gaussian width coefficients", "fcj_profile_uvw": "FCJ Gaussian width coefficients",
    "fcj_axial_zero_polarization": "FCJ geometry and corrections", "profiled_wls_lattice": "Lattice gradient of profile loss",
}


def checked_inputs(directory):
    manifest = json.loads((directory / "figures.json").read_text())
    if manifest["schema"] != "diffpxrd-paper1-plotting-inputs-v2":
        raise ValueError("unknown plotting-data schema")
    files = {}
    for name in ("engine_accuracy.csv", "engine_derivatives.csv", "ca_windows.csv"):
        path = directory / name
        if name not in manifest["files"]:
            raise ValueError(f"Undeclared source data: {name}")
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                files[name] = list(csv.DictReader(stream))
        else:
            files[name] = json.loads(path.read_text())
    return files


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"expected an explicit CSV boolean, received {value!r}")
    return value == "True"


def numbers(rows, field):
    values = np.asarray([float(row[field]) for row in rows])
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"empty or nonfinite numeric column: {field}")
    return values


def scientific(value):
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def write_table(directory, name, rows, headers, display, *, columns=None):
    with (directory / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = columns or "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{tabular}{@{}" + columns + "@{}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule",
             *[" & ".join(map(str, row)) + r" \\" for row in display],
             r"\bottomrule", r"\end{tabular}", ""]
    (directory / f"{name}.tex").write_text("\n".join(lines))


def accuracy_table(source, output):
    identities = [(row["case_id"], row["config"]) for row in source]
    cohort = {row["case_id"] for row in source}
    expected = {(case, configuration) for case in cohort for configuration in CONFIGURATIONS}
    if len(cohort) != 200 or len(identities) != 1800 or set(identities) != expected:
        raise ValueError("accuracy table requires the same 200 structures in all nine configurations")
    if any(row["status"] != "completed" or not boolean(row["passes_all_gates"]) for row in source):
        raise ValueError("accuracy table cannot omit or reclassify failed comparisons")
    rows, display = [], []
    fields = ("position_error_max_deg", "intensity_nrmse", "profile_nrmse")
    for configuration, labels in CONFIGURATIONS.items():
        group = [row for row in source if row["config"] == configuration]
        maxima = [float(numbers(group, field).max()) for field in fields]
        rows.append(dict(configuration=configuration, label=labels[0], **dict(zip(fields, maxima, strict=True))))
        display.append([labels[1], *map(scientific, maxima)])
    write_table(output, "table_1_accuracy", rows,
                ["Configuration", r"Position ($^\circ$)", "Intensity NRMSE", "Profile NRMSE"], display)


def derivative_table(source, output):
    identities = [(row["case_id"], row["parameter_block"]) for row in source]
    cohort = {row["case_id"] for row in source}
    if len(cohort) != 10 or len(identities) != 100 or set(identities) != {(case, block) for case in cohort for block in PARAMETERS}:
        raise ValueError("derivative table requires ten structures and all ten parameter groups")
    if any(row["status"] not in {"pass", "pass_zero_sensitivity"} or row["dtype"] != "float64" for row in source):
        raise ValueError("derivative checks must all pass in double precision")
    rows, display = [], []
    for block, name in PARAMETERS.items():
        group = [row for row in source if row["parameter_block"] == block and not boolean(row["near_zero_sensitivity"])]
        selected = float(numbers(group, "best_relative_error").max())
        all_steps = float(numbers(group, "max_step_relative_error").max())
        rows.append({"parameter_group": name, "nonzero": len(group), "selected_step_max_relative_error": selected,
                     "all_steps_max_relative_error": all_steps})
        display.append([name, len(group), scientific(selected), scientific(all_steps)])
    write_table(output, "table_2_derivatives", rows,
                ["Parameter group", "Nonzero", "Selected step", "All steps"], display)




def window_table(source, output):
    selected = sorted((row for row in source if boolean(row["fitted"]) or int(row["window_index"]) == 4),
                      key=lambda row: float(row["center_two_theta_deg"]))
    if len(selected) != 11 or sum(boolean(row["fitted"]) for row in selected) != 10:
        raise ValueError("expected ten fitting windows and one additional evaluation window")
    rows, display = [], []
    for row in selected:
        signed = [tuple(map(int, item.split())) for item in re.findall(r"\(([^)]+)\)", row["families_ever_within_outer"])]
        if not signed or any(len(hkl) != 3 for hkl in signed):
            raise ValueError("invalid reflection membership")
        indices = sorted({tuple(abs(value) for value in hkl) for hkl in signed})
        center, use = float(row["center_two_theta_deg"]), "Fit" if boolean(row["fitted"]) else "Evaluate"
        rows.append({"center_two_theta_deg": center, "use": use,
                     "reflection_indices": ", ".join("(" + " ".join(map(str, hkl)) + ")" for hkl in indices)})
        labels = ", ".join("$(" + r"\,".join(map(str, hkl)) + ")$" for hkl in indices)
        display.append([f"{center:.3f}", use, labels])
    write_table(output, "table_4_reflection_windows", rows,
                ["Center (degree)", "Use", "Reflection indices"], display, columns=r"rlp{0.61\linewidth}")






def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="checksummed frozen plotting-data packet")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for five CSV/LaTeX table pairs")
    args = parser.parse_args()
    data = checked_inputs(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, builder in (("engine_accuracy.csv", accuracy_table), ("engine_derivatives.csv", derivative_table),
                          ("ca_windows.csv", window_table)):
        builder(data[name], args.output_dir)
    timing_tables(*(pd.read_csv(args.data_dir / name, float_precision="round_trip") for name in (
        "engine_complete_timings.csv", "engine_peak_counts.csv", "engine_repeated_timings.csv",
        "engine_reference_timings.csv")), args.output_dir)


if __name__ == "__main__":
    main()
