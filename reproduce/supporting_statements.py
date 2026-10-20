"""Check the paper's supporting statements from compact saved records, without refits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numbers(values):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), "Empty or nonfinite numerical evidence")
    return values


def flags(values):
    # In particular, bool('False') must never count as a successful comparison.
    values = pd.Series(values).astype(str).str.lower()
    require(values.isin(["true", "false"]).all(), "Missing or invalid Boolean evidence")
    return values.eq("true").to_numpy()


def load(root, name):
    return json.loads((root / name).read_text())


def table(root, name):
    return pd.read_csv(root / name, float_precision="round_trip")


def rounded(value, precision, expected):
    actual = f"{float(value):.{precision}f}"
    require(actual == expected, f"Manuscript displays {expected}, exported evidence gives {actual} ({value})")


def engine_cases(root):
    cases = load(root, "index.json")["engine_case_ids"]
    require(len(cases) == 200 and len(set(cases)) == 200, "Incomplete frozen structure roster")
    return set(cases)


def cu_benchmark(root):
    rows = load(root, "cu_benchmark.json")
    arms = {"gsasii_sequential_cpu", "diffpxrd_serial_cpu", "diffpxrd_batched_cpu", "diffpxrd_batched_gpu"}
    require(len(rows) == 24 and {(r["repeat"], r["arm"]) for r in rows} ==
            {(repeat, arm) for repeat in range(6) for arm in arms}, "Incomplete Cu benchmark repetitions")
    for row in rows:
        require(row["patterns"] == 17 and row["converged"] is True, "Cu benchmark failed convergence")
        expected = 17 if row["arm"] in {"gsasii_sequential_cpu", "diffpxrd_serial_cpu"} else 1
        require(len(row["termination"]) == expected, "Missing Cu termination records")
        for terminal in row["termination"]:
            if row["arm"] == "gsasii_sequential_cpu":
                converged = terminal["status"] in (1, 2, 3, 4) and terminal["finite_solution_and_covariance"] is True
            else:
                gradient, change = numbers([terminal["gradient_max"], terminal["relative_terminal_change"]])
                converged = gradient <= 1e-5 or (terminal["iterations"] < 200 and change <= 1e-10)
            require(converged and terminal["converged"] is True, "Saved Cu termination does not satisfy its criterion")
    return {"repetitions_per_arm": 6, "patterns_per_repetition": 17, "all_converged": True,
            "boundary": "Batched L-BFGS termination is batch-level, not a separate stationarity test for each pattern."}


def cu_starts(root):
    config = load(root, "cu_starts.json")
    rows = table(root, "cu_starts.csv")
    require(config["smoke"] is False and config["distances_a"] == [.02, .05, .1, .2, .3]
            and config["directions_per_distance"] == 3, "Wrong Cu displaced-start protocol")
    require(config["same_solution_a"] == .005 and config["same_rwp_pp"] == .01, "Changed Cu solution definition")
    plan = {row["start"]: row for row in config["starts"]}
    expected = {(start, program, pattern) for start in plan for program in ("gsasii", "diffpxrd") for pattern in range(17)}
    keys = list(rows[["start", "program", "pattern"]].itertuples(index=False, name=None))
    require(len(plan) == 16 and len(keys) == len(expected) and set(keys) == expected, "Incomplete Cu start roster")
    require(flags(rows.converged).all(), "Not every displaced-start fit reports convergence")
    for start, group in rows.groupby("start"):
        require((group.distance_a == plan[start]["distance_a"]).all(), "Cu start distance label mismatch")
    lattice = numbers(config["lattice_first_pattern_a"])
    require(lattice.shape == (3, 3), "Cu Cartesian distance needs a 3x3 cell matrix")
    displaced = rows.loc[rows.distance_a > 0]
    gsas = displaced.loc[displaced.program.eq("gsasii")].set_index(["start", "pattern"]).sort_index()
    diff = displaced.loc[displaced.program.eq("diffpxrd")].set_index(["start", "pattern"]).loc[gsas.index]
    distance = np.linalg.norm((numbers(gsas[["x", "y", "z"]]) - numbers(diff[["x", "y", "z"]])) @ lattice, axis=1)
    matching = int((distance <= config["same_solution_a"]).sum())
    require(len(distance) == 255 and matching == 251, "Cu cross-program position count is not 251/255")
    return {"matching_positions": matching, "pattern_pairs": len(distance), "all_reported_converged": True,
            "definition": "Cartesian distance between programs <= 0.005 angstrom, using the saved first-pattern cell; no symmetry remapping. Not the saved within-program same_solution flag."}


def merged_equivalence(root):
    config = load(root, "merged_equivalence.json")
    rows = table(root, "merged_equivalence.csv")
    require(config["smoke"] is False and config["dtype"] == "float64", "Not the full float64 equivalence probe")
    cases = engine_cases(root)
    keys = list(rows[["case_id", "device"]].itertuples(index=False, name=None))
    require(len(keys) == 400 and set(keys) ==
            {(case, device) for case in cases for device in ("cpu", "cuda")}, "Incomplete equivalence roster")
    groups = numbers(rows.spacegroup_number)
    require(((groups >= 1) & (groups <= 230) & (groups == np.floor(groups))).all() and
            rows.groupby("case_id").spacegroup_number.nunique().eq(1).all(), "Invalid or inconsistent space groups")
    require(config["points"] == {"reference": [1., 1.], "isotropic": [1.003, 1.003], "mixed": [1.003, .997]},
            "Equivalence strain states differ")
    maxima = {}
    for point in ("reference", "isotropic", "mixed"):
        # A mixed a/b-versus-c strain breaks cubic symmetry. Keep those results in
        # the export, but do not use them to test the symmetry-preserving claim.
        group = rows.loc[rows.spacegroup_number < 195] if point == "mixed" else rows
        for kind in ("profile_max_rel_diff", "loss_rel_diff", "grad_rel_diff"):
            values = numbers(group[f"{point}_{kind}"])
            require((values >= 0).all(), "Negative discrepancy")
            maxima[f"{point}_{kind}"] = float(values.max())
    maximum = max(maxima.values())
    require(maximum <= 6e-10, f"Symmetry-preserving merged/separate discrepancy {maximum} exceeds 6e-10")
    return {"maximum_relative_difference": maximum, "by_state_and_observable": maxima}


def device_parity(root):
    rows = table(root, "device_parity.csv")
    cases = engine_cases(root)
    require(set(rows.case_id) == cases and rows.dtype.eq("float64").all(), "Device parity needs the frozen 200 float64 structures")
    expected = {(case, kernel, mode, batch, observable) for case in cases
                for kernel, batches in (("fcj_merged", (1, 4, 16, 32)), ("fcj_exact", (1, 4)))
                for batch in batches for mode in ("forward", "wls_gradient")
                for observable in (("profile", "x_scales", "y_scales") if mode == "forward" else
                                   ("profile", "loss", "gradient", "x_scales", "y_scales"))}
    keys = list(rows[["case_id", "kernel", "mode", "batch_size", "observable"]].itertuples(index=False, name=None))
    require(len(keys) == len(expected) and set(keys) == expected, "Missing or duplicate device-parity observables")
    require(flags(rows.passed).all(), "A saved full-array CPU/A100 comparison failed")
    numeric = rows.loc[rows.observable.isin(["profile", "loss", "gradient"])]
    require(flags(numeric.shape_match).all() and flags(numeric.finite).all(), "Invalid CPU/A100 arrays")
    require((numeric.rtol == 5e-9).all() and (numeric.normalized_atol == 5e-11).all(), "Changed device tolerances")
    differences = numbers(numeric[["max_absolute_difference", "max_peak_normalized_difference", "relative_rms_difference"]])
    require((differences >= 0).all(), "Negative CPU/A100 discrepancy")
    reference = numbers(numeric.reference_at_max_difference)
    candidate = numbers(numeric.candidate_at_max_difference)
    require(np.allclose(np.abs(candidate - reference), differences[:, 0], rtol=1e-12, atol=0),
            "Saved maximum-difference witnesses disagree")
    require(rows.loc[rows.observable.isin(["x_scales", "y_scales"]), "comparison"].eq("exact_input_equality").all(),
            "Lattice inputs were not compared exactly")
    pairs = rows.groupby(["case_id", "kernel", "mode", "batch_size"]).ngroups
    require(pairs == 2400, "CPU/A100 setting-pair count is not 2400")
    return {"passing_setting_pairs": pairs, "relative_tolerance": 5e-9, "normalized_absolute_tolerance": 5e-11,
            "boundary": "Count recomputed from all saved full-array per-observable decisions; maximum-difference witnesses checked. Full arrays are not reconstructed from these summaries."}


def nca_quality(root):
    packet = load(root, "nca_quality.json")
    require(packet["config"] == {"smoke": False, "warmups": 1, "repeats": 3}, "Wrong NCA timing protocol")
    fits = packet["fits"]
    require(len(fits) == 8 and {(r["device"], r["repeat"]) for r in fits} ==
            {(device, repeat) for device in ("cpu", "cuda:0") for repeat in range(-1, 3)}, "Incomplete NCA quality roster")
    checks = []
    for fit in fits:
        serial, batched = fit["serial"], fit["batched"]
        arrays = [numbers(method[key]) for method in (serial, batched) for key in ("c_values", "profile_losses")]
        require(all(a.shape == (239,) for a in arrays), "NCA timing must retain all 239 patterns")
        require((arrays[1] >= 0).all() and (arrays[3] >= 0).all() and arrays[1].mean() > 0, "Invalid NCA losses")
        ratio = float(arrays[3].mean() / arrays[1].mean())
        rms = float(np.sqrt(np.mean((arrays[2] - arrays[0]) ** 2)))
        saved = fit["saved_quality"]
        require(np.allclose([ratio, rms], [saved["batched_to_serial_loss_ratio"], saved["batched_to_serial_c_rmse_A"]],
                            rtol=1e-12, atol=0), "NCA saved quality disagrees with its per-pattern arrays")
        require(ratio <= 1.05 and rms <= .02, f"NCA quality fails for {fit['device']} repeat {fit['repeat']}")
        checks.append({"device": fit["device"], "repeat": fit["repeat"], "loss_ratio": ratio, "lattice_rms_angstrom": rms})
    return {"comparisons": checks}


def ca_starts(root):
    packet = load(root, "ca_starts.json")
    frame = table(root, "ca_starts.csv")
    names = {"external_neutron", "near_acaa", "one_and_half_external", "reversed_external", "mixed_0", "mixed_1", "mixed_2"}
    require(packet["config"]["smoke"] is False and len(packet["starts"]) == 7 and
            {r["start"] for r in packet["starts"]} == names, "Incomplete Ca start roster")
    require(set(frame.start) == names | {"displayed"}, "Ca trajectories differ from objective roster")
    reference = frame.loc[frame.start.eq("displayed")].set_index(["cycle", "scan"])
    require(len(reference) == 42 and reference.index.is_unique, "Displayed Ca trajectory must have 42 distinct nodes")
    baseline = float(numbers([packet["displayed_objective"]])[0])
    require(baseline > 0, "Invalid displayed Ca objective")
    results = []
    fixed = {"external_neutron": 1., "near_acaa": .1, "one_and_half_external": 1.5, "reversed_external": -1.}
    for record in packet["starts"]:
        factors = numbers(record["factors"])
        require(factors.shape == (3,), "Ca starts need three displacement factors")
        if record["start"] in fixed:
            require((factors == fixed[record["start"]]).all(), "Wrong Ca start scaling")
        else:
            require(((factors >= -1.5) & (factors <= 1.5)).all(), "Ca random scale outside declared interval")
        part = frame.loc[frame.start.eq(record["start"])].set_index(["cycle", "scan"])
        require(len(part) == 42 and part.index.is_unique and set(part.index) == set(reference.index), "Ca node mismatch")
        distance = numbers(part.loc[reference.index, "structural_rms_from_acaa_angstrom"]) - numbers(reference.structural_rms_from_acaa_angstrom)
        rms = float(np.sqrt(np.mean(distance ** 2)))
        objective = float(numbers([record["objective"]])[0])
        if record["start"] != "external_neutron":
            results.append({"start": record["start"], "rms_angstrom": rms,
                            "objective_change_percent": 100 * (objective / baseline - 1), "converged": record["converged"]})
    close = [r for r in results if float(f"{r['rms_angstrom']:.4f}") <= .0034]
    require(len(close) == 5, "Not five of six Ca starts within the displayed 0.0034 angstrom RMS")
    rounded(min(r["objective_change_percent"] for r in close), 1, "-0.6")
    rounded(max(r["objective_change_percent"] for r in close), 1, "1.3")
    other = next(r for r in results if r not in close)
    rounded(other["objective_change_percent"], 0, "22")
    return {"additional_starts": results,
            "definition": "RMS difference of scalar total displacement magnitudes at the 42 fitted nodes, relative to the displayed trajectory; not RMS atomic-coordinate distance or evidence of optimizer stationarity."}


def ca_profiles(root):
    packet = load(root, "ca_profiles.json")
    require(packet["cycle"] == "heating" and packet["window"] == 19 and len(packet["profiles"]) == 2,
            "Wrong Ca profile examples")
    results = []
    for row, temperature, expected in zip(packet["profiles"], ("314", "1069"), ("0.024", "0.051"), strict=True):
        x, observed, predicted = (numbers(row[key]) for key in ("two_theta_deg", "observed", "predicted"))
        require(x.ndim == 1 and x.shape == observed.shape == predicted.shape and observed.max() > 0 and
                (np.diff(x) > 0).all(), "Invalid saved Ca window profiles")
        nrmse = float(np.sqrt(np.mean((observed - predicted) ** 2)) / observed.max())
        rounded(row["temperature_k"], 0, temperature)
        rounded(nrmse, 3, expected)
        results.append({"temperature_k": row["temperature_k"], "profile_nrmse": nrmse})
    return {"profiles": results}


def ca_ratio_scan(root):
    rows = table(root, "ca_ratio_scan.csv")
    rows = rows.loc[rows.temperature_k.map(lambda t: f"{t:.0f}" == "314")]
    require(len(rows) == 49 and rows.amplitude_fraction.is_unique, "Missing 314 K displacement scan")
    require(np.allclose(np.sort(numbers(rows.amplitude_fraction)), np.linspace(0, 1.2, 49), rtol=0, atol=1e-14),
            "Wrong displacement fractions")
    numbers(rows.ratio)
    result = []
    for fraction, expected in ((0., "0.193"), (.25, "0.160"), (1., "0.455")):
        value = rows.loc[np.isclose(rows.amplitude_fraction, fraction, rtol=0, atol=1e-14), "ratio"].item()
        rounded(value, 3, expected)
        result.append({"amplitude_fraction": fraction, "ratio": value})
    return {"ratios_at_314_k": result}


def lisulfur_lines(root):
    config = load(root, "lisulfur_lines.json")
    peaks = table(root, "lisulfur_lines.csv")
    frames = table(root, "lisulfur_frames.csv")
    require(len(frames) == 131 and frames.frame.tolist() == list(range(131)), "Missing measured Li-S frames")
    require(config["window_deg"] == [30.5, 32.5] and config["evaluation_cycles"] == [5, 6, 7, 8], "Wrong Li-S window/split")
    require(config["threshold_counts_above_running_median"] == 400 and config["running_median_width_deg"] == .6,
            "Changed sharp-peak detector")
    require(set(frames.cycle) == set(range(1, 9)), "Missing Li-S cycles")
    joined = peaks.merge(frames[["frame", "cycle"]], on="frame", validate="many_to_one", suffixes=("", "_roster"))
    require(len(joined) == len(peaks) and joined.cycle.eq(joined.cycle_roster).all(), "Peak/frame cycle mismatch")
    require((numbers(peaks.excess_counts) > 400).all(), "Saved peak below the declared threshold")
    inside = peaks.two_theta_deg.between(*config["window_deg"])
    sulfur = peaks.loc[inside & peaks.nearest_line.eq("S8 044")]
    reflections = pd.read_csv(root / "lisulfur_reflections.csv", dtype={"hkl": str}, float_precision="round_trip")
    angle = reflections.loc[reflections.hkl.eq("044"), "two_theta_deg"].item()
    require(30.5 <= angle <= 32.5 and np.allclose(numbers(sulfur.two_theta_deg) - numbers(sulfur.offset_deg),
                                                angle, rtol=0, atol=1e-12), "Inconsistent sulfur 044 assignment")
    require(set(sulfur.frame) == set(range(6)) and sulfur.cycle.isin([1, 2, 3, 4]).all(), "S8 044 not confined to the first six training patterns")
    evaluation_count = int((inside & peaks.cycle.isin(config["evaluation_cycles"])).sum())
    require(evaluation_count == 0, "A detected sharp line enters the 200 window in an evaluation cycle")
    return {"sulfur_044_frames_zero_based": sorted(int(i) for i in sulfur.frame.unique()),
            "evaluation_window_sharp_peaks": evaluation_count,
            "boundary": "Absence means no sharp peak detected above 400 counts over a 0.6-degree running median, not proof of zero sulfur."}


def validation_settings(root):
    values = load(root, "validation_settings.json")
    rounded(values["standardization_symprec_angstrom"], 2, "0.01")
    rounded(values["standardization_angle_tolerance_deg"], 0, "5")
    rounded(values["11_id_wavelength_angstrom"], 6, "0.413263")
    return values


CHECKS = {"cu_benchmark": (1, cu_benchmark), "cu_starts": (1, cu_starts),
          "merged_equivalence": (2, merged_equivalence), "device_parity": (3, device_parity),
          "nca_quality": (4, nca_quality), "ca_starts": (5, ca_starts), "ca_profiles": (6, ca_profiles),
          "ca_ratio_scan": (7, ca_ratio_scan), "lisulfur_lines": (8, lisulfur_lines),
          "validation_settings": (9, validation_settings)}


def check(data_dir, output):
    root = data_dir / "supporting"
    index = load(root, "index.json")
    results = []
    for name, (statement, function) in CHECKS.items():
        row = {"statement": statement, "record": name}
        if name in index["missing"]:
            row.update(passed=False, status="missing_evidence", reason=index["missing"][name]["reason"])
        else:
            try:
                row.update(values=function(root), passed=True, status="checked")
            except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
                row.update(passed=False, status="failed", reason=str(error))
        results.append(row)
    report = {"passed": all(r["passed"] for r in results), "checks": results,
              "new_fits": 0, "new_timing_measurements": 0, "new_forward_calculations": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.data_dir, args.output)
    raise SystemExit(0 if result["passed"] else f"Supporting-statement checks failed: {args.output}")
