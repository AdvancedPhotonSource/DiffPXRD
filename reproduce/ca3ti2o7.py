"""Refine Ca3Ti2O7 collective displacements from prepared diffraction profiles.

Run ``python reproduce/ca3ti2o7.py --help``. Input preparation and the scientific
meaning of each comparison are documented in docs/ca3ti2o7.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ca3ti2o7_model import Problem, radial_bound
from ca3ti2o7_structure import Descriptors, reference_doublet
from pymatgen.core import Lattice

CASES = {
    "reference": {"width": "separate", "ratio_weight": 1., "omit_doublet": False},
    "profile_only": {"width": "separate", "ratio_weight": 0., "omit_doublet": False},
    "omit_doublet": {"width": "separate", "ratio_weight": 0., "omit_doublet": True},
    "common_width": {"width": "common", "ratio_weight": 1., "omit_doublet": False},
    "common_profile_only": {"width": "common", "ratio_weight": 0., "omit_doublet": False},
    "common_omit_doublet": {"width": "common", "ratio_weight": 0., "omit_doublet": True},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(problem, case, kind, initial, extras, output, *, smoke=False, structural_check=False):
    output.mkdir(parents=True, exist_ok=False)
    options = problem.config["optimization"]
    raw = torch.nn.Parameter(initial.detach().clone().contiguous())
    variables = {key: value.detach().clone().contiguous().requires_grad_(not structural_check)
                 for key, value in extras.items()}
    parameters = [raw, *([] if structural_check else variables.values())]
    evaluate = problem.objective(case)
    block = 2 if smoke else 25 if structural_check else options["block_iterations"]
    budget = block if smoke else 100 if structural_check else options["max_iterations"]
    optimizer = torch.optim.LBFGS(
        parameters, lr=options["learning_rate"], max_iter=block, max_eval=2 * block,
        tolerance_grad=options["gradient_tolerance"], tolerance_change=1e-12,
        history_size=options["history_size"], line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        loss, _ = evaluate(problem.decode(raw, kind), variables)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite objective")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("nonfinite gradient")
        calls += 1
        return loss

    started = time.monotonic()
    previous, stable_blocks, history = None, 0, []
    for block_index in range(int(np.ceil(budget / block))):
        optimizer.step(closure)
        closure()
        with torch.no_grad():
            amplitudes = problem.decode(raw, kind).detach().clone()
            loss, diagnostics = evaluate(amplitudes, variables)
            terms = {key: value.detach().clone() for key, value in problem.terms(variables).items()}
        gradient = max(float(p.grad.abs().max()) for p in parameters if p.grad is not None)
        row = {"block": block_index + 1, "iterations": optimizer.state[raw].get("n_iter", 0),
                   "evaluations": calls, "objective": float(loss), "profile": float(diagnostics["profile"]),
                   "gradient_max_abs": gradient, "seconds": time.monotonic() - started}
        if previous is not None:
            relative = abs(previous["objective"] - float(loss)) / max(1., abs(previous["objective"]))
            movement = float(torch.sqrt(torch.mean((previous["amplitudes"] - amplitudes)**2)))
            nuisance = max(float((previous["terms"][key] - value).abs().max()) for key, value in terms.items())
            stable = (relative < options["relative_objective_tolerance"]
                      and movement < options["amplitude_change_tolerance_angstrom"]
                      and nuisance < options["nuisance_change_tolerance"])
            stable_blocks = stable_blocks + 1 if stable else 0
            row.update(relative_objective_change=relative, amplitude_rms_change_angstrom=movement,
                       maximum_bounded_nuisance_change=nuisance)
        reason = "continue"
        if gradient <= options["gradient_tolerance"]:
            reason = "gradient_converged"
        elif (structural_check and history and row["iterations"] <= history[-1]["iterations"] + 1
              or not structural_check and stable_blocks >= options["stable_blocks_required"]):
            reason = "stalled_nonstationary"
        elif (block_index + 1) * block >= budget:
            reason = "smoke_budget" if smoke else "iteration_budget"
        row.update(termination=reason, gradient_converged=reason == "gradient_converged" and not smoke)
        history.append(row)
        write_json(output / "history.json", history)
        np.savez_compressed(output / "parameters.npz", raw=raw.detach().cpu().numpy(),
                            amplitudes=amplitudes.cpu().numpy(),
                            **{key: value.detach().cpu().numpy() for key, value in variables.items()})
        print(json.dumps({"fit": output.name, **row}), flush=True)
        previous = {"objective": float(loss), "amplitudes": amplitudes, "terms": terms}
        if reason != "continue":
            break
    return {"raw": raw.detach(), "extras": {key: value.detach() for key, value in variables.items()},
                "amplitudes": amplitudes, "diagnostics": {key: value.detach() for key, value in diagnostics.items()},
                "objective": float(loss), "history": history, "kind": kind}


def summarize(problem, result, output, case):
    d, diag = problem.data, result["diagnostics"]
    terms = problem.terms(result["extras"])
    lengths = d["lattice"] * np.exp(terms["lattice_log_scale"].cpu().numpy())
    amplitudes = result["amplitudes"].cpu().numpy()
    descriptors = Descriptors(problem)
    rows = []
    for i in range(42):
        rows.append(dict(cycle=str(d["cycle"][i]), scan=int(d["scan"][i]), temperature_k=float(d["temperature"][i]),
                         frame_index=int(d["source_index"][i]),
                         **{f"component_{j + 1}": float(amplitudes[i, j]) for j in range(3)},
                         **descriptors(amplitudes[i], lengths[i]), ratio=float(diag["ratio"][i]),
                         measured_ratio=float(d["ratio"][i]), profile_deviance=float(diag["deviance_per_frame"][i])))
    write_csv(output / "trajectory.csv", rows)
    arrays = {"calculated": diag["calculated"].cpu().numpy(), "fitted": diag["fitted_profile"].cpu().numpy(),
              "residual": diag["residual"].cpu().numpy(), "scale": diag["scale"].cpu().numpy(),
              **{key: value.cpu().numpy() for key, value in terms.items()}}
    scores = []
    for index in dict.fromkeys([*problem.config["fit_windows"], problem.config["evaluation_window"]]):
        with torch.no_grad():
            prediction, deviance, mask = problem.predict_window(diag["calculated"], diag["scale"], index)
        unfitted = index == problem.config["evaluation_window"] or (case["omit_doublet"] and index == problem.config["doublet_window"])
        arrays[f"window_{index}_prediction"] = prediction.cpu().numpy()
        for i in range(42):
            keep = mask[i]
            observed, var, predicted = (value[i, keep].cpu().numpy() for value in (problem.observed, problem.variance, prediction))
            rwp = 100 * np.sqrt(np.sum((predicted - observed)**2 / var) / np.sum(observed**2 / var))
            record = {"cycle": rows[i]["cycle"], "scan": rows[i]["scan"], "temperature_k": rows[i]["temperature_k"],
                          "window": index, "unfitted": unfitted, "mean_standardized_deviance": float(deviance[i]), "rwp_percent": float(rwp)}
            if index == problem.config["doublet_window"] and "doublet_parameters" in d:
                reference = reference_doublet(d["doublet_parameters"][i], d["two_theta"][keep.cpu().numpy()], observed, var)
                record["measured_doublet_rwp_percent"] = float(100 * np.sqrt(np.sum((reference - observed)**2 / var) / np.sum(observed**2 / var)))
            else:
                record["measured_doublet_rwp_percent"] = ""
            scores.append(record)
    write_csv(output / "window_scores.csv", scores)
    np.savez_compressed(output / "profiles.npz", **arrays)
    summary = {"objective": result["objective"], "profile": float(diag["profile"]),
               "ratio_rms_standardized_residual": float(torch.sqrt(diag["ratio_terms"].mean())),
               "terminal": result["history"][-1], "kind": result["kind"]}
    write_json(output / "summary.json", summary)
    return summary


def predict_temperatures(problem, result, output):
    """Forward-only structures at the remaining measured temperatures."""
    d, rows, scan_rows = problem.data, [], []
    terms = problem.terms(result["extras"])
    fitted_keys = set(zip(d["cycle"], d["scan"], strict=True))
    for b, cycle in enumerate(("heating", "cooling")):
        selected = (d["series_cycle"] == cycle) & d["series_eligible"]
        temperatures = d["series_temperature"][selected]
        lengths = d["series_lattice"][selected]
        cells = np.stack([Lattice.orthorhombic(*cell).matrix for cell in lengths])
        base = problem.batch_for(lengths, d["series_cycle"][selected], temperatures)
        spline = problem.spline(temperatures)
        amplitudes = radial_bound(spline.amplitudes(result["raw"][b]), problem.config["radius"])
        part = problem.parts[b]
        fitted_scales = np.exp(terms["lattice_log_scale"][part].cpu().numpy())
        scales = np.column_stack([np.interp(temperatures, d["temperature"][part], fitted_scales[:, j]) for j in range(3)])
        log_scale = problem.tensor(np.log(scales))
        with torch.no_grad():
            _, _, ratio = problem.intensities(problem.structure_batch(amplitudes, log_scale, base=base, cells=cells))
            _, _, acaa = problem.intensities(problem.structure_batch(torch.zeros_like(amplitudes), log_scale, base=base, cells=cells))
        for i, source in enumerate(np.flatnonzero(selected)):
            rows.append({"cycle": cycle, "scan": int(d["series_scan"][source]), "temperature_k": float(temperatures[i]),
                             "fitted_node": (cycle, d["series_scan"][source]) in fitted_keys,
                             "measured_ratio": float(d["series_ratio"][source]), "ratio_sigma": float(d["series_ratio_sigma"][source]),
                             "calculated_ratio": float(ratio[i]), "acaa_ratio": float(acaa[i])})
        if cycle == "heating":
            endpoint = np.asarray([np.argmin(temperatures), np.argmax(temperatures)])
            endpoint_base = problem.batch_for(lengths[endpoint], np.asarray([cycle, cycle]), temperatures[endpoint])
            for factor in np.linspace(0, 1.2, 49):
                with torch.no_grad():
                    _, _, value = problem.intensities(problem.structure_batch(
                        amplitudes[endpoint] * factor, log_scale[endpoint], base=endpoint_base, cells=cells[endpoint]
                    ))
                for j, index in enumerate(endpoint):
                    scan_rows.append({"temperature_k": float(temperatures[index]), "displacement_factor": float(factor), "ratio": float(value[j])})
    if len(rows) != 343 or sum(not row["fitted_node"] for row in rows) != 301:
        raise ValueError("expected 42 fitted and 301 unfitted eligible temperatures")
    write_csv(output / "temperature_predictions.csv", rows)
    write_csv(output / "displacement_ratio_scan.csv", scan_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory with metadata.json and measurements.npz")
    parser.add_argument("--neutron-tables", type=Path, help="reader-prepared NPZ; see docs/ca3ti2o7_neutron_tables.md")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", default="cpu", help="cpu or explicit CUDA device; never falls back")
    parser.add_argument("--case", choices=[*CASES, "all"], default="reference")
    parser.add_argument("--smoke", action="store_true", help="two iterations, full input set; not a scientific result")
    parser.add_argument("--structural-check", action="store_true", help="also continue each external-start joint fit for 100 iterations with nonlinear nuisances fixed")
    args = parser.parse_args()
    problem = Problem(args.data_dir, args.device, args.neutron_tables)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "input_metadata.json", problem.config)
    write_json(args.output_dir / "run.json", {"device": str(problem.device), "dtype": "float64", "smoke": args.smoke,
                                             "case": args.case, "torch_version": torch.__version__,
                                             "basis_fraction": problem.basis_fraction[:3].tolist()})
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        case = CASES[name]
        destination = args.output_dir / name
        destination.mkdir()
        summaries, results = {}, {}
        kinds = ["joint"] if args.smoke or case["width"] == "common" or case["omit_doublet"] else ["joint", "framewise"]
        for kind in kinds:
            for start, scale in [("neutron", 1.), ("half_neutron", .5)][:1 if args.smoke else 2]:
                label = kind + "_" + start
                result = fit(problem, case, kind, problem.initial(kind, scale), problem.extras(case["width"]),
                             destination / label, smoke=args.smoke)
                results[label] = result
                summaries[label] = summarize(problem, result, destination / label, case)
        best = min((key for key in results if key.startswith("joint_")), key=lambda key: results[key]["objective"])
        for start, extras in [("default", problem.extras(case["width"])), ("joint_nuisance", results[best]["extras"])][:1 if args.smoke else 2]:
            label = "acaa_" + start
            result = fit(problem, case, "acaa", problem.initial("acaa", 0), extras, destination / label, smoke=args.smoke)
            summaries[label] = summarize(problem, result, destination / label, case)
        if args.structural_check:
            source = results["joint_neutron"]
            result = fit(problem, case, "joint", source["raw"], source["extras"], destination / "structural_check",
                         smoke=args.smoke, structural_check=True)
            summaries["structural_check"] = summarize(problem, result, destination / "structural_check", case)
        if name == "reference":
            predict_temperatures(problem, results["joint_neutron"], destination)
        write_json(destination / "summary.json", {"case": case, "fits": summaries, "displayed_start": "joint_neutron"})


if __name__ == "__main__":
    main()
