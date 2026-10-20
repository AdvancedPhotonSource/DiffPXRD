"""Reproduce the NCA 003 lattice-parameter analysis from the paper.

Run ``python reproduce/nca.py --help``; input preparation is described in
``docs/nca.md``. The numerical settings are fixed to the paper recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.interpolate import BSpline
import torch

from diffpxrd import ModularCWForwardModel, solved_profile_wls_residuals_torch


A, C0, OXYGEN_Z = 2.85756, 14.2, 0.25383
WAVELENGTH, C_LOG_BOUND = 1.5406, 0.05
INSTRUMENT = {"U": 0.004, "V": 0., "W": 0.004, "X": 0., "Y": 0.015,
              "Z": 0., "SH/L": 0.02, "Zero": 0.01813,
              "Polariz.": 0.5, "Azimuth": 0.}
SIGMA_BOUNDS, GAMMA_BOUNDS = (0.001, 0.3), (0.001, 0.5)
WIDTH_START, RETRY_START = (0.06, 0.10), (0.12, 0.08)
MAX_STEPS, BASIS_COUNT, RANDOM_SEED = 200, 12, 20260724
INPUT_KEYS = ("scan_numbers", "time_sec", "capacity", "state", "two_theta_deg", "raw_counts")
ARRAY_KEYS = ("c_values", "sigma_deg", "gamma_deg", "coefficients", "profile_losses",
              "scale", "background_coeffs", "design_rank", "design_condition",
              "fitted_profile", "residual", "calculated", "peak_initial_c",
              "c_near_lower_bound", "c_near_upper_bound")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_numpy(value):
    return value.detach().cpu().numpy().copy()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def normalize(counts):
    rows = []
    for row in np.asarray(counts, dtype=np.float64):
        low, high = np.quantile(row, [0.02, 0.995])
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("profile normalization requires finite nonconstant counts")
        rows.append(np.clip((row - low) / (high - low), 0., None))
    return np.stack(rows)


def validate_data(data):
    counts, x = data["raw_counts"], data["two_theta_deg"]
    if (counts.shape != (239, 114) or x.shape != (114,)
            or any(data[key].shape != (239,) for key in INPUT_KEYS[:4])
            or not np.array_equal(data["scan_numbers"], np.arange(1, 240))
            or np.count_nonzero(data["state"] == "C") != 155
            or np.count_nonzero(data["state"] == "D") != 84):
        raise ValueError("expected all 239 scans, 155 charge / 84 discharge, and 114 bins")
    if (not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.array_equal(counts, np.rint(counts))
            or not np.isfinite(data["capacity"]).all()
            or not np.isfinite(data["time_sec"]).all()
            or not np.isfinite(x).all() or not np.all(np.diff(x) > 0)
            or x[0] < 18.1 or x[-1] > 19.8):
        raise ValueError("invalid measured counts, electrochemistry, or 003 domain")


def prepare(data_dir, output_dir, input_format):
    """Select the published angular bins; do not use published fitted structures."""
    if input_format == "archive":
        source = data_dir / "science_trajectories.npz"
        with np.load(source, allow_pickle=False) as packet:
            data = {key: packet[key].copy() for key in INPUT_KEYS}
        sources = {source.name: sha256(source)}
    else:
        import pandas as pd

        def scan_number(path):
            match = re.search(r"\.(\d{3})\.xy$", path.name)
            if match is None:
                raise ValueError(f"cannot read scan number: {path.name}")
            return int(match.group(1))

        files = sorted((data_dir / "XRD/Operando files").glob("*_back2.*.xy"), key=scan_number)
        scans = np.asarray([scan_number(path) for path in files], dtype=np.int64)
        if not np.array_equal(scans, np.arange(1, 240)):
            raise ValueError("expected exactly the 239 public *_back2.*.xy patterns")
        electrochemistry = data_dir / "XRD/icixrd.csv"
        rows = pd.read_csv(electrochemistry)
        if len(rows) != 466:
            raise ValueError("expected 466 rows in the published electrochemistry table")
        rows = rows.iloc[:239].reset_index(drop=True)
        if not np.array_equal(np.asarray(rows["rest"], dtype=np.int64), scans):
            raise ValueError("electrochemistry and diffraction scan indices differ")
        profiles, domain = [], None
        for path in files:
            xy = np.loadtxt(path)
            selected = xy[(xy[:, 0] >= 18.1) & (xy[:, 0] <= 19.8)]
            if domain is None:
                domain = selected[:, 0]
            elif selected[:, 0].shape != domain.shape or not np.allclose(selected[:, 0], domain):
                raise ValueError(f"angular grid mismatch: {path.name}")
            profiles.append(selected[:, 1])
        data = {"scan_numbers": scans, "two_theta_deg": domain,
                "raw_counts": np.stack(profiles),
                "time_sec": np.asarray(rows["t"], dtype=np.float64),
                "capacity": np.asarray(rows["Q"], dtype=np.float64),
                "state": np.asarray(rows["state"], dtype=str)}
        sources = {str(path.relative_to(data_dir)): sha256(path) for path in [*files, electrochemistry]}
    validate_data(data)
    np.savez_compressed(output_dir / "nca.npz", **data)
    write_json(output_dir / "input.json", {"source_doi": "10.5281/zenodo.7049855",
        "input_format": input_format, "source_sha256": sources,
        "nca_sha256": sha256(output_dir / "nca.npz"),
        "selection": "Scans 1 through 239; inclusive 18.1 to 19.8 degree window; no rebinning."})


def spline_design(capacity, state):
    design = np.zeros((len(capacity), 2 * BASIS_COUNT), dtype=np.float64)
    knots = np.r_[np.zeros(4), np.linspace(0., 1., BASIS_COUNT - 2)[1:-1], np.ones(4)]
    for block, branch in enumerate(("C", "D")):
        mask = state == branch
        values = np.asarray(capacity[mask], dtype=np.float64)
        lower, upper = values.min(), values.max()
        if upper <= lower:
            raise ValueError("each branch requires a nonconstant capacity range")
        scaled = np.clip((values - lower) / (upper - lower), 0., 1.)
        design[mask, block * BASIS_COUNT:(block + 1) * BASIS_COUNT] = BSpline.design_matrix(
            scaled, knots, 3, extrapolate=False).toarray()
    return design


def setup(data_dir, device):
    with np.load(data_dir / "nca.npz", allow_pickle=False) as packet:
        data = {key: packet[key].copy() for key in INPUT_KEYS}
    validate_data(data)
    structure = Structure.from_spacegroup("R-3mH", Lattice.hexagonal(A, C0),
        [{"Li": 1.}, {"Ni": 0.90, "Co": 0.05, "Al": 0.05}, {"O": 1.}],
        [[0., 0., 0.5], [0., 0., 0.], [0., 0., OXYGEN_Z]])
    q = [4 * math.pi * math.sin(math.radians(float(angle) / 2)) / WAVELENGTH
         for angle in data["two_theta_deg"][[0, -1]]]
    model = ModularCWForwardModel(q_min=float(q[0]), q_max=float(q[1]),
        wavelength=WAVELENGTH, device=device)
    batch = model.structure_to_batch(structure, dtype=torch.float32)
    hkls = model.compile_reflections(batch).hkl_pts.detach().clone()
    if hkls.numel() == 0:
        raise ValueError("no reflections compiled for the 003 window")
    # Preserve the paper's rounded setup geometry/domain before float64 fitting.
    domain = torch.as_tensor(data["two_theta_deg"], dtype=torch.float32, device=device).double()
    return {"data": data, "observed": normalize(data["raw_counts"]), "model": model,
            "batch": batch, "hkls": hkls, "domain": domain,
            "design": spline_design(data["capacity"], data["state"])}


def repeated_batch(batch, c):
    count = len(c)
    a = torch.full_like(c, A)
    result = {"lengths": torch.stack((a, a, c), dim=1),
        "angles": batch["angles"].to(c.device, c.dtype).repeat(count, 1),
        "num_atoms": batch["num_atoms"].to(c.device).repeat(count),
        "frac_coords": batch["frac_coords"].to(c.device, c.dtype).repeat(count, 1),
        "atom_types": batch["atom_types"].to(c.device).repeat(count)}
    for key in ("site_uiso", "site_uaniso", "site_species_zs", "site_species_occs",
                "site_species_ff_a", "site_species_ff_b", "site_species_ff_c",
                "site_species_fp", "site_species_fpp"):
        if key in batch:
            tensor = batch[key].to(c.device)
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(c.dtype)
            result[key] = tensor.repeat((count,) + (1,) * (tensor.ndim - 1))
    return result


def objective(c, sigma, gamma, observed, context, *, diagnostics=False):
    model, domain, hkls = (context[key] for key in ("model", "domain", "hkls"))
    instrument = {**INSTRUMENT, "U": 0., "V": 0., "W": (100 * sigma) ** 2,
                  "X": 0., "Y": 0., "Z": 100 * gamma}
    batch = repeated_batch(context["batch"], c)
    selected = hkls[0] if hkls.ndim == 3 and hkls.shape[0] == 1 else hkls
    reflections = model.compute_reflection_layer_from_hkls(batch, selected)
    scattering = model.compute_scattering_layer(batch, reflections)
    corrections = model.compute_correction_layer(reflections, scattering, instrument=instrument)
    profile = model.compute_fcj_profile_layer(corrections, instrument=instrument, domain=domain,
        num_steps=len(domain), backend="bucketed", return_diagnostics=diagnostics)
    histogram = torch.clamp(profile.histogram, min=0.)
    calculated = histogram / torch.clamp(histogram.amax(dim=-1, keepdim=True), min=1.e-12)
    residuals, nuisance = solved_profile_wls_residuals_torch(observed, calculated,
        weights=torch.ones_like(observed), x=domain, fit_scale=True, fit_background=True,
        background_degree=1, constrain_scale_nonnegative=False)
    losses = residuals.square().mean(dim=-1)
    arrays = {"calculated": calculated, "profile_losses": losses,
              **{key: nuisance[key] for key in ("fitted_profile", "residual", "scale",
                 "background_coeffs", "design_rank", "design_condition")}}
    return losses.mean(), arrays


def peak_initialization(observed, context):
    domain = as_numpy(context["domain"])
    values = []
    for row in observed:
        background = np.linspace(np.mean(row[:10]), np.mean(row[-10:]), len(row))
        signal = np.maximum(row - background, 0.)
        peak = int(np.argmax(signal))
        if signal[peak] <= 0:
            raise ValueError("no positive observed peak for initialization")
        corrected = float(domain[peak]) - INSTRUMENT["Zero"]
        values.append(3 * WAVELENGTH / (2 * np.sin(np.deg2rad(corrected / 2))))
    return np.asarray(values)


def encode_width(value, bounds):
    low, high = np.log(bounds)
    fraction = (np.log(value) - low) / (high - low)
    return float(np.log(fraction / (1 - fraction)))


def decode_width(raw, bounds):
    low, high = np.log(bounds)
    return torch.exp(low + (high - low) * torch.sigmoid(raw))


def initial_coefficients(design, peak):
    log_delta = np.log(np.clip(np.asarray(peak, dtype=np.float64), 1.e-12, None) / C0)
    target = C_LOG_BOUND * np.arctanh(np.clip(log_delta / C_LOG_BOUND, -0.999, 0.999))
    # Preserve the paper's float32 initializer before float64 optimization.
    return np.linalg.lstsq(np.asarray(design, dtype=np.float64), target, rcond=None)[0].astype(np.float32)


def optimize(design, observed, peak, context, *, widths=WIDTH_START, fixed_c=None):
    device = context["model"].device
    y = torch.as_tensor(observed, dtype=torch.float64, device=device)
    coefficients = None
    if fixed_c is None:
        if design.shape[0] != len(observed) or np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("trajectory design must have full column rank")
        initial = initial_coefficients(design, peak).astype(np.float64)
        coefficients = torch.nn.Parameter(torch.as_tensor(initial / C_LOG_BOUND, device=device))
        design = torch.as_tensor(design, dtype=torch.float64, device=device)
    else:
        fixed_c = torch.as_tensor(fixed_c, dtype=torch.float64, device=device)
    sigma_raw = torch.nn.Parameter(torch.full((len(y),), encode_width(widths[0], SIGMA_BOUNDS),
                                            dtype=torch.float64, device=device))
    gamma_raw = torch.nn.Parameter(torch.full((len(y),), encode_width(widths[1], GAMMA_BOUNDS),
                                            dtype=torch.float64, device=device))
    parameters = ([] if coefficients is None else [coefficients]) + [sigma_raw, gamma_raw]

    def decoded():
        c = fixed_c if coefficients is None else C0 * torch.exp(C_LOG_BOUND * torch.tanh(design @ coefficients))
        return c, decode_width(sigma_raw, SIGMA_BOUNDS), decode_width(gamma_raw, GAMMA_BOUNDS)

    with torch.no_grad():
        initial_loss = float(objective(*decoded(), y, context)[0])
    optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=MAX_STEPS,
        max_eval=500, history_size=50, line_search_fn="strong_wolfe",
        tolerance_grad=1.e-7, tolerance_change=1.e-12)
    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(*decoded(), y, context)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite profile objective")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("missing or nonfinite profile gradient")
        gradient = torch.cat([p.grad.reshape(-1) for p in parameters])
        history.append({"loss": float(loss), "gradient_max_abs": float(gradient.abs().max())})
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    loss, packet = objective(*decoded(), y, context, diagnostics=True)
    loss.backward()
    if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("nonfinite final objective or gradient")
    arrays = {key: as_numpy(value) for key, value in packet.items()}
    c, sigma, gamma = (as_numpy(value) for value in decoded())
    arrays.update(c_values=c, sigma_deg=sigma, gamma_deg=gamma,
        coefficients=np.empty(0) if coefficients is None else as_numpy(coefficients * C_LOG_BOUND))
    if fixed_c is None:
        arrays["peak_initial_c"] = np.asarray(peak).copy()
    for label, value, bounds in (("c", c, C0 * np.exp(np.array([-1., 1.]) * C_LOG_BOUND)),
                                ("sigma", sigma, SIGMA_BOUNDS), ("gamma", gamma, GAMMA_BOUNDS)):
        relative = (value - bounds[0]) / (bounds[1] - bounds[0])
        if np.any(relative < -1.e-12) or np.any(relative > 1 + 1.e-12):
            raise ValueError(f"{label} outside the prescribed interval")
        arrays[f"{label}_near_lower_bound"] = relative <= 0.001
        arrays[f"{label}_near_upper_bound"] = relative >= 0.999
    if any(not np.isfinite(arrays[key]).all() for key in
           ("c_values", "sigma_deg", "gamma_deg", "fitted_profile", "residual", "profile_losses", "scale", "background_coeffs")):
        raise FloatingPointError("nonfinite fitted result")
    maximum_gradient = float(torch.cat([p.grad.reshape(-1) for p in parameters]).abs().max())
    return {"arrays": arrays, "metadata": {"closure_evaluations": len(history),
        "initial_loss": initial_loss,
        "optimizer_iterations": int(optimizer.state[parameters[0]].get("n_iter", 0)),
        "gradient_stationary": maximum_gradient <= 1.e-7,
        "final_gradient_max_abs": maximum_gradient, "initial_widths_deg": list(widths), "history": history}}


def fit(design, observed, peak, context, *, fixed_c=None):
    result = optimize(design, observed, peak, context, fixed_c=fixed_c)
    bad = np.flatnonzero(result["arrays"]["scale"] <= 0)
    attempts = [result["metadata"]]
    if len(bad):
        rejected = {f"rejected_{key}": value.copy() for key, value in result["arrays"].items()}
        if fixed_c is None and len(observed) > 1 and np.array_equal(design, np.eye(len(observed))):
            for index in bad:
                one = optimize(np.ones((1, 1)), observed[index:index + 1], peak[index:index + 1], context)
                if one["arrays"]["scale"][0] <= 0:
                    raise RuntimeError("nonpositive scale after fresh isolated refinement")
                attempts.append(one["metadata"])
                for key, value in result["arrays"].items():
                    if value.shape[0] == len(observed):
                        value[index] = one["arrays"][key][0]
            result["metadata"] = {"gradient_stationary": None, "note": "assembled isolated refinements"}
        else:
            result = optimize(design, observed, peak, context, widths=RETRY_START, fixed_c=fixed_c)
            attempts.append(result["metadata"])
        result["arrays"].update(rejected)
    if np.any(result["arrays"]["scale"] <= 0):
        raise RuntimeError("nonpositive scale remains after the prescribed retry")
    result["metadata"] = {**result["metadata"], "retry_used": bool(len(bad)), "attempts": attempts}
    return result


def fit_method(method, observed, context, design=None):
    peak = peak_initialization(observed, context)
    if method == "serial":
        fits = [fit(np.ones((1, 1)), row[None, :], peak[index:index + 1], context)
                for index, row in enumerate(observed)]
        arrays = {key: np.concatenate([item["arrays"][key] for item in fits]) for key in ARRAY_KEYS}
        arrays.update({f"frame_{index}_{key}": value for index, item in enumerate(fits)
                       for key, value in item["arrays"].items() if key.startswith("rejected_")})
        return {"arrays": arrays, "metadata": [item["metadata"] for item in fits]}
    return fit(np.eye(len(observed)) if method == "batched" else design, observed, peak, context)


def original_profile_score(arrays, context):
    device = context["model"].device
    values = [torch.as_tensor(arrays[key], dtype=torch.float64, device=device)
              for key in ("c_values", "sigma_deg", "gamma_deg")]
    observed = torch.as_tensor(context["observed"], dtype=torch.float64, device=device)
    with torch.no_grad():
        _, packet = objective(*values, observed, context)
    return {key: as_numpy(value) for key, value in packet.items()}


def save_fit(output, result):
    output.mkdir()
    np.savez_compressed(output / "profiles.npz", **result["arrays"])
    write_json(output / "optimizer.json", result["metadata"])


def noise(context, output, *, check=False):
    trajectories, records = {}, []
    replicates = [0] if check else range(20)
    for replicate in [None, *replicates]:
        name = "original" if replicate is None else f"counts_10_percent_{replicate:02d}"
        if replicate is None:
            observed = context["observed"]
        else:
            rng = np.random.default_rng(RANDOM_SEED + 100 + 10007 * replicate)
            observed = normalize(rng.binomial(np.rint(context["data"]["raw_counts"]).astype(np.int64), 0.1))
        folder = output / name
        folder.mkdir()
        np.savez_compressed(folder / "observations.npz", observed=observed)
        trajectories[name] = {}
        for method, label in (("serial", "independent"), ("joint", "joint")):
            result = fit_method(method, observed, context, context["design"])
            score = original_profile_score(result["arrays"], context)
            result["arrays"]["original_fitted_profile"] = score["fitted_profile"]
            save_fit(folder / label, result)
            arrays = result["arrays"]
            records.append({"scenario": name, "replicate": replicate, "method": label,
                "fit_profile_loss": float(np.mean(arrays["profile_losses"])),
                "original_profile_loss": float(np.mean(score["profile_losses"])),
                "nonpositive_scale_count": int(np.count_nonzero(arrays["scale"] <= 0)),
                "deficient_rank_count": int(np.count_nonzero(arrays["design_rank"] != 3)),
                "c_bound_hit_count": int(np.count_nonzero(arrays["c_near_lower_bound"] | arrays["c_near_upper_bound"]))})
            trajectories[name][label] = arrays["c_values"]
        write_json(output / "progress.json", {"completed_scenarios": len(trajectories), "last": name})
    stability = []
    for label in ("independent", "joint"):
        stack = np.stack([trajectories[f"counts_10_percent_{rep:02d}"][label] for rep in replicates])
        original = trajectories["original"][label]
        stability.append({"method": label,
            "cross_replicate_c_std_rms_A": float(np.sqrt(np.mean(np.std(stack, axis=0, ddof=0) ** 2))),
            "median_rmse_to_own_original_A": float(np.median(np.sqrt(np.mean((stack - original) ** 2, axis=1))))})
    loss_ratios = []
    for replicate in replicates:
        losses = {row["method"]: row["original_profile_loss"] for row in records if row["replicate"] == replicate}
        loss_ratios.append(losses["joint"] / losses["independent"])
    denominator = stability[0]["cross_replicate_c_std_rms_A"]
    valid = all(not (row["nonpositive_scale_count"] or row["deficient_rank_count"]
                    or row["c_bound_hit_count"]) for row in records)
    write_json(output / "summary.json", {"diagnostic_only": check, "replicates": len(replicates),
        "records": records, "stability": stability, "all_physical_checks_pass": valid,
        "joint_to_independent_stability_ratio": stability[1]["cross_replicate_c_std_rms_A"] / denominator if denominator else None,
        "median_paired_original_profile_loss_ratio": float(np.median(loss_ratios))})
    np.savez_compressed(output / "trajectories.npz", **context["data"],
        **{f"{name}_{method}": values for name, methods in trajectories.items() for method, values in methods.items()})


def holdout(context, output):
    state, capacity = context["data"]["state"], context["data"]["capacity"]
    omitted = np.zeros(len(state), dtype=bool)
    for branch in ("C", "D"):
        indices = np.flatnonzero(state == branch)
        selected = indices[np.arange(len(indices)) % 5 == 2]
        omitted[selected[(selected != indices[0]) & (selected != indices[-1])]] = True
    if omitted.sum() != 48:
        raise ValueError("expected 191 training and 48 omitted profiles")
    records = {}
    for method, label in (("batched", "independent"), ("joint", "joint")):
        result = fit_method(method, context["observed"][~omitted], context, context["design"][~omitted])
        save_fit(output / f"training_{label}", result)
        if method == "joint":
            predicted = C0 * np.exp(C_LOG_BOUND * np.tanh(context["design"] @ result["arrays"]["coefficients"] / C_LOG_BOUND))
        else:
            predicted = np.empty(len(state))
            training = np.flatnonzero(~omitted)
            for branch in ("C", "D"):
                positions = np.flatnonzero(state[training] == branch)
                order = np.argsort(capacity[training[positions]])
                predicted[state == branch] = np.interp(capacity[state == branch],
                    capacity[training[positions]][order], result["arrays"]["c_values"][positions][order])
        scored = fit(None, context["observed"][omitted], None, context, fixed_c=predicted[omitted])
        save_fit(output / f"omitted_{label}", scored)
        records[label] = float(np.mean(scored["arrays"]["profile_losses"]))
        np.savez_compressed(output / f"predicted_{label}.npz", c_values=predicted, omitted=omitted)
    write_json(output / "summary.json", {"conditional_profile_losses": records,
        "joint_to_independent_loss_ratio": records["joint"] / records["independent"],
        "evaluation": "c is predicted; widths, scale, and background use each omitted profile."})


def timing(data_dir, output, device):
    records, quality = [], []
    for repeat in range(-1, 3):
        fitted = {}
        for method in ("serial", "batched", "joint"):
            synchronize(device)
            started = time.perf_counter()
            context = setup(data_dir, device)
            result = fit_method(method, context["observed"], context, context["design"])
            arrays = {key: result["arrays"][key] for key in ("c_values", "sigma_deg", "gamma_deg", "profile_losses")}
            arrays.update({key: value for key, value in result["arrays"].items() if "rejected_" in key})
            np.savez_compressed(output / f"{method}_{repeat}.npz", **arrays)
            synchronize(device)
            records.append({"method": method, "repeat": repeat, "warmup": repeat < 0,
                "seconds": time.perf_counter() - started, "mean_loss": float(np.mean(arrays["profile_losses"]))})
            fitted[method] = arrays
        ratio = np.mean(fitted["batched"]["profile_losses"]) / np.mean(fitted["serial"]["profile_losses"])
        rmse = np.sqrt(np.mean((fitted["batched"]["c_values"] - fitted["serial"]["c_values"]) ** 2))
        quality.append({"repeat": repeat, "batched_to_serial_loss_ratio": float(ratio),
            "batched_to_serial_c_rmse_A": float(rmse), "passes": bool(ratio <= 1.05 and rmse <= 0.02)})
        write_json(output / "progress.json", {"completed_repetitions_including_warmup": repeat + 2})
    write_json(output / "summary.json", {"device": device, "cpu_threads": 4, "records": records,
        "quality": quality, "all_quality_checks_pass": all(row["passes"] for row in quality),
        "scope": "Portable prepared-input workflow. Archived paper timings include additional archival validation I/O."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prepare", "noise", "holdout", "timing", "check"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--input-format", choices=("raw", "archive"), default="raw",
                        help="Used only by prepare; archive extracts measured arrays, never fitted trajectories.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.task == "prepare":
        prepare(args.data_dir, args.output_dir, args.input_format)
        return
    input_hash = sha256(args.data_dir / "nca.npz")
    write_json(args.output_dir / "settings.json", {"task": args.task, "device": args.device,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "cpu_threads": 4, "input_sha256": input_hash, "driver_sha256": sha256(Path(__file__)),
        "spline_coefficients_per_branch": BASIS_COUNT, "max_iterations": MAX_STEPS,
        "width_start_deg": WIDTH_START, "retry_start_deg": RETRY_START,
        "random_seed": RANDOM_SEED, "retained_fraction": 0.1})
    if args.task == "timing":
        timing(args.data_dir, args.output_dir, args.device)
    else:
        context = setup(args.data_dir, args.device)
        write_json(args.output_dir / "reflections.json", as_numpy(context["hkls"]).tolist())
        if args.task == "holdout":
            holdout(context, args.output_dir)
        else:
            noise(context, args.output_dir, check=args.task == "check")
    if sha256(args.data_dir / "nca.npz") != input_hash:
        raise RuntimeError("input packet changed during execution")


if __name__ == "__main__":
    main()
