"""Reproduce Li2S intensity evolution using shared kinetic and spline models."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lisulfur_data import (SOURCE_SHA256, amplitudes, calibrate_template, capacity_progress,
                          load_sequence, residualize, sample_counts, window)


SHAPE_BOUNDS = ((0.25, 4.0), (0.0, 0.8), (0.05, 20.0), (0.25, 4.0))
SHAPE_NAMES = ("birth_n", "birth_onset", "death_k", "death_n")


def inverse_sigmoid(value):
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(clipped / (1.0 - clipped))


def kinetic_shapes(progress, raw):
    exponent, onset, rate, decay_exponent = [lower + (upper - lower) * torch.sigmoid(value)
                                           for value, (lower, upper) in zip(raw, SHAPE_BOUNDS, strict=True)]
    birth_progress = torch.clamp((progress - onset) / torch.clamp(1.0 - onset, min=1e-6), 0.0, 1.0)
    birth = torch.pow(birth_progress, exponent)
    epsilon = torch.as_tensor(1e-6, dtype=progress.dtype, device=progress.device)
    coordinate = epsilon + (1.0 - epsilon) * progress
    survival = torch.exp(-torch.pow(rate * coordinate, decay_exponent))
    start = torch.exp(-torch.pow(rate * epsilon, decay_exponent))
    stop = torch.exp(-torch.pow(rate, decay_exponent))
    death = (survival - stop) / torch.clamp(start - stop, min=1e-12)
    return birth, death


def linear_basis(progress, count=6):
    values = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    knots = np.linspace(0.0, 1.0, count, dtype=np.float64)
    basis = np.zeros((values.size, count), dtype=np.float64)
    for row, value in enumerate(values):
        if value >= 1.0:
            basis[row, -1] = 1.0
            continue
        left = max(min(int(np.searchsorted(knots, value, side="right") - 1), count - 2), 0)
        fraction = (value - knots[left]) / (knots[left + 1] - knots[left])
        basis[row, left], basis[row, left + 1] = 1.0 - fraction, fraction
    return basis


def monotone_knots(raw):
    increments = torch.nn.functional.softplus(raw) + 1e-6
    cumulative = torch.cumsum(increments, dim=0)
    normalized = cumulative / cumulative[-1]
    return torch.cat((torch.zeros_like(normalized[:1]), normalized), dim=0)


def fit_shape(observations, template, cycles, half_cycles, progress, training, method, device, max_steps):
    y_numpy = np.asarray(observations[training], dtype=np.float64)
    cycle_numpy = np.rint(cycles[training]).astype(np.int64)
    unique_cycles = np.unique(cycle_numpy)
    independent = amplitudes(y_numpy, template)
    initial = np.asarray([max(float(np.max(independent[cycle_numpy == cycle])), 1.0)
                          for cycle in unique_cycles], dtype=np.float64)
    scale = max(float(np.median(initial)), 1.0)
    y = torch.as_tensor(y_numpy / scale, dtype=torch.float64, device=device)
    template_t = torch.as_tensor(template, dtype=torch.float64, device=device)
    progress_t = torch.as_tensor(progress[training], dtype=torch.float64, device=device)
    increasing = torch.as_tensor(np.rint(half_cycles[training]).astype(np.int64) % 2 == 0,
                                  dtype=torch.bool, device=device)
    lookup = torch.as_tensor(np.searchsorted(unique_cycles, cycle_numpy), dtype=torch.long, device=device)
    raw_amplitudes = torch.nn.Parameter(torch.as_tensor(
        np.log(np.expm1(np.clip(initial / scale, 1e-8, None))), dtype=torch.float64, device=device))
    parameters = [raw_amplitudes]
    if method == "kinetic":
        raw_shape = torch.nn.Parameter(torch.tensor([
            inverse_sigmoid((value - lower) / (upper - lower))
            for value, (lower, upper) in zip((1.5, 0.30, 2.0, 1.0), SHAPE_BOUNDS, strict=True)
        ], dtype=torch.float64, device=device))
        parameters.append(raw_shape)
    else:
        basis = torch.as_tensor(linear_basis(progress[training]), dtype=torch.float64, device=device)
        raw_birth = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64, device=device))
        raw_death = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64, device=device))
        parameters.extend((raw_birth, raw_death))

    def shape_values():
        if method == "kinetic":
            birth, death = kinetic_shapes(progress_t, raw_shape)
        else:
            birth, death = basis @ monotone_knots(raw_birth), 1.0 - basis @ monotone_knots(raw_death)
        return torch.where(increasing, birth, death)

    normalizer = torch.clamp(torch.mean(torch.square(y)), min=1e-12)

    def objective():
        activity = torch.nn.functional.softplus(raw_amplitudes)[lookup] * shape_values()
        prediction = activity.unsqueeze(1) * template_t.unsqueeze(0)
        return torch.mean(torch.square(y - prediction)) / normalizer

    optimizer = torch.optim.LBFGS(parameters, lr=0.5, max_iter=max_steps, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-12)
    closure_count = 0

    def closure():
        nonlocal closure_count
        closure_count += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Li-S fitting objective")
        loss.backward()
        return loss

    optimizer.step(closure)
    result = {"method": method, "training_loss": float(objective().detach().cpu()),
              "closure_evaluations": closure_count}
    if method == "kinetic":
        named = {name: float((lower + (upper - lower) * torch.sigmoid(value)).detach().cpu())
                 for name, value, (lower, upper) in zip(SHAPE_NAMES, raw_shape, SHAPE_BOUNDS, strict=True)}
        result["parameters"] = named
        # Reconstruct through bounded parameters, as in the original evaluation.
        raw = torch.tensor([inverse_sigmoid((named[name] - lower) / (upper - lower))
                            for name, (lower, upper) in zip(SHAPE_NAMES, SHAPE_BOUNDS, strict=True)],
                           dtype=torch.float64)
        with torch.no_grad():
            birth, death = kinetic_shapes(torch.as_tensor(progress, dtype=torch.float64), raw)
        birth, death = birth.numpy(), death.numpy()
    else:
        birth_knots = monotone_knots(raw_birth).detach().cpu().numpy().astype(np.float64)
        death_knots = monotone_knots(raw_death).detach().cpu().numpy().astype(np.float64)
        result.update(birth_knots=birth_knots.tolist(), death_knots=death_knots.tolist())
        birth, death = linear_basis(progress) @ birth_knots, 1.0 - linear_basis(progress) @ death_knots
    return np.where(np.rint(half_cycles).astype(np.int64) % 2 == 0, birth, death), result


def fit_cycle_amplitudes(observations, template, shape, cycles):
    cycles = np.rint(cycles).astype(np.int64)
    activity = np.zeros(cycles.size, dtype=np.float64)
    fitted = np.zeros(int(np.max(cycles)) + 1, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        design = shape[mask, None] * template[None, :]
        amplitude = max(float(np.sum(observations[mask] * design)) /
                        max(float(np.sum(np.square(design))), 1e-12), 0.0)
        fitted[cycle], activity[mask] = amplitude, amplitude * shape[mask]
    return activity, fitted


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    sequence = load_sequence(args.source_zip)
    cycles = np.rint(sequence.cycle_number).astype(np.int64)
    training, evaluation = np.isin(cycles, (1, 2, 3, 4)), np.isin(cycles, (5, 6, 7, 8))
    reference_x, counts = window(sequence, (30.5, 32.5))
    target_x, target_counts = window(sequence, (25.5, 28.0))
    manual_training = np.isin(sequence.reference_cycle_number, (1, 2, 3, 4))
    sigma = float(np.clip(np.median(sequence.reference_fwhm_deg[manual_training]) / 2.35482004503, 0.05, 0.50))
    template = calibrate_template(reference_x, counts, target_x, target_counts, training, sigma, args.device)
    progress = capacity_progress(sequence.half_cycle, sequence.capacity_mah)
    metric_rows, trajectory_rows, fits = [], [], []
    scenarios = [(1.0, 0)] + [(0.10, replicate) for replicate in range(args.replicates)]
    manual_mask = np.isin(sequence.reference_cycle_number, (5, 6, 7, 8))
    for fraction, replicate in scenarios:
        sampled = sample_counts(counts, fraction, replicate, args.seed)
        observations, _ = residualize(reference_x, sampled)
        predictions = {"framewise": amplitudes(observations, template["reference_template"])}
        for method in ("spline", "kinetic"):
            shape, fit = fit_shape(observations, template["reference_template"], sequence.cycle_number,
                                   sequence.half_cycle, progress, training, method, args.device, args.max_steps)
            predictions[method], cycle_amplitudes = fit_cycle_amplitudes(
                observations, template["reference_template"], shape, sequence.cycle_number)
            fits.append({"fraction": fraction, "replicate": replicate,
                         **fit, "cycle_amplitudes": cycle_amplitudes.tolist()})
        for method, activity in predictions.items():
            full = activity / fraction
            prediction = template["target_scale"] * full[:, None] * template["target_template"][None, :]
            loss = np.mean(np.square(template["target_y"] - prediction), axis=1)
            reference_prediction = full[:, None] * template["reference_template"][None, :]
            reference_loss = np.mean(np.square(template["reference_y"] - reference_prediction), axis=1)
            corr = float(np.corrcoef(full[sequence.reference_frame_index[manual_mask]],
                                    sequence.reference_peak_area[manual_mask])[0, 1])
            metric_rows.append({"fraction": fraction, "replicate": replicate, "method": method,
                                "reference_loss": float(np.mean(reference_loss[evaluation])),
                                "heldout_loss": float(np.mean(loss[evaluation])), "manual_area_correlation": corr})
            trajectory_rows.append(pd.DataFrame({
                "fraction": fraction, "replicate": replicate, "method": method,
                "time_h": sequence.scan_time_h, "cycle": cycles, "half_cycle": sequence.half_cycle,
                "capacity_progress": progress, "activity": full,
            }))
    metrics, trajectories = pd.DataFrame(metric_rows), pd.concat(trajectory_rows, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    trajectories.to_csv(output / "trajectories.csv", index=False)
    weak = trajectories[(trajectories["fraction"] == 0.1) & trajectories["cycle"].isin((5, 6, 7, 8))]
    variability = {}
    for method in ("framewise", "spline", "kinetic"):
        pivot = weak[weak["method"] == method].pivot(index="replicate", columns="time_h", values="activity")
        variability[method] = float(np.sqrt(np.mean(np.square(pivot.std(axis=0, ddof=1)))))
    summary = {"source_sha256": SOURCE_SHA256, "seed": args.seed, "replicates": args.replicates,
               "max_steps": args.max_steps, "device": args.device,
               "paper_recipe": args.seed == 20260804 and args.replicates == 20 and args.max_steps == 120,
               "training_cycles": [1, 2, 3, 4], "evaluation_cycles": [5, 6, 7, 8],
               "template_lattice_a_angstrom": template["lattice_a_angstrom"],
               "template_sigma_deg": sigma, "target_scale": template["target_scale"],
               "trajectory_variability": variability, "fits": fits}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True, help="Unmodified Zenodo 3514967 data.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    if args.replicates < 2 or args.max_steps < 1:
        parser.error("At least two noise replicates and one optimizer step are required")
    summary = run(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "fits"}, indent=2))


if __name__ == "__main__":
    main()
