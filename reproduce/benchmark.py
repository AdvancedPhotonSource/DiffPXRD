"""Measure matched repeated FCJ evaluation workloads, not refinement time."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from statistics import median
import time

import numpy as np
from pymatgen.core import Lattice, Structure
import torch

from diffpxrd import ModularCWForwardModel, normalized_chebyshev_basis_torch
from diffpxrd import solved_profile_wls_loss_torch


WAVELENGTH = 0.49593679373280103
BATCH_SIZES = (1, 2, 4, 8)
MODES = ("forward", "loss", "loss_gradient")
INSTRUMENT = {"Type": "PXC", "Lam": WAVELENGTH, "U": 1.163, "V": -0.126,
    "W": 0.063, "X": 0., "Y": 0., "Z": 0., "SH/L": 0.002,
    "Zero": 0., "Polariz.": 0.99, "Azimuth": 0.}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_from_payload(payload):
    species = payload.get("species") or payload.get("species_string")
    if species is None:
        raise ValueError("structure payload requires species")
    return Structure(Lattice(np.asarray(payload["lattice_matrix"], dtype=np.float64)), species,
        np.asarray(payload["frac_coords"], dtype=np.float64), coords_are_cartesian=False,
        site_properties={key: list(values) for key, values in (payload.get("site_properties") or {}).items()})


def load_structures(path, limit=None):
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records = []
    for file in files:
        value = json.loads(file.read_text())
        records.extend(value["structures"] if "structures" in value else [value])
    if (not records or any(not isinstance(row.get("case_id"), str) or not row["case_id"] for row in records)
            or len({row["case_id"] for row in records}) != len(records)
            or any(not isinstance(row.get("structure"), dict) for row in records)):
        raise ValueError("expected unique case_id values and compact structure objects")
    if limit is None and len(records) != 200:
        raise ValueError("the complete workload requires 200 structures; use --limit for an explicit diagnostic")
    if limit is not None:
        if not 1 <= limit <= len(records):
            raise ValueError("--limit must select a nonempty available subset")
        records = records[:limit]
    return records, {str(file): sha256(file) for file in files}


def prepare(structure):
    domain = torch.linspace(5., 35., 4096, dtype=torch.float64)
    q = [4 * math.pi * math.sin(math.radians(angle / 2)) / WAVELENGTH for angle in (5., 35.)]
    model = ModularCWForwardModel(q_min=q[0], q_max=q[1], wavelength=WAVELENGTH, device="cpu")
    batch = model.structure_to_batch(structure, dtype=torch.float64)
    hkls = model.compile_reflections(batch).hkl_pts
    if hkls.numel() == 0:
        raise ValueError("structure has no reflections in the benchmark window")
    prepared = {"model": model, "batch": batch, "hkls": hkls, "domain": domain,
        "background": normalized_chebyshev_basis_torch(domain.unsqueeze(0), 3),
        "scales": {size: (torch.linspace(0.99, 1.01, size, dtype=torch.float64),
                          torch.linspace(1.01, 0.99, size, dtype=torch.float64)) for size in BATCH_SIZES}}
    with torch.no_grad():
        prepared["target"] = profile(prepared, batch).detach()
    floor = torch.clamp(torch.max(prepared["target"]) * 1.e-4, min=1.)
    prepared["weights"] = 1. / torch.clamp(prepared["target"], min=floor)
    return prepared


def on_device(prepared, device):
    model = prepared["model"]
    moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in prepared.items()}
    moved["model"] = ModularCWForwardModel(q_min=model.q_min, q_max=model.q_max,
        wavelength=model.wavelength, device=device)
    moved["batch"] = {key: value.to(device) if torch.is_tensor(value) else value
                      for key, value in prepared["batch"].items()}
    moved["scales"] = {size: tuple(value.to(device) for value in values)
                       for size, values in prepared["scales"].items()}
    return moved


def scaled_batch(batch, x, y):
    count = len(x)
    candidate = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            candidate[key] = value
        elif key == "lengths":
            lengths = value.expand(count, -1)
            candidate[key] = torch.stack((lengths[:, 0] * x, lengths[:, 1] * x, lengths[:, 2] * y), dim=1)
        elif key == "angles":
            candidate[key] = value.repeat(count, 1)
        elif key == "num_atoms":
            candidate[key] = value.repeat(count)
        else:
            candidate[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
    return candidate


def profile(prepared, batch):
    model, hkls = prepared["model"], prepared["hkls"]
    count = int(batch["lengths"].shape[0])
    if hkls.ndim == 3 and hkls.shape[0] == 1 and count > 1:
        hkls = hkls[0].unsqueeze(0).expand(count, -1, -1)
    reflections = model.compute_reflection_layer_from_hkls(batch, hkls)
    scattering = model.compute_scattering_layer(batch, reflections)
    corrections = model.compute_correction_layer(reflections, scattering, instrument=INSTRUMENT)
    return model.compute_fcj_profile_layer(corrections, instrument=INSTRUMENT,
        domain=prepared["domain"], num_steps=4096, backend=None, return_diagnostics=False).histogram


def evaluate(prepared, mode, batch_size, diagnostics=None):
    x, y = (value.clone() for value in prepared["scales"][batch_size])
    if mode == "loss_gradient":
        x.requires_grad_(True)
        y.requires_grad_(True)
    candidate = scaled_batch(prepared["batch"], x, y)
    calculated = profile(prepared, candidate)
    gradients = None
    if mode == "forward":
        value = calculated.sum()
    else:
        value, _ = solved_profile_wls_loss_torch(prepared["target"].expand_as(calculated), calculated,
            weights=prepared["weights"].expand_as(calculated), x=prepared["domain"],
            background_basis=prepared["background"], fit_scale=True, fit_background=True,
            background_degree=3, reduction="sum")
        if diagnostics is not None:
            diagnostics["loss"] = value.detach().cpu().numpy()
        if mode == "loss_gradient":
            gradients = torch.autograd.grad(value, (x, y))
            value = value + 1.e-12 * (gradients[0].sum() + gradients[1].sum())
    if diagnostics is not None:
        tensors = {**prepared["batch"], **candidate, "domain": prepared["domain"],
                   "target": prepared["target"], "weights": prepared["weights"],
                   "background": prepared["background"], "x": x, "y": y,
                   "calculated": calculated, "objective": value}
        if any(tensor.is_floating_point() and tensor.dtype != torch.float64
               for tensor in tensors.values() if torch.is_tensor(tensor)):
            raise ValueError("the benchmark requires float64 for every floating-point input and result")
        diagnostics.update(profile=calculated.detach().cpu().numpy(),
            x_scales=x.detach().cpu().numpy(), y_scales=y.detach().cpu().numpy())
        if gradients is not None:
            diagnostics["gradient"] = torch.stack(gradients).detach().cpu().numpy()
        if not all(np.isfinite(array).all() for array in diagnostics.values()):
            raise FloatingPointError("nonfinite profile, loss, or gradient")
    return float(value.detach().cpu()), float(calculated.shape[0])


def compare(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if (reference.shape != candidate.shape or not reference.size
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        return {"passed": False, "reason": "shape or finite-value check failed"}
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    allowed = 5.e-11 * scale + 5.e-9 * np.abs(reference)
    return {"passed": bool(np.all(np.abs(difference) <= allowed)), "rtol": 5.e-9,
        "normalized_atol": 5.e-11, "max_absolute_difference": float(np.max(np.abs(difference))),
        "max_peak_normalized_difference": float(np.max(np.abs(difference)) / scale),
        "relative_rms_difference": float(np.linalg.norm(difference.ravel()) /
            max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny))}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run(args, report):
    records, input_hashes = load_structures(args.structures, args.limit)
    devices = ("cpu", "cuda:0") if args.device == "both" else (args.device,)
    if "cuda:0" in devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no device fallback")
    torch.set_num_threads(4)
    report.update(input_sha256=input_hashes, case_ids=[row["case_id"] for row in records],
        diagnostic=args.limit is not None, device_request=args.device,
        hardware={"platform": platform.platform(), "processor": platform.processor(),
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if "cuda:0" in devices else None,
                  "cpu_threads": torch.get_num_threads()},
        settings={"wavelength_A": WAVELENGTH, "instrument": INSTRUMENT, "points": 4096,
                  "start_deg": 5., "stop_deg": 35., "dtype": "float64",
                  "warmups": 3, "repeats": 10, "batches": BATCH_SIZES, "modes": MODES})
    for record in records:
        cpu = prepare(structure_from_payload(record["structure"]))
        inputs = {"cpu": cpu}
        if "cuda:0" in devices:
            inputs["cuda:0"] = on_device(cpu, torch.device("cuda:0"))
        for mode in MODES:
            for size in BATCH_SIZES:
                checks, diagnostics = {}, {}
                for device in dict.fromkeys(("cpu", *devices)):
                    elements = size * cpu["hkls"].shape[-2] * 4096
                    limit = 32_000_000_000 if device == "cpu" else 10_000_000_000
                    if elements > limit:
                        raise MemoryError(f"{record['case_id']} exceeds the declared {device} work-element cap; no smaller-batch substitution")
                    diagnostics[device] = {}
                    evaluate(inputs[device], mode, size, diagnostics[device])
                if "cuda:0" in diagnostics:
                    for key, reference in diagnostics["cpu"].items():
                        checks[key] = ({"passed": bool(np.array_equal(reference, diagnostics["cuda:0"][key]))}
                            if key in ("x_scales", "y_scales") else compare(reference, diagnostics["cuda:0"][key]))
                    if not all(row["passed"] for row in checks.values()):
                        write_json(args.output_dir / "failed_accuracy.json", {"case_id": record["case_id"],
                            "mode": mode, "batch_size": size, "checks": checks})
                        raise ValueError("CPU/GPU full-array agreement failed before timing")
                for device in devices:
                    prepared = inputs[device]
                    for _ in range(3):
                        value, _ = evaluate(prepared, mode, size)
                        synchronize(device)
                        if not math.isfinite(value):
                            raise FloatingPointError("nonfinite warm-up result")
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    samples = []
                    for _ in range(10):
                        synchronize(device)
                        started = time.perf_counter()
                        value, _ = evaluate(prepared, mode, size)
                        synchronize(device)
                        samples.append(time.perf_counter() - started)
                        if not math.isfinite(value):
                            raise FloatingPointError("nonfinite timed result")
                    report["rows"].append({"case_id": record["case_id"], "mode": mode,
                        "batch_size": size, "device": device, "reflection_count": int(cpu["hkls"].shape[-2]),
                        "frozen_hkls_sha256": hashlib.sha256(cpu["hkls"].numpy().tobytes()).hexdigest(),
                        "accuracy": checks, "cpu_gpu_checked": bool(checks), "runtime_samples_sec": samples,
                        "median_runtime_sec": median(samples), "profiles_per_sec": size / median(samples),
                        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else None})
                    write_json(args.output_dir / "progress.json", report)
    report["completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Explicit diagnostic subset; not the 200-structure paper benchmark")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "rows": [], "scope": "Repeated FCJ evaluation only; no GSAS-II or optimization timings."}
    try:
        run(args, report)
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.output_dir / "summary.json", report)


if __name__ == "__main__":
    main()
