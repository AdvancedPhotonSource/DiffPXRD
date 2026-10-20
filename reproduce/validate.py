"""Recalculate the paper's forward benchmark against prepared GSAS-II references.

Inputs are a checksummed, separately distributed benchmark packet. The packet
contains structural inputs and GSAS-II reference arrays, not diffPXRD predictions.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from diffpxrd import ModularCWForwardModel, __version__, fit_weighted_profile_numpy
from diffpxrd.cw.contracts import get_reciprocal_orbit_cached


SCHEMA = "diffpxrd_forward_validation_v1"
GATES = {"profile_nrmse": 5e-3, "profile_cosine": 0.995,
         "position_error_deg": 5e-5, "intensity_nrmse": 2e-7}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, item):
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
        raise ValueError(f"Missing, changed, or uncontained packet member: {item['path']}")
    return path


def _finite(name, array, *, dtype=None, shape=None):
    if not np.isfinite(array).all() or (dtype is not None and array.dtype != dtype):
        raise ValueError(f"Invalid values or dtype in {name}: {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Invalid shape in {name}: {array.shape}, expected {shape}")


def forward(metadata, arrays, *, device="cpu"):
    """Re-enumerate reflections and calculate scattering, corrections, and profiles."""
    if metadata.get("schema") != SCHEMA or metadata.get("dtype") != "float64":
        raise ValueError("This benchmark requires its float64 prepared-input schema")
    domain = arrays["two_theta_deg"]
    _finite("two_theta_deg", domain, dtype=np.float64)
    if domain.ndim != 1 or domain.size < 2 or not np.all(np.diff(domain) > 0):
        raise ValueError("Diffraction grid must be a strictly increasing vector")
    if metadata["profile_comparison"] != {"fit_scale": True, "fit_background": False,
                                           "weights": "uniform", "reference": "signal"}:
        raise ValueError("The frozen comparison uses one scale, no background fit, and signal-only profiles")
    wavelengths, weights = metadata["wavelengths"], metadata["weights"]
    if not wavelengths or len(wavelengths) != len(weights) or min(wavelengths) <= 0:
        raise ValueError("Invalid wavelength-component configuration")
    expected_q = 4 * np.pi * np.sin(np.deg2rad(domain[[0, -1]] / 2)) / np.array(
        [max(wavelengths), min(wavelengths)])
    np.testing.assert_allclose(np.array([metadata["q_min"], metadata["q_max"]]), expected_q,
                               rtol=1e-14, atol=1e-14)
    model = ModularCWForwardModel(q_min=metadata["q_min"], q_max=metadata["q_max"],
                                 wavelength=wavelengths[0], device=device)
    batch = {}
    for key, array in arrays.items():
        if key.startswith("batch__"):
            _finite(key, array, dtype=np.int64 if key[7:] in ("num_atoms", "atom_types", "site_species_zs")
                    else np.float64)
            batch[key[7:]] = torch.as_tensor(array, device=device)
    if not batch or batch["lengths"].shape != (1, 3) or int(batch["num_atoms"].sum()) != len(batch["frac_coords"]):
        raise ValueError("Each benchmark member must contain one complete structural batch")
    reflection = model.compute_reflection_layer(batch)
    scattering = model.compute_scattering_layer(batch, reflection, store_intermediates=True)
    components = []
    symbol = metadata["spacegroup_symbol"]
    for wavelength, weight in zip(wavelengths, weights, strict=True):
        correction = model.compute_correction_layer(
            reflection, scattering, wavelength=wavelength, instrument=metadata["instrument"],
            sample_parameters=metadata["sample_parameters"], spacegroup_symbol=symbol,
        )
        pattern, hkls, cell = correction.pattern, correction.hkl_pts, correction.cell_matrix
        if metadata["reflection_source"] == "contract_families":
            families = model.compute_reflection_contract_layer(
                reflection, correction, spacegroup_symbols=symbol, overlap_tolerance_deg=1e-5,
            )
            pattern = model.build_pattern_tensor(families.family_positions[0], families.family_intensities[0])
            hkls, cell = families.family_hkls[0].unsqueeze(0), reflection.cell_matrix
        elif metadata["reflection_source"] != "enumerated":
            raise ValueError("Unknown reflection source")
        methods = {"tch": model.compute_tch_profile_layer_from_pattern,
                   "fcj_exact": model.compute_fcj_profile_layer_from_pattern}
        profile = methods[metadata["profile_model"]](
            pattern, instrument=metadata["instrument"], wavelength=wavelength,
            sample_profile=metadata["sample_profile"], peak_hkls=hkls, cell_matrix=cell,
            spacegroup_symbol=symbol, domain=domain, num_steps=len(domain),
        )
        np.testing.assert_array_equal(profile.pattern_domain.detach().cpu().numpy(), domain)
        components.append({
            "weight": weight,
            "hkls": np.rint(hkls.detach().cpu().numpy().reshape(-1, 3)).astype(np.int64),
            "positions": pattern[0].detach().cpu().numpy().reshape(-1),
            "intensities": pattern[1].detach().cpu().numpy().reshape(-1),
            "sigma_sq": profile.sigma_sq_centideg2.detach().cpu().numpy().reshape(-1),
            "gamma": profile.gamma_centideg.detach().cpu().numpy().reshape(-1),
            "fwhm": profile.fwhm.detach().cpu().numpy().reshape(-1),
            "histogram": profile.histogram[0].detach().cpu().numpy(),
        })
    signal = np.sum([component["weight"] * component["histogram"] for component in components], axis=0)
    for component in components:
        for name, value in component.items():
            if name != "weight":
                _finite(name, value)
    return signal, components


def _hkl_map(hkls, symbol):
    mapping = {}
    for index, hkl in enumerate(hkls):
        key = min(get_reciprocal_orbit_cached(symbol, tuple(map(int, hkl)), include_friedel_pairs=True))
        mapping.setdefault(key, index)
    return mapping


def compare(metadata, arrays, signal, components):
    """Apply the original six gates, including the open primary-wavelength interval."""
    target, domain = arrays["gsasii_signal"], arrays["two_theta_deg"]
    _finite("gsasii_signal", target, dtype=np.float64, shape=domain.shape)
    fit = fit_weighted_profile_numpy(target, signal, weights=np.ones_like(target),
                                    fit_scale=True, fit_background=False)
    first = components[0]
    mask = (first["positions"] > domain[0]) & (first["positions"] < domain[-1])
    reference_hkls = arrays["gsasii_hkls"]
    _finite("gsasii_hkls", reference_hkls, dtype=np.int64)
    if reference_hkls.ndim != 2 or reference_hkls.shape[1] != 3:
        raise ValueError("GSAS reflection indices must have shape (N, 3)")
    for key in ("positions", "intensities", "sigma_sq", "gamma", "fwhm"):
        _finite(f"gsasii_{key}", arrays[f"gsasii_{key}"], dtype=np.float64, shape=(len(reference_hkls),))
    expected = _hkl_map(reference_hkls, metadata["spacegroup_symbol"])
    observed = _hkl_map(first["hkls"][mask], metadata["spacegroup_symbol"])
    common = sorted(expected.keys() & observed.keys())
    missing, extra = sorted(expected.keys() - observed.keys()), sorted(observed.keys() - expected.keys())
    result = {"profile_nrmse": fit.nrmse_vs_observed_max, "profile_cosine": fit.cosine_similarity,
              "profile_scale": fit.scale, "missing_reflections": len(missing), "extra_reflections": len(extra),
              "missing_hkls": missing, "extra_hkls": extra, "overlap_reflections": len(common),
              "position_error_max_deg": None, "intensity_nrmse": None, "intensity_scale": None}
    if common:
        left, right = np.array([expected[key] for key in common]), np.array([observed[key] for key in common])
        reference = arrays["gsasii_intensities"][left]
        calculated = first["intensities"][mask][right]
        denominator = float(np.dot(calculated, calculated))
        scale = 1. if denominator <= 1e-12 else float(np.dot(reference, calculated) / denominator)
        result.update(
            position_error_max_deg=float(np.max(np.abs(first["positions"][mask][right] - arrays["gsasii_positions"][left]))),
            intensity_nrmse=float(np.sqrt(np.mean((scale * calculated - reference)**2)) /
                                  max(float(np.max(np.abs(reference))), 1e-12)), intensity_scale=scale,
        )
        result["width_diagnostics"] = {
            key + "_maximum_absolute_error": float(np.max(np.abs(first[key][mask][right] - arrays[f"gsasii_{key}"][left])))
            for key in ("sigma_sq", "gamma", "fwhm")}
    result["checks"] = {
        "missing_reflections": not missing, "extra_reflections": not extra,
        "profile_nrmse": result["profile_nrmse"] <= GATES["profile_nrmse"],
        "profile_cosine": result["profile_cosine"] >= GATES["profile_cosine"],
        "position_error": bool(common) and result["position_error_max_deg"] <= GATES["position_error_deg"],
        "intensity_nrmse": bool(common) and result["intensity_nrmse"] <= GATES["intensity_nrmse"],
    }
    result["passes_all_gates"] = all(result["checks"].values())
    return result


def validate_case(metadata, arrays, *, device="cpu"):
    with torch.no_grad():
        signal, components = forward(metadata, arrays, device=device)
    return compare(metadata, arrays, signal, components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Prepared packet directory")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--case", action="append", help="Explicit structural subset; may be repeated")
    parser.add_argument("--config", action="append", help="Explicit configuration subset; may be repeated")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("gates") != GATES:
        raise ValueError("Packet schema or prespecified tolerances do not match this validator")
    provenance = json.loads(checked_path(args.inputs, manifest["provenance"]).read_text())
    if provenance.get("export_succeeded") is not True:
        raise ValueError("Packet export did not finish successfully")
    cases, configs = manifest["case_ids"], manifest["configurations"]
    if len(set(cases)) != 200 or len(cases) != 200 or len(set(configs)) != 9 or len(configs) != 9:
        raise ValueError("Packet must declare the frozen 200-structure, nine-configuration universe")
    selected_cases, selected_configs = set(args.case or cases), set(args.config or configs)
    if not selected_cases.issubset(cases) or not selected_configs.issubset(configs):
        raise ValueError("Requested case or configuration is absent from the frozen universe")
    expected = {(case, config) for case in selected_cases for config in selected_configs}
    universe = {(case, config) for case in cases for config in configs}
    index = {(entry["case_id"], entry["configuration"]): entry for entry in manifest["members"]}
    if len(index) != len(manifest["members"]) or not expected.issubset(index) or not index.keys() <= universe:
        raise ValueError("Missing or duplicate packet members; request an explicit available subset for a diagnostic run")
    args.output.mkdir(parents=True, exist_ok=False)
    results, started = [], time.monotonic()
    with (args.output / "results.jsonl").open("w") as stream:
        for identity in sorted(expected):
            row = {"case_id": identity[0], "configuration": identity[1], "status": "error", "passes_all_gates": False}
            try:
                entry = index[identity]
                metadata = json.loads(checked_path(args.inputs, entry["metadata"]).read_text())
                if (metadata["case_id"], metadata["configuration"]) != identity:
                    raise ValueError("Metadata identity differs from manifest")
                with np.load(checked_path(args.inputs, entry["arrays"]), allow_pickle=False) as data:
                    row.update(validate_case(metadata, dict(data), device=args.device), status="completed")
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            results.append(row)
    summary = {"scope": "full_200_by_9" if len(expected) == 1800 else "explicit_subset",
               "expected": len(expected), "completed": sum(row["status"] == "completed" for row in results),
               "passed": sum(row["passes_all_gates"] for row in results), "device": args.device,
               "gates": GATES, "input_manifest_sha256": sha256(manifest_path),
               "elapsed_seconds": time.monotonic() - started,
               "versions": {"diffpxrd": __version__,
                            **{name: version(name) for name in ("numpy", "torch", "gemmi")}}}
    summary["all_passed"] = summary["passed"] == summary["expected"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
