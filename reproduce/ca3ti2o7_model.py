"""Diffraction and structural geometry for the Ca3Ti2O7 example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffpxrd.trajectory.spline import CubicSplineTrajectory
from prepare_ca3ti2o7_neutron import load_prepared
from pymatgen.core import Lattice, Structure

from diffpxrd import ModularCWForwardModel
from diffpxrd.cw.profile import CW_PROFILE_WIDTH_FLOOR_CENTIDEG


def neutron_basis(data):
    """Uncentered heating-path SVD, after removing rigid translations."""
    delta = data["neutron_fractional"] - data["acaa_fractional"]
    delta -= np.round(delta)
    common_lengths = (data["neutron_lattice"] + data["acaa_lattice"]) / 2
    displacement = delta * common_lengths[:, None, :]
    displacement -= displacement.mean(axis=1, keepdims=True)
    matrix = displacement[data["neutron_cycle"] == "heating"].reshape(-1, 144)
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    basis = vectors[:3].copy()
    for vector in basis:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    basis = basis.reshape(3, 48, 3) * np.sqrt(48)
    amplitudes = np.einsum("ksd,nsd->nk", basis, displacement) / 48
    return basis, amplitudes, singular**2 / np.sum(singular**2)


def interpolate_neutron(data, values, cycles, temperatures):
    result = np.empty((len(temperatures), *values.shape[1:]))
    flat = values.reshape(len(values), -1)
    for cycle in ("heating", "cooling"):
        source = np.flatnonzero(data["neutron_cycle"] == cycle)
        source = source[np.argsort(data["neutron_temperature"][source])]
        selected = cycles == cycle
        result[selected] = np.column_stack([
            np.interp(temperatures[selected], data["neutron_temperature"][source], flat[source, i])
            for i in range(flat.shape[1])
        ]).reshape((-1, *values.shape[1:]))
    return result


def radial_bound(raw, radius):
    return radius * raw / torch.sqrt(radius**2 + raw.square().sum(-1, keepdim=True))


def inverse_radial_bound(amplitudes, radius):
    squared = np.sum(amplitudes**2, axis=-1, keepdims=True)
    if np.any(squared >= radius**2):
        raise ValueError("neutron initialization exceeds the displacement bound")
    return radius * amplitudes / np.sqrt(radius**2 - squared)


def profile_objective(observed, calculated, variance, background, mask):
    """Profile one nonnegative scale and local linear backgrounds per frame."""
    design = torch.cat((calculated.unsqueeze(-1), background), dim=-1)
    weight = mask.to(observed.dtype) * torch.rsqrt(variance)
    weighted = design * weight.unsqueeze(-1)
    column_norm = torch.linalg.vector_norm(weighted, dim=1)
    column_norm = torch.where(column_norm > 1e-12, column_norm, torch.ones_like(column_norm))
    scaled = weighted / column_norm.unsqueeze(1)
    gram = scaled.transpose(1, 2) @ scaled
    rhs = (scaled.transpose(1, 2) @ (observed * weight).unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(design.shape[-1], dtype=observed.dtype, device=observed.device)
    free = torch.linalg.solve(gram + 1e-12 * eye, rhs.unsqueeze(-1)).squeeze(-1)
    zero_background = torch.linalg.solve(
        gram[:, 1:, 1:] + 1e-12 * eye[1:, 1:], rhs[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    zero = torch.cat((torch.zeros_like(zero_background[:, :1]), zero_background), dim=1)
    free, zero = free / column_norm, zero / column_norm
    use_free = (free[:, 0] >= -1e-10).detach()
    coefficients = torch.where(use_free[:, None], free, zero)
    fitted = (design @ coefficients.unsqueeze(-1)).squeeze(-1)
    residual = fitted - observed
    deviance = (residual.square() / variance * mask).sum(1) / mask.sum(1)
    return deviance.mean(), {
        "scale": coefficients[:, 0].clamp_min(0), "fitted_profile": fitted,
        "residual": residual, "deviance_per_frame": deviance,
        "background_coefficients": coefficients[:, 1:],
    }


class Problem:
    """Prepared 42-frame problem; the neutron basis is rebuilt, not loaded."""

    def __init__(self, directory: Path, device: str, neutron_tables: Path | None = None):
        neutron = load_prepared(neutron_tables)
        self.config = json.loads((directory / "metadata.json").read_text())
        if self.config["schema"] != "diffpxrd-ca3ti2o7-inputs-1":
            raise ValueError("unknown Ca3Ti2O7 input schema")
        path = directory / "measurements.npz"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.config["measurements_sha256"]:
            raise ValueError("prepared input checksum mismatch")
        with np.load(path, allow_pickle=False) as packet:
            self.data = {key: packet[key] for key in packet.files}
        if self.data.keys() & neutron.keys():
            raise ValueError("Measurements must not bundle neutron tables; use the current deposit")
        self.data.update(neutron)
        self.validate()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=self.device)
        d, c = self.data, self.config
        self.domain = self.tensor(d["two_theta"])
        self.observed, self.variance = self.tensor(d["observed"]), self.tensor(d["variance"])
        self.basis, external, self.basis_fraction = neutron_basis(d)
        self.external = interpolate_neutron(d, external, d["cycle"], d["temperature"])
        self.model = ModularCWForwardModel(
            q_min=c["q_min"], q_max=c["q_max"], wavelength=c["wavelength"],
            device=self.device,
        )
        self.hkls = self.tensor(d["hkls"])
        self.contract = self.model.compile_reflection_contract(
            self.hkls, spacegroup_symbol="P1", static_full_families=True
        )
        self.base = self.batch_for(d["lattice"], d["cycle"], d["temperature"])
        self.splines = tuple(self.spline(d["temperature"][d["cycle"] == cycle])
                             for cycle in ("heating", "cooling"))
        self.parts = tuple(np.flatnonzero(d["cycle"] == cycle) for cycle in ("heating", "cooling"))
        distances = np.abs(d["two_theta"][None, :, None] - d["window_centers"][:, None, :])
        self.owner = np.where(distances.min(-1) <= c["window_half_width"], distances.argmin(-1), -1)
        backgrounds = []
        for i in range(d["window_centers"].shape[1]):
            mask = self.owner == i
            backgrounds.extend((mask.astype(float), np.where(
                mask, (d["two_theta"][None, :] - d["window_centers"][:, i, None]) / c["window_half_width"], 0
            )))
        self.background = self.tensor(np.stack(backgrounds, axis=-1))
        integer_hkls = np.rint(d["hkls"].reshape(-1, 3)).astype(int)
        self.members = [torch.as_tensor(np.flatnonzero(np.all(np.abs(integer_hkls) == hkl, axis=1)),
                                        device=self.device) for hkl in ([2, 1, 3], [1, 1, 7])]
        if any(len(indices) == 0 for indices in self.members):
            raise ValueError("frozen reflections omit a doublet family")
        with torch.no_grad():
            profile, _ = self.render(self.tensor(np.zeros((42, 3))), self.extras("common"), normalize=False)
            self.normalization = profile.max(1, keepdim=True).values
        if torch.any(self.normalization <= 0):
            raise ValueError("nonpositive Acaa profile normalization")

    def validate(self):
        d = self.data
        required = {"two_theta", "observed", "variance", "temperature", "cycle", "scan", "source_index",
                    "lattice", "window_centers", "hkls", "ratio", "ratio_sigma", "species", "acaa_fractional",
                    "acaa_lattice", "neutron_fractional", "neutron_lattice", "neutron_u_iso",
                    "neutron_temperature", "neutron_cycle", "neighbor_reference_fractional",
                    "neighbor_reference_lattice", "series_temperature", "series_cycle", "series_scan",
                    "series_lattice", "series_ratio", "series_ratio_sigma", "series_eligible", "doublet_parameters"}
        if not required.issubset(d):
            raise ValueError(f"missing input arrays: {sorted(required - d.keys())}")
        for name, array in d.items():
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"nonfinite {name}")
        if d["observed"].shape != (42, len(d["two_theta"])) or d["variance"].shape != d["observed"].shape:
            raise ValueError("the paper recipe requires all 42 aligned profiles")
        if np.any(np.diff(d["two_theta"]) <= 0) or np.any(d["variance"] <= 0):
            raise ValueError("angles must increase and variances must be positive")
        if d["lattice"].shape != (42, 3) or d["acaa_fractional"].shape != (48, 3):
            raise ValueError("expected orthorhombic lattices and 48 ordered sites")
        if d["species"].shape != (48,) or d["neutron_fractional"].shape[1:] != (48, 3):
            raise ValueError("neutron structures must share the 48-site ordering")
        neutron_count = len(d["neutron_fractional"])
        if d["neutron_lattice"].shape != (neutron_count, 3) or d["neutron_u_iso"].shape != (neutron_count, 48):
            raise ValueError("neutron coordinates, cells and displacement factors must align")
        if d["hkls"].ndim != 3 or d["hkls"].shape[0] != 1 or d["hkls"].shape[-1] != 3:
            raise ValueError("frozen reflections must have shape (1, H, 3)")
        if not np.array_equal(d["hkls"], np.rint(d["hkls"])):
            raise ValueError("reflection indices must be integral")
        if d["window_centers"].ndim != 2 or d["window_centers"].shape[0] != 42:
            raise ValueError("tracked centers must align with the selected frames")
        if d["series_lattice"].shape != (346, 3) or len(d["series_temperature"]) != 346:
            raise ValueError("expected the complete 346-frame single-phase series")
        if d["series_eligible"].dtype != np.bool_ or int(d["series_eligible"].sum()) != 343:
            raise ValueError("expected a boolean mask retaining 343 eligible temperatures")
        if d["doublet_parameters"].shape != (42, 6):
            raise ValueError("expected six measured doublet parameters per fitted frame")
        if list(d["cycle"]) != ["heating"] * 21 + ["cooling"] * 21:
            raise ValueError("frames must contain 21 heating then 21 cooling nodes")
        for cycle, cap in (("heating", 1073), ("cooling", 1055)):
            values = d["temperature"][d["cycle"] == cycle]
            if np.any(np.diff(values) < 0) or values.min() < 313 or values.max() > cap:
                raise ValueError(f"invalid {cycle} temperature selection")
        if np.any(d["lattice"] <= 0) or np.any(d["neutron_u_iso"] < 0) or np.any(d["ratio_sigma"] <= 0) or np.any(d["ratio"] <= 0):
            raise ValueError("invalid lattice, thermal displacement or ratio uncertainty")

    def spline(self, temperatures):
        return CubicSplineTrajectory.from_state_values(
            temperatures, full_knots=self.config["knots"], input_min=313., input_max=1073.
        )

    def batch_for(self, lengths, cycles, temperatures):
        uiso = interpolate_neutron(self.data, self.data["neutron_u_iso"], cycles, temperatures)
        structures = tuple(Structure(Lattice.orthorhombic(*cell), self.data["species"],
                                     self.data["acaa_fractional"], site_properties={"u_iso": u.tolist()})
                           for cell, u in zip(lengths, uiso, strict=True))
        return {key: value.to(dtype=torch.float64, device=self.device) if value.is_floating_point()
                else value.to(self.device) for key, value in self.model.structures_to_batch(structures).items()}

    def structure_batch(self, amplitudes, lattice_log_scale, *, base=None, cells=None):
        base = self.base if base is None else base
        cartesian = torch.einsum("fm,msd->fsd", amplitudes, self.tensor(self.basis))
        if cells is None:
            cells = np.stack([Lattice.orthorhombic(*cell).matrix for cell in self.data["lattice"]])
        fractional = torch.einsum("fsd,fde->fse", cartesian, torch.linalg.inv(self.tensor(cells)))
        return {**base, "frac_coords": (base["frac_coords"].reshape(-1, 48, 3) + fractional).reshape(-1, 3),
                "lengths": base["lengths"] * torch.exp(lattice_log_scale)}

    def extras(self, width="separate"):
        result = {"lattice": self.observed.new_zeros((42, 3)), "zero": self.observed.new_zeros(42)}
        if width == "separate":
            fraction = self.config["gaussian_start"] / self.config["gaussian_max"]
            result.update(gaussian=self.observed.new_full((42,), np.log(fraction / (1 - fraction))),
                          lorentzian=self.observed.new_zeros(42))
        else:
            result["width"] = self.observed.new_zeros(42)
        return result

    def terms(self, extras):
        limits = self.config["limits"]
        result = {"lattice_log_scale": limits["lattice_log_scale"] * torch.tanh(extras["lattice"]),
                  "zero_shift_deg": limits["zero_shift_deg"] * torch.tanh(extras["zero"])}
        if "gaussian" in extras:
            result["gaussian_fwhm_deg"] = self.config["gaussian_max"] * torch.sigmoid(extras["gaussian"])
        for key in ("width", "lorentzian"):
            if key in extras:
                result[key + "_log_scale"] = limits["width_log_scale"] * torch.tanh(extras[key])
        return result

    def intensities(self, batch, *, instrument=None):
        reflection = self.model.compute_reflection_layer_from_hkls(batch, self.hkls)
        scattering = self.model.compute_scattering_layer(batch, reflection, scattering_backend=self.config["scattering_backend"])
        correction = self.model.compute_correction_layer(
            reflection, scattering, wavelength=self.config["wavelength"],
            instrument=self.config["instrument"] if instrument is None else instrument, spacegroup_symbol="P1"
        )
        ratio = correction.corrected_intensities[:, self.members[0]].sum(1) / correction.corrected_intensities[:, self.members[1]].sum(1)
        return reflection, correction, ratio

    def render(self, amplitudes, extras, *, normalize=True):
        terms = self.terms(extras)
        batch = self.structure_batch(amplitudes, terms["lattice_log_scale"])
        instrument = dict(self.config["instrument"])
        instrument["Zero"] += terms["zero_shift_deg"]
        if "gaussian_fwhm_deg" in terms:
            instrument.update(U=0., V=0., W=CW_PROFILE_WIDTH_FLOOR_CENTIDEG + (100 * terms["gaussian_fwhm_deg"] / 2.35482004503)**2)
            lorentzian = torch.exp(terms["lorentzian_log_scale"])
        else:
            for key in ("U", "V", "W"):
                instrument[key] *= torch.exp(2 * terms["width_log_scale"])
            lorentzian = torch.exp(terms["width_log_scale"])
        for key in ("X", "Y", "Z"):
            instrument[key] *= lorentzian
        reflection, correction, _ = self.intensities(batch, instrument=instrument)
        # The profile refines peak shifts; the family-area ratio uses the fixed instrument.
        _, _, ratio = self.intensities(batch)
        pattern, hkls = self.model.apply_compiled_reflection_contract_to_pattern(
            self.contract, correction.two_thetas, correction.corrected_intensities
        )
        profile = self.model.compute_tch_profile_layer_from_pattern(
            pattern, instrument=instrument, wavelength=self.config["wavelength"], peak_hkls=hkls,
            cell_matrix=reflection.cell_matrix, spacegroup_symbol="P1", domain=self.domain,
            num_steps=len(self.domain), peak_chunk_size=self.config["peak_chunk_size"], return_diagnostics=False,
        ).histogram
        return (profile / self.normalization if normalize else profile), ratio

    def windows(self, indices):
        columns = [j for i in indices for j in (2 * i, 2 * i + 1)]
        return torch.as_tensor(np.isin(self.owner, indices), device=self.device), self.background[:, :, columns]

    def decode(self, raw, kind):
        if kind == "acaa":
            return self.observed.new_zeros((42, 3))
        if kind == "joint":
            raw = torch.cat([spline.amplitudes(raw[i]) for i, spline in enumerate(self.splines)])
        return radial_bound(raw, self.config["radius"])

    def initial(self, kind, scale):
        raw = inverse_radial_bound(scale * self.external, self.config["radius"])
        if kind == "joint":
            raw = np.stack([np.linalg.lstsq(spline.design_matrix, raw[part], rcond=None)[0].T
                            for spline, part in zip(self.splines, self.parts, strict=True)])
        elif kind == "acaa":
            raw = np.zeros(1)
        return self.tensor(raw).contiguous()

    def objective(self, case):
        indices = [i for i in self.config["fit_windows"] if not (case["omit_doublet"] and i == self.config["doublet_window"])]
        mask, background = self.windows(indices)
        if case["omit_doublet"] and case["ratio_weight"]:
            raise ValueError("an omitted doublet cannot contribute its measured ratio")

        def evaluate(amplitudes, extras):
            calculated, ratio = self.render(amplitudes, extras)
            profile, diagnostics = profile_objective(self.observed, calculated, self.variance, background, mask)
            ratio_terms = ((ratio - self.tensor(self.data["ratio"])) / self.tensor(self.data["ratio_sigma"]))**2
            diagnostics.update(profile=profile, calculated=calculated, ratio=ratio, ratio_terms=ratio_terms)
            return profile + case["ratio_weight"] * ratio_terms.sum(), diagnostics

        return evaluate

    def predict_window(self, calculated, scale, index):
        mask, background = self.windows([index])
        weight = mask.to(calculated.dtype) * torch.rsqrt(self.variance)
        design = background * weight.unsqueeze(-1)
        residual = (self.observed - scale[:, None] * calculated) * weight
        coefficients = torch.linalg.lstsq(design, residual.unsqueeze(-1)).solution
        prediction = scale[:, None] * calculated + (background @ coefficients).squeeze(-1)
        deviance = (((prediction - self.observed)**2 / self.variance) * mask).sum(1) / mask.sum(1)
        return prediction, deviance, mask
