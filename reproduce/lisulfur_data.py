"""Read the Li-S source archive and calibrate its fixed Li2S diffraction template."""

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
import torch

from diffpxrd import ModularCWForwardModel


SOURCE_URL = "https://zenodo.org/api/records/3514967/files/data.zip/content"
SOURCE_SHA256 = "e7e738408dbbf52eea2c7bba41b24953ae1c0788dab87bc10dd9ac01451aeb7a"


@dataclass
class Sequence:
    two_theta_deg: np.ndarray
    raw_counts: np.ndarray
    scan_time_h: np.ndarray
    half_cycle: np.ndarray
    cycle_number: np.ndarray
    capacity_mah: np.ndarray
    voltage_v: np.ndarray
    reference_frame_index: np.ndarray
    reference_peak_area: np.ndarray
    reference_fwhm_deg: np.ndarray
    reference_cycle_number: np.ndarray


def load_sequence(path: Path) -> Sequence:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError("Li-S archive differs from the published input checksum")
    pattern = re.compile(r"YC65A_op_back\.(\d{3})_R01\.xyd$")
    with zipfile.ZipFile(path) as archive:
        indexed = sorted((int(match.group(1)), name) for name in archive.namelist()
                         if (match := pattern.search(name)))
        if [index for index, _ in indexed] != list(range(1, 132)):
            raise ValueError("Expected the complete ordered 131-pattern sequence")
        arrays = [np.loadtxt(archive.open(name)) for _, name in indexed]
        grid = np.asarray(arrays[0][:, 0], dtype=np.float64)
        if any(array.shape != arrays[0].shape or not np.array_equal(array[:, 0], grid) for array in arrays):
            raise ValueError("Li-S patterns do not share an exact grid")
        counts = np.stack([array[:, 1] for array in arrays]).astype(np.float64)
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.max(np.abs(counts - np.rint(counts))) > 1e-6:
            raise ValueError("Li-S source must contain finite nonnegative integer counts")
        raw = archive.read("data/operando_cell.mpt")
        match = re.search(rb"Nb header lines\s*:\s*(\d+)", raw[:2048])
        if match is None:
            raise ValueError("Electrochemistry header length is missing")
        electrochem = pd.read_csv(io.BytesIO(raw), sep="\t", skiprows=int(match.group(1)) - 1,
                                 decimal=",", encoding="latin1")
        manual = pd.read_csv(archive.open("data/peakarea.txt"), sep="\t", decimal=",")
    scan_time = np.arange(1, 132, dtype=np.float64) - 0.5
    electrochem_time = electrochem["time/s"].to_numpy(dtype=np.float64) / 3600.0

    def interpolate(column):
        return np.interp(scan_time, electrochem_time, electrochem[column].to_numpy(dtype=np.float64))

    manual = manual.dropna(subset=["time", "peak_area", "fwhm"])
    manual = manual[manual["time"].between(scan_time[0], scan_time[-1])]
    manual_time = manual["time"].to_numpy(dtype=np.float64)
    sequence = Sequence(
        grid, counts, scan_time, interpolate("half cycle"), interpolate("cycle number"),
        interpolate("(Q-Qo)/mA.h"), interpolate("Ewe/V"),
        np.rint(manual_time + 0.5).astype(np.int64) - 1,
        manual["peak_area"].to_numpy(dtype=np.float64), manual["fwhm"].to_numpy(dtype=np.float64),
        np.rint(np.interp(manual_time, electrochem_time,
                         electrochem["cycle number"].to_numpy(dtype=np.float64))).astype(np.int64),
    )
    for labels in (sequence.cycle_number, sequence.half_cycle):
        if not np.allclose(labels, np.rint(labels), atol=1e-8):
            raise ValueError("Interpolated cycle labels are not integer states")
    if set(np.rint(sequence.cycle_number).astype(int)) != set(range(1, 9)):
        raise ValueError("Frames must cover cycles 1 through 8")
    return sequence


def window(sequence, bounds):
    mask = (sequence.two_theta_deg >= bounds[0]) & (sequence.two_theta_deg <= bounds[1])
    if mask.sum() < 20:
        raise ValueError(f"Too few points in reflection window {bounds}")
    return sequence.two_theta_deg[mask], sequence.raw_counts[:, mask]


def residualize(grid, counts):
    scaled = (grid - float(np.mean(grid))) / float(np.ptp(grid))
    basis, _ = np.linalg.qr(np.column_stack((np.ones_like(grid), scaled)))
    return counts - (counts @ basis) @ basis.T, basis


def amplitudes(observations, template):
    denominator = float(template @ template)
    if denominator <= 1e-12:
        raise ValueError("Diffraction template is confounded with the linear background")
    return np.maximum((observations @ template) / denominator, 0.0)


def calibrate_template(reference_x, reference_counts, target_x, target_counts, calibration, sigma, device):
    reference_y, reference_q = residualize(reference_x, reference_counts)
    target_y, target_q = residualize(target_x, target_counts)
    model = ModularCWForwardModel(q_min=4.0 * math.pi * math.sin(math.radians(10.0)) / 1.5406,
                                  q_max=4.0 * math.pi * math.sin(math.radians(17.5)) / 1.5406,
                                  wavelength=1.5406, device=device)
    # Preserve the source analysis precision: float32 template, float64 shape fits.
    domain = torch.as_tensor(np.concatenate((target_x, reference_x)), dtype=torch.float32, device=device)
    best = None
    with torch.no_grad():
        for lattice_a in np.linspace(5.60, 5.95, 15):
            structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(float(lattice_a)),
                                                  ["S", "Li"], [[0, 0, 0], [0.25, 0.25, 0.25]])
            batch = model.structure_to_batch(structure)
            reflection = model.compute_reflection_layer(batch)
            scattering = model.compute_scattering_layer(batch, reflection)
            instrument = {"Zero": 0.0, "U": 0.0, "V": 0.0, "W": (100.0 * float(sigma)) ** 2,
                          "X": 0.0, "Y": 0.0, "Z": 0.0}
            correction = model.compute_correction_layer(reflection, scattering, instrument=instrument)
            profile = model.compute_tch_profile_layer(correction, instrument=instrument, domain=domain)
            raw = profile.histogram[0].cpu().numpy().astype(np.float64)
            target = raw[:target_x.size] - target_q @ (target_q.T @ raw[:target_x.size])
            reference = raw[target_x.size:] - reference_q @ (reference_q.T @ raw[target_x.size:])
            combined = np.concatenate((target, reference))
            combined_y = np.concatenate((target_y, reference_y), axis=1)
            scale = amplitudes(combined_y[calibration], combined)
            loss = float(np.mean(np.square(combined_y[calibration] - scale[:, None] * combined[None, :])))
            if best is None or loss < best["calibration_loss"]:
                best = {"lattice_a_angstrom": float(lattice_a), "calibration_loss": loss,
                        "reference_template": reference, "target_template": target,
                        "reference_y": reference_y, "target_y": target_y}
    reference_scale = amplitudes(reference_y, best["reference_template"])
    target_scale = amplitudes(target_y, best["target_template"])
    denominator = float(reference_scale[calibration] @ reference_scale[calibration])
    best["target_scale"] = max(float(reference_scale[calibration] @ target_scale[calibration]) /
                               max(denominator, 1e-12), 0.0)
    return best


def capacity_progress(half_cycle, capacity):
    labels = np.rint(half_cycle).astype(np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, labels.size]
    progress = np.empty(labels.size, dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment = np.asarray(capacity[start:stop], dtype=np.float64)
        monotone = np.maximum.accumulate(np.abs(segment - segment[0]))
        span = float(monotone[-1])
        if span <= 1e-10:
            raise ValueError("Capacity does not change within a half-cycle")
        progress[start:stop] = monotone / span
    return progress


def sample_counts(counts, fraction, replicate, seed):
    if fraction == 1.0:
        return np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(int(seed) + int(round(1000.0 * fraction)) + 10007 * int(replicate))
    return rng.binomial(np.rint(counts).astype(np.int64), fraction).astype(np.float64)
