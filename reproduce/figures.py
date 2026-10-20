"""Render Figures 1-5 and S1 from deposited source data.

No refinement or benchmark runs are performed. See docs/figures.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figure_panels as panels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import GREEN, INDIGO, apply_style, panel_label
from matplotlib.colors import to_rgba
from prepare_ca3ti2o7_neutron import load_prepared


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class PlottingPacket:
    def __init__(self, root):
        self.root = root.resolve()
        manifest = json.loads((self.root / "figures.json").read_text())
        if manifest.get("schema") != "diffpxrd-paper1-plotting-inputs-v2":
            raise ValueError("unsupported plotting packet")
        self.files = manifest["files"]
        self.consumed = {}

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or name not in self.files:
            raise ValueError(f"undeclared plotting input: {name}")
        self.consumed[name] = self.files[name]
        return path

    def json(self, name):
        return json.loads(self.path(name).read_text())

    def table(self, name):
        return pd.read_csv(self.path(name), float_precision="round_trip")

    def arrays(self, name):
        with np.load(self.path(name), allow_pickle=False) as packet:
            return {key: packet[key].copy() for key in packet.files}


def load_icsd_structure(path):
    """Read a user-supplied ICSD 427040 cell without changing its setting or origin."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifParser

    raw = path.read_bytes()
    declared_id = None
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("ICSD JSON must contain one compact structure object")
        declared_id = payload.get("icsd_id")
        required = {"lattice_matrix", "species", "frac_coords"}
        if not required <= payload.keys():
            raise ValueError("ICSD JSON requires lattice_matrix, species, and frac_coords")
        payload = {key: payload[key] for key in required} | {"site_properties": payload.get("site_properties", {})}
    elif path.suffix.lower() == ".cif":
        parser = CifParser.from_str(raw.decode("utf-8-sig"), frac_tolerance=0.)
        blocks = parser.as_dict()
        if len(blocks) != 1:
            raise ValueError("Supply a CIF containing only ICSD entry 427040")
        block = next(iter(blocks.values()))
        declared_id = next((value for key, value in block.items()
                            if key.lower() == "_database_code_icsd"), None)
        structures = parser.parse_structures(primitive=False)
        if len(structures) != 1 or not structures[0].is_ordered:
            raise ValueError("ICSD 427040 requires one ordered, full-occupancy structure")
        structure = structures[0]
        payload = {"lattice_matrix": structure.lattice.matrix.tolist(),
                   "species": [site.specie.symbol for site in structure],
                   "frac_coords": structure.frac_coords.tolist(), "site_properties": {}}
    else:
        raise ValueError("--icsd-structure must be a compact .json or a .cif file")
    if declared_id is not None and str(declared_id).strip() != "427040":
        raise ValueError(f"Figure 2 requires ICSD entry 427040, not {declared_id}")
    lattice = np.asarray(payload["lattice_matrix"], dtype=float)
    fractional = np.asarray(payload["frac_coords"], dtype=float)
    species = payload["species"]
    if (lattice.shape != (3, 3) or fractional.shape != (80, 3)
            or not np.isfinite(lattice).all() or not np.isfinite(fractional).all()
            or abs(np.linalg.det(lattice)) < 1e-12):
        raise ValueError("ICSD 427040 requires a finite, nonsingular 80-site cell")
    if (not isinstance(species, list) or not all(isinstance(value, str) for value in species)
            or Counter(species) != {"Ho": 32, "Ru": 8, "Br": 40}):
        raise ValueError("ICSD 427040 requires 80 ordered sites: Ho32 Ru8 Br40 (Ho4RuBr5)")
    structure = Structure(Lattice(lattice), species, fractional, coords_are_cartesian=False)
    if structure.composition.reduced_formula != "Ho4RuBr5":
        raise ValueError("The Figure 2 structure must have formula Ho4RuBr5")
    metadata = {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "required_icsd_id": "427040", "declared_icsd_id": declared_id,
                "formula": "Ho4RuBr5", "site_count": 80,
                "identity_check": "composition and site count; database origin is supplied by the user",
                "cell_policy": "input setting and origin retained; no primitive or standard-cell conversion"}
    return payload, metadata


def save_figure(figure, output, stem, formats=("svg", "pdf", "png")):
    paths = {}
    for suffix in formats:
        path = output / f"{stem}.{suffix}"
        with plt.rc_context({"savefig.bbox": None}):
            figure.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches=None)
        paths[suffix] = path.name
    plt.close(figure)
    return paths


def render_overview(packet, output):
    source = packet.path("overview.svg")
    tree = ET.fromstring(source.read_bytes())
    identifiers = {node.get("id") for node in tree.iter() if node.get("id")}
    for node in tree.iter():
        if node.tag.rsplit("}", 1)[-1] in ("script", "foreignObject"):
            raise ValueError("the overview must be a standalone static SVG")
        for key, value in node.attrib.items():
            if key.rsplit("}", 1)[-1] == "href" and (not value.startswith("#") or value[1:] not in identifiers):
                raise ValueError("the overview contains an external or unresolved reference")
    target = output / "figure_1.svg"
    shutil.copyfile(source, target)
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError("Figure 1 PDF/PNG export requires rsvg-convert (librsvg)")
    font_config = ET.Element("fontconfig")
    ET.SubElement(font_config, "include", ignore_missing="yes").text = "/etc/fonts/fonts.conf"
    ET.SubElement(font_config, "dir").text = str(packet.root / "fonts")
    with tempfile.TemporaryDirectory(prefix="diffpxrd-figure-fonts-") as temporary:
        config = Path(temporary) / "fonts.conf"
        ET.ElementTree(font_config).write(config, encoding="utf-8", xml_declaration=True)
        for suffix in ("pdf", "png"):
            subprocess.run(["rsvg-convert", f"--format={suffix}", "--dpi-x=600", "--dpi-y=600",
                            "--output", str(output / f"figure_1.{suffix}"), str(target)],
                           env={**os.environ, "FONTCONFIG_FILE": str(config)}, check=True)
    return {suffix: f"figure_1.{suffix}" for suffix in ("svg", "pdf", "png")}


def render_engine(data, output_dir):
    accuracy = data["accuracy"]
    complete = data["end"]
    statistics = data["statistics"]
    stem = "figure_2"
    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.22, 1, 0.82), width_ratios=(0.85, 1.65))
    panels.draw_accuracy_panel(fig.add_subplot(grid[0, :]), accuracy)
    ax = fig.add_subplot(grid[1, 0])
    rng = np.random.default_rng(20260916)
    medians = {}
    for index, (engine, color) in enumerate((("diffpxrd_cpu_pymatgen", INDIGO), ("diffpxrd_gpu_pymatgen", GREEN))):
        values = complete.loc[complete.engine.eq(engine), "median_sec"].to_numpy(float)
        if len(values) != 200:
            raise ValueError("Complete-profile panel requires 200 structures on each device")
        box = ax.boxplot([values], positions=[index], widths=0.45, patch_artist=True,
                         showfliers=False, whis=(0, 100), manage_ticks=False)
        box["boxes"][0].set(facecolor=to_rgba(color, 0.20), edgecolor=color, linewidth=0.8)
        for key in ("whiskers", "caps", "medians"):
            for artist in box[key]:
                artist.set(color=color, linewidth=0.8)
        ax.scatter(index + rng.uniform(-0.16, 0.16, len(values)), values, color=color, s=6, alpha=0.2, linewidths=0)
        medians[engine] = float(np.median(values))
        ax.text(index, medians[engine], f"{medians[engine]:.3g} s", ha="center", va="bottom", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4})
    ax.set_xticks([0, 1], ["CPU", "A100"])
    ax.set_yscale("log")
    ax.set_ylabel("Complete-profile time [s]")
    panel_label(ax, "b", x=-0.24)

    ax = fig.add_subplot(grid[1, 1])
    reference = data["reference_statistics"]
    mode_names = {"forward": "profile", "wls_gradient": "loss + gradient"}
    styles = {"forward": "-", "wls_gradient": ":"}
    for device, color, name in (("cpu", INDIGO, "CPU"), ("cuda:0", GREEN, "A100")):
        for mode, mode_name in mode_names.items():
            group = statistics[statistics.device.eq(device) & statistics["mode"].eq(mode)].sort_values("batch_size")
            x = group.batch_size.to_numpy(int)
            median = group.median_seconds_per_profile.to_numpy(float)
            ax.fill_between(x, group.q25_seconds_per_profile.to_numpy(float), group.q75_seconds_per_profile.to_numpy(float),
                            color=color, alpha=0.07, linewidth=0)
            ax.plot(x, median, color=color, linestyle=styles[mode], linewidth=1.0, marker="o", markersize=2.7,
                    label=f"{name}, {mode_name}")
        # Every reflection broadened separately: the earlier refinement path, shown for reference only.
        points = reference[reference.device.eq(device) & reference["mode"].eq("forward")]
        # Offset sideways so the crosses do not sit on the merged-path markers at the same batch size.
        offset = 0.88 if device == "cpu" else 1.12
        ax.scatter(points.batch_size.to_numpy(int) * offset, points.median_seconds_per_profile.to_numpy(float),
                   color=color, marker="x", s=22, linewidths=1.0, alpha=0.85, label=f"{name}, profile, per reflection")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(data["batches"], [str(batch) for batch in data["batches"]])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Repeated time per profile [s]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
              fontsize=7, ncol=2, handlelength=1.8, columnspacing=1.2)
    panel_label(ax, "c", x=-0.13)
    panels.draw_cucr2o4_panel(fig, grid[2, :], data["cucr2o4"])
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if artist.get_fontsize() < 7:
            artist.set_fontsize(7)
    # Preserve the approved September 26 PDF geometry above row d. Constrained
    # layout otherwise resizes all three rows when the timing axes are removed.
    fig.set_layout_engine("none")
    boxes_pt = (
        (44.6505606891, 347.2709924942, 467.0268567912, 175.3779890621),
        (44.6505606891, 142.0959990417, 142.1665865877, 143.7524500509),
        (235.7069846925, 142.0959990417, 275.9704327878, 143.7524500509),
        (36.984615, 24.422115, 478.7553024803, 78.7820432302),
    )
    width_pt, height_pt = fig.get_size_inches() * 72
    for axis, (left, bottom, width, height) in zip(fig.axes, boxes_pt, strict=True):
        axis.set_position((left / width_pt, bottom / height_pt, width / width_pt, height / height_pt))
    return save_figure(fig, output_dir, stem, formats=("pdf", "svg", "png"))


def render_nca(packet, output):
    trajectories = packet.arrays("nca_trajectories.npz")
    aggregate = packet.json("nca_settings.json")
    reduction = packet.json("nca_summary.json")["reduction"]
    packet.path("nca_timing/timing_summary.csv")
    packet.path("nca_timing/hardware_manifest.json")
    fig = plt.figure(figsize=(183 / 25.4, 146 / 25.4))
    grid = fig.add_gridspec(3, 2, left=.085, right=.985, bottom=.09, top=.96,
        width_ratios=(1, 1.15), height_ratios=(.9, 1.25, .9), wspace=.48, hspace=.62)
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])]
    panels.plot_measured_series(axes[0], aggregate, trajectories)
    panels.plot_branch_markers(axes[0], trajectories)
    panels.plot_capacity_loop(axes[1], trajectories)
    shown = panels.plot_trajectory_robustness(axes[2], trajectories, fraction="0.100",
                                            label="10%", legend=True, replicate_count=20)
    np.testing.assert_allclose(shown, reduction, rtol=1e-12)
    lower, upper = axes[2].get_ylim()
    axes[2].set_ylim(lower, upper + .12 * (upper - lower))
    axes[2].text(.97, .98, f"{100 * reduction:.1f}% lower variability", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=7, color=INDIGO)
    panels.plot_timing(axes[3], packet.root / "nca_timing")
    axes[3].set_xlabel("Analysis time [s]")
    for label, axis in zip("abcd", axes, strict=True):
        panel_label(axis, label, x=-.17, y=1.04)
    for axis in fig.axes:
        for artist in [*axis.texts, *axis.get_xticklabels(), *axis.get_yticklabels()]:
            if artist.get_fontsize() < 7:
                artist.set_fontsize(7)
        if axis.get_legend():
            for artist in axis.get_legend().get_texts():
                artist.set_fontsize(7)
    return save_figure(fig, output, "figure_3")


def render_ca(data, output, neutron):
    rows, states, measured, predictions = (data[key] for key in ("rows", "states", "measured", "predictions"))
    domain, measured_image, cartesian, species = (
        data["arrays"][key] for key in ("domain", "measured_image", "cartesian", "species"))
    acaa = neutron["acaa_fractional"] @ neutron["neighbor_reference_lattice"]
    lengths = np.linalg.norm(neutron["neighbor_reference_lattice"], axis=1)
    figure = plt.figure(figsize=(183 / 25.4, 168 / 25.4))
    grid = figure.add_gridspec(3, 2, left=0.085, right=0.92, bottom=0.07, top=0.945,
                              height_ratios=(1.25, 0.78, 0.78), hspace=0.62, wspace=0.42)
    panels.plot_cartoon(figure, grid[0, 0], rows, cartesian, acaa, species, lengths)
    image_axis = figure.add_subplot(grid[0, 1])
    panels.plot_measured_map(figure, image_axis, states, domain, measured_image)
    motion = figure.add_subplot(grid[1, :])
    panels.plot_descriptor(motion, rows, panels.DISPLACEMENT, "Displacement [Å RMS]", legend=True)
    ratio_axis = figure.add_subplot(grid[2, :])
    panels.plot_ratio(ratio_axis, measured, rows, predictions, legend=True, footer_space=True)
    position = grid[0, 0].get_position(figure)
    figure.text(position.x0 - 0.04, position.y1 + 0.018, "a", fontsize=9, fontweight="bold")
    for letter, axis in zip("bcd", [image_axis, motion, ratio_axis], strict=True):
        panel_label(axis, letter, x=-0.15 if letter == "b" else -0.06, y=1.08)
    qa = panels.check_text(figure)
    (output / "ca_figure_geometric_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    return save_figure(figure, output, "figure_5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=["1", "2", "3", "4", "5", "S1"],
                        default=["1", "2", "3", "4", "5", "S1"])
    parser.add_argument("--icsd-structure", type=Path,
        help="User-licensed ICSD entry 427040 as compact JSON or CIF; required for Figure 2")
    parser.add_argument("--neutron-tables", type=Path,
        help="Reader-prepared NPZ for Figure 5; see docs/ca3ti2o7_neutron_tables.md")
    args = parser.parse_args()
    if "2" in args.figures and args.icsd_structure is None:
        parser.error("Figure 2 requires --icsd-structure for licensed ICSD entry 427040 (Ho4RuBr5, 80 sites). "
                     "The public packet contains identifiers only; use --figures 1 3 4 5 to render without it.")
    try:
        neutron = load_prepared(args.neutron_tables) if "5" in args.figures else None
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    packet = PlottingPacket(args.data_dir)
    for name in ("Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Italic.ttf", "OFL.txt"):
        packet.path("fonts/" + name)
    apply_style(packet.root / "fonts")
    report = {"complete": False, "new_fits": 0, "figures": {}, "inputs": {}}
    try:
        for number in args.figures:
            if number == "1":
                outputs = render_overview(packet, args.output_dir)
            elif number == "2":
                data = packet.json("engine.json")
                data["accuracy"]["structure"], report["icsd_structure"] = load_icsd_structure(args.icsd_structure)
                data["accuracy"]["arrays"] = packet.arrays("engine_profiles.npz")
                data["end"] = packet.table("engine_complete_timings.csv")
                data["statistics"] = packet.table("engine_batch_statistics.csv")
                data["reference_statistics"] = packet.table("engine_reference_statistics.csv")
                cu = packet.arrays("cucr2o4_profiles.npz")
                data["cucr2o4"] = packet.json("cucr2o4.json")
                data["cucr2o4"]["arrays"] = {arm: {key.removeprefix(arm + "__"): value
                    for key, value in cu.items() if key.startswith(arm + "__")}
                    for arm in ("gsasii_sequential_cpu", "diffpxrd_batched_gpu")}
                outputs = render_engine(data, args.output_dir)
            elif number == "3":
                outputs = render_nca(packet, args.output_dir)
            elif number == "4":
                outputs, qa, _ = panels.render_lisulfur(packet.table("lisulfur_trajectories.csv"),
                    packet.table("lisulfur_metrics.csv"), packet.json("lisulfur_summary.json"),
                    packet.arrays("lisulfur_profiles.npz"), args.output_dir)
                write_json(args.output_dir / "figure_4_qa.json", qa)
            elif number == "S1":
                display = packet.arrays("ca_transition_display.npz")
                panels.render_transition_figure(args.output_dir, packet.table("ca_doublet_fits.csv"),
                                                display["domain"], display["image"])
                outputs = {suffix: f"figure_S1.{suffix}" for suffix in ("pdf", "svg", "png")}
            else:
                data = {key: packet.table(name) for key, name in (
                    ("rows", "ca_trajectories.csv"), ("states", "ca_states.csv"),
                    ("measured", "ca_measured_ratios.csv"), ("predictions", "ca_predictions.csv"))}
                data["arrays"] = packet.arrays("ca_display.npz")
                outputs = render_ca(data, args.output_dir, neutron)
            report["figures"][str(number)] = outputs
        report["complete"] = True
    finally:
        report["inputs"] = packet.consumed
        write_json(args.output_dir / "rendering.json", report)


if __name__ == "__main__":
    main()
