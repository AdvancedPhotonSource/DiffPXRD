"""Pure rendering functions retained from the accepted Paper 1 figures.

Functions are grouped by figure. Input preparation and evidence access belong
in figures.py and the separately deposited, checksummed plotting packet.
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    GREEN,
    GRID,
    INDIGO,
    INK,
    LIGHTBLUE,
    MUTED,
    RED,
    SLATE,
    TIMES,
    apply_style,
    chip_legend,
    flat_atoms,
    panel_label,
)
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

# Figure 2: numerical agreement and computational cost.

def _clean_glyph_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_error_scale(ax: plt.Axes, values, axis: str = "x") -> None:
    """Keep exact zero errors visible rather than clipping them on a log axis."""
    values = np.asarray(values, dtype=float)
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if np.any(values == 0):
        positive = values[values > 0]
        setter("symlog", linthresh=float(positive.min()) / 10 if positive.size else 1e-12)
        ticks, labels = [0.0], ["0"]
        if positive.size:
            decades = np.unique(np.rint(np.linspace(
                np.floor(np.log10(positive.min())), np.ceil(np.log10(positive.max())), 3,
            )).astype(int))
            ticks.extend(10.0 ** decades)
            labels.extend(f"$10^{{{decade}}}$" for decade in decades)
        error_axis = ax.xaxis if axis == "x" else ax.yaxis
        error_axis.set_ticks(ticks, labels=labels)
        error_axis.set_ticks([], minor=True)
        if axis == "x":
            ax.set_xlim(left=0)
        else:
            ax.set_ylim(bottom=0)
    else:
        setter("log")


def project_unit_cell(structure: dict[str, Any]) -> dict[str, np.ndarray]:
    """Orthographic a-c-plane projection of atoms and all twelve cell edges."""
    lattice = np.asarray(structure["lattice_matrix"], dtype=float)
    frac = np.asarray(structure["frac_coords"], dtype=float)
    if (lattice.shape != (3, 3) or frac.ndim != 2 or frac.shape[1] != 3
            or not np.isfinite(lattice).all() or not np.isfinite(frac).all()
            or abs(np.linalg.det(lattice)) < 1e-12):
        raise ValueError("Invalid specimen geometry")
    horizontal = lattice[0] / np.linalg.norm(lattice[0])
    vertical = lattice[2] - np.dot(lattice[2], horizontal) * horizontal
    vertical /= np.linalg.norm(vertical)
    basis = np.column_stack((horizontal, vertical))
    vertices = np.asarray(list(itertools.product((0, 1), repeat=3)))
    edges = np.asarray([(i, j) for i in range(8) for j in range(i + 1, 8)
                        if np.abs(vertices[i] - vertices[j]).sum() == 1])
    return {"atoms": (frac % 1.0) @ lattice @ basis, "corners": vertices @ lattice @ basis,
            "edges": edges, "basis": basis, "depth": (frac % 1.0) @ lattice @ np.cross(horizontal, vertical)}


def draw_accuracy_panel(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Full saved profile, complete population, and the actual maximum-error ICSD cell."""
    arrays = data["arrays"]
    worst_summary = data["worst"]["record"]
    completed = [row for row in data["rows"] if row["status"] == "completed"]
    unsupported_count = sum(row["status"] == "unsupported" for row in data["rows"])
    balanced_errors = np.asarray([row["profile_nrmse"] for row in completed])
    denominator = len(data["rows"])
    if len(completed) + unsupported_count != denominator or not np.isfinite(balanced_errors).all():
        raise ValueError("Accuracy distribution requires explicit support accounting without finite-value filtering")
    x = arrays["two_theta_deg"]
    target = arrays["gsasii_profile"]
    calculated = arrays["diffpxrd_profile"]
    scale = max(float(np.max(np.abs(target))), 1.0e-12)
    ax.plot(x, calculated / scale, color=INDIGO, linewidth=1.5, zorder=2)
    ax.plot(x, target / scale, color=SLATE, linewidth=0.7, zorder=3)
    offset = -0.24
    ax.axhline(offset, color=GRID, linewidth=0.5)
    ax.plot(
        x,
        arrays["residual"] / scale + offset,
        color=MUTED,
        linewidth=0.5,
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    # Headroom above the strongest peak keeps the specimen, its label, and the
    # error-distribution inset clear of the profiles.
    low = min(-0.37, float(np.min(arrays["residual"] / scale + offset)) - 0.08,
              float(np.min(calculated / scale)) - 0.08, float(np.min(target / scale)) - 0.08)
    high = max(1.0, float(np.max(calculated / scale)), float(np.max(arrays["residual"] / scale + offset)))
    ax.set_ylim(low, high + 0.72 * max(high, 1.0))
    ax.set_xlabel("2θ [deg]")
    ax.set_yticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_ylabel("Normalized intensity")
    direct_label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.82,
        "pad": 0.45,
    }
    ax.text(
        0.335,
        0.47,
        "diffPXRD",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.5,
        color=INDIGO,
        bbox=direct_label_box,
    )
    ax.text(
        0.335,
        0.39,
        "GSAS-II",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.5,
        color=SLATE,
        bbox=direct_label_box,
    )
    ax.text(0.97, 0.10, "difference, same scale", transform=ax.transAxes, ha="right", va="center", fontsize=6.5, color=MUTED)
    # Logarithmic bins retain the full error range without crowding small errors.
    distribution_ax = ax.inset_axes([0.72, 0.68, 0.26, 0.30])
    sorted_errors = np.sort(np.asarray(balanced_errors, dtype=float))
    positive = sorted_errors[sorted_errors > 0]
    if not len(positive):
        raise ValueError("The logarithmic error distribution requires positive errors")
    bins = np.geomspace(positive.min() * .95, positive.max() * 1.05, 19)
    if np.any(sorted_errors == 0):
        bins = np.r_[0, bins]
    distribution_ax.hist(sorted_errors, bins=bins, color=to_rgba(INDIGO, .40),
                         edgecolor=INDIGO, linewidth=.4)
    distribution_ax.text(
        0.04,
        0.86,
        f"{data['branch_summary']['passes_all_gates']:,}/{denominator:,} pass"
        + (f"\n{unsupported_count} unsupported (31m strain)" if unsupported_count else ""),
        transform=distribution_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        color=INDIGO,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3},
        zorder=4,
    )
    set_error_scale(distribution_ax, sorted_errors)
    distribution_ax.margins(y=.25)
    distribution_ax.set_xlabel("Profile error", fontsize=6.5, labelpad=1)
    distribution_ax.set_ylabel("Comparisons", fontsize=6.5, labelpad=1)
    distribution_ax.tick_params(axis="x", labelsize=6.5, pad=1)
    distribution_ax.tick_params(axis="y", labelsize=6.5, pad=1)
    distribution_ax.grid(False)
    distribution_ax.set_facecolor("white")
    for side in ("right", "top"):
        distribution_ax.spines[side].set_visible(False)
    failed = np.asarray([not row["passes_all_gates"] for row in completed])[np.argsort(balanced_errors)]
    if failed.any():
        distribution_ax.hist(sorted_errors[failed], bins=bins, color=RED, alpha=.65)

    # Embedded specimen: the balanced sweep's largest-error structure.
    worst_structure = data["structure"]
    projected = project_unit_cell(worst_structure)
    specimen_ax = ax.inset_axes([0.34, 0.66, 0.105, 0.32])
    _clean_glyph_axis(specimen_ax)
    elements = sorted(set(worst_structure["species"]))
    specimen_colors = {element: (RED, LIGHTBLUE, "#B9C0CA", SLATE)[index % 4] for index, element in enumerate(elements)}
    for left, right in projected["edges"]:
        segment = projected["corners"][[left, right]]
        specimen_ax.plot(segment[:, 0], segment[:, 1], color=SLATE, linewidth=0.55, zorder=1)
    radius = 0.035 * float(np.linalg.norm(np.asarray(worst_structure["lattice_matrix"]), axis=1).min())
    for index in np.argsort(projected["depth"]):
        position = projected["atoms"][index]
        flat_atoms(specimen_ax, float(position[0]), float(position[1]), radius,
                     specimen_colors[worst_structure["species"][index]], zorder=2)
    bounds = np.vstack((projected["atoms"], projected["corners"]))
    specimen_ax.set_xlim(bounds[:, 0].min() - 2 * radius, bounds[:, 0].max() + 2 * radius)
    specimen_ax.set_ylim(bounds[:, 1].min() - 2 * radius, bounds[:, 1].max() + 2 * radius)
    specimen_ax.set_aspect("equal", adjustable="box")
    formula = re.sub(r"(\d+)", lambda match: "$_" + (match[0] if len(match[0]) == 1 else "{" + match[0] + "}") + "$",
                     data["worst"]["selection"]["formula"])
    specimen_label = f"{formula}\nNRMSE {worst_summary['profile_nrmse']:.5f}"
    if data.get("profile_label"):
        specimen_label += "\n" + data["profile_label"]
    ax.text(
        0.46,
        0.85,
        specimen_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        color=MUTED,
    )
    for column, (name, color) in enumerate(specimen_colors.items()):
        chip_x = 0.46 + 0.05 * (column % 4)
        chip_y = 0.69 - 0.07 * (column // 4)
        ax.scatter(
            [chip_x],
            [chip_y],
            s=14,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
            zorder=7,
        )
        ax.text(
            chip_x + 0.011,
            chip_y,
            name,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            color=MUTED,
            zorder=7,
        )
    panel_label(ax, "a", x=-0.088)


# Figure 3: NCA lattice evolution and count-retention comparison.

def _two_theta_003(c_ang: np.ndarray, wavelength_ang: float) -> np.ndarray:
    arg = 3.0 * float(wavelength_ang) / (2.0 * np.asarray(c_ang, dtype=np.float64))
    return np.degrees(2.0 * np.arcsin(np.clip(arg, -1.0, 1.0)))


def plot_measured_series(
    ax: plt.Axes,
    aggregate: dict[str, Any],
    trajectories: dict[str, np.ndarray],
) -> plt.Axes:
    x = np.asarray(trajectories["two_theta_deg"], dtype=np.float64)
    scans = np.asarray(trajectories["scan_numbers"], dtype=np.float64)
    counts = np.asarray(trajectories["raw_counts"], dtype=np.float64)

    ax.grid(False)
    measured_ramp = LinearSegmentedColormap.from_list(
        "measured_slate",
        ["#FFFFFF", "#D7DAE0", "#9AA3AE", SLATE],
    )
    image = ax.imshow(
        np.log1p(np.clip(counts, 0.0, None)),
        origin="lower",
        aspect="auto",
        extent=(float(x.min()), float(x.max()), float(scans.min()), float(scans.max())),
        cmap=measured_ramp,
        interpolation="nearest",
        rasterized=True,
    )
    overlay_z = 10

    wavelength = float(aggregate["science"]["config"]["wavelength"])
    independent_tt = _two_theta_003(trajectories["original_independent_c"], wavelength)
    spline_tt = _two_theta_003(trajectories["original_capacity_state_spline_c"], wavelength)
    outline = [pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]
    ax.plot(
        independent_tt[::4],
        scans[::4],
        linestyle="none",
        marker="s",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor=GREEN,
        markeredgewidth=0.7,
        label="framewise",
        zorder=overlay_z,
    )
    ax.plot(
        spline_tt,
        scans,
        color=INDIGO,
        linewidth=1.5,
        label="joint refinement",
        path_effects=outline,
        zorder=overlay_z + 1,
    )
    ax.set_xlabel(r"$2\theta$ [deg]")
    ax.set_ylabel("Measurement index")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(scans.min()), float(scans.max()))
    chip_legend(
        ax,
        loc="lower left",
        bbox_to_anchor=(0.14, 1.005),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=0.8,
    )
    cbar = ax.figure.colorbar(image, ax=ax, pad=0.014, fraction=0.042)
    cbar.set_label("log(1 + recorded intensity)")
    cbar.ax.tick_params(labelsize=7)
    return cbar.ax


def plot_branch_markers(ax: plt.Axes, trajectories: dict[str, np.ndarray]) -> None:
    """Charge and discharge arrows along the frame axis of the measured map."""
    scans = np.asarray(trajectories["scan_numbers"], dtype=np.float64)
    state = np.asarray(trajectories["state"]).astype(str)
    x = np.asarray(trajectories["two_theta_deg"], dtype=np.float64)
    span = float(x.max() - x.min())
    x_arrow = float(x.max()) - 0.05 * span
    for branch, label in (("C", "charge"), ("D", "discharge")):
        index = np.flatnonzero(state == branch)
        if index.size == 0 or np.any(np.diff(index) != 1):
            raise ValueError(f"the {label} branch is not one contiguous block of frames")
        low, high = float(scans[index[0]]), float(scans[index[-1]])
        ax.annotate(
            "",
            xy=(x_arrow, high),
            xytext=(x_arrow, low),
            arrowprops={"arrowstyle": "-|>", "color": SLATE, "linewidth": 0.8,
                        "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6},
            zorder=12,
        )
        ax.text(
            x_arrow - 0.03 * span,
            0.5 * (low + high),
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=6.5,
            color=SLATE,
            zorder=12,
        )


def plot_capacity_loop(ax: plt.Axes, trajectories: dict[str, np.ndarray]) -> dict[str, float]:
    """Layer spacing against capacity: the shared spline and framewise values on both branches."""
    capacity = np.asarray(trajectories["capacity"], dtype=np.float64)
    state = np.asarray(trajectories["state"]).astype(str)
    framewise = np.asarray(trajectories["original_independent_c"], dtype=np.float64)
    spline = np.asarray(trajectories["original_capacity_state_spline_c"], dtype=np.float64)
    if not (np.isfinite(capacity).all() and np.isfinite(framewise).all() and np.isfinite(spline).all()):
        raise ValueError("Invalid NCA capacity or trajectory arrays")
    if set(np.unique(state)) != {"C", "D"}:
        raise ValueError("The NCA series must hold one charge and one discharge branch")
    charge_maximum = float(capacity[state == "C"].max())
    x = capacity / charge_maximum
    for branch in ("C", "D"):
        mask = state == branch
        ax.plot(
            x[mask],
            framewise[mask],
            linestyle="none",
            marker="s",
            markersize=2.4,
            markerfacecolor="white",
            markeredgecolor=GREEN,
            markeredgewidth=0.6,
            label="framewise" if branch == "C" else None,
            zorder=2,
        )
        ax.plot(
            x[mask],
            spline[mask],
            color=INDIGO,
            linewidth=1.5,
            label="joint refinement" if branch == "C" else None,
            zorder=3,
        )
    charge, discharge = state == "C", state == "D"
    # Label each branch at a point where the two curves are apart.
    xc, yc = x[charge], spline[charge]
    xd, yd = x[discharge], spline[discharge]
    i = int(np.argmin(np.abs(xc - 0.30)))
    j = int(np.argmin(np.abs(xd - 0.78)))
    ax.annotate("charge \u2192", xy=(xc[i], yc[i]), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color=SLATE)
    ax.annotate("\u2190 discharge", xy=(xd[j], yd[j]), xytext=(-30, -12), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, color=SLATE)
    ax.set_xlabel("Capacity / charge maximum")
    ax.set_ylabel("$c$ [\N{LATIN CAPITAL LETTER A WITH RING ABOVE}]")
    ax.set_xlim(-0.02, 1.02)
    chip_legend(ax, loc="lower left", handlelength=1.4, borderaxespad=0.4, fontsize=6.5)
    hysteresis = np.interp(xd, xc, yc) - yd
    return {"charge_maximum": charge_maximum,
            "discharge_capacity_fraction_range": [float(xd.min()), float(xd.max())],
            "spline_hysteresis_angstrom_max_abs": float(np.max(np.abs(hysteresis)))}


def _trajectory_stack(
    trajectories: dict[str, np.ndarray],
    fraction: str,
    method: str,
    replicate_count: int = 5,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                trajectories[
                    f"exposure_{fraction}_replicate_{replicate}_{method}"
                ],
                dtype=np.float64,
            )
            for replicate in range(replicate_count)
        ],
        axis=0,
    )


def plot_trajectory_robustness(
    ax: plt.Axes,
    trajectories: dict[str, np.ndarray],
    *,
    fraction: str,
    label: str,
    legend: bool = False,
    replicate_count: int = 5,
) -> float:
    frames = np.asarray(trajectories["scan_numbers"], dtype=np.float64)
    framewise = _trajectory_stack(trajectories, fraction, "independent_c", replicate_count)
    joint = _trajectory_stack(
        trajectories,
        fraction,
        "capacity_state_spline_c",
        replicate_count,
    )
    full_count = np.asarray(
        trajectories["original_capacity_state_spline_c"],
        dtype=np.float64,
    )

    ax.fill_between(
        frames,
        np.min(framewise, axis=0) - full_count,
        np.max(framewise, axis=0) - full_count,
        color=GREEN,
        linewidth=0.0,
        alpha=0.15,
    )
    for replicate in framewise:
        ax.plot(frames, replicate - full_count, color=GREEN, linewidth=0.5, alpha=0.45)
    ax.fill_between(
        frames,
        np.min(joint, axis=0) - full_count,
        np.max(joint, axis=0) - full_count,
        color=INDIGO,
        linewidth=0.0,
        alpha=0.25,
    )
    ax.plot(
        frames,
        np.median(joint, axis=0) - full_count,
        color=INDIGO,
        linewidth=1.5,
    )
    reference_line = ax.axhline(0, color=INK, linestyle="--", linewidth=.7,
                               label="Original-data joint reference")

    framewise_variation = float(
        np.sqrt(np.mean(np.square(np.std(framewise, axis=0))))
    )
    joint_variation = float(np.sqrt(np.mean(np.square(np.std(joint, axis=0)))))
    reduction = 1.0 - joint_variation / framewise_variation
    extent = 1.12 * max(float(np.max(np.abs(framewise - full_count))),
                        float(np.max(np.abs(joint - full_count))), .005)
    ax.set_ylim(-extent, extent)
    ax.set_ylabel(r"$\Delta c$ [$\AA$]")
    ax.set_xlabel("Measurement index")
    ax.set_xlim(float(frames.min()), float(frames.max()))
    ax.margins(y=0.04)
    if legend:
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower - 0.34 * (upper - lower), upper)
        handles = [
            Line2D([], [], color=GREEN, linewidth=0.8, alpha=0.7),
            Line2D([], [], color=INDIGO, linewidth=1.5),
            reference_line,
        ]
        ax.legend(
            handles=handles,
            labels=[
                f"Framewise, {label} counts",
                f"Joint refinement, {label} counts",
                "Original-count reference",
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            handlelength=1.2,
            columnspacing=0.7,
            borderaxespad=0.0,
            frameon=False,
            fontsize=6.5,
        )
    return reduction


def _timing_gpu_label(data_dir: Path) -> str:
    hardware = json.loads((data_dir / "hardware_manifest.json").read_text())
    if len(hardware["gpus"]) != 1 or hardware["gpus"][0]["index"] != 0:
        raise ValueError("NCA timing labels require the measured cuda:0 hardware")
    name = hardware["gpus"][0]["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("NCA timing GPU name is missing")
    return name.removeprefix("NVIDIA ").split()[0].split("-")[0]


def plot_timing(ax: plt.Axes, data_dir: Path) -> None:
    timing = pd.read_csv(data_dir / "timing_summary.csv")
    workflows = [
        "serial_independent",
        "batched_independent",
        "batched_capacity_spline",
    ]
    labels = ["serial\nframewise", "batched\nframewise", "joint\nrefinement"]
    y = np.arange(len(workflows), dtype=float)
    devices = (("cpu", INDIGO, "CPU", -0.19), ("cuda:0", GREEN, _timing_gpu_label(data_dir), 0.19))
    longest = 0.0
    for device, color, label, offset in devices:
        records = [
            timing.loc[
                timing["device"].eq(device) & timing["workflow"].eq(workflow)
            ].iloc[0]
            for workflow in workflows
        ]
        values = np.asarray([float(record["median_runtime_sec"]) for record in records])
        minimum = np.asarray([float(record["min_runtime_sec"]) for record in records])
        maximum = np.asarray([float(record["max_runtime_sec"]) for record in records])
        longest = max(longest, float(maximum.max()))
        axis_y = y + offset
        # Bars show the median; whiskers span the measured repetitions.
        ax.barh(
            axis_y,
            values,
            height=0.36,
            xerr=np.vstack((values - minimum, maximum - values)),
            color=color,
            label=label,
            error_kw={"elinewidth": 0.8, "capsize": 2.0, "ecolor": INK},
            zorder=3,
        )
        for index, value in enumerate(values):
            # Anchor past the whisker end so long repeat ranges cannot run
            # through the printed median.
            ax.annotate(
                f"{value:.1f} s",
                xy=(float(maximum[index]), axis_y[index]),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.5,
                color=INK,
            )
    ax.set_xlim(0.0, 1.25 * longest)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end time [s], lower is faster")
    ax.grid(False, axis="y")
    ax.legend(
        loc="lower right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        handlelength=1.0,
        borderaxespad=0.3,
        frameon=False,
        fontsize=6.5,
    )


# Figure 4: lithium-sulfur intensity evolution.

STEM = "figure_4"
TEST_CYCLES = (5, 6, 7, 8)
BAND_ALPHA = 0.22
FITTED_WINDOW_DEG = (30.5, 32.5)
HELDOUT_WINDOW_DEG = (25.5, 28.0)
COUNTS_RAMP = LinearSegmentedColormap.from_list(
    "counts_slate", ["#FFFFFF", "#E6E8EB", "#C1C6CC", "#8D97A2", SLATE, "#252B34"])


def paired_metric_ratios(
    metrics: pd.DataFrame, numerator: str, denominator: str
) -> np.ndarray:
    weak = metrics[np.isclose(metrics["exposure_fraction"], 0.10)]
    pivot = weak.pivot(
        index="replicate", columns="method", values="heldout_target_loss_mean"
    )
    if numerator not in pivot or denominator not in pivot:
        raise ValueError(f"missing methods for paired ratio: {numerator}, {denominator}")
    values = (pivot[numerator] / pivot[denominator]).to_numpy(dtype=float)
    if values.size != 20 or not np.all(np.isfinite(values)):
        raise ValueError("expected 20 finite paired 10% count ratios")
    return values


def _normalise(values: np.ndarray) -> np.ndarray:
    maximum = float(np.nanmax(values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("cannot normalise a nonpositive trajectory")
    return values / maximum


def _manual_area(profile_source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(
        np.rint(profile_source["manual_cycle_number"]).astype(int), TEST_CYCLES
    )
    time_h = profile_source["scan_time_h"][
        profile_source["manual_frame_index"].astype(int)[mask]
    ]
    return np.asarray(time_h, dtype=float), _normalise(
        np.asarray(profile_source["manual_peak_area"][mask], dtype=float)
    )


WINDOW_MARKS = {"Fit 200": ((31.08, "Li$_2$S 200"),),
                "Evaluate 111": ((25.92, "S"), (26.83, "Li$_2$S 111"), (27.79, "S"))}


def _plot_reflection_map(axis, colorbar_axis, profile_source, window_deg, label, boundary_h, *, time_labels):
    """Show one measured reflection window with its own explicit count scale."""
    angles = profile_source["two_theta_deg"]
    time = profile_source["scan_time_h"]
    selected = (angles >= window_deg[0]) & (angles <= window_deg[1])
    counts = profile_source["raw_counts"][:, selected]
    low, high = np.quantile(counts, [.02, .995])
    if not np.isfinite(counts).all() or high <= low:
        raise ValueError("The reflection window must have finite, varying intensities")
    image = axis.pcolormesh(angles[selected], time, counts, vmin=low, vmax=high,
                            cmap=COUNTS_RAMP, rasterized=True)
    axis.set(xlim=window_deg, ylim=(0, float(time.max())), xlabel=r"$2\theta$ [deg]")
    axis.set_yticks((0, 25, 50, 75, 100, 125))
    axis.tick_params(axis="y", labelleft=time_labels)
    if time_labels:
        axis.set_ylabel("Elapsed experiment time [h]")
    axis.grid(False)
    axis.text(.5, 1.08, label, transform=axis.transAxes, ha="center", va="bottom", fontsize=7)
    marks = axis.secondary_xaxis("top")
    marks.set_xticks([position for position, _ in WINDOW_MARKS[label]],
                     labels=[text for _, text in WINDOW_MARKS[label]])
    marks.tick_params(labelsize=6.5, length=2.5, pad=1)
    axis.axhline(boundary_h, color=INK, linestyle=(0, (3, 2)), linewidth=.7)
    if time_labels:
        for offset, text, valign in ((1.5, "Cycles 5–8", "bottom"), (-1.5, "Cycles 1–4", "top")):
            axis.text(window_deg[0] + .06, boundary_h + offset, text, va=valign, fontsize=7,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .5})
    bar = axis.figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    bar.set_ticks([low, high], labels=[f"{low:,.0f}", f"{high:,.0f}"])
    bar.set_label("Counts", fontsize=7, labelpad=1)
    bar.ax.tick_params(labelsize=7, pad=1)
    return float(low), float(high)


def _plot_original(
    axis: plt.Axes,
    trajectories: pd.DataFrame,
    profile_source: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    local = trajectories[trajectories["scenario"].eq("original")]
    evaluation = local["cycle_number"].round().astype(int).isin(TEST_CYCLES)
    # The spline is drawn dashed above the kinetic law: the two nearly coincide,
    # and that coincidence is the finding, so both must stay visible.
    styles = (
        ("framewise", "Framewise", GREEN, 0.7, "-", 2),
        ("threshold_jmak", "Kinetic model", INDIGO, 1.5, "-", 3),
        ("spline", "Monotone spline", LIGHTBLUE, 0.9, (0, (2.2, 1.6)), 4),
    )
    for method, label, color, width, linestyle, zorder in styles:
        row = local[local["method"].eq(method) & evaluation]
        axis.plot(
            row["scan_time_h"],
            _normalise(row["activity_full_equivalent"].to_numpy(dtype=float)),
            color=color,
            linewidth=width,
            linestyle=linestyle,
            zorder=zorder,
            label=label,
        )
    manual_time, manual_area = _manual_area(profile_source)
    axis.plot(
        manual_time,
        manual_area,
        "o",
        markerfacecolor="white",
        markeredgecolor=SLATE,
        markeredgewidth=0.8,
        markersize=3.5,
        label="Published Li$_2$S area",
        zorder=5,
    )
    threshold_path = local[local["method"].eq("threshold_jmak") & evaluation]
    for cycle in TEST_CYCLES:
        cycle_path = threshold_path[
            threshold_path["cycle_number"].round().astype(int).eq(cycle)
        ]
        if len(cycle_path["half_cycle"].unique()) != 2:
            raise ValueError(
                f"cycle {cycle} does not contain precipitation and disappearance halves"
            )
    axis.set_xlabel("Elapsed experiment time [h]")
    axis.set_ylabel("Li$_2$S signal / maximum")
    # dead bands: legend below zero, staggered regime labels above one
    axis.set_ylim(-0.36, 1.24)
    axis.set_yticks((0.0, 0.25, 0.50, 0.75, 1.0))
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.3,
        borderaxespad=0.0,
        frameon=False,
        fontsize=6.5,
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))


def _plot_low_count_bands(axis: plt.Axes, trajectories: pd.DataFrame,
                          profile_source: dict[str, np.ndarray]) -> None:
    weak = trajectories[np.isclose(trajectories["exposure_fraction"], 0.10)]
    weak = weak[weak["cycle_number"].round().astype(int).isin(TEST_CYCLES)]
    # Two bands keep the kinetic/framewise comparison legible. The spline
    # median remains visible and its variability is reported in the SI.
    styles = (
        ("framewise", GREEN, 0.7, "-", 1.0),
        ("threshold_jmak", INDIGO, 1.5, "-", 1.5),
        ("spline", LIGHTBLUE, 0.9, (0, (2.2, 1.6)), None),
    )
    medians: list[tuple[np.ndarray, np.ndarray, str, float, Any]] = []
    for method, color, width, linestyle, band in styles:
        local = weak[weak["method"].eq(method)]
        pivot = local.pivot(
            index="scan_time_h", columns="replicate", values="activity_full_equivalent"
        ).sort_index()
        values = pivot.to_numpy(dtype=float)
        # A fixed, method-specific full-count maximum puts both panels on the
        # same scale without normalizing away fluctuations in each realization.
        reference = trajectories.loc[trajectories.scenario.eq("original") & trajectories.method.eq(method)
                                     & trajectories.cycle_number.round().astype(int).isin(TEST_CYCLES),
                                     "activity_full_equivalent"].to_numpy(dtype=float)
        if reference.size == 0 or not np.isfinite(reference).all() or reference.max() <= 0:
            raise ValueError("Low-count normalization requires the original evaluation trajectory")
        values = values / reference.max()
        time_h = pivot.index.to_numpy(dtype=float)
        if band is not None:
            # The kinetic band is drawn above the framewise band so that it
            # stays visible inside the wider framewise range.
            lower, upper = np.quantile(values, (0.025, 0.975), axis=1)
            axis.fill_between(
                time_h,
                lower,
                upper,
                color=color,
                alpha=BAND_ALPHA,
                linewidth=0.0,
                zorder=band,
            )
        medians.append((time_h, np.median(values, axis=1), color, width, linestyle))
    labels = ("Framewise", "Kinetic model", "Monotone spline")
    for (time_h, median, color, width, linestyle), label in zip(medians, labels):
        axis.plot(
            time_h,
            median,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            zorder=2,
            label=label,
        )
    manual_time, manual_area = _manual_area(profile_source)
    axis.plot(manual_time, manual_area, "o", markerfacecolor="white", markeredgecolor=SLATE,
              markeredgewidth=0.7, markersize=3.2, label="Published Li$_2$S area", zorder=5)
    axis.legend(
        loc="lower center",
        ncol=2,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0.2,
        frameon=False,
        fontsize=6.5,
    )
    axis.set_ylim(-0.42, 1.55)
    axis.set_xlabel("Elapsed experiment time [h]")
    axis.set_ylabel("Li$_2$S signal / full-count maximum")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    axis.set_yticks((0, 0.4, 0.8, 1.2))


def _figure_qa(figure: plt.Figure) -> dict[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.get_window_extent(renderer)
    clipped: list[str] = []
    font_sizes: list[float] = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        font_sizes.append(float(artist.get_fontsize()))
        box = artist.get_window_extent(renderer)
        if not figure_box.contains(box.x0, box.y0) or not figure_box.contains(
            box.x1, box.y1
        ):
            clipped.append(artist.get_text())
    return {
        "minimum_font_size_pt": min(font_sizes),
        "clipped_text": clipped,
        "width_mm": 183.0,
    }


def render_lisulfur(
    trajectories: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    profile_source: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, float]]:
    apply_style()
    ratio_spline = paired_metric_ratios(metrics, "threshold_jmak", "spline")
    ratio_framewise = paired_metric_ratios(metrics, "threshold_jmak", "framewise")
    loss_medians = [float(np.median(ratio_spline)), float(np.median(ratio_framewise))]

    weak = metrics[np.isclose(metrics["exposure_fraction"], 0.10)]
    correlation_groups = [
        weak[weak["method"].eq(method)]["manual_peak_area_corr_evaluation"].to_numpy(
            dtype=float
        )
        for method in ("framewise", "spline", "threshold_jmak")
    ]
    correlation_medians = [float(np.median(values)) for values in correlation_groups]
    variation_ratio = float(
        summary["findings"]["weak_variability_ratio_threshold_jmak_to_framewise"]
    )

    values = {
        "loss_ratio_to_spline_median": loss_medians[0],
        "loss_ratio_to_framewise_median": loss_medians[1],
        "framewise_manual_correlation_median": correlation_medians[0],
        "spline_manual_correlation_median": correlation_medians[1],
        "physics_manual_correlation_median": correlation_medians[2],
        "physics_manual_correlation_ratio_to_framewise": (
            correlation_medians[2] / correlation_medians[0]
        ),
        "physics_variation_ratio_to_framewise": variation_ratio,
        "physics_variation_reduction_vs_framewise_pct": 100.0 * (1.0 - variation_ratio),
    }

    figure = plt.figure(figsize=(183 / 25.4, 4.75), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.965,
        bottom=0.23,
        top=0.90,
        width_ratios=(.9, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.30,
        hspace=0.44,
    )
    left = grid[:, 0].subgridspec(1, 2, wspace=.18)
    fitted_ax = figure.add_subplot(left[0, 0])
    heldout_ax = figure.add_subplot(left[0, 1])
    fitted_bar = fitted_ax.inset_axes([.1, -.17, .8, .018])
    heldout_bar = heldout_ax.inset_axes([.1, -.17, .8, .018])
    original_ax = figure.add_subplot(grid[0, 1])
    low_count_ax = figure.add_subplot(grid[1, 1])
    original = trajectories[
        trajectories["scenario"].eq("original")
        & trajectories["method"].eq("threshold_jmak")
    ]
    cycle = original["cycle_number"].round().astype(int)
    boundary_h = 0.5 * (
        float(original.loc[cycle.eq(4), "scan_time_h"].max())
        + float(original.loc[cycle.eq(5), "scan_time_h"].min())
    )
    values["fitted_window_display_counts"] = list(_plot_reflection_map(
        fitted_ax, fitted_bar, profile_source, FITTED_WINDOW_DEG, "Fit 200", boundary_h, time_labels=True))
    values["evaluation_window_display_counts"] = list(_plot_reflection_map(
        heldout_ax, heldout_bar, profile_source, HELDOUT_WINDOW_DEG, "Evaluate 111", boundary_h, time_labels=False))
    _plot_original(original_ax, trajectories, profile_source, summary)
    _plot_low_count_bands(low_count_ax, trajectories, profile_source)
    low_count_ax.text(
        0.03,
        0.96,
        "20 draws at 10% counts",
        transform=low_count_ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6.5,
    )

    # The full-height panel a needs a lower letter offset than the default,
    # or the 9 pt glyph tops out above the figure boundary.
    panel_label(fitted_ax, "a", x=-.42, y=1.012)
    for label, axis in zip("bc", (original_ax, low_count_ax), strict=True):
        panel_label(axis, label, x=-0.16)

    qa = _figure_qa(figure)
    if qa["clipped_text"]:
        raise RuntimeError(f"figure contains clipped text: {qa['clipped_text']}")
    if qa["minimum_font_size_pt"] < 6.5:
        raise RuntimeError(
            f"figure font floor violated: {qa['minimum_font_size_pt']:.2f} pt"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{STEM}.{extension}"
        with plt.rc_context({"savefig.bbox": None}):
            figure.savefig(
                path,
                dpi=600 if extension == "png" else None,
                bbox_inches=None,
            )
        outputs[extension] = str(path)
    plt.close(figure)

    return outputs, qa, values


# Figure 5: Ca3Ti2O7 collective structural evolution.

DISPLACEMENT = "structural_rms_from_acaa_angstrom"
MAP_REFLECTIONS = (("213", 20.86), ("117", 21.10))
TEMPERATURE_TICKS = (400, 600, 800)
MEASURED_RAMP = LinearSegmentedColormap.from_list(
    "measured_slate", ["#FFFFFF", "#D7DAE0", "#9AA3AE", SLATE])
ELEMENT_COLORS = {"Ca": LIGHTBLUE, "Ti": SLATE, "O": RED}
TIO_BOND_MAXIMUM_ANGSTROM = 2.6


def _tio6_bonds(
    reference_cartesian: np.ndarray,
    species: np.ndarray,
    lengths: np.ndarray,
) -> list[tuple[int, int, tuple[int, int, int]]]:
    """Ti-O pairs within the bond cutoff in the reference display state.

    Every octahedron in the 48-site cell crosses a periodic boundary, so the
    search runs over the 27 neighbouring cell translations and each bond
    records the integer shift of its oxygen image. The topology is fixed once
    from the reference so every state draws the same octahedral framework.
    """
    titanium = np.flatnonzero(species == "Ti")
    oxygen = np.flatnonzero(species == "O")
    shifts = [
        (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
    ]
    bonds: list[tuple[int, int, tuple[int, int, int]]] = []
    for ti in titanium:
        for o in oxygen:
            for shift in shifts:
                image = reference_cartesian[o] + np.asarray(shift, dtype=float) * lengths
                if float(np.linalg.norm(image - reference_cartesian[ti])) <= TIO_BOND_MAXIMUM_ANGSTROM:
                    bonds.append((int(ti), int(o), shift))
    return bonds


def _octahedron_members(
    atomic: dict[str, Any],
    bonds: list[tuple[int, int, tuple[int, int, int]]],
) -> tuple[int, list[tuple[int, tuple[int, int, int]]]]:
    """Choose one complete, central TiO6 unit shared by every square panel."""
    states = atomic["states"]
    reference_state = next(state for state in states if state["is_reference"])
    reference = reference_state["cartesian"] - reference_state["displacement"]
    lengths = np.asarray(reference_state["lengths"], dtype=float)
    bond_entries: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
    for ti, o, shift in bonds:
        bond_entries.setdefault(ti, []).append((o, shift))
    complete = {ti: entries for ti, entries in bond_entries.items() if len(entries) == 6}
    if not complete:
        best = max(len(values) for values in bond_entries.values())
        complete = {
            ti: entries for ti, entries in bond_entries.items() if len(entries) == best
        }
    cell_center = lengths / 2.0
    chosen_ti = min(
        complete,
        key=lambda ti: float(np.linalg.norm(reference[ti] - cell_center)),
    )
    return chosen_ti, complete[chosen_ti]


def thermal_order(rows):
    if set(rows.cycle) != {"heating", "cooling"}:
        raise ValueError("The thermal figure requires heating and cooling")
    return pd.concat([rows.loc[rows.cycle.eq(cycle)].sort_values(
        "temperature_k", ascending=cycle == "heating")
        for cycle in ("heating", "cooling")], ignore_index=True)


def thermal_x(rows):
    # Temperature, not row index, sets the spacing along each thermal leg of the measured map.
    low, high = rows.temperature_k.min(), rows.temperature_k.max()
    return np.where(rows.cycle.eq("heating"), rows.temperature_k - low,
                    2 * high - low - rows.temperature_k) / (high - low)


def thermal_axis(axis, rows):
    """Vertical measurement-order axis of the map: heating down to the turnaround, then cooling."""
    high = rows.temperature_k.max()
    start = rows.loc[rows.cycle.eq("heating"), "temperature_k"].min()
    end = rows.loc[rows.cycle.eq("cooling"), "temperature_k"].min()
    labels = [f"{start:.0f} K", f"{high:.0f} K", f"{end:.0f} K"]
    axis.set(yticks=[0, 1, 2], yticklabels=labels, ylim=(2.04, -0.04))
    axis.get_yticklabels()[0].set_va("top")
    axis.get_yticklabels()[-1].set_va("bottom")
    for y, label in ((.5, "Heating"), (1.5, "Cooling")):
        axis.text(-.13, y, label, transform=axis.get_yaxis_transform(),
                  rotation=90, ha="center", va="center", fontsize=7, color=SLATE)





def measured_states(measured):
    states = measured[["cycle", "scan", "temperature_k"]].drop_duplicates()
    if (states.groupby("cycle").size().to_dict() != {"cooling": 173, "heating": 173}
            or states.duplicated(["cycle", "scan"]).any()
            or not np.isfinite(states.temperature_k).all()):
        raise ValueError("The measured map must retain all 346 states with unique scan temperatures")
    return thermal_order(states)


def ratio_residual_rms(rows, column):
    """Relative RMS difference between a fitted ratio column and the measured ratio at the fitted nodes."""
    measured = rows.measured_ratio.to_numpy(dtype=float)
    fitted = rows[column].to_numpy(dtype=float)
    if not np.isfinite(measured).all() or np.any(measured <= 0) or not np.isfinite(fitted).all():
        raise ValueError("Invalid ratio values at the fitted nodes")
    return float(np.sqrt(np.mean(np.square(fitted / measured - 1.0))))


def _octahedron_views(axes, reference, fitted, ti, entries, lengths, amplification):
    """One TiO6 unit viewed along c and along the in-plane axis in which its apical oxygens move."""

    shifts = np.asarray([shift for _, shift in entries], dtype=float) * lengths
    members = [o for o, _ in entries]
    ref_o = reference[members] + shifts
    fit_o = fitted[members] + shifts
    shown_o = ref_o + amplification * (fit_o - ref_o)
    ref_ti = reference[ti]
    shown_ti = ref_ti + amplification * (fitted[ti] - ref_ti)
    order = np.argsort(np.abs(ref_o[:, 2] - ref_ti[2]))
    equatorial, apical = order[:4], order[4:]
    angles = np.arctan2(ref_o[equatorial, 1] - ref_ti[1], ref_o[equatorial, 0] - ref_ti[0])
    equatorial = equatorial[np.argsort(angles)]
    ring = np.append(equatorial, equatorial[0])
    # The side view looks along whichever in-plane axis leaves the apical motion visible.
    horizontal = int(np.argmax([np.abs(shown_o[apical, k] - ref_o[apical, k]).sum() for k in (0, 1)]))
    if not np.isfinite(shown_o).all() or apical.size != 2:
        raise ValueError("The cartoon needs a complete TiO6 unit with two apical oxygens")
    top, side = axes
    for axis in (top, side):
        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
        axis.set_anchor("C")
    top.plot(ref_o[ring, 0], ref_o[ring, 1], color=SLATE, linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    top.plot(shown_o[ring, 0], shown_o[ring, 1], color=SLATE, linewidth=1.0, zorder=1)
    top.scatter(ref_o[equatorial, 0], ref_o[equatorial, 1], s=30, facecolor="white", edgecolor=SLATE,
                linewidths=0.7, zorder=3)
    flat_atoms(top, shown_o[equatorial, 0], shown_o[equatorial, 1], 0.24, RED, zorder=4)
    flat_atoms(top, (shown_ti[0],), (shown_ti[1],), 0.28, ELEMENT_COLORS["Ti"], zorder=4)
    half = 2.6
    top.set_xlim(ref_ti[0] - half, ref_ti[0] + half)
    top.set_ylim(ref_ti[1] - half, ref_ti[1] + half)
    top.plot([0.12, 0.30], [0.10, 0.10], transform=top.transAxes, color=SLATE, linewidth=0.6)
    top.plot([0.12, 0.12], [0.10, 0.28], transform=top.transAxes, color=SLATE, linewidth=0.6)
    top.text(0.34, 0.10, "$a$", transform=top.transAxes, fontsize=6.5, va="center", color=SLATE)
    top.text(0.12, 0.32, "$b$", transform=top.transAxes, fontsize=6.5, ha="center", color=SLATE)
    axis_points = np.stack([ref_o[apical[0]], ref_ti, ref_o[apical[1]]])
    shown_axis_points = np.stack([shown_o[apical[0]], shown_ti, shown_o[apical[1]]])
    axis_points = axis_points[np.argsort(axis_points[:, 2])]
    shown_axis_points = shown_axis_points[np.argsort(shown_axis_points[:, 2])]
    side.plot(axis_points[:, horizontal], axis_points[:, 2], color=SLATE, linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    side.plot(shown_axis_points[:, horizontal], shown_axis_points[:, 2], color=SLATE, linewidth=1.0, zorder=1)
    for o in equatorial:
        side.plot((shown_ti[horizontal], shown_o[o, horizontal]), (shown_ti[2], shown_o[o, 2]), color=SLATE,
                  linewidth=0.5, alpha=0.6, zorder=1)
    side.scatter(ref_o[:, horizontal], ref_o[:, 2], s=30, facecolor="white", edgecolor=SLATE, linewidths=0.7, zorder=3)
    flat_atoms(side, shown_o[:, horizontal], shown_o[:, 2], 0.24, RED, zorder=4)
    flat_atoms(side, (shown_ti[horizontal],), (shown_ti[2],), 0.28, ELEMENT_COLORS["Ti"], zorder=4)
    side.set_xlim(ref_ti[horizontal] - half, ref_ti[horizontal] + half)
    side.set_ylim(ref_ti[2] - half, ref_ti[2] + half)
    return "ba"[horizontal]


def plot_cartoon(figure, slot, rows, cartesian, reference, species, lengths, amplification=3):
    """One TiO6 unit at the first and hottest heating nodes against the nonpolar Acaa reference."""
    reference_state = {"cartesian": reference, "displacement": np.zeros_like(reference),
                       "lengths": lengths, "is_reference": True}
    ti, entries = _octahedron_members({"states": [reference_state]}, _tio6_bonds(reference, species, lengths))
    if len(entries) != 6:
        raise ValueError("The cartoon must show a complete TiO6 unit")
    heating = np.flatnonzero(rows.cycle.eq("heating").to_numpy())
    columns = [int(heating[np.argmin(rows.temperature_k.to_numpy()[heating])]),
               int(heating[np.argmax(rows.temperature_k.to_numpy()[heating])])]
    grid = slot.subgridspec(2, 2, wspace=0.05, hspace=0.05)
    axes, display_rows, views = [], [], set()
    for column, index in enumerate(columns):
        pair = [figure.add_subplot(grid[0, column]), figure.add_subplot(grid[1, column])]
        views.add(_octahedron_views(pair, reference, cartesian[index], ti, entries, lengths, amplification))
        row = rows.iloc[index]
        pair[0].set_title(f"{row.temperature_k:.0f} K", fontsize=6.8, pad=2)
        axes.extend(pair)
        display_rows.extend({"state": column, "temperature_k": float(row.temperature_k), "cycle": row.cycle,
            "site_index": site, "element": species[site],
            **{f"reference_{name}": float(reference[site, j]) for j, name in enumerate("xyz")},
            **{f"fitted_{name}": float(cartesian[index, site, j]) for j, name in enumerate("xyz")},
            **{f"display_{name}": float(reference[site, j] + amplification * (cartesian[index, site, j] - reference[site, j]))
               for j, name in enumerate("xyz")}}
            for site in range(len(species)))
    if len(views) != 1:
        raise ValueError("The two displayed states must share one side-view axis")
    view = views.pop()
    for axis, label in zip((axes[0], axes[1]), ("rotation, along $c$", f"tilt, along ${view}$"), strict=True):
        axis.text(-0.08, 0.5, label, transform=axis.transAxes, rotation=90, ha="right", va="center",
                  fontsize=6.5, color=SLATE)
    axes[0].legend(handles=[Line2D([], [], color=SLATE, linestyle="--", marker="o", markerfacecolor="white",
                                   markeredgecolor=SLATE, markersize=3.5, label="$\\mathit{Acaa}$ reference"),
                            Line2D([], [], color=SLATE, marker="o", markerfacecolor=RED,
                                   markeredgecolor="none", markersize=3.5, label=f"fitted, motion ${TIMES}${amplification}")],
                   loc="lower center", bbox_to_anchor=(1.03, 1.18), ncol=2, frameon=False,
                   fontsize=6.5, handlelength=1.6, columnspacing=1)
    return axes, display_rows, view


def plot_measured_map(figure, axis, states, domain, measured):
    for cycle in ("heating", "cooling"):
        mask = states.cycle.eq(cycle).to_numpy()
        image = axis.pcolormesh(domain, thermal_x(states)[mask],
            np.ma.masked_invalid(np.ma.masked_less_equal(measured[mask], 0)),
            shading="nearest", cmap=MEASURED_RAMP, norm=LogNorm(
                vmin=float(np.nanmin(measured[measured > 0])), vmax=float(np.nanmax(measured))), rasterized=True)
    thermal_axis(axis, states)
    axis.set(xlabel=r"$2\theta$ [deg]", xlim=(20.7, 21.3), xticks=[20.7, 20.9, 21.1, 21.3])
    axis.grid(False)
    for name, position in MAP_REFLECTIONS:
        axis.text(position, 0.05, name, ha="center", va="top", fontsize=6.5, color=INDIGO,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4}, zorder=5)
    cax = axis.inset_axes([1.04, 0.10, 0.025, 0.80])
    colorbar = figure.colorbar(image, cax=cax)
    ticks = np.geomspace(image.norm.vmin, image.norm.vmax, 3)
    colorbar.set_ticks(ticks, labels=[f"{value / 1e4:.1f}" for value in ticks])
    colorbar.set_label(r"Intensity [$10^4$ a.u.]", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.minorticks_off()
    return cax


def plot_descriptor(axis, rows, metric, ylabel, *, legend=False):
    for column in (f"joint_{metric}", f"framewise_{metric}"):
        if column not in rows or not np.isfinite(rows[column]).all():
            raise ValueError(f"The fitted descriptor {column} is missing")
    for cycle, color in BRANCH_COLORS.items():
        part = rows.loc[rows.cycle.eq(cycle)].sort_values("temperature_k")
        axis.plot(part.temperature_k, part[f"joint_{metric}"], color=color, linewidth=1.5, zorder=3)
        axis.plot(part.temperature_k, part[f"framewise_{metric}"], linestyle="none", marker="o",
                  markersize=2.8, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8, zorder=4)
    temperature_axis(axis, rows)
    axis.set_ylabel(ylabel)
    if legend:
        axis.legend(handles=branch_handles(framewise=True), loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=1.2)


def plot_ratio(axis, measured, rows, predictions, *, legend=False, footer_space=False):
    node_keys = set(zip(rows.cycle, rows.scan, strict=True))
    is_node = np.array([(r.cycle, r.scan) in node_keys for r in measured.itertuples()])
    if is_node.sum() != 42:
        raise ValueError("Exactly 42 measured ratios must be identified as fitted nodes")
    fitted = measured.loc[is_node]
    unfitted = measured.loc[measured.fitted_interval & ~is_node]
    excluded = measured.loc[~measured.fitted_interval]
    for cycle, color in BRANCH_COLORS.items():
        part = unfitted.loc[unfitted.cycle.eq(cycle)]
        axis.errorbar(part.temperature_k, part.ratio, yerr=part.ratio_sigma, linestyle="none",
                      marker="o", markersize=1.6, markerfacecolor="white", markeredgewidth=0.4,
                      color=color, elinewidth=0.4, zorder=2)
        part = fitted.loc[fitted.cycle.eq(cycle)]
        axis.errorbar(part.temperature_k, part.ratio, yerr=part.ratio_sigma, linestyle="none",
                      marker="o", markersize=3.4, markerfacecolor=color, markeredgewidth=0.5,
                      color=color, elinewidth=0.5, zorder=4)
        part = predictions.loc[predictions.cycle.eq(cycle)].sort_values("temperature_k")
        axis.plot(part.temperature_k, part.spline_ratio, color=color, linewidth=1.15, zorder=3)
    if len(excluded):
        axis.plot(excluded.temperature_k, excluded.ratio, linestyle="none", marker="x", markersize=2.6,
                  markeredgewidth=0.5, color="#B0B0B0", zorder=1)
    temperature_axis(axis, rows)
    axis.set_ylabel(r"$I(213)\,/\,I(117)$")
    if footer_space:
        values = np.concatenate([measured.ratio.to_numpy(dtype=float), rows.joint_ratio.to_numpy(dtype=float)])
        low, high = float(values.min()), float(values.max())
        axis.set_ylim(low - 0.18 * (high - low), high + 0.12 * (high - low))
    if legend:
        axis.legend(handles=branch_handles(measured=True), loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=5, frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=1.0)
    return ratio_residual_rms(rows, "joint_ratio")


def check_text(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    artists = list(figure.texts)
    axes = list(figure.axes)
    for axis in axes:
        axes.extend(child for child in axis.child_axes if child not in axes)
        artists.extend(axis.texts)
        artists.append(axis.title)
        if axis.get_legend() is not None:
            artists.extend(axis.get_legend().get_texts())
        if not axis.axison:
            continue
        artists.extend([axis.xaxis.label, axis.yaxis.label])
        for direction, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            artists.extend(label for tick in direction.get_major_ticks()
                           if min(limits) <= tick.get_loc() <= max(limits)
                           for label in (tick.label1, tick.label2))
    artists = [a for a in artists if a.get_visible() and a.get_text().strip()]
    findings = []
    boxes = []
    for artist in artists:
        box = artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if artist.get_fontsize() < 6.49:
            findings.append({"small_text": artist.get_text(), "font_pt": artist.get_fontsize()})
        if not (figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1)):
            findings.append({"clipped_text": artist.get_text()})
        for other, previous in boxes:
            intersection = Bbox.intersection(box, previous)
            if intersection is not None and intersection.width > 1 and intersection.height > 1:
                findings.append({"overlap": [artist.get_text(), other]})
        boxes.append((artist.get_text(), box))
    return {"width_mm": float(figure.get_figwidth() * 25.4), "findings": findings}


def temperature_axis(axis, rows):
    """One temperature axis for both thermal branches, so that heating and cooling overlay."""
    low, high = float(rows.temperature_k.min()), float(rows.temperature_k.max())
    axis.set_xlim(low - 0.02 * (high - low), high + 0.02 * (high - low))
    axis.set_xlabel("Temperature [K]")


def branch_handles(*, framewise=False, measured=False):
    handles = [Line2D([], [], color=color, linewidth=1.5, label=cycle.capitalize())
               for cycle, color in BRANCH_COLORS.items()]
    if framewise:
        handles.append(Line2D([], [], color=SLATE, linewidth=1.5, label="Joint refinement"))
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3.2, markerfacecolor="white",
                              markeredgecolor=SLATE, label="Framewise refinement"))
    if measured:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3.4, color=SLATE,
                              label="Measured, 42 fitted"))
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=1.8, markerfacecolor="white",
                              markeredgecolor=SLATE, label="Measured, 301 unfitted"))
        handles.append(Line2D([], [], color=SLATE, linewidth=1.15, label="Joint refinement"))
    return handles


def transition_onsets(doublet, factor=3.0):
    """Temperatures above 1000 K where the two-peak doublet description fails, per branch."""
    result = {}
    for cycle in BRANCH_COLORS:
        part = doublet.loc[doublet.cycle.eq(cycle)]
        baseline = float(part.loc[part.temperature_k.between(313, 1000), "reduced_chi_square"].median())
        failed = part.loc[(part.temperature_k > 1000) & (part.reduced_chi_square > factor * baseline)]
        result[cycle] = {"baseline_reduced_chi_square": baseline, "threshold": factor * baseline,
                         "failed_patterns": len(failed), "lowest_k": float(failed.temperature_k.min()),
                         "highest_k": float(failed.temperature_k.max())}
    return result

BRANCH_COLORS = {"heating": GREEN, "cooling": INDIGO}


def draw_cucr2o4_panel(fig, slot, series: dict[str, Any]) -> dict[str, Any]:
    """Compare refined oxygen positions across the measured CuCr2O4 series."""
    temperatures, cells = np.asarray(series["temperatures"], float), np.asarray(series["cells"], float)
    gsas = np.asarray(series["arrays"]["gsasii_sequential_cpu"]["coordinates"], float)
    diff = np.asarray(series["arrays"]["diffpxrd_batched_gpu"]["coordinates"], float)
    if gsas.shape != (17, 3) or diff.shape != (17, 3) or cells.shape != (17, 3) or temperatures.shape != (17,):
        raise ValueError("Figure 2d requires 17 refined oxygen positions from each program")
    reference = gsas[0]
    between = float(np.max(np.linalg.norm((diff - gsas) * cells, axis=1)))
    ax = fig.add_subplot(slot)
    for index, (name, color) in enumerate(zip("xyz", (INDIGO, LIGHTBLUE, GREEN), strict=True)):
        ax.plot(temperatures, (diff[:, index] - reference[index]) * cells[:, index], color=color, linewidth=1.2)
        ax.plot(temperatures, (gsas[:, index] - reference[index]) * cells[:, index], linestyle="none", marker="o",
                markersize=3.4, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8)
    ax.axhline(0, color="#B0B0B0", linewidth=0.6, zorder=0)
    ax.set(xlabel="Temperature [K]", ylabel="O shift from 6.8 K [Å]")
    handles = [Line2D([], [], color=color, linewidth=1.2, label=f"along {name}")
               for name, color in zip(("$a$", "$b$", "$c$"), (INDIGO, LIGHTBLUE, GREEN), strict=True)]
    handles += [Line2D([], [], color=SLATE, linewidth=1.2, label="diffPXRD"),
                Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=SLATE,
                       markersize=3.4, label="GSAS-II")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7, ncol=2, handlelength=1.4,
              columnspacing=0.9)
    ax.text(0.02, 0.04, f"$R_{{\\mathrm{{wp}}}}$ within {series['max_rwp_difference_pp']:.2f} pp\n"
            f"positions within {np.ceil(between * 1e4) / 1e4:.4f} Å", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    panel_label(ax, "d", x=-0.065)
    return {"max_rwp_difference_pp": float(series["max_rwp_difference_pp"]), "max_position_difference_a": between}


def render_transition_figure(output, doublet, domain, image):
    """Every measured pattern, 296-1223 K: the doublet window and how well two peaks describe it."""
    states = doublet[["cycle", "scan", "temperature_k"]]
    onsets = transition_onsets(doublet)
    figure = plt.figure(figsize=(183 / 25.4, 120 / 25.4))
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.5), left=.075, right=.985, bottom=.1, top=.93,
                               wspace=.45, hspace=.45)
    positive = image[image > 0]
    norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    for column, cycle in enumerate(BRANCH_COLORS):
        axis = figure.add_subplot(grid[:, column])
        mask = states.cycle.eq(cycle).to_numpy()
        order = np.argsort(states.temperature_k.to_numpy()[mask])
        temperatures = states.temperature_k.to_numpy()[mask][order]
        axis.pcolormesh(domain, temperatures, np.ma.masked_invalid(np.ma.masked_less_equal(image[mask][order], 0)),
                        shading="nearest", cmap=MEASURED_RAMP, norm=norm, rasterized=True)
        limit = 1073 if cycle == "heating" else 1055
        axis.axhline(limit, color=INDIGO, linewidth=0.8, linestyle=(0, (3, 2)))
        axis.text(21.28, limit - 8, "fitted below", fontsize=6.5, color=INDIGO, va="top", ha="right")
        axis.axhspan(onsets[cycle]["lowest_k"], onsets[cycle]["highest_k"], color=BRANCH_COLORS[cycle], alpha=.18, lw=0)
        axis.set(xlabel=r"$2\theta$ [deg]", xlim=(20.7, 21.3), xticks=[20.8, 21.0, 21.2], title=cycle.capitalize())
        axis.set_ylabel("Temperature [K]")
        axis.grid(False)
        panel_label(axis, "ab"[column], x=-.3, y=1.04)
    chi = figure.add_subplot(grid[0, 2])
    width = figure.add_subplot(grid[1, 2], sharex=chi)
    for cycle, color in BRANCH_COLORS.items():
        part = doublet.loc[doublet.cycle.eq(cycle)].sort_values("temperature_k")
        chi.plot(part.temperature_k, part.reduced_chi_square, marker="o", markersize=1.6, linewidth=.6, color=color,
                 label=cycle.capitalize())
        width.plot(part.temperature_k, part.fwhm_deg, marker="o", markersize=1.6, linewidth=.6, color=color)
    for cycle, color in BRANCH_COLORS.items():
        # Each branch has its own fitted-interval end and its own failure threshold.
        for axis in (chi, width):
            axis.axvline(1073 if cycle == "heating" else 1055, color=color, linewidth=.7, linestyle=(0, (3, 2)))
        chi.axhline(onsets[cycle]["threshold"], color=color, linewidth=.6, linestyle=":")
    chi.set(yscale="log", ylabel="Two-peak reduced $\\chi^2$")
    chi.tick_params(labelbottom=False)
    chi.legend(frameon=False, fontsize=6.5, loc="upper left")
    width.set(xlabel="Temperature [K]", ylabel="Two-peak FWHM [deg]")
    panel_label(chi, "c", x=-.16, y=1.06)
    panel_label(width, "d", x=-.16, y=1.06)
    qa = check_text(figure)
    from figures import save_figure
    save_figure(figure, output, "figure_S1")
    return {"onsets": onsets, "patterns": len(states), "geometric_qa": qa,
            "claim_boundary": "Model-free view of the measured series; the single-phase fit uses 313-1073 K heating "
                              "and 313-1055 K cooling. The shaded bands mark where a two-peak description fails."}
