"""Frozen plotting style used by the accepted Paper 1 figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib import transforms as mtransforms
from matplotlib.colors import to_rgb

INDIGO = "#6465E9"
GREEN = "#95CA42"
LIGHTBLUE = "#38BDF8"
RED = "#E14F62"
SLATE = "#495567"
INK = "#000000"
MUTED = "#808080"
GRID = "#D9D9D9"
CHIP_FACE = "#F0F0F0"

TIMES = "\N{MULTIPLICATION SIGN}"
FONT_DIR = None


def apply_style(font_dir: Path | None = None) -> None:
    """Apply the vivid rcParams contract. Call once before building a figure."""
    global FONT_DIR
    if font_dir is not None:
        FONT_DIR = Path(font_dir)
    if FONT_DIR is None:
        raise ValueError("Set the checked plotting-packet font directory before rendering")
    font_paths = [
        FONT_DIR / "Lato-Regular.ttf",
        FONT_DIR / "Lato-Bold.ttf",
        FONT_DIR / "Lato-Italic.ttf",
    ]
    missing = [path for path in font_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing manuscript fonts: {missing}")
    for path in font_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "mathtext.default": "regular",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Lato",
            "mathtext.it": "Lato:italic",
            "mathtext.bf": "Lato:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 3.0,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 0.7,
            "legend.frameon": True,
            "legend.facecolor": CHIP_FACE,
            "legend.edgecolor": "#FFFFFF",
            "legend.framealpha": 1.0,
            "legend.labelspacing": 0.25,
            "legend.borderpad": 0.35,
            "legend.handlelength": 1.3,
            "figure.dpi": 110,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.facecolor": "none",
        }
    )


def chip_legend(ax: plt.Axes, **kwargs) -> plt.legend:
    """Gray chip legend. Place inside genuine whitespace or outside the axes."""
    legend = ax.legend(**kwargs)
    frame = legend.get_frame()
    frame.set_facecolor(CHIP_FACE)
    frame.set_edgecolor("#FFFFFF")
    frame.set_linewidth(0.0)
    return legend


class _CanvasClampedAxesFraction(mtransforms.Transform):
    """Axes-fraction transform whose output never leaves the figure canvas.

    A panel letter anchored above tightly-margined axes can extend past the
    fixed-size canvas and print decapitated. Clamping the anchor at draw time
    keeps the letter's full text box inside the canvas whatever the final
    subplot layout is; anchors with enough headroom pass through unchanged.
    """

    input_dims = 2
    output_dims = 2

    def __init__(self, ax: plt.Axes, box_height_points: float) -> None:
        super().__init__()
        self._ax = ax
        self._box_height_points = box_height_points
        self.set_children(ax.transAxes)

    def transform_non_affine(self, values):
        points = self._ax.transAxes.transform(np.atleast_2d(values))
        figure = self._ax.figure
        ceiling = figure.bbox.height - self._box_height_points * figure.dpi / 72.0
        points[:, 1] = np.minimum(points[:, 1], ceiling)
        return points

    def inverted(self):
        return self._ax.transAxes.inverted()


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.045) -> None:
    """Venue panel letter: bold lowercase outside the top-left corner."""
    fontsize = 9.0
    ax.text(
        x,
        y,
        label,
        transform=_CanvasClampedAxesFraction(ax, fontsize * 1.45),
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def shaded_atoms(
    ax: plt.Axes,
    x,
    y,
    radius: float,
    color: str,
    *,
    zorder: float = 3.0,
    layers: int = 6,
) -> None:
    """Draw atoms as vector-shaded spheres in data units.

    Concentric circles run from a darkened rim to a lightened highlight
    drifting toward the upper left, giving the rendered-structure depth of
    the vivid exemplar corpus without rasterisation. The axes must hold an
    equal aspect ratio for the circles to stay round.
    """
    base = np.asarray(to_rgb(color), dtype=float)
    rim = base * 0.72
    highlight = base + (1.0 - base) * 0.72
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    ys = np.atleast_1d(np.asarray(y, dtype=float))
    for cx, cy in zip(xs, ys):
        for index in range(layers):
            t = index / (layers - 1)
            r = radius * (1.0 - 0.78 * t)
            if t < 0.4:
                shade = rim + (base - rim) * (t / 0.4)
            else:
                shade = base + (highlight - base) * ((t - 0.4) / 0.6)
            ax.add_patch(
                plt.Circle(
                    (cx - 0.30 * radius * t, cy + 0.34 * radius * t),
                    r,
                    facecolor=tuple(shade),
                    edgecolor="none",
                    zorder=zorder + 0.001 * index,
                )
            )


def flat_atoms(
    ax: plt.Axes,
    x,
    y,
    radius: float,
    color: str,
    *,
    zorder: float = 3.0,
) -> None:
    """Draw atoms as flat circles of one palette colour with a thin slate outline, in data units.

    Unlike shaded_atoms, no darker or lighter shades are introduced, so a figure keeps one shade per hue.
    The axes must hold an equal aspect ratio for the circles to stay round.
    """
    for cx, cy in zip(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))):
        ax.add_patch(plt.Circle((cx, cy), radius, facecolor=color, edgecolor=SLATE, linewidth=0.4, zorder=zorder))
