"""Shared visual style for the GCG-collusion article figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

FONT_FAMILY = ["Noto Sans CJK JP", "Noto Sans Thai", "DejaVu Sans"]

TEXT = "#1F2937"
MUTED = "#4B5563"
GRID = "#D9D9D9"
SPINE = "#9CA3AF"

CONDITION_ORDER = (
    "no_channel",
    "channel_optional",
    "channel_optional_hint",
    "channel_forced",
    "channel_forced_strategy",
)
CONDITION_LABELS = {
    "no_channel": "No channel",
    "channel_optional": "Optional",
    "channel_optional_hint": "Optional + hint",
    "channel_forced": "Forced",
    "channel_forced_strategy": "Forced + hint",
}
CONDITION_COLORS = {
    "no_channel": "#4D4D4D",
    "channel_optional": "#2878B5",
    "channel_optional_hint": "#E17C05",
    "channel_forced": "#2B8C56",
    "channel_forced_strategy": "#B23A48",
}

SHORT_LABELS = {
    "no_channel": "No channel",
    "channel_optional": "Optional",
    "channel_optional_hint": "Optional + hint",
    "channel_forced": "Forced",
    "channel_forced_strategy": "Forced + hint",
}
HEATMAP_LABELS = {
    "no_channel": "No ch.",
    "channel_optional": "Optional",
    "channel_optional_hint": "Opt.+hint",
    "channel_forced": "Forced",
    "channel_forced_strategy": "Forced+hint",
}

SUFFIX_COLORS = {
    "ない": "#1F77B4",
    "ある": "#9467BD",
}
REFERENCE_COLORS = {
    "optional": "#111111",
    "no_channel": "#7F7F7F",
}

SCHEDULE_ORDER = ("all-false", "mixed-40", "mixed-80", "all-true")
SCHEDULE_LABELS = {
    "all-false": "all-false",
    "mixed-40": "mixed-40",
    "mixed-80": "mixed-80",
    "all-true": "all-true",
}
SCHEDULE_STYLE = {
    "all-false": {"color": "#8C3B3B", "marker": "o"},
    "mixed-40": {"color": "#C47F17", "marker": "s"},
    "mixed-80": {"color": "#4F7A5A", "marker": "^"},
    "all-true": {"color": "#2F5D8C", "marker": "D"},
}

ADV_CMAP = "RdBu_r"
MI_CMAP = "PuOr_r"

RUN_COLORS = (
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
)

FIGURE_WIDTH = 7.0


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })


def panel_label(ax, letter: str, *, x: float = -0.16, y: float = 1.10,
                fontsize: float = 11) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def style_axes(ax, *, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.55, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def condition_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=CONDITION_COLORS[c], linewidth=2.0,
               label=CONDITION_LABELS[c])
        for c in CONDITION_ORDER
    ]


def suffix_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, linewidth=2.0, label=suffix)
        for suffix, color in SUFFIX_COLORS.items()
    ]


def reference_handles(reference_kind: str = "v4") -> list[Line2D]:
    if reference_kind == "v4":
        optional_label = "Optional channel — v4 reference"
        no_channel_label = "No channel — v4 reference"
    else:
        optional_label = "Optional channel (no suffix)"
        no_channel_label = "No channel"
    return [
        Line2D([0], [0], color=REFERENCE_COLORS["optional"], linewidth=1.8,
               linestyle="--", label=optional_label),
        Line2D([0], [0], color=REFERENCE_COLORS["no_channel"], linewidth=1.8,
               linestyle=":", label=no_channel_label),
    ]


def schedule_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SCHEDULE_STYLE[s]["color"], linewidth=1.8,
               marker=SCHEDULE_STYLE[s]["marker"], markersize=3.5,
               label=SCHEDULE_LABELS[s])
        for s in SCHEDULE_ORDER
    ]


def add_legend(ax, handles, *, ncol: int, loc: str = "center",
               fontsize: float = 7, title: str | None = None):
    return ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        title=title,
        title_fontsize=fontsize,
        handlelength=1.8,
        columnspacing=1.1,
        labelspacing=0.35,
        borderaxespad=0.0,
    )


def save(fig, output_stem: Path, *, png_dpi: int = 250) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return [pdf_path, png_path]


def symmetric_norm(values, minimum: float = 1e-6) -> Normalize:
    extent = max(float(abs(v)) for v in values) if len(values) else 0.0
    return Normalize(vmin=-max(extent, minimum), vmax=max(extent, minimum))


def contrast_color(value: float, norm: Normalize, cmap_name: str) -> str:
    r, g, b, _ = matplotlib.colormaps[cmap_name](norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 0.55 else "#FFFFFF"


__all__ = [
    "FIGURE_WIDTH", "TEXT", "MUTED", "GRID",
    "CONDITION_ORDER", "CONDITION_LABELS", "CONDITION_COLORS", "SHORT_LABELS",
    "SUFFIX_COLORS", "REFERENCE_COLORS",
    "SCHEDULE_ORDER", "SCHEDULE_LABELS", "SCHEDULE_STYLE",
    "ADV_CMAP", "MI_CMAP", "RUN_COLORS",
    "apply_style", "panel_label", "style_axes",
    "condition_handles", "suffix_handles", "reference_handles", "schedule_handles",
    "add_legend", "save", "symmetric_norm", "contrast_color",
]
