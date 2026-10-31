#!/usr/bin/env python3
"""Regenerate the five article figures from the existing data.

Usage:
    python make_figures.py [--figure all|1|2|3|S1|S2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import FIGURE_DIR
from . import _data as data
from . import _style as style
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

QWEN = "Qwen2.5-32B"


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------

def draw_curve(ax, x, curve, color, *, linestyle="-", linewidth=1.6,
               band_alpha=0.15):
    mean, q25, q75, _ = curve
    ax.plot(x, mean, color=color, linewidth=linewidth, linestyle=linestyle)
    ax.fill_between(x, q25, q75, color=color, alpha=band_alpha, linewidth=0)


def condition_labels() -> list[str]:
    return [style.HEATMAP_LABELS[c] for c in data.CONDITIONS]


def draw_pairwise_heatmap(ax, means_delta, means_mi, delta_norm, mi_norm, *,
                          show_xlab=True, show_ylab=True, label_size=6.2,
                          value_size=6.2, aspect="auto"):
    delta_diff = data.pairwise_differences(means_delta, lower=True)
    mi_diff = data.pairwise_differences(means_mi, lower=False)
    lower_mask = np.tril(np.ones((5, 5), dtype=bool), k=-1)
    upper_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)

    ax.imshow(np.ma.array(delta_diff, mask=~lower_mask), cmap=style.ADV_CMAP,
              norm=delta_norm, aspect=aspect)
    ax.imshow(np.ma.array(mi_diff, mask=~upper_mask), cmap=style.MI_CMAP,
              norm=mi_norm, aspect=aspect)

    for i in range(5):
        for j in range(5):
            if i > j:
                value = delta_diff[i, j]
            elif i < j:
                value = mi_diff[i, j]
            else:
                continue
            cmap = style.ADV_CMAP if i > j else style.MI_CMAP
            norm = delta_norm if i > j else mi_norm
            ax.text(j, i, f"{value:+.02f}", ha="center", va="center",
                    fontsize=value_size,
                    color=style.contrast_color(value, norm, cmap))

    labels = condition_labels()
    ax.set_xticks(range(5), labels, rotation=32, ha="right", fontsize=label_size)
    ax.set_yticks(range(5), labels, fontsize=label_size)
    if not show_xlab:
        ax.set_xticklabels([])
    if not show_ylab:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_aspect(aspect)
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="black", linewidth=0.9, clip_on=False)


def draw_significance(ax, p_delta, p_mi, *, show_xlab=True, show_ylab=True,
                      label_size=6.2, star_size=8.4, aspect="auto"):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    for i in range(5):
        for j in range(5):
            if i > j:
                text = data.stars(p_delta[i, j])
            elif i < j:
                text = data.stars(p_mi[i, j])
            else:
                text = "\u2014"
            significant = text not in ("ns", "\u2014")
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=star_size if significant else label_size + 1,
                    fontweight="bold" if significant else "normal",
                    color="#111111" if significant else "#777777")
    for k in range(6):
        ax.axhline(k - 0.5, color="#CCCCCC", linewidth=0.55)
        ax.axvline(k - 0.5, color="#CCCCCC", linewidth=0.55)
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="black", linewidth=0.9)

    labels = condition_labels()
    ax.set_xticks(range(5), labels, rotation=32, ha="right", fontsize=label_size)
    ax.set_yticks(range(5), labels, fontsize=label_size)
    if not show_xlab:
        ax.set_xticklabels([])
    if not show_ylab:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_aspect(aspect)


def add_horizontal_colorbar(fig, ax, cmap, norm, label, *, label_size=6.4,
                            tick_size=6.0):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(label, fontsize=label_size, labelpad=2)
    cb.ax.tick_params(labelsize=tick_size, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    return cb


# ---------------------------------------------------------------------------
# Figure 1 — RPS overview and AgentCollusion scores
# ---------------------------------------------------------------------------

def make_fig1(out_stem: Path) -> list[Path]:
    style.apply_style()
    metrics = data.collect_rps_metrics()
    ac = data.collect_ac_scores()

    fig = plt.figure(figsize=(7.0, 6.0))
    outer = fig.add_gridspec(
        2, 2, height_ratios=[1.02, 0.80], width_ratios=[1.45, 1.05],
        left=0.10, right=0.985, top=0.95, bottom=0.12,
        hspace=0.22, wspace=0.32,
    )

    # --- A: cumulative pair advantage, three models -----------------------
    gs_a = outer[0, :].subgridspec(2, 3, height_ratios=[1.0, 0.10],
                                   hspace=0.52, wspace=0.13)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    legend_a = fig.add_subplot(gs_a[1, :])
    legend_a.axis("off")

    for index, (model, campaign) in enumerate(data.RPS_CAMPAIGNS.items()):
        ax = axes_a[index]
        grouped = data.load_campaign(campaign)
        for condition in data.CONDITIONS:
            curve = data.curve_from_games(grouped[condition])
            x = np.arange(len(curve[0]))
            draw_curve(ax, x, curve, style.CONDITION_COLORS[condition],
                       linewidth=1.5, band_alpha=0.14)
        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.55)
        ax.set_title(model, fontsize=8.5, pad=4)
        ax.set_xlim(0, 59)
        ax.set_xlabel("Round", fontsize=7.5)
        ax.set_xticks(range(0, 61, 20))
        style.style_axes(ax)
        if index == 0:
            ax.set_ylabel("Cumulative \u0394 = \u03a3[(A+B)/2 \u2212 C] (points)",
                          fontsize=7.5)
            style.panel_label(ax, "A", x=-0.20, y=1.14)
    style.add_legend(legend_a, style.condition_handles(), ncol=5, fontsize=6.8)

    # --- B: Qwen pairwise heatmap and significance ------------------------
    gs_b = outer[1, 0].subgridspec(2, 2, height_ratios=[1.0, 0.085],
                                   hspace=0.30, wspace=0.40)
    ax_heat = fig.add_subplot(gs_b[0, 0])
    ax_sig = fig.add_subplot(gs_b[0, 1])
    cax_delta = fig.add_subplot(gs_b[1, 0])
    cax_mi = fig.add_subplot(gs_b[1, 1])

    draw_pairwise_heatmap(
        ax_heat, metrics["means"][QWEN]["delta"], metrics["means"][QWEN]["mi"],
        metrics["delta_norm"], metrics["mi_norm"], aspect="equal",
    )
    draw_significance(
        ax_sig, metrics["pvalues"][QWEN]["delta"], metrics["pvalues"][QWEN]["mi"],
        show_ylab=False, aspect="equal",
    )
    style.panel_label(ax_heat, "B", x=-0.28, y=1.12)
    ax_heat.text(0.0, 1.10, QWEN, transform=ax_heat.transAxes,
                 fontsize=8, ha="left", va="bottom")

    add_horizontal_colorbar(
        fig, cax_delta, style.ADV_CMAP, metrics["delta_norm"],
        "\u0394 score (row \u2212 column, pts/round)",
    )
    add_horizontal_colorbar(
        fig, cax_mi, style.MI_CMAP, metrics["mi_norm"],
        "\u0394 MI (row \u2212 column, bits)",
    )

    # --- C: AgentCollusion normalized scores ------------------------------
    ax_c = fig.add_subplot(outer[1, 1])
    for schedule in style.SCHEDULE_ORDER:
        rows = ac[schedule]["rows"]
        xs = [row["episode"] for row in rows]
        mean = [row["normalized_mean"] for row in rows]
        low = [row["normalized_min"] for row in rows]
        high = [row["normalized_max"] for row in rows]
        s = style.SCHEDULE_STYLE[schedule]
        ax_c.fill_between(xs, low, high, color=s["color"], alpha=0.12, linewidth=0)
        ax_c.plot(xs, mean, color=s["color"], marker=s["marker"], markersize=3.2,
                  linewidth=1.6, label=style.SCHEDULE_LABELS[schedule])
    ax_c.axhline(1.0, color=style.MUTED, linestyle="--", linewidth=1.0,
                 label="honest play")
    ax_c.set_xlim(0.5, data.AC_EPISODES + 0.5)
    ax_c.set_xticks(range(1, data.AC_EPISODES + 1))
    ax_c.set_xlabel("Episode", fontsize=7.5)
    ax_c.set_ylabel("Normalized score\n(observed / honest)", fontsize=7.5)
    style.style_axes(ax_c)
    style.add_legend(ax_c, style.schedule_handles()
                     + [Line2D([0], [0], color=style.MUTED, linestyle="--",
                               linewidth=1.0, label="honest play")],
                     ncol=2, loc="upper left", fontsize=6.4)
    style.panel_label(ax_c, "C", x=-0.20, y=1.06)

    return style.save(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 2 — GCG optimisation and validation
# ---------------------------------------------------------------------------

def _chrono_palette(counts):
    palette = {
        "ない": style.SUFFIX_COLORS["ない"],
        "ある": style.SUFFIX_COLORS["ある"],
        "quoi": "#AEC7E8",
        "在网络": "#FFBB78",
        "IllegalArgumentException": "#FF9896",
    }
    fallbacks = ("#F7B6D2", "#C49C94", "#DBDB8D", "#9EDAE5")
    fallback_iter = iter(fallbacks)
    for suffix, count in counts.most_common():
        if count > 4 and suffix not in palette:
            palette[suffix] = next(fallback_iter, "#D0D0D0")
    return palette


def _draw_chrono_panel(ax, chrono, losses, palette, *, title, xlabels, mean):
    for run_id, blocks in chrono["runs"].items():
        values = losses[run_id]
        xs = [b["block_index"] for b, v in zip(blocks, values) if v is not None]
        ys = [v for v in values if v is not None]
        suffixes = [data.suffix_of(b) for b, v in zip(blocks, values) if v is not None]
        ax.plot(xs, ys, color="#B8B8B8", linewidth=0.9, alpha=0.85, zorder=1)
        ax.scatter(xs, ys, s=24,
                   facecolors=[palette.get(s, "#D0D0D0") for s in suffixes],
                   edgecolors="#4D4D4D", linewidths=0.45, zorder=2)
    xs_mean = np.arange(1, len(mean) + 1)
    ax.plot(xs_mean, mean, color="black", linewidth=2.0, zorder=3)
    ax.scatter(xs_mean, mean, s=12, color="black", zorder=4)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    ax.set_ylabel("KL loss (bits)", fontsize=7.5)
    ax.set_xticks(range(1, len(xlabels) + 1), xlabels, fontsize=6.6)
    style.style_axes(ax)
    ax.margins(x=0.04)


def _draw_ac_loss(ax, result):
    losses = [float(v) for v in result["losses"]]
    initial = result.get("initial") or {}
    baseline = initial.get("mean_nll")
    xs = list(range(1, len(losses) + 1))
    ys = losses
    if baseline is not None:
        xs = [0] + xs
        ys = [float(baseline)] + ys
    losses_only = losses
    best_index = min(range(len(losses_only)), key=losses_only.__getitem__)
    best_step = best_index + 1
    best_loss = losses_only[best_index]

    ax.plot(xs, ys, color="#2F5D8C", linewidth=1.4, zorder=2)
    ax.scatter(xs, ys, s=9, color="#4F7A5A", edgecolor="white",
               linewidth=0.4, zorder=3)
    ax.scatter([best_step], [best_loss], s=42, color="#8C3B3B",
               edgecolor="white", linewidth=0.7, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("GCG iteration", fontsize=7.5)
    ax.set_ylabel("Loss (mean NLL, log)", fontsize=7.5)
    ax.set_xlim(-0.8, max(xs) + 0.8)
    ax.set_xticks(range(0, max(xs) + 1, 5))
    if baseline is not None:
        labels = ["0\nno suffix" if t == 0 else str(t) for t in range(0, max(xs) + 1, 5)]
        ax.set_xticklabels(labels, fontsize=6.4)
    style.style_axes(ax)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.annotate(
        f"best {best_loss:.1e} (step {best_step})",
        xy=(best_step, best_loss),
        xytext=(0.46, 0.34), textcoords="axes fraction",
        fontsize=6.6, color="#8C3B3B", va="center",
        arrowprops={"arrowstyle": "->", "color": "#8C3B3B", "linewidth": 0.6},
    )


def _draw_ac_validation(ax, rows, gcg):
    xs = [row["episode"] for row in rows]
    mean = [row["mean"] for row in rows]
    low = [row["min"] for row in rows]
    high = [row["max"] for row in rows]
    ax.fill_between(xs, low, high, color="#8C3B3B", alpha=0.11, linewidth=0)
    ax.plot(xs, mean, color="#8C3B3B", marker="o", markersize=3.2,
            linewidth=1.7, label="all-false mean (25 runs)")
    gcg_style = {
        "GCG suffix": {"color": "#C47F17", "marker": "s"},
        "GCG suffix v2": {"color": "#2F5D8C", "marker": "D"},
    }
    for label, run in gcg.items():
        s = gcg_style[label]
        ex = [e["episode"] for e in run["episodes"]]
        ey = [e["normalized"] for e in run["episodes"]]
        ax.plot(ex, ey, color=s["color"], marker=s["marker"], markersize=3.2,
                linewidth=1.5, label=label)
    ax.axhline(1.0, color=style.MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlim(0.5, data.AC_EPISODES + 0.5)
    ax.set_xticks(range(1, data.AC_EPISODES + 1))
    ax.set_xlabel("Episode", fontsize=7.5)
    ax.set_ylabel("Normalized score", fontsize=7.5)
    style.style_axes(ax)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], color=style.MUTED, linestyle="--", linewidth=1.0,
               label="honest play")
    ]
    leg = style.add_legend(ax, handles, ncol=1, loc="center right", fontsize=6.2)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D1D5DB")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.92)


def make_fig2(out_stem: Path) -> list[Path]:
    style.apply_style()
    chrono = data.load_chronological()
    ac_loss = data.load_ac_gcg_loss()
    ac_rows, ac_gcg = data.collect_ac_validation()
    fresh_rows = data.collect_fresh_heatmap()

    palette = _chrono_palette(chrono["suffix_counts"])
    train_mean = np.mean(np.array([chrono["train"][r] for r in chrono["runs"]], dtype=float), axis=0)
    val_matrix = np.array([chrono["val"][r][:5] for r in chrono["runs"]], dtype=float)
    val_mean = np.nanmean(val_matrix, axis=0)

    fig = plt.figure(figsize=(7.0, 8.4))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.15, 0.85],
        left=0.095, right=0.975, top=0.965, bottom=0.055,
        hspace=0.26,
    )
    top = outer[0].subgridspec(2, 2, width_ratios=[1.12, 1.0],
                               height_ratios=[1.0, 1.0],
                               hspace=0.62, wspace=0.36)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.60, 0.75],
                                  wspace=0.42)

    # --- A: chronological train / validation loss -------------------------
    gs_a = top[0:2, 0].subgridspec(3, 1, height_ratios=[1.0, 1.0, 0.24],
                                     hspace=0.40)
    ax_train = fig.add_subplot(gs_a[0])
    ax_val = fig.add_subplot(gs_a[1])
    legend_a = fig.add_subplot(gs_a[2])
    legend_a.axis("off")

    train_labels = [f"B{b}" for b in range(1, 7)]
    val_labels = [f"B{b}\u2192{b + 1}" for b in range(1, 6)]
    _draw_chrono_panel(ax_train, chrono, chrono["train"], palette,
                       title="Train", xlabels=train_labels, mean=train_mean)
    _draw_chrono_panel(ax_val, chrono, chrono["val"], palette,
                       title="Validation", xlabels=val_labels, mean=val_mean)
    ax_val.set_xlabel("Block (10 rounds each)", fontsize=7.5)
    style.panel_label(ax_train, "A", x=-0.16, y=1.13)

    handles = [Line2D([0], [0], color="black", linewidth=2.0, label="Mean")]
    for suffix, count in chrono["suffix_counts"].most_common():
        if count <= 4:
            continue
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=palette.get(suffix, "#D0D0D0"),
            markeredgecolor="#4D4D4D", markeredgewidth=0.45,
            label=f"{suffix} ({count})",
        ))
    handles.append(Line2D(
        [0], [0], marker="o", linestyle="none", markersize=5,
        markerfacecolor="#D0D0D0", markeredgecolor="#4D4D4D",
        markeredgewidth=0.45, label="Other",
    ))
    style.add_legend(legend_a, handles, ncol=4, fontsize=6.0)

    # --- B: AC GCG loss descent -------------------------------------------
    ax_b = fig.add_subplot(top[0, 1])
    _draw_ac_loss(ax_b, ac_loss)
    style.panel_label(ax_b, "B", x=-0.24, y=1.13)

    # --- C: Qwen fresh suffix advantage vs v4 references ------------------
    gs_c = bottom[0].subgridspec(2, 2, height_ratios=[1.0, 0.17],
                                 hspace=0.32, wspace=0.16)
    axes_c = [fig.add_subplot(gs_c[0, 0]), fig.add_subplot(gs_c[0, 1])]
    legend_c = fig.add_subplot(gs_c[1, :])
    legend_c.axis("off")
    x = np.arange(11)
    optional_ref = data.v4_curve("qwen32b", "channel_optional", max_rounds=10)
    no_channel_ref = data.v4_curve("qwen32b", "no_channel", max_rounds=10)
    for idx, (suffix_key, suffix_text) in enumerate(data.SUFFIXES):
        ax = axes_c[idx]
        draw_curve(ax, x, data.fresh_curve("qwen32b", f"optional_gcg_{suffix_key}"),
                   style.SUFFIX_COLORS[suffix_text], linewidth=1.8)
        draw_curve(ax, x, optional_ref, style.REFERENCE_COLORS["optional"],
                   linestyle="--", linewidth=1.4, band_alpha=0.10)
        draw_curve(ax, x, no_channel_ref, style.REFERENCE_COLORS["no_channel"],
                   linestyle=":", linewidth=1.4, band_alpha=0.10)
        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_title(suffix_text, fontsize=8, pad=3)
        ax.set_xlabel("Round", fontsize=7.5)
        style.style_axes(ax)
        if idx == 0:
            ax.set_ylabel("Cumulative \u0394 (points)", fontsize=7.5)
            style.panel_label(ax, "C", x=-0.26, y=1.14)
    handles_c = style.suffix_handles() + style.reference_handles("v4")
    style.add_legend(legend_c, handles_c, ncol=2, fontsize=6.2)

    # --- D: Qwen fresh suffix heatmap vs old channel ----------------------
    gs_d = bottom[1].subgridspec(3, 1, height_ratios=[0.08, 1.0, 0.11],
                                 hspace=0.22)
    ax_d = fig.add_subplot(gs_d[1])
    gs_d_cb = gs_d[2].subgridspec(1, 2, wspace=0.45)
    cax_d_adv = fig.add_subplot(gs_d_cb[0, 0])
    cax_d_mi = fig.add_subplot(gs_d_cb[0, 1])

    qwen_rows = [row for row in fresh_rows if row["model_key"] == "qwen32b"]
    all_adv = [row["delta_adv"] for row in fresh_rows]
    all_mi = [row["delta_mi"] for row in fresh_rows]
    adv_norm = style.symmetric_norm(all_adv)
    mi_norm = style.symmetric_norm(all_mi)
    suffix_labels = [row["suffix"] for row in qwen_rows]
    for row_index, row in enumerate(qwen_rows):
        for col, value, norm, cmap in (
            (0, row["delta_adv"], adv_norm, style.ADV_CMAP),
            (1, row["delta_mi"], mi_norm, style.MI_CMAP),
        ):
            ax_d.imshow(np.ma.array([[value]]), cmap=cmap, norm=norm,
                        extent=[col, col + 1, row_index + 1, row_index],
                        aspect="auto")
            ax_d.text(col + 0.5, row_index + 0.34, f"{value:+.3f}",
                      ha="center", va="center", fontsize=8.0,
                      color=style.contrast_color(value, norm, cmap))
            if col == 0:
                ax_d.text(col + 0.5, row_index + 0.70, f"p={row['p_adv']:.3f}",
                          ha="center", va="center", fontsize=6.2,
                          color=style.contrast_color(value, norm, cmap), alpha=0.9)
    ax_d.axvline(1, color="white", linewidth=1.4)
    for yline in range(3):
        ax_d.axhline(yline, color="white", linewidth=2.0 if yline in (0, 2) else 0.9)
    ax_d.set_xlim(0, 2)
    ax_d.set_ylim(2, 0)
    ax_d.set_yticks([0.5, 1.5], suffix_labels, fontsize=7)
    ax_d.set_xticks([])
    ax_d.tick_params(length=0)
    ax_d.text(0.25, 1.02, "\u0394 adv", transform=ax_d.transAxes,
              ha="center", va="bottom", fontsize=7)
    ax_d.text(0.75, 1.02, "\u0394 MI", transform=ax_d.transAxes,
              ha="center", va="bottom", fontsize=7)
    style.panel_label(ax_d, "D", x=-0.20, y=1.10)
    add_horizontal_colorbar(fig, cax_d_adv, style.ADV_CMAP, adv_norm,
                            "\u0394 adv", label_size=6.0, tick_size=5.6)
    add_horizontal_colorbar(fig, cax_d_mi, style.MI_CMAP, mi_norm,
                            "\u0394 MI", label_size=6.0, tick_size=5.6)

    # --- E: AC validation runs --------------------------------------------
    ax_e = fig.add_subplot(top[1, 1])
    _draw_ac_validation(ax_e, ac_rows, ac_gcg)
    style.panel_label(ax_e, "E", x=-0.24, y=1.13)

    return style.save(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 3 — transfer of the RPS suffixes
# ---------------------------------------------------------------------------

def make_fig3(out_stem: Path) -> list[Path]:
    style.apply_style()
    rows = data.collect_fresh_heatmap()

    fig = plt.figure(figsize=(7.0, 7.35))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.94],
        left=0.125, right=0.98, top=0.955, bottom=0.065, hspace=0.22,
    )

    # --- A: suffix advantage, suffix rows x model columns -----------------
    gs_a = outer[0].subgridspec(3, 3, height_ratios=[1.0, 1.0, 0.14],
                                hspace=0.60, wspace=0.13)
    axes_a = [[fig.add_subplot(gs_a[r, c]) for c in range(3)] for r in range(2)]
    legend_a = fig.add_subplot(gs_a[2, :])
    legend_a.axis("off")

    x = np.arange(11)
    refs = {
        model_key: (
            data.v4_curve(model_key, "channel_optional", max_rounds=10),
            data.v4_curve(model_key, "no_channel", max_rounds=10),
        )
        for model_key, _ in data.FRESH_MODELS
    }
    for row_index, (suffix_key, suffix_text) in enumerate(data.SUFFIXES):
        for col_index, (model_key, model_label) in enumerate(data.FRESH_MODELS):
            ax = axes_a[row_index][col_index]
            draw_curve(ax, x, data.fresh_curve(model_key, f"optional_gcg_{suffix_key}"),
                       style.SUFFIX_COLORS[suffix_text], linewidth=1.8)
            draw_curve(ax, x, refs[model_key][0], style.REFERENCE_COLORS["optional"],
                       linestyle="--", linewidth=1.3, band_alpha=0.09)
            draw_curve(ax, x, refs[model_key][1], style.REFERENCE_COLORS["no_channel"],
                       linestyle=":", linewidth=1.3, band_alpha=0.09)
            ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_xlim(0, 10)
            ax.set_xticks(range(0, 11, 2))
            style.style_axes(ax)
            if row_index == 0:
                ax.set_title(model_label, fontsize=8.5, pad=4)
            if col_index == 0:
                ax.set_ylabel(f"{suffix_text}\nCumulative \u0394 (points)",
                              fontsize=7.2)
            if row_index == 1:
                ax.set_xlabel("Round", fontsize=7.5)
            if row_index == 0 and col_index == 0:
                style.panel_label(ax, "A", x=-0.22, y=1.14)
    handles_a = style.suffix_handles() + style.reference_handles("v4")
    style.add_legend(legend_a, handles_a, ncol=4, fontsize=6.0)

    # --- B: compact fresh suffix heatmap ----------------------------------
    gs_b = outer[1].subgridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.40)
    ax_b = fig.add_subplot(gs_b[0])
    gs_cb = gs_b[1].subgridspec(1, 2, wspace=0.45)
    cax_adv = fig.add_subplot(gs_cb[0, 0])
    cax_mi = fig.add_subplot(gs_cb[0, 1])

    all_adv = [row["delta_adv"] for row in rows]
    all_mi = [row["delta_mi"] for row in rows]
    adv_norm = style.symmetric_norm(all_adv)
    mi_norm = style.symmetric_norm(all_mi)
    models = [label for _, label in data.FRESH_MODELS]
    suffixes = [text for _, text in data.SUFFIXES]
    # columns: nai adv, nai mi, aru adv, aru mi
    by_key = {(row["model_key"], row["suffix_key"]): row for row in rows}
    for row_index, (model_key, model_label) in enumerate(data.FRESH_MODELS):
        for suffix_index, (suffix_key, _) in enumerate(data.SUFFIXES):
            row = by_key[(model_key, suffix_key)]
            for metric_index, (value, cmap, norm, offset) in enumerate((
                (row["delta_adv"], style.ADV_CMAP, adv_norm, 0),
                (row["delta_mi"], style.MI_CMAP, mi_norm, 1),
            )):
                col = suffix_index * 2 + metric_index
                ax_b.imshow(np.ma.array([[value]]), cmap=cmap, norm=norm,
                            extent=[col, col + 1, row_index + 1, row_index],
                            aspect="auto")
                ax_b.text(col + 0.5, row_index + 0.34, f"{value:+.3f}",
                          ha="center", va="center", fontsize=6.8,
                          color=style.contrast_color(value, norm, cmap))
                if metric_index == 0:
                    ax_b.text(col + 0.5, row_index + 0.70, f"p={row['p_adv']:.3f}",
                              ha="center", va="center", fontsize=5.6,
                              color=style.contrast_color(value, norm, cmap), alpha=0.9)
    for xline in (0, 1, 2, 3, 4):
        ax_b.axvline(xline, color="white", linewidth=1.2)
    for yline in range(4):
        ax_b.axhline(yline, color="white", linewidth=2.0 if yline in (0, 3) else 0.9)
    ax_b.axvline(2, color="#444444", linewidth=1.2)
    ax_b.set_xlim(0, 4)
    ax_b.set_ylim(3, 0)
    ax_b.set_yticks(np.arange(3) + 0.5, models, fontsize=7)
    ax_b.set_xticks([0.5, 1.5, 2.5, 3.5],
                    ["\u0394 adv", "\u0394 MI", "\u0394 adv", "\u0394 MI"],
                    fontsize=6.8)
    ax_b.tick_params(length=0)
    for suffix_index, suffix_text in enumerate(suffixes):
        ax_b.text(suffix_index * 2 + 1.0, -0.30, suffix_text, ha="center",
                  va="bottom", fontsize=8, fontweight="bold")
    style.panel_label(ax_b, "B", x=-0.09, y=1.13)
    add_horizontal_colorbar(fig, cax_adv, style.ADV_CMAP, adv_norm,
                            "\u0394 pair advantage (pts/round)",
                            label_size=6.2, tick_size=5.8)
    add_horizontal_colorbar(fig, cax_mi, style.MI_CMAP, mi_norm,
                            "\u0394 MI (bits)", label_size=6.2, tick_size=5.8)

    return style.save(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure S1 — mutual-information curves
# ---------------------------------------------------------------------------

def make_fig_s1(out_stem: Path) -> list[Path]:
    style.apply_style()
    curves = data.collect_mi_curves()

    fig = plt.figure(figsize=(7.0, 3.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.20],
                          left=0.09, right=0.99, top=0.90, bottom=0.06,
                          hspace=0.34, wspace=0.13)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend = fig.add_subplot(gs[1, :])
    legend.axis("off")

    for index, model in enumerate(data.RPS_CAMPAIGNS):
        ax = axes[index]
        for condition in data.CONDITIONS:
            ends, observed, low, high, _ = curves[model][condition]
            ax.plot(ends, observed, color=style.CONDITION_COLORS[condition],
                    linewidth=1.6)
            ax.fill_between(ends, low, high, color=style.CONDITION_COLORS[condition],
                            alpha=0.15, linewidth=0)
        ax.set_title(model, fontsize=8.5, pad=4)
        ax.set_xlabel(f"Round (end of {data.WINDOW}-round window)", fontsize=7.2)
        ax.set_xlim(data.WINDOW - 2, 61)
        ax.set_ylim(bottom=0.0)
        style.style_axes(ax)
        style.panel_label(ax, "ABC"[index], x=-0.20, y=1.14)
        if index == 0:
            ax.set_ylabel("I(A;B | previous joint action) (bits)", fontsize=7.2)
    style.add_legend(legend, style.condition_handles(), ncol=5, fontsize=6.8)

    return style.save(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure S2 — pairwise heatmaps and significance for three models
# ---------------------------------------------------------------------------

def make_fig_s2(out_stem: Path) -> list[Path]:
    style.apply_style()
    metrics = data.collect_rps_metrics()

    fig = plt.figure(figsize=(7.0, 8.1))
    outer = fig.add_gridspec(
        4, 2, height_ratios=[1.0, 1.0, 1.0, 0.11],
        left=0.115, right=0.975, top=0.955, bottom=0.075,
        hspace=0.40, wspace=0.26,
    )
    models = list(data.RPS_CAMPAIGNS)
    for row, model in enumerate(models):
        ax_heat = fig.add_subplot(outer[row, 0])
        ax_sig = fig.add_subplot(outer[row, 1])
        draw_pairwise_heatmap(
            ax_heat, metrics["means"][model]["delta"], metrics["means"][model]["mi"],
            metrics["delta_norm"], metrics["mi_norm"],
            show_xlab=(row == len(models) - 1),
            show_ylab=True,
        )
        draw_significance(
            ax_sig, metrics["pvalues"][model]["delta"], metrics["pvalues"][model]["mi"],
            show_xlab=(row == len(models) - 1),
            show_ylab=False,
        )
        ax_heat.set_title(model, fontsize=8, pad=3)
        if row == 0:
            style.panel_label(ax_heat, "A", x=-0.32, y=1.13)
            style.panel_label(ax_sig, "B", x=-0.16, y=1.13)

    gs_cb = outer[3, :].subgridspec(1, 2, wspace=0.45)
    cax_delta = fig.add_subplot(gs_cb[0, 0])
    cax_mi = fig.add_subplot(gs_cb[0, 1])
    add_horizontal_colorbar(fig, cax_delta, style.ADV_CMAP, metrics["delta_norm"],
                            "\u0394 score (row \u2212 column, pts/round)")
    add_horizontal_colorbar(fig, cax_mi, style.MI_CMAP, metrics["mi_norm"],
                            "\u0394 MI (row \u2212 column, bits)")

    return style.save(fig, out_stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BUILDERS = {
    "1": ("fig1_rps_ac_overview", make_fig1),
    "2": ("fig2_gcg_optimisation", make_fig2),
    "3": ("fig3_suffix_transfer", make_fig3),
    "S1": ("figS1_mi_curves", make_fig_s1),
    "S2": ("figS2_pairwise_heatmaps", make_fig_s2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", default="all", choices=("all", *BUILDERS))
    args = parser.parse_args()

    selected = list(BUILDERS) if args.figure == "all" else [args.figure]
    for key in selected:
        stem, builder = BUILDERS[key]
        written = builder(FIGURE_DIR / stem)
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
