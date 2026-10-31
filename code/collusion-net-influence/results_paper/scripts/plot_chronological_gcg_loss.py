#!/usr/bin/env python3
"""Plot chronological GCG training and validation loss by run."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results_paper" / "data" / "gcg_qwen_pilot"
OUT = ROOT / "figures_to_keep" / "chronological_gcg_train_val_loss_by_run.pdf"
FILES = [
    DATA / "qwen32b_chronological_bang5_runs_0_4.json",
    DATA / "qwen32b_chronological_bang5_runs_5_9.json",
]
EXCLUDED_RUNS = {8}
FREQUENT_COLORS = {
    "quoi": "#AEC7E8",
    "在网络": "#FFBB78",
    "ない": "#98DF8A",
    "ある": "#C5B0D5",
    "IllegalArgumentException": "#FF9896",
}
FALLBACK_COLORS = ("#F7B6D2", "#C49C94", "#DBDB8D", "#9EDAE5", "#FFBB78", "#C5B0D5")
OTHER_COLOR = "#E0E0E0"
RUN_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)
LOG2 = math.log(2)


def load_runs() -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = {}
    for path in FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run_id, run in payload["runs"].items():
            run_id = int(run_id)
            if run_id in EXCLUDED_RUNS:
                continue
            runs[run_id] = run["blocks"]
    return dict(sorted(runs.items()))


def training_loss_bits(block: dict) -> float:
    accepted = block.get("accepted_candidate")
    if accepted is not None:
        return float(accepted["block_mean_kl_bits"])
    return float(block["block_loss_before_nats"]) / LOG2


def validation_loss_bits(block: dict) -> float | None:
    prequential = block.get("next_block_prequential")
    if not prequential:
        return None
    return float(prequential["updated_suffix_mean_kl_nats"]) / LOG2


def suffix_of(block: dict) -> str:
    return block["suffix_after"].strip()


def build_suffix_palette(runs: dict[int, list[dict]]) -> tuple[dict[str, str], Counter]:
    counts = Counter(suffix_of(block) for blocks in runs.values() for block in blocks)
    palette = dict(FREQUENT_COLORS)
    fallback = iter(FALLBACK_COLORS)
    for suffix, count in counts.most_common():
        if count > 4 and suffix not in palette:
            palette[suffix] = next(fallback, OTHER_COLOR)
    return palette, counts


def draw_panel(ax, runs, losses, suffix_palette, run_colors, x_labels,
               title, mean_values):
    for run_id, blocks in runs.items():
        values = losses[run_id]
        xs = [block["block_index"] for block, value in zip(blocks, values) if value is not None]
        ys = [value for value in values if value is not None]
        suffixes = [suffix_of(block) for block, value in zip(blocks, values) if value is not None]
        ax.plot(xs, ys, color=run_colors[run_id], linewidth=1.25, alpha=0.9, zorder=1)
        ax.scatter(
            xs, ys, s=58,
            facecolors=[suffix_palette.get(suffix, OTHER_COLOR) for suffix in suffixes],
            edgecolors="#4d4d4d", linewidths=0.6, zorder=2,
        )

    xs_mean = np.arange(1, len(mean_values) + 1)
    ax.plot(xs_mean, mean_values, color="black", linewidth=2.6, zorder=3)
    ax.scatter(xs_mean, mean_values, s=22, color="black", zorder=4)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("KL loss (bits)")
    ax.set_xticks(range(1, len(x_labels) + 1), x_labels)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.margins(x=0.04)


def main() -> None:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "Noto Sans Thai", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    runs = load_runs()
    suffix_palette, suffix_counts = build_suffix_palette(runs)
    run_colors = {run_id: RUN_COLORS[index] for index, run_id in enumerate(runs)}

    train_losses = {
        run_id: [training_loss_bits(block) for block in blocks]
        for run_id, blocks in runs.items()
    }
    val_losses = {
        run_id: [validation_loss_bits(block) for block in blocks]
        for run_id, blocks in runs.items()
    }

    train_mean = np.mean(np.array([train_losses[run_id] for run_id in runs], dtype=float), axis=0)
    val_matrix = np.array([val_losses[run_id][:5] for run_id in runs], dtype=float)
    val_mean = np.nanmean(val_matrix, axis=0)

    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(11.8, 9.2))
    train_labels = [f"{b}\nR{(b - 1) * 10 + 1}–{b * 10}" for b in range(1, 7)]
    val_labels = [f"{b}→{b + 1}\nR{b * 10 + 1}–{(b + 1) * 10}" for b in range(1, 6)]

    draw_panel(
        ax_train, runs, train_losses, suffix_palette, run_colors, train_labels,
        "Training loss — suffix after update on its block", train_mean,
    )
    draw_panel(
        ax_val, runs, val_losses, suffix_palette, run_colors, val_labels,
        "Validation loss — suffix from block b evaluated on block b+1", val_mean,
    )

    run_handles = [
        Line2D([0], [0], color=run_colors[run_id], linewidth=2.2, label=f"Run {run_id}")
        for run_id in runs
    ]
    run_handles.append(Line2D([0], [0], color="black", linewidth=3.0, label="Mean"))
    suffix_handles = []
    for suffix, count in suffix_counts.most_common():
        if count <= 4:
            continue
        suffix_handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=suffix_palette.get(suffix, OTHER_COLOR),
            markeredgecolor="#4d4d4d", markeredgewidth=0.6,
            label=f"{suffix} ({count})",
        ))
    suffix_handles.append(Line2D(
        [0], [0], marker="o", linestyle="none", markersize=8,
        markerfacecolor=OTHER_COLOR, markeredgecolor="#4d4d4d", markeredgewidth=0.6,
        label="Other",
    ))

    fig.suptitle("Chronological GCG loss by run (runs 0–7, starting from `!!!!!`)", fontsize=13)
    run_legend = fig.legend(
        handles=run_handles, loc="upper center", bbox_to_anchor=(0.5, 0.947),
        ncol=len(run_handles), frameon=False, fontsize=8, title="Run (line color)",
        title_fontsize=8,
    )
    fig.add_artist(run_legend)
    fig.legend(
        handles=suffix_handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=len(suffix_handles), frameon=False, fontsize=8, title="Suffix (point color)",
        title_fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.065, 1, 0.90])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
