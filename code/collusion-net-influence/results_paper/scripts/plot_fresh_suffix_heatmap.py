#!/usr/bin/env python3
"""Compact heatmap of fresh suffix deltas against the old channel reference."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_fresh_vs_old import (  # noqa: E402
    FRESH_PREFIX,
    MODELS,
    RESULTS,
    SUFFIXES,
    V4_CAMPAIGNS,
    load_games,
    permutation_pvalue,
    run_advantage,
    run_mi,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_to_keep" / "fresh_suffix_heatmap_vs_old_channel.pdf"
SEED = 20261026
ADV_CMAP = "RdBu_r"
MI_CMAP = "PuOr_r"


def collect(rng: np.random.Generator):
    rows = []
    for model_key, model_label in MODELS:
        old_games = [
            game for game in load_games(RESULTS / V4_CAMPAIGNS[model_key] / "games")
            if game.get("condition") == "channel_optional"
        ]
        old_adv = np.asarray([run_advantage(game) for game in old_games])
        old_mi = np.asarray([run_mi(game) for game in old_games])

        for suffix_key, suffix_text in SUFFIXES:
            games = load_games(
                RESULTS / f"{FRESH_PREFIX}_{model_key}_optional_gcg_{suffix_key}" / "games"
            )
            fresh_adv = np.asarray([run_advantage(game) for game in games])
            fresh_mi = np.asarray([run_mi(game) for game in games])
            rows.append({
                "model": model_label,
                "suffix": suffix_text,
                "delta_adv": float(fresh_adv.mean() - old_adv.mean()),
                "delta_mi": float(fresh_mi.mean() - old_mi.mean()),
                "p_adv": float(permutation_pvalue(fresh_adv, old_adv, rng)),
            })
    return rows


def contrast_color(value: float, norm: Normalize, cmap_name: str) -> str:
    r, g, b, _ = plt.get_cmap(cmap_name)(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 0.55 else "#ffffff"


def main() -> None:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "Noto Sans Thai", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    rng = np.random.default_rng(SEED)
    rows = collect(rng)
    n = len(rows)

    adv_values = np.asarray([row["delta_adv"] for row in rows])
    mi_values = np.asarray([row["delta_mi"] for row in rows])
    adv_max = max(float(np.abs(adv_values).max()), 1e-6)
    mi_max = max(float(np.abs(mi_values).max()), 1e-6)
    norm_adv = Normalize(vmin=-adv_max, vmax=adv_max)
    norm_mi = Normalize(vmin=-mi_max, vmax=mi_max)

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.imshow(adv_values.reshape(n, 1), cmap=ADV_CMAP, norm=norm_adv,
              extent=[0, 1, n, 0], aspect="auto")
    ax.imshow(mi_values.reshape(n, 1), cmap=MI_CMAP, norm=norm_mi,
              extent=[1, 2, n, 0], aspect="auto")

    for index, row in enumerate(rows):
        adv_color = contrast_color(row["delta_adv"], norm_adv, ADV_CMAP)
        mi_color = contrast_color(row["delta_mi"], norm_mi, MI_CMAP)
        ax.text(0.5, index + 0.38, f"{row['delta_adv']:+.3f}", ha="center", va="center",
                fontsize=9.5, color=adv_color)
        ax.text(0.5, index + 0.69, f"p={row['p_adv']:.3f}", ha="center", va="center",
                fontsize=7.0, color=adv_color, alpha=0.90)
        ax.text(1.5, index + 0.50, f"{row['delta_mi']:+.3f}", ha="center", va="center",
                fontsize=9.5, color=mi_color)

    for x in (0, 1, 2):
        ax.axvline(x, color="white", linewidth=1.4)
    for y in range(n + 1):
        width = 2.2 if y in (0, 2, 4, n) else 0.9
        ax.axhline(y, color="white", linewidth=width)

    ax.set_xlim(0, 2)
    ax.set_ylim(n, 0)
    ax.set_xticks([0.5, 1.5], ["Δ Pair advantage\n(pts/round)", "Δ MI\n(bits)"], fontsize=9)
    ax.set_yticks(np.arange(n) + 0.5,
                  [f"{row['model']} · {row['suffix']}" for row in rows], fontsize=9)
    ax.tick_params(length=0)
    ax.set_title("Fresh suffix runs vs old channel reference — first 10 rounds",
                 fontsize=12, pad=10)

    sm_adv = ScalarMappable(norm=norm_adv, cmap=ADV_CMAP)
    sm_mi = ScalarMappable(norm=norm_mi, cmap=MI_CMAP)
    cax_adv = fig.add_axes([0.875, 0.57, 0.016, 0.29])
    cb_adv = fig.colorbar(sm_adv, cax=cax_adv, orientation="vertical")
    cb_adv.set_label("Δ pair advantage (pts/round)", fontsize=8)
    cb_adv.ax.tick_params(labelsize=7)
    cax_mi = fig.add_axes([0.875, 0.19, 0.016, 0.29])
    cb_mi = fig.colorbar(sm_mi, cax=cax_mi, orientation="vertical")
    cb_mi.set_label("Δ MI (bits)", fontsize=8)
    cb_mi.ax.tick_params(labelsize=7)

    caption = (
        "Cells: suffix mean minus old `channel_optional` mean. Pair advantage = mean per-round "
        "(A+B)/2 − C. MI = mean per-run I(A;B | previous joint action), bits, over 10 rounds.\n"
        "p-values under the pair-advantage cells: two-sided permutation test on the difference of "
        "means (10,000 permutations, no correction). MI has no p-value by request.\n"
        "Suffix runs: fresh screen, 3 games × 10 rounds. Old reference: original v4 campaign, "
        "20 games, first 10 rounds. Runs are not matched (different seeds); exploratory only."
    )
    fig.text(0.42, 0.008, caption, ha="center", va="bottom", fontsize=7.2, color="#444444")
    fig.subplots_adjust(top=0.91, bottom=0.17, left=0.24, right=0.84)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
