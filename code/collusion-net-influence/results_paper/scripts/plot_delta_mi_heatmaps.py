#!/usr/bin/env python3
"""Plot pairwise delta/MI differences and permutation-test significance by model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from plot_mi_curves import (  # noqa: E402
    CAMPAIGNS,
    CONDITIONS,
    LABELS,
    cmi_bits,
    game_rows,
    load_games,
    row_counts,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_to_keep" / "delta_mi_heatmaps_and_significance.pdf"
WINDOW = 20
STEP = 5
N_PERM = 10_000
SEED = 20261026
SHORT_LABELS = {
    "no_channel": "No channel",
    "channel_optional": "Optional",
    "channel_optional_hint": "Optional + hint",
    "channel_forced": "Forced",
    "channel_forced_strategy": "Forced + hint",
}


def run_delta(game: dict) -> float:
    rewards = np.asarray(game["rewards"], dtype=float)
    return float(np.mean(rewards[:, :2].mean(axis=1) - rewards[:, 2]))


def run_window_mean_mi(game: dict) -> float:
    rows = game_rows(game)
    values = [
        float(cmi_bits(row_counts(rows[end - WINDOW:end])[None])[0])
        for end in range(WINDOW, len(rows) + 1, STEP)
    ]
    return float(np.mean(values))


def collect_metrics() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model, campaign in CAMPAIGNS.items():
        grouped = load_games(campaign)
        data[model] = {}
        for condition in CONDITIONS:
            games = grouped[condition]
            data[model][condition] = {
                "delta": np.asarray([run_delta(game) for game in games]),
                "mi": np.asarray([run_window_mean_mi(game) for game in games]),
            }
    return data


def permutation_pvalue(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                       n_perm: int = N_PERM) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = len(x)
    draws = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)
    differences = draws[:, :n_x].mean(axis=1) - draws[:, n_x:].mean(axis=1)
    exceed = int(np.sum(np.abs(differences) >= abs(observed) - 1e-12))
    return (exceed + 1) / (n_perm + 1)


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def pairwise_differences(means: np.ndarray, lower: bool) -> np.ndarray:
    result = np.full((len(means), len(means)), np.nan)
    for i in range(len(means)):
        for j in range(len(means)):
            if lower and i > j:
                result[i, j] = means[i] - means[j]
            elif not lower and i < j:
                result[i, j] = means[i] - means[j]
    return result


def shortcut_labels() -> list[str]:
    return [SHORT_LABELS[condition] for condition in CONDITIONS]


def draw_heatmap(ax, means_delta, means_mi, delta_norm, mi_norm, model):
    delta_diff = pairwise_differences(means_delta, lower=True)
    mi_diff = pairwise_differences(means_mi, lower=False)
    lower_mask = np.tril(np.ones((5, 5), dtype=bool), k=-1)
    upper_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)

    ax.imshow(np.ma.array(delta_diff, mask=~lower_mask), cmap="RdBu_r", norm=delta_norm)
    ax.imshow(np.ma.array(mi_diff, mask=~upper_mask), cmap="PuOr_r", norm=mi_norm)

    for i in range(5):
        for j in range(5):
            if i > j:
                value = delta_diff[i, j]
            elif i < j:
                value = mi_diff[i, j]
            else:
                continue
            ax.text(j, i, f"{value:+.02f}", ha="center", va="center", fontsize=7.5,
                    color="#1a1a1a")

    labels = shortcut_labels()
    ax.set_xticks(range(5), labels, rotation=28, ha="right")
    ax.set_yticks(range(5), labels)
    ax.tick_params(length=0)
    ax.set_title(f"{model} — pairwise differences", fontsize=11)
    ax.set_xlabel("Trajectory (column)")
    ax.set_ylabel("Trajectory (row)")
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="black", linewidth=1.1, clip_on=False)


def draw_significance(ax, p_delta, p_mi, model):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    for i in range(5):
        for j in range(5):
            if i > j:
                text = stars(p_delta[i, j])
            elif i < j:
                text = stars(p_mi[i, j])
            else:
                text = "—"
            significant = text not in ("ns", "—")
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11 if significant else 9,
                    fontweight="bold" if significant else "normal",
                    color="#111111" if significant else "#777777")
    for k in range(6):
        ax.axhline(k - 0.5, color="#cccccc", linewidth=0.6)
        ax.axvline(k - 0.5, color="#cccccc", linewidth=0.6)
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="black", linewidth=1.1)

    labels = shortcut_labels()
    ax.set_xticks(range(5), labels, rotation=28, ha="right")
    ax.set_yticks(range(5), labels)
    ax.tick_params(length=0)
    ax.set_title(f"{model} — permutation tests", fontsize=11)
    ax.set_xlabel("Trajectory (column)")
    ax.set_ylabel("Trajectory (row)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = collect_metrics()

    delta_extent = 0.0
    mi_extent = 0.0
    means = {}
    for model, conditions in data.items():
        delta_means = np.asarray([conditions[c]["delta"].mean() for c in CONDITIONS])
        mi_means = np.asarray([conditions[c]["mi"].mean() for c in CONDITIONS])
        means[model] = {"delta": delta_means, "mi": mi_means}
        for i in range(5):
            for j in range(5):
                if i > j:
                    delta_extent = max(delta_extent, abs(delta_means[i] - delta_means[j]))
                elif i < j:
                    mi_extent = max(mi_extent, abs(mi_means[i] - mi_means[j]))

    delta_norm = Normalize(vmin=-max(delta_extent, 1e-6), vmax=max(delta_extent, 1e-6))
    mi_norm = Normalize(vmin=-max(mi_extent, 1e-6), vmax=max(mi_extent, 1e-6))

    fig, axes = plt.subplots(3, 2, figsize=(11.6, 15.0))
    for row, model in enumerate(CAMPAIGNS):
        conditions = data[model]
        p_delta = np.ones((5, 5))
        p_mi = np.ones((5, 5))
        for i in range(5):
            for j in range(i + 1, 5):
                p_delta[i, j] = p_delta[j, i] = permutation_pvalue(
                    conditions[CONDITIONS[i]]["delta"], conditions[CONDITIONS[j]]["delta"], rng
                )
                p_mi[i, j] = p_mi[j, i] = permutation_pvalue(
                    conditions[CONDITIONS[i]]["mi"], conditions[CONDITIONS[j]]["mi"], rng
                )
        draw_heatmap(axes[row, 0], means[model]["delta"], means[model]["mi"],
                     delta_norm, mi_norm, model)
        draw_significance(axes[row, 1], p_delta, p_mi, model)

    sm_delta = plt.cm.ScalarMappable(norm=delta_norm, cmap="RdBu_r")
    sm_mi = plt.cm.ScalarMappable(norm=mi_norm, cmap="PuOr_r")
    fig.subplots_adjust(top=0.95, bottom=0.19, left=0.08, right=0.98,
                        hspace=0.42, wspace=0.27)

    cax_delta = fig.add_axes([0.14, 0.095, 0.30, 0.008])
    cb_delta = fig.colorbar(sm_delta, cax=cax_delta, orientation="horizontal")
    cb_delta.set_label("Δ delta (row − column, points/round)", fontsize=8)
    cb_delta.ax.tick_params(labelsize=7)

    cax_mi = fig.add_axes([0.56, 0.095, 0.30, 0.008])
    cb_mi = fig.colorbar(sm_mi, cax=cax_mi, orientation="horizontal")
    cb_mi.set_label("Δ MI (row − column, bits)", fontsize=8)
    cb_mi.ax.tick_params(labelsize=7)

    fig.suptitle("Pairwise trajectory differences and permutation-test significance by model",
                 fontsize=13)
    caption = (
        "Lower triangle: pairwise difference in mean delta ((A+B)/2 − C per round); "
        "upper triangle: pairwise difference in mean window-MI. "
        "The diagonal separates delta (below) from MI (above).\n"
        "MI = I(A;B | previous joint action), averaged over trailing 20-round windows "
        "(step 5) for each run.\n"
        "Significance: two-sided permutation test on the difference of means "
        f"({N_PERM:,} permutations, no correction).\n"
        "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant."
    )
    fig.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=7.5, color="#444444")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
