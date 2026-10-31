#!/usr/bin/env python3
"""Plot pooled mutual-information curves for the three v4 model campaigns."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {
    "Qwen2.5-32B": ROOT / "results/llm_rps_v2/20261025-1834_same_model_qwen32b_big-v4",
    "GLM-4-32B": ROOT / "results/llm_rps_v2/20261025-1837_same_model_glm32b_big-v4",
    "Mixtral-8x7B": ROOT / "results/llm_rps_v2/20261025-1853_same_model_mixtral8x7b_big-v4",
}
CONDITIONS = (
    "no_channel",
    "channel_optional",
    "channel_optional_hint",
    "channel_forced",
    "channel_forced_strategy",
)
LABELS = {
    "no_channel": "No channel",
    "channel_optional": "Optional channel",
    "channel_optional_hint": "Optional + hint",
    "channel_forced": "Forced channel",
    "channel_forced_strategy": "Forced + hint",
}
COLORS = {
    "no_channel": "#4d4d4d",
    "channel_optional": "#2878b5",
    "channel_optional_hint": "#e17c05",
    "channel_forced": "#2b8c56",
    "channel_forced_strategy": "#b23a48",
}
N_STATES = 28
OUT = ROOT / "figures_to_keep" / "mi_curves_3models_5conditions.pdf"


def game_rows(game: dict) -> list[tuple[int, int, int]]:
    rows = []
    state = 0
    for moves in game["actions"]:
        a, b, c = (int(x) for x in moves)
        rows.append((state, a, b))
        state = 1 + 9 * a + 3 * b + c
    return rows


def row_counts(rows: list[tuple[int, int, int]]) -> np.ndarray:
    counts = np.zeros((N_STATES, 3, 3), dtype=np.float64)
    for state, a, b in rows:
        counts[state, a, b] += 1.0
    return counts


def cmi_bits(counts: np.ndarray) -> np.ndarray:
    n = counts.sum(axis=(-1, -2))
    total = n.sum(axis=-1)
    row = counts.sum(axis=-1)
    col = counts.sum(axis=-2)
    denom = row[..., :, :, None] * col[..., :, None, :]
    valid = (counts > 0) & (denom > 0)
    ratio = np.ones_like(counts)
    n_broadcast = np.broadcast_to(n[..., None, None], counts.shape)
    ratio[valid] = counts[valid] * n_broadcast[valid] / denom[valid]
    terms = np.zeros_like(counts)
    terms[valid] = counts[valid] * np.log2(ratio[valid])
    return terms.sum(axis=(-1, -2, -3)) / total


def load_games(campaign: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((campaign / "games").glob("*_run*.json")):
        game = json.loads(path.read_text())
        grouped[game["condition"]].append(game)
    return grouped


def window_ends(n_rounds: int, window: int, step: int) -> list[int]:
    ends = list(range(window, n_rounds + 1, step))
    if ends[-1] != n_rounds:
        ends.append(n_rounds)
    return ends


def pooled_curve(games: list[dict], window: int, step: int,
                 n_boot: int, rng: np.random.Generator):
    rows = [game_rows(game) for game in games]
    n_rounds = len(rows[0])
    ends = window_ends(n_rounds, window, step)
    per_run = np.stack([
        np.stack([row_counts(game[start - window:start]) for start in ends])
        for game in rows
    ])

    pooled = per_run.sum(axis=0)
    observed = cmi_bits(pooled)
    full = cmi_bits(row_counts([row for game in rows for row in game])[None])[0]

    n_runs = per_run.shape[0]
    weights = rng.multinomial(n_runs, np.full(n_runs, 1.0 / n_runs), size=n_boot)
    flat = per_run.reshape(n_runs, len(ends), -1)
    boot_flat = np.einsum("br,rwk->bwk", weights, flat)
    boot = cmi_bits(boot_flat.reshape(n_boot, len(ends), N_STATES, 3, 3))
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return np.asarray(ends), observed, low, high, full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.8), sharey=True)
    handles = None
    for index, (model, campaign) in enumerate(CAMPAIGNS.items()):
        ax = axes[index]
        grouped = load_games(campaign)
        for condition in CONDITIONS:
            ends, observed, low, high, full = pooled_curve(
                grouped[condition], args.window, args.step, args.bootstrap, rng
            )
            print(
                f"{model:12s} {LABELS[condition]:18s} full_pooled={full:.4f} "
                f"window_mean={observed.mean():.4f}"
            )
            line = ax.plot(ends, observed, color=COLORS[condition], linewidth=2.1,
                           label=LABELS[condition])[0]
            ax.fill_between(ends, low, high, color=COLORS[condition], alpha=0.17,
                            linewidth=0)
            if index == 0:
                handles = handles or []
                handles.append(line)

        ax.set_title(model)
        ax.set_xlabel(f"Round (end of trailing {args.window}-round window)")
        ax.set_xlim(args.window - 2, len(game_rows(grouped[CONDITIONS[0]][0])) + 1)
        ax.grid(alpha=0.22)
        ax.margins(x=0.02)
        ax.set_ylim(bottom=0.0)
        if index == 0:
            ax.set_ylabel("Pooled I(A;B | previous joint action) (bits)")

    fig.suptitle("Action mutual information by condition — pooled rolling-window estimate",
                 fontsize=13)
    fig.legend(
        handles=handles, labels=[line.get_label() for line in handles],
        loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=5, frameon=False,
    )
    caption = (
        f"Estimator: pooled I(A;B | previous joint action), bits. Each point pools 20 runs "
        f"over a trailing {args.window}-round window (step {args.step}); ribbons: 95% "
        f"bootstrap CI resampling complete runs ({args.bootstrap} draws).\n"
        "State = previous round's joint (A,B,C) move; round 1 uses a start state. "
        "Windowed pooled estimates are not per-round MI and inflate low-MI levels; "
        "compare shapes, not absolute levels, with full-game values."
    )
    fig.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
