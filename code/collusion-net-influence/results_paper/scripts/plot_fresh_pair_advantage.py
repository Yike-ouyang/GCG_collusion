#!/usr/bin/env python3
"""Plot cumulative pair advantage for the fresh GCG suffix screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "llm_rps_v2"
FRESH_PREFIX = "fresh_gcg_transfer_10r3g_screen_v2_20261026"
V4_CAMPAIGNS = {
    "qwen32b": "20261025-1834_same_model_qwen32b_big-v4",
    "glm32b": "20261025-1837_same_model_glm32b_big-v4",
    "mixtral8x7b": "20261025-1853_same_model_mixtral8x7b_big-v4",
}
MODELS = (
    ("qwen32b", "Qwen2.5-32B"),
    ("glm32b", "GLM-4-32B"),
    ("mixtral8x7b", "Mixtral-8x7B"),
)
MODEL_LABELS = dict(MODELS)
SUFFIXES = (
    ("nai", "ない", "#1f77b4"),
    ("aru", "ある", "#9467bd"),
)
ARM_COLORS = {
    "optional_empty": "#111111",
    "no_channel": "#7f7f7f",
}
OUTPUTS = {
    "fresh": ROOT / "figures_to_keep" / "fresh_pair_advantage_suffixes_3models.pdf",
    "v4_truncated": ROOT / "figures_to_keep" / "fresh_pair_advantage_suffixes_3models_v4refs_10rounds.pdf",
}


def load_json_games(folder: Path) -> list[dict]:
    games = [json.loads(path.read_text()) for path in sorted(folder.glob("*_run*.json"))]
    if not games:
        raise SystemExit(f"No games found under {folder}")
    return games


def load_fresh_games(model: str, arm: str) -> list[dict]:
    return load_json_games(RESULTS / f"{FRESH_PREFIX}_{model}_{arm}" / "games")


def load_v4_games(model: str, condition: str) -> list[dict]:
    return load_json_games(RESULTS / V4_CAMPAIGNS[model] / "games")


def cumulative_delta(game: dict, max_rounds: int | None = None) -> np.ndarray:
    rewards = np.asarray(game["rewards"], dtype=float)
    if rewards.ndim != 2 or rewards.shape[1] != 3:
        raise ValueError(f"Expected rewards with shape (rounds, 3): run {game.get('run')}")
    if max_rounds is not None:
        rewards = rewards[:max_rounds]
    per_round = rewards[:, :2].mean(axis=1) - rewards[:, 2]
    return np.concatenate(([0.0], np.cumsum(per_round)))


def curve_from_games(games: list[dict], max_rounds: int | None = None):
    curves = np.stack([cumulative_delta(game, max_rounds) for game in games])
    mean = curves.mean(axis=0)
    q25, q75 = np.percentile(curves, [25, 75], axis=0)
    return mean, q25, q75, len(curves)


def fresh_curve(model: str, arm: str):
    return curve_from_games(load_fresh_games(model, arm))


def v4_curve(model: str, condition: str):
    return curve_from_games(load_v4_games(model, condition), max_rounds=10)


def draw_curve(ax, x, curve, color, linestyle="-", alpha=0.16):
    mean, q25, q75, _ = curve
    ax.plot(x, mean, color=color, linewidth=2.2, linestyle=linestyle)
    ax.fill_between(x, q25, q75, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", choices=("fresh", "v4_truncated"), default="fresh",
                        help="fresh: matched fresh-screen references; v4_truncated: original v4 references truncated to 10 rounds")
    parser.add_argument("--models", default=",".join(key for key, _ in MODELS),
                        help="comma-separated model keys to plot (default: all three)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    selected = [(key, MODEL_LABELS[key]) for key in (x.strip() for x in args.models.split(",")) if key]
    unknown = [key for key, _ in selected if key not in MODEL_LABELS]
    if not selected or unknown:
        raise SystemExit(f"--models must be a subset of {sorted(MODEL_LABELS)}, got {args.models!r}")

    if args.out:
        out = args.out
    elif len(selected) == len(MODELS):
        out = OUTPUTS[args.reference]
    else:
        stem = "_".join(key for key, _ in selected)
        out = OUTPUTS[args.reference].with_name(
            f"{OUTPUTS[args.reference].stem.rsplit('_3models', 1)[0]}_{stem}"
            f"{OUTPUTS[args.reference].stem.rsplit('_3models', 1)[1]}.pdf"
        )
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "Noto Sans Thai", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    x = np.arange(11)
    ncols = len(selected)
    horizontal = ncols == 1
    if horizontal:
        fig, axes = plt.subplots(1, len(SUFFIXES), figsize=(10.0, 5.6), sharex=True, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(2, ncols, figsize=(5.0 * ncols, 8.4), sharex=True, sharey=True, squeeze=False)

    for row, (suffix_key, suffix_text, suffix_color) in enumerate(SUFFIXES):
        for col, (model_key, model_label) in enumerate(selected):
            ax = axes[0, row] if horizontal else axes[row, col]
            draw_curve(ax, x, fresh_curve(model_key, f"optional_gcg_{suffix_key}"), suffix_color)
            if args.reference == "fresh":
                optional_ref = fresh_curve(model_key, "optional_empty")
                no_channel_ref = fresh_curve(model_key, "no_channel")
            else:
                optional_ref = v4_curve(model_key, "channel_optional")
                no_channel_ref = v4_curve(model_key, "no_channel")
            draw_curve(ax, x, optional_ref, ARM_COLORS["optional_empty"], "--")
            draw_curve(ax, x, no_channel_ref, ARM_COLORS["no_channel"], ":")

            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlim(0, 10)
            ax.set_xticks(range(0, 11, 2))
            ax.grid(True, alpha=0.22)
            ax.set_title(f"{model_label} — {suffix_text}", fontsize=11)
            ax.margins(x=0)

            if row == 1 or horizontal:
                ax.set_xlabel("Round")
            if col == 0 and (not horizontal or row == 0):
                ax.set_ylabel("Cumulative Δ = Σ[(A+B)/2 − C] (points)")

    if args.reference == "fresh":
        optional_label = "Optional channel (no suffix)"
        no_channel_label = "No channel"
        suptitle = "Fresh-game cumulative pair advantage by suffix — 3 games × 10 rounds per arm"
        caption = (
            "Each line is the mean across 3 complete games; ribbons show the 25th–75th percentile "
            "across games. Δ_t = (r_A + r_B)/2 − r_C, plotted cumulatively.\n"
            "Suffix arms append ない or ある to each A/B move request. Optional channel without a "
            "suffix is the no-suffix reference; no channel is a separate behavioral reference. "
            "Small exploratory screen (n=3), not a confirmatory test."
        )
    else:
        optional_label = "Optional channel — original v4 (first 10 rounds, unmatched)"
        no_channel_label = "No channel — original v4 (first 10 rounds, unmatched)"
        suptitle = "Fresh suffix arms vs original v4 references — first 10 rounds (supplemental)"
        caption = (
            "Suffix lines: fresh screen, mean across 3 games; ribbons: 25th–75th percentile across games. "
            "Reference lines: original v4 campaigns, mean across 20 games truncated to the first 10 rounds; "
            "ribbons: 25th–75th percentile across games.\n"
            "Suffix and reference games are not matched (different runs and seed offsets), so this "
            "supplemental figure is descriptive only. Δ_t = (r_A + r_B)/2 − r_C, cumulative."
        )

    handles = [
        Line2D([0], [0], color=SUFFIXES[0][2], linewidth=2.2, label="Suffix ない"),
        Line2D([0], [0], color=SUFFIXES[1][2], linewidth=2.2, label="Suffix ある"),
        Line2D([0], [0], color=ARM_COLORS["optional_empty"], linewidth=2.0,
               linestyle="--", label=optional_label),
        Line2D([0], [0], color=ARM_COLORS["no_channel"], linewidth=2.0,
               linestyle=":", label=no_channel_label),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.13 if horizontal else 0.075),
               ncol=2 if horizontal else 4, frameon=False, fontsize=9)

    fig.suptitle(suptitle, fontsize=13)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color="#444444")
    if horizontal:
        fig.subplots_adjust(top=0.80, bottom=0.30, left=0.09, right=0.98, wspace=0.12)
    else:
        fig.subplots_adjust(top=0.91, bottom=0.19, left=0.07, right=0.99,
                            hspace=0.25, wspace=0.10)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
