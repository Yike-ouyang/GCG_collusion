"""Plot cumulative pair advantage (A+B)/2 - C with IQR ribbons across runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = (
    "no_channel",
    "channel_optional",
    "channel_optional_hint",
    "channel_forced",
    "channel_forced_strategy",
)
CONDITION_LABELS = {
    "no_channel": "No channel",
    "channel_optional": "Optional channel",
    "channel_optional_hint": "Optional + hint",
    "channel_forced": "Forced channel",
    "channel_forced_strategy": "Forced + hint",
}
CONDITION_COLORS = {
    "no_channel": "#4d4d4d",
    "channel_optional": "#2878b5",
    "channel_optional_hint": "#e17c05",
    "channel_forced": "#2b8c56",
    "channel_forced_strategy": "#b23a48",
}
MODEL_LABELS = {
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "zai-org/GLM-4-32B-0414": "GLM-4-32B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
}


def load_games(campaign: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((campaign / "games").glob("*_run*.json")):
        game = json.loads(path.read_text())
        grouped[game["condition"]].append(game)
    if not grouped:
        raise SystemExit(f"No game JSON files found under {campaign / 'games'}")
    return grouped


def cumulative_delta(game: dict) -> np.ndarray:
    rewards = np.asarray(game["rewards"], dtype=float)
    if rewards.ndim != 2 or rewards.shape[1] != 3:
        raise ValueError(f"Expected per-round rewards with shape (rounds, 3): run {game.get('run')}")
    per_round = rewards[:, :2].mean(axis=1) - rewards[:, 2]
    return np.concatenate(([0.0], np.cumsum(per_round)))


def campaign_model(campaign: Path) -> str:
    config_path = campaign / "config.json"
    if not config_path.exists():
        return ""
    settings = json.loads(config_path.read_text()).get("settings", {})
    agents = settings.get("agents", "")
    return agents.split(",")[0].split("@", 1)[0] if agents else ""


def model_label(campaign: Path) -> str:
    model = campaign_model(campaign)
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return model.split("/")[-1] if model else campaign.name


def plot_campaign(ax: plt.Axes, campaign: Path, show_ylabel: bool) -> list:
    grouped = load_games(campaign)
    present = [condition for condition in CONDITION_ORDER if condition in grouped]
    if not present:
        raise SystemExit(f"No known conditions under {campaign / 'games'}")

    handles = []
    rounds = None
    for condition in present:
        curves = np.stack([cumulative_delta(game) for game in grouped[condition]])
        rounds = np.arange(curves.shape[1])
        mean = curves.mean(axis=0)
        q25, q75 = np.percentile(curves, [25, 75], axis=0)
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]
        line = ax.plot(rounds, mean, color=color, linewidth=2.2, label=label)[0]
        ax.fill_between(rounds, q25, q75, color=color, alpha=0.18, linewidth=0)
        handles.append(line)
        print(
            f"{model_label(campaign):14s} {label:20s} runs={len(curves):2d} "
            f"final_mean={mean[-1]:+.3f} final_q25={q25[-1]:+.3f} final_q75={q75[-1]:+.3f}"
        )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(model_label(campaign))
    ax.set_xlabel("Round")
    if show_ylabel:
        ax.set_ylabel("Cumulative Δ = Σ[(A+B)/2 − C] (points)")
    ax.set_xlim(0, int(rounds.shape[0]) - 1)
    ax.grid(True, alpha=0.22)
    ax.margins(x=0)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path, help="llm_rps_v2 campaign directories")
    parser.add_argument("--out", type=Path, help="output PDF path")
    parser.add_argument("--title", default="Cumulative pair advantage (A+B)/2 − C")
    args = parser.parse_args()

    n = len(args.campaigns)
    if args.out is None:
        if n == 1:
            slug = model_label(args.campaigns[0]).lower().replace(".", "").replace(" ", "_")
            out = Path(f"figures_to_keep/{slug}_pair_advantage_cumulative_iqr.pdf")
        else:
            out = Path("figures_to_keep/pair_advantage_cumulative_iqr_3models.pdf")
    else:
        out = args.out

    figsize = (10.5, 5.8) if n == 1 else (5.9 * n, 5.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    handles = None
    for index, (ax, campaign) in enumerate(zip(axes, args.campaigns)):
        current = plot_campaign(ax, campaign, show_ylabel=index == 0)
        if index == 0:
            handles = current

    fig.suptitle(args.title, fontsize=13)
    fig.legend(handles, [handle.get_label() for handle in handles], loc="outside lower center",
               ncol=len(handles), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
