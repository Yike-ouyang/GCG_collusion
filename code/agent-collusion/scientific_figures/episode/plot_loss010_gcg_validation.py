#!/usr/bin/env python3
"""Plot the two GCG validation runs against the all-false mean.

Same style and normalization as ``plot_loss010_scores.py``. The all-false mean
and min-max band use the 25 available runs at 10% loss (5 initial + 20 reruns);
the two GCG runs are single reinjections on Qwen3.6-27B.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_loss010_scores as base

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "scientific_figures/episode"
DEFAULT_STEM = "loss010_gcg_validation"

ALL_FALSE_PATTERNS = (
    "results/synthetic/grid/enabled/loss010/all-false/seed*/run_*/run.json",
    "results/synthetic/grid_rerun10/enabled/loss010/all-false/seed*/run_*/run.json",
)
GCG_RUN_DIRS = {
    "GCG suffix": "results/synthetic/gcg_poc/suffix",
    "GCG suffix_v2": "results/synthetic/gcg_poc/suffix_v2",
}

EPISODES = 10
LOSS_RATE = 0.10
HONEST_ALL_FALSE_SCORE = 10.0

STYLE = {
    "all-false mean": {"color": "#8C3B3B", "marker": "o"},
    "GCG suffix": {"color": "#C47F17", "marker": "s"},
    "GCG suffix_v2": {"color": "#2F5D8C", "marker": "D"},
}


def _load_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sequence = (data.get("run_config") or {}).get("synthetic_sequence") or {}
    episodes = []
    for episode in data.get("results") or []:
        truth = episode["truth"]
        honest = base._honest_agent_reward(
            list(truth["alice_parts"])
        ) + base._honest_agent_reward(list(truth["bob_parts"]))
        alice = int(episode["agents"]["alice"]["reward_total"])
        bob = int(episode["agents"]["bob"]["reward_total"])
        episodes.append(
            {
                "episode": int(episode["episode_index"]),
                "score": alice + bob,
                "honest_score": honest,
                "normalized": (alice + bob) / HONEST_ALL_FALSE_SCORE,
            }
        )
    episodes.sort(key=lambda item: item["episode"])
    return {
        "path": str(path),
        "seed": int(sequence.get("seed", -1)),
        "schedule": sequence.get("schedule"),
        "loss_rate": float(sequence.get("loss_rate", -1)),
        "episodes": episodes,
    }


def _validate_run(run: dict[str, Any], label: str) -> None:
    if run["schedule"] != "all-false":
        raise SystemExit(f"{label}: expected all-false, found {run['schedule']!r}")
    if run["loss_rate"] != LOSS_RATE:
        raise SystemExit(
            f"{label}: expected loss rate {LOSS_RATE}, found {run['loss_rate']}"
        )
    if len(run["episodes"]) != EPISODES:
        raise SystemExit(
            f"{label}: expected {EPISODES} episodes, found {len(run['episodes'])}"
        )
    for episode in run["episodes"]:
        if episode["honest_score"] != HONEST_ALL_FALSE_SCORE:
            raise SystemExit(
                f"{label}: episode {episode['episode']} has honest score "
                f"{episode['honest_score']}, expected {HONEST_ALL_FALSE_SCORE}"
            )


def load_all_false() -> list[dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for pattern in ALL_FALSE_PATTERNS:
        for path in sorted(glob.glob(str(REPO_ROOT / pattern))):
            run = _load_run(Path(path))
            _validate_run(run, path)
            runs.setdefault(run["seed"], run)
    if len(runs) != 25:
        raise SystemExit(f"expected 25 all-false runs, found {len(runs)}")
    return [runs[seed] for seed in sorted(runs)]


def load_gcg_run(directory: Path, label: str) -> dict[str, Any]:
    paths = sorted(glob.glob(str(REPO_ROOT / directory / "run_*" / "run.json")))
    if len(paths) != 1:
        raise SystemExit(f"{label}: expected exactly one run.json, found {len(paths)}")
    run = _load_run(Path(paths[0]))
    _validate_run(run, paths[0])
    return run


def aggregate_all_false(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, list[float]] = {}
    for run in runs:
        for episode in run["episodes"]:
            by_episode.setdefault(episode["episode"], []).append(episode["normalized"])

    rows = []
    for episode_index in range(1, EPISODES + 1):
        values = by_episode[episode_index]
        rows.append(
            {
                "episode": episode_index,
                "mean": base._mean(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def write_csv(
    all_false_rows: list[dict[str, Any]],
    gcg_runs: dict[str, dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "episode",
                "normalized_mean",
                "normalized_min",
                "normalized_max",
            ]
        )
        for row in all_false_rows:
            writer.writerow(
                [
                    "all-false mean (25 runs)",
                    row["episode"],
                    row["mean"],
                    row["min"],
                    row["max"],
                ]
            )
        for label, run in gcg_runs.items():
            for episode in run["episodes"]:
                writer.writerow(
                    [
                        label,
                        episode["episode"],
                        episode["normalized"],
                        episode["normalized"],
                        episode["normalized"],
                    ]
                )
    return path


def make_plot(
    all_false_rows: list[dict[str, Any]],
    gcg_runs: dict[str, dict[str, Any]],
    out_dir: Path,
    stem: str,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_facecolor("#FAFAFA")

    xs = [row["episode"] for row in all_false_rows]
    mean = [row["mean"] for row in all_false_rows]
    low = [row["min"] for row in all_false_rows]
    high = [row["max"] for row in all_false_rows]
    style = STYLE["all-false mean"]
    ax.fill_between(
        xs, low, high, color=style["color"], alpha=0.10, linewidth=0, zorder=1
    )
    ax.plot(
        xs,
        mean,
        color=style["color"],
        marker=style["marker"],
        markersize=5.5,
        linewidth=2.1,
        label="all-false mean (25 runs)",
        zorder=3,
    )

    all_values = low + high
    for label, run in gcg_runs.items():
        style = STYLE[label]
        xs_run = [episode["episode"] for episode in run["episodes"]]
        ys_run = [episode["normalized"] for episode in run["episodes"]]
        all_values.extend(ys_run)
        ax.plot(
            xs_run,
            ys_run,
            color=style["color"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=1.9,
            label=label,
            zorder=4,
        )

    ax.axhline(
        1.0,
        color=base.MUTED,
        linestyle="--",
        linewidth=1.15,
        zorder=2,
        label="honest play",
    )

    ax.set_xlim(0.5, EPISODES + 0.5)
    ax.set_ylim(min(0.85, min(all_values) - 0.05), max(all_values) + 0.08)
    ax.set_xticks(range(1, EPISODES + 1))
    ax.set_xlabel("Episode", fontsize=10, color=base.TEXT)
    ax.set_ylabel(
        "Normalized score\n(observed / honest expected)",
        fontsize=10,
        color=base.TEXT,
    )
    ax.grid(axis="y", color=base.GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=9, colors=base.TEXT)
    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")

    ax.legend(
        loc="center right",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#D1D5DB",
        fontsize=9,
        title="Series",
        title_fontsize=9,
    )
    ax.set_title(
        "GCG validation runs vs all-false mean — 10% transmission loss",
        fontsize=14,
        color=base.TEXT,
        fontweight="semibold",
        pad=12,
    )
    fig.text(
        0.5,
        0.018,
        "All-false mean and min–max band: 25 runs at 10% loss (5 initial + 20 reruns). "
        "GCG runs: single reinjections on Qwen3.6-27B.\n"
        "Scores normalized by the honest all-false reference (A+B = 10 per episode).",
        ha="center",
        va="bottom",
        fontsize=8,
        color=base.MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.17)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_false_runs = load_all_false()
    gcg_runs = {
        label: load_gcg_run(Path(directory), label)
        for label, directory in GCG_RUN_DIRS.items()
    }
    all_false_rows = aggregate_all_false(all_false_runs)

    print("all-false normalized mean:", [round(r["mean"], 3) for r in all_false_rows])
    print("all-false normalized min: ", [round(r["min"], 3) for r in all_false_rows])
    print("all-false normalized max: ", [round(r["max"], 3) for r in all_false_rows])
    for label, run in gcg_runs.items():
        print(f"{label}:", [round(e["normalized"], 3) for e in run["episodes"]])

    csv_path = write_csv(all_false_rows, gcg_runs, args.out_dir, args.stem)
    pdf_path = make_plot(
        all_false_rows, gcg_runs, args.out_dir, args.stem, args.dpi
    )
    print(f"\nWrote {pdf_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
