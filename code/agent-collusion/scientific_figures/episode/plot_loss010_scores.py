#!/usr/bin/env python3
"""Plot normalized Alice+Bob rewards for the four 10% loss schedules.

Each configuration is the mean of the five initial runs (seeds 7-11), with a
min-max band over those runs at every episode. Scores are divided by the mean
reward the two agents would have earned by playing honestly on the same
sequences.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT = "#1F2937"
MUTED = "#4B5563"
GRID = "#E5E7EB"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = REPO_ROOT / "results/synthetic/grid/enabled/loss010"
DEFAULT_OUT_DIR = REPO_ROOT / "scientific_figures/episode"
DEFAULT_STEM = "loss010_scores_normalized"

SCHEDULES = ("all-false", "mixed-40", "mixed-80", "all-true")
EXPECTED_SEEDS = (7, 8, 9, 10, 11)
EXPECTED_EPISODES = 10
EXPECTED_LOSS_RATE = 0.10

STYLE = {
    "all-false": {"color": "#8C3B3B", "marker": "o", "label": "all-false"},
    "mixed-40": {"color": "#C47F17", "marker": "s", "label": "mixed-40"},
    "mixed-80": {"color": "#4F7A5A", "marker": "^", "label": "mixed-80"},
    "all-true": {"color": "#2F5D8C", "marker": "D", "label": "all-true"},
}


def _sequence_config(data: dict[str, Any]) -> dict[str, Any]:
    return (data.get("run_config") or {}).get("synthetic_sequence") or {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _honest_agent_reward(parts: list[bool]) -> int:
    return 5 + 10 * int(all(parts))


def load_runs(base_dir: Path) -> dict[str, list[dict[str, Any]]]:
    schedules: dict[str, list[dict[str, Any]]] = {}
    for schedule in SCHEDULES:
        paths = sorted(glob.glob(str(base_dir / schedule / "seed*" / "run_*" / "run.json")))
        if not paths:
            raise SystemExit(f"No run.json found for {schedule} under {base_dir}")

        runs = []
        for path in paths:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            sequence = _sequence_config(data)
            seed = int(sequence.get("seed", -1))
            loss_rate = float(
                sequence.get("loss_rate", data.get("run_config", {}).get("synthetic_loss_rate", -1))
            )
            episodes = []
            for episode in data.get("results") or []:
                truth = episode["truth"]
                alice_parts = list(truth["alice_parts"])
                bob_parts = list(truth["bob_parts"])
                alice = int(episode["agents"]["alice"]["reward_total"])
                bob = int(episode["agents"]["bob"]["reward_total"])
                honest_alice = _honest_agent_reward(alice_parts)
                honest_bob = _honest_agent_reward(bob_parts)
                episodes.append(
                    {
                        "episode": int(episode["episode_index"]),
                        "alice": alice,
                        "bob": bob,
                        "run_score": (alice + bob) / 2,
                        "honest_alice": honest_alice,
                        "honest_bob": honest_bob,
                        "honest_run_score": (honest_alice + honest_bob) / 2,
                    }
                )
            episodes.sort(key=lambda item: item["episode"])
            runs.append(
                {
                    "path": path,
                    "seed": seed,
                    "loss_rate": loss_rate,
                    "episodes": episodes,
                }
            )
        runs.sort(key=lambda run: run["seed"])
        schedules[schedule] = runs
    return schedules


def validate(schedules: dict[str, list[dict[str, Any]]]) -> None:
    for schedule, runs in schedules.items():
        if len(runs) != len(EXPECTED_SEEDS):
            raise SystemExit(
                f"{schedule}: expected {len(EXPECTED_SEEDS)} runs, found {len(runs)}"
            )
        seeds = tuple(run["seed"] for run in runs)
        if seeds != EXPECTED_SEEDS:
            raise SystemExit(f"{schedule}: expected seeds {EXPECTED_SEEDS}, found {seeds}")
        for run in runs:
            if run["loss_rate"] != EXPECTED_LOSS_RATE:
                raise SystemExit(
                    f"{run['path']}: expected loss rate {EXPECTED_LOSS_RATE}, "
                    f"found {run['loss_rate']}"
                )
            if len(run["episodes"]) != EXPECTED_EPISODES:
                raise SystemExit(
                    f"{run['path']}: expected {EXPECTED_EPISODES} episodes, "
                    f"found {len(run['episodes'])}"
                )


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        for episode in run["episodes"]:
            by_episode.setdefault(episode["episode"], []).append(episode)

    honest_scalar = _mean(
        [
            episode["honest_run_score"]
            for run in runs
            for episode in run["episodes"]
        ]
    )

    rows = []
    for episode_index in range(1, EXPECTED_EPISODES + 1):
        items = by_episode[episode_index]
        scores = [item["run_score"] for item in items]
        mean_score = _mean(scores)
        rows.append(
            {
                "episode": episode_index,
                "mean_score": mean_score,
                "min_score": min(scores),
                "max_score": max(scores),
                "honest_scalar": honest_scalar,
                "normalized_mean": mean_score / honest_scalar,
                "normalized_min": min(scores) / honest_scalar,
                "normalized_max": max(scores) / honest_scalar,
            }
        )
    return {"honest_scalar": honest_scalar, "rows": rows}


def write_csvs(
    schedules: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{stem}_raw.csv"
    summary_path = out_dir / f"{stem}_summary.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schedule",
                "seed",
                "episode",
                "alice_score",
                "bob_score",
                "run_score",
                "honest_alice",
                "honest_bob",
                "honest_run_score",
            ]
        )
        for schedule in SCHEDULES:
            for run in schedules[schedule]:
                for episode in run["episodes"]:
                    writer.writerow(
                        [
                            schedule,
                            run["seed"],
                            episode["episode"],
                            episode["alice"],
                            episode["bob"],
                            episode["run_score"],
                            episode["honest_alice"],
                            episode["honest_bob"],
                            episode["honest_run_score"],
                        ]
                    )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schedule",
                "episode",
                "mean_score",
                "min_score",
                "max_score",
                "honest_scalar",
                "normalized_mean",
                "normalized_min",
                "normalized_max",
            ]
        )
        for schedule in SCHEDULES:
            aggregate_data = aggregates[schedule]
            for row in aggregate_data["rows"]:
                writer.writerow(
                    [
                        schedule,
                        row["episode"],
                        row["mean_score"],
                        row["min_score"],
                        row["max_score"],
                        row["honest_scalar"],
                        row["normalized_mean"],
                        row["normalized_min"],
                        row["normalized_max"],
                    ]
                )
    return raw_path, summary_path


def make_plot(
    aggregates: dict[str, dict[str, Any]],
    out_dir: Path,
    stem: str,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_facecolor("#FAFAFA")

    all_values = []
    for schedule in SCHEDULES:
        rows = aggregates[schedule]["rows"]
        xs = [row["episode"] for row in rows]
        mean = [row["normalized_mean"] for row in rows]
        low = [row["normalized_min"] for row in rows]
        high = [row["normalized_max"] for row in rows]
        all_values.extend(low + high)
        style = STYLE[schedule]
        ax.fill_between(
            xs,
            low,
            high,
            color=style["color"],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        honest = aggregates[schedule]["honest_scalar"]
        ax.plot(
            xs,
            mean,
            color=style["color"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=1.9,
            label=f"{style['label']} (honest ref {honest:.1f})",
            zorder=3,
        )

    ax.axhline(
        1.0,
        color=MUTED,
        linestyle="--",
        linewidth=1.15,
        zorder=2,
        label="honest play",
    )

    lower = min(0.65, min(all_values) - 0.05)
    upper = max(all_values) + 0.08
    ax.set_xlim(0.5, EXPECTED_EPISODES + 0.5)
    ax.set_ylim(lower, upper)
    ax.set_xticks(range(1, EXPECTED_EPISODES + 1))
    ax.set_xlabel("Episode", fontsize=10, color=TEXT)
    ax.set_ylabel(
        "Normalized score\n(observed / honest expected)",
        fontsize=10,
        color=TEXT,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=9, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")

    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9,
        title="Configuration (honest reference)",
        title_fontsize=9,
    )
    ax.set_title(
        "10% transmission loss — normalized mean score (Alice + Bob)",
        fontsize=14,
        color=TEXT,
        fontweight="semibold",
        pad=12,
    )
    fig.text(
        0.5,
        0.018,
        "Mean Alice/Bob scores over the 5 initial runs (seeds 7–11) per configuration; "
        "band = min–max across the 5 runs.\n"
        "Honest reference per agent and episode: 5 + 10·1[group entirely valid], "
        "then averaged over the 5 runs × 10 episodes.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.17)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def report_findings(aggregates: dict[str, dict[str, Any]]) -> None:
    print("Honest scalar by configuration (mean reward of a truthful pair):")
    for schedule in SCHEDULES:
        print(f"  {schedule:10s} {aggregates[schedule]['honest_scalar']:.3f}")

    all_true = [row["normalized_mean"] for row in aggregates["all-true"]["rows"]]
    if max(abs(value - 1.0) for value in all_true) > 1e-9:
        raise SystemExit("all-true should be normalized to 1.0 on every episode")

    all_false = [row["normalized_mean"] for row in aggregates["all-false"]["rows"]]
    if min(all_false[2:]) < 1.0:
        raise SystemExit("all-false normalized score should exceed 1 from episode 3")
    print("\nall-false normalized means:", [round(value, 3) for value in all_false])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedules = load_runs(args.base_dir)
    validate(schedules)
    aggregates = {schedule: aggregate(schedules[schedule]) for schedule in SCHEDULES}
    report_findings(aggregates)
    raw_path, summary_path = write_csvs(schedules, aggregates, args.out_dir, args.stem)
    pdf_path = make_plot(aggregates, args.out_dir, args.stem, args.dpi)
    print(f"\nWrote {pdf_path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
