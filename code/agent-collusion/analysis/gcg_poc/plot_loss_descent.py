#!/usr/bin/env python3
"""Plot the GCG loss descent for the collusion suffix.

Y axis: GCG loss (mean NLL of the target token), log scale.
X axis: GCG iteration.
The final suffix is written above the axes. When the result file contains the
per-iteration ``strings`` field (nanogcg >= 0.3), the beginning of the suffix is
also annotated every ``--annotate-every`` steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

TEXT = "#1F2937"
MUTED = "#4B5563"
GRID = "#E5E7EB"
LINE = "#3B5B8C"
POINT = "#4F7A5A"
BEST = "#8C3B3B"
ANNOT_BG = "#F8FAFC"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = REPO_ROOT / "analysis/results/gcg_poc/result.json"
DEFAULT_OUT = REPO_ROOT / "analysis/results/gcg_poc/gcg_loss_descent"


def _snippet(text: str, tokens: int = 4) -> str:
    """Return the beginning of a suffix, token-ish and single-line."""
    cleaned = " ".join(str(text).split())
    parts = cleaned.split(" ")
    head = " ".join(parts[:tokens])
    if len(parts) > tokens:
        head += " …"
    return head


def _final_suffix(result: dict[str, Any]) -> str:
    suffix = str(result.get("suffix") or "").strip()
    return suffix


def load_series(result: dict[str, Any], *, baseline: bool) -> tuple[list[int], list[float]]:
    losses = [float(value) for value in result.get("losses") or []]
    if not losses:
        raise SystemExit("result.json has no 'losses' field")
    xs = list(range(1, len(losses) + 1))
    ys = losses
    if baseline:
        initial = result.get("initial") or {}
        mean_nll = initial.get("mean_nll")
        if mean_nll is not None:
            xs = [0] + xs
            ys = [float(mean_nll)] + ys
    return xs, ys


def plot(
    result: dict[str, Any],
    output: Path,
    *,
    baseline: bool,
    annotate_every: int,
    annotate_tokens: int,
) -> list[Path]:
    xs, ys = load_series(result, baseline=baseline)
    strings = result.get("strings") or []
    suffix = _final_suffix(result)

    step_offset = 1 if baseline else 0
    losses_only = ys[step_offset:]
    best_index = min(range(len(losses_only)), key=losses_only.__getitem__)
    best_step = best_index + 1
    best_loss = losses_only[best_index]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, ys, color=LINE, linewidth=1.6, zorder=2)
    ax.scatter(xs, ys, s=22, color=POINT, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(
        [best_step],
        [best_loss],
        s=90,
        color=BEST,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("GCG iteration", fontsize=10, color=TEXT, labelpad=8)
    ax.set_ylabel("Loss (mean NLL of target token, log scale)", fontsize=10, color=TEXT)
    ax.set_xlim(-0.8, max(xs) + 0.8)
    ticks = list(range(0, max(xs) + 1, 5))
    if 0 not in ticks:
        ticks = [0] + ticks
    ax.set_xticks(ticks)
    if baseline:
        ax.set_xticklabels(
            ["0\nno suffix" if tick == 0 else str(tick) for tick in ticks]
        )
    ax.tick_params(axis="both", labelsize=8.5, colors=MUTED, length=3)

    ax.grid(axis="y", which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(axis="y", which="minor", color="#F1F5F9", linewidth=0.5, zorder=0)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    init_str = result.get("optim_str_init")
    if init_str:
        ax.annotate(
            f"init: {_snippet(str(init_str), annotate_tokens)}",
            xy=(1, losses_only[0]),
            xytext=(3.2, losses_only[0] * 2.9),
            fontsize=7.5,
            color=MUTED,
            va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": ANNOT_BG, "edgecolor": GRID},
            arrowprops={"arrowstyle": "->", "color": "#94A3B8", "linewidth": 0.7},
        )

    if strings and len(strings) >= len(losses_only):
        last_annotated = None
        for index in range(0, len(losses_only), max(1, annotate_every)):
            step = index + 1
            if step in (1, best_step):
                continue
            xy = (step, losses_only[index])
            ax.annotate(
                _snippet(str(strings[index]), annotate_tokens),
                xy=xy,
                xytext=(step + 0.5, losses_only[index] * 4.0),
                fontsize=6.5,
                color=MUTED,
                rotation=0,
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": GRID,
                    "alpha": 0.9,
                },
                arrowprops={"arrowstyle": "-", "color": "#CBD5E1", "linewidth": 0.6},
            )
            last_annotated = step

    best_label = f"best loss = {best_loss:.2e}\n(step {best_step})"
    if suffix:
        best_label += f"\nsuffix: {_snippet(suffix, annotate_tokens)}"
    ax.annotate(
        best_label,
        xy=(best_step, best_loss),
        xytext=(best_step - 9.5, best_loss * 3.2),
        fontsize=8,
        color=BEST,
        fontweight="bold",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#FDF2F2",
            "edgecolor": "#E7C4C4",
        },
        arrowprops={"arrowstyle": "->", "color": BEST, "linewidth": 0.8},
    )

    if suffix:
        token_count = int(result.get("suffix_token_count") or 0)
        label = "Final suffix"
        if token_count:
            label += f" ({token_count} tokens)"
        fig.text(
            0.5,
            0.955,
            label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )
        fig.text(
            0.5,
            0.912,
            suffix,
            ha="center",
            va="top",
            fontsize=12.5,
            color=TEXT,
            family="monospace",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": ANNOT_BG,
                "edgecolor": GRID,
            },
        )

    subtitle = (
        f"{result.get('model_id', 'model')} · target {result.get('target', '?')!r} · "
        f"{len(losses_only)} GCG steps"
    )
    fig.text(0.5, 0.868, subtitle, ha="center", va="top", fontsize=8, color=MUTED)

    if not strings:
        fig.text(
            0.5,
            0.016,
            "Per-iteration suffixes were not persisted in result.json; "
            "only the initialization and final suffix are shown.",
            ha="center",
            va="bottom",
            fontsize=7,
            color=MUTED,
            style="italic",
        )

    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.10, right=0.97)
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    path = output.with_suffix(".pdf")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not prepend the empty-suffix baseline as iteration 0.",
    )
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=3,
        help="Annotate every N iterations when per-step strings are available.",
    )
    parser.add_argument(
        "--annotate-tokens",
        type=int,
        default=4,
        help="Number of suffix tokens to show in each annotation.",
    )
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    paths = plot(
        result,
        args.output,
        baseline=not args.no_baseline,
        annotate_every=args.annotate_every,
        annotate_tokens=args.annotate_tokens,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
