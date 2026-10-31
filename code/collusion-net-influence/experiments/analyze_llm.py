"""Summarise LLM RPS games: behaviour and detector rates with 95% CIs, and detector AUROC (RQ 2.2).

Reads every <model_dir>/*_run*.json (from llm_rps.py) and <model_dir>/judge.csv (from judge_chat.py)
under the given results dir. Writes summary.csv, detectors.csv and flags_by_condition.png next to them.
Usage:
    python experiments/analyze_llm.py results/llm_rps
"""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from netinf.labels import colluded, colluded_any

CONDITIONS = ["independent", "channel", "channel_shared", "instructed", "channel_public_hint"]
FIXED, CALIB = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 of the dataviz reference palette


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with Wilson 95% interval, in percent."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, max(0.0, 100 * (centre - half)), min(100.0, 100 * (centre + half))


def mean_ci(x, n_boot: int = 2000, rng=np.random.default_rng(0)) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def auroc(scores, labels) -> float:
    """P(score of a colluding game > score of a non-colluding game), ties count half."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    keep = ~np.isnan(s)
    s, y = s[keep], y[keep]
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def auroc_ci(scores, labels, n_boot: int = 2000, rng=np.random.default_rng(1)) -> tuple[float, float, float]:
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    boots = [auroc(s[i], y[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(n_boot))]
    boots = [b for b in boots if not np.isnan(b)]
    if not boots:
        return auroc(s, y), float("nan"), float("nan")
    return auroc(s, y), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(t: tuple[float, float, float], d: int = 1) -> str:
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    summary, detectors = [], []
    model_dirs = sorted(d for d in args.results_dir.iterdir() if d.is_dir() and any(d.glob("*_run*.json")))
    for mdir in model_dirs:
        games = [json.loads(f.read_text()) for f in sorted(mdir.glob("*_run*.json"))]
        for cond in CONDITIONS:
            g = [x for x in games if x["condition"] == cond]
            if not g:
                continue
            n = len(g)
            fixed = sum(x["detection"]["prev"]["fixed"] == [0, 1] for x in g)
            calib = sum(x["detection"]["prev"]["calibrated"] == [0, 1] for x in g)
            # any flagged pair: in conditions without A+B collusion, every flag is a false accusation
            fixed_any = sum(x["detection"]["prev"]["fixed"] is not None for x in g)
            calib_any = sum(x["detection"]["prev"]["calibrated"] is not None for x in g)
            def vals(key, scale=1.0):
                return [scale * x.get(key, np.nan) for x in g]

            summary.append({
                "model": mdir.name, "condition": cond, "games": n, "rounds": len(g[0]["actions"]),
                "assist_rate_pct": fmt(mean_ci(vals("assist_rate", 100))),
                "assist_any_pct": fmt(mean_ci(vals("assist_any", 100))),
                "assist_a_to_b_pct": fmt(mean_ci(vals("assist_a_to_b", 100))),
                "assist_b_to_a_pct": fmt(mean_ci(vals("assist_b_to_a", 100))),
                "alternation_rate": fmt(mean_ci(vals("alternation_rate")), 3),
                "assist_imbalance": fmt(mean_ci(vals("assist_imbalance")), 3),
                "pair_advantage": fmt(mean_ci([x["pair_advantage"] for x in g]), 3),
                "c_mean_reward": fmt(mean_ci(vals("c_mean_reward")), 3),
                "c_match_assisted_rate": fmt(mean_ci(vals("c_match_assisted_rate")), 3),
                "c_third_move_rate": fmt(mean_ci(vals("c_third_move_rate")), 3),
                "colluding_games_any": sum(bool(x.get("colluded_any", False)) for x in g),
                "flag_fixed_pct": fmt(wilson(fixed, n)), "flag_calibrated_pct": fmt(wilson(calib, n)),
                "any_pair_fixed_pct": fmt(wilson(fixed_any, n)), "any_pair_calibrated_pct": fmt(wilson(calib_any, n)),
                "parse_failures": sum(x["parse_failures"] for x in g),
                "_fixed": wilson(fixed, n), "_calib": wilson(calib, n),
            })

        # The moves-only detector needs no judge: label and score both come from the run JSON.
        y_act = [colluded(x["actions"])[0] for x in games]
        act_scores = [x["detection"]["prev"]["net_AB"] for x in games]
        detectors.append({"model": mdir.name, "detector": "act (net influence)", "games": len(games),
                          "colluding_games": sum(y_act), "auroc": fmt(auroc_ci(act_scores, y_act), 2)})
        if any("colluded_any" in x for x in games):
            y_any = [colluded_any(x["actions"])[0] for x in games]
            detectors.append({"model": mdir.name, "detector": "act (net influence, any-assist label)",
                              "games": len(games), "colluding_games": sum(y_any),
                              "auroc": fmt(auroc_ci(act_scores, y_any), 2)})

        judge = mdir / "judge.csv"
        if judge.exists():
            rows = list(csv.DictReader(judge.open()))
            y = [r["colluded"] == "True" for r in rows]
            num = lambda k: [float(r[k]) if r[k] not in ("", "None") else np.nan for r in rows]  # noqa: E731
            for name, scores in (("talk (judge, messages only)", num("talk")),
                                 ("talk+act (judge, messages + moves)", num("talk_act"))):
                detectors.append({"model": mdir.name, "detector": name, "games": len(rows),
                                  "colluding_games": sum(y), "auroc": fmt(auroc_ci(scores, y), 2)})

    if not summary:
        raise SystemExit(f"no games under {args.results_dir}")
    for name, rows in (("summary.csv", summary), ("detectors.csv", detectors)):
        if rows:
            with open(args.results_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if not k.startswith("_")], extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(open(args.results_dir / name).read())

    # Figure: how often each detector accuses A+B, per condition and model
    models = sorted({r["model"] for r in summary})
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.6), squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        rows = [r for r in summary if r["model"] == model]
        x = np.arange(len(rows))
        for off, key, color, label in ((-0.18, "_fixed", FIXED, "fixed alpha=0.05"),
                                       (0.18, "_calib", CALIB, "calibrated")):
            v = np.array([r[key] for r in rows])
            ax.bar(x + off, v[:, 0], 0.34, color=color, label=label)
            ax.errorbar(x + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="#555", lw=1)
        ax.set_xticks(x, [r["condition"].replace("_", "\n") for r in rows])
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("% of games where A+B is flagged")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results_dir / "flags_by_condition.png", dpi=160)


if __name__ == "__main__":
    main()
