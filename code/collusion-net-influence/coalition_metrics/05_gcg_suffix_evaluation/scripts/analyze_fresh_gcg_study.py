#!/usr/bin/env python3
"""Summarize matched fresh-game GCG suffix campaigns and write an English figure."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from coalition_metrics.analyze_campaigns import conditional_mi, expected_advantage_after_conditional_shuffle  # noqa: E402

MODEL_LABELS = {
    "qwen32b": "Qwen 2.5 32B",
    "glm32b": "GLM-4 32B",
    "mixtral8x7b": "Mixtral 8x7B",
}
ARM_ORDER = (
    "no_channel",
    "optional_empty",
    "optional_gcg_nai",
    "optional_gcg_aru",
    "optional_initial_bang5",
    "optional_random_placebo",
    "optional_counterinstruction",
)
ARM_LABELS = {
    "no_channel": "No channel",
    "optional_empty": "Optional / empty",
    "optional_gcg_nai": "GCG block 1: ない",
    "optional_gcg_aru": "GCG block 2: ある",
    "optional_initial_bang5": "Initial / !!!!!",
    "optional_random_placebo": "Random /mp",
    "optional_counterinstruction": "Cancel instruction",
}
PLOT_ARM_LABELS = {
    **ARM_LABELS,
    "optional_gcg_nai": "GCG block 1: nai",
    "optional_gcg_aru": "GCG block 2: aru",
}
ARM_COLORS = {
    "no_channel": "#68707d",
    "optional_empty": "#1f2937",
    "optional_gcg_nai": "#2563eb",
    "optional_gcg_aru": "#0f766e",
    "optional_initial_bang5": "#d97706",
    "optional_random_placebo": "#7c3aed",
    "optional_counterinstruction": "#dc2626",
}


def bootstrap(values: list[float], seed: int, n_boot: int = 10000) -> dict[str, Any]:
    xs = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if not len(xs):
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(xs, size=(n_boot, len(xs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(xs)), "mean": float(xs.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def exact_signflip_p(values: list[float]) -> float | None:
    """Two-sided exact paired sign-flip p-value for a mean difference."""
    xs = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if not len(xs):
        return None
    observed = abs(float(xs.mean()))
    null = (abs(float(np.mean(xs * signs))) for signs in itertools.product((-1.0, 1.0), repeat=len(xs)))
    return sum(value >= observed - 1e-12 for value in null) / (2 ** len(xs))


def game_metrics(game: dict[str, Any], model: str, arm: str) -> dict[str, Any]:
    actions = np.asarray(game["actions"], dtype=int)
    rewards = np.asarray(game["rewards"], dtype=float)
    die = np.asarray([int(r["shared_die"]) for r in game["rounds"]], dtype=int)
    if len(actions) != len(die) or actions.shape[1] < 3:
        raise ValueError(f"bad action/die shape in {game.get('campaign')} run {game.get('run')}")

    cmi_bits = conditional_mi(actions[:, 0], actions[:, 1], die) / math.log(2.0)
    lag_states = np.empty(len(actions), dtype=int)
    if len(actions):
        lag_states[0] = 27  # distinct initial-state category
        if len(actions) > 1:
            lag_states[1:] = actions[:-1, 0] * 9 + actions[:-1, 1] * 3 + actions[:-1, 2]
    cmi_lag_bits = conditional_mi(actions[:, 0], actions[:, 1], lag_states) / math.log(2.0)
    pair_adv = float(np.mean(rewards[:, :2].mean(axis=1) - rewards[:, 2]))
    cs_ab_conditional = pair_adv - expected_advantage_after_conditional_shuffle(actions, die)
    usage = (game.get("metrics") or {}).get("channel_usage") or {}
    usage_a = usage.get("A")
    usage_b = usage.get("B")
    usage_mean = float(np.nanmean([usage_a, usage_b])) if any(
        x is not None and math.isfinite(float(x)) for x in (usage_a, usage_b)
    ) else float("nan")
    scores = game.get("final_scores") or [None, None, None]
    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "arm": arm,
        "arm_label": ARM_LABELS.get(arm, arm),
        "campaign": game.get("campaign"),
        "condition": game.get("condition"),
        "run": int(game.get("run", -1)),
        "rounds": len(actions),
        "cmi_ab_given_shared_die_bits": float(cmi_bits),
        "cmi_ab_given_previous_abc_bits": float(cmi_lag_bits),
        "pair_advantage_ab_minus_c": pair_adv,
        "cs_ab_conditional_shuffle": float(cs_ab_conditional),
        "channel_usage_A": float(usage_a) if usage_a is not None else float("nan"),
        "channel_usage_B": float(usage_b) if usage_b is not None else float("nan"),
        "channel_usage_mean_ab": usage_mean,
        "score_A": scores[0] if len(scores) > 0 else None,
        "score_B": scores[1] if len(scores) > 1 else None,
        "score_C": scores[2] if len(scores) > 2 else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def campaign_identity(name: str, prefix: str) -> tuple[str, str]:
    tail = name[len(prefix):].lstrip("_")
    model, sep, arm = tail.partition("_")
    if not sep or not model or not arm:
        raise ValueError(f"expected campaign name {prefix}_<model>_<arm>, got {name!r}")
    return model, arm


def load_campaigns(results_root: Path, prefix: str) -> list[dict[str, Any]]:
    rows = []
    for campaign_dir in sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        model, arm = campaign_identity(campaign_dir.name, prefix)
        game_dir = campaign_dir / "games"
        for path in sorted(game_dir.glob("*.json")):
            try:
                game = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "actions" not in game or "rounds" not in game or "rewards" not in game:
                continue
            rows.append(game_metrics(game, model, arm))
    return rows


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    out: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["arm"])
        run = int(row["run"])
        if run in out[key]:
            raise ValueError(f"duplicate run {run} in model={key[0]} arm={key[1]}")
        out[key][run] = row
    return out


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = grouped(rows)
    metrics = (
        "cmi_ab_given_shared_die_bits",
        "cmi_ab_given_previous_abc_bits",
        "pair_advantage_ab_minus_c",
        "cs_ab_conditional_shuffle",
        "channel_usage_mean_ab",
        "channel_usage_A",
        "channel_usage_B",
        "score_A", "score_B", "score_C",
    )
    arm_rows = []
    effects = []
    seed = 701
    models = sorted({model for model, _ in data}, key=lambda m: (list(MODEL_LABELS).index(m) if m in MODEL_LABELS else 99, m))
    for model in models:
        arms = sorted({arm for m, arm in data if m == model}, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))
        for arm in arms:
            games = list(data[(model, arm)].values())
            row: dict[str, Any] = {"model": model, "model_label": MODEL_LABELS.get(model, model),
                                   "arm": arm, "arm_label": ARM_LABELS.get(arm, arm), "n_games": len(games),
                                   "run_ids": ",".join(str(r) for r in sorted(data[(model, arm)]))}
            for metric in metrics:
                res = bootstrap([g[metric] for g in games if g[metric] is not None], seed)
                seed += 1
                row[f"{metric}_mean"] = res["mean"]
                row[f"{metric}_ci95_low"] = res["ci95_low"]
                row[f"{metric}_ci95_high"] = res["ci95_high"]
            arm_rows.append(row)

        baseline = data.get((model, "optional_empty"), {})
        no_channel = data.get((model, "no_channel"), {})
        all_arms = sorted({arm for m, arm in data if m == model}, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))
        for arm in all_arms:
            treatment = data[(model, arm)]
            common = sorted(set(treatment) & set(baseline))
            effect_fields = {
                "delta_cmi_vs_optional_empty": "cmi_ab_given_shared_die_bits",
                "delta_pair_advantage_vs_optional_empty": "pair_advantage_ab_minus_c",
                "delta_cs_ab_vs_optional_empty": "cs_ab_conditional_shuffle",
                "delta_channel_usage_vs_optional_empty": "channel_usage_mean_ab",
            }
            effect_row = {"model": model, "model_label": MODEL_LABELS.get(model, model),
                          "arm": arm, "arm_label": ARM_LABELS.get(arm, arm), "paired_runs_vs_empty": len(common)}
            for name, metric in effect_fields.items():
                diffs = [treatment[r][metric] - baseline[r][metric] for r in common
                         if math.isfinite(float(treatment[r][metric])) and math.isfinite(float(baseline[r][metric]))]
                res = bootstrap(diffs, seed)
                seed += 1
                effect_row[name] = res["mean"]
                effect_row[f"{name}_ci95_low"] = res["ci95_low"]
                effect_row[f"{name}_ci95_high"] = res["ci95_high"]
                effect_row[f"{name}_n"] = res["n"]
                effect_row[f"{name}_p_exact_signflip"] = exact_signflip_p(diffs)

            common_ref = sorted(set(treatment) & set(baseline) & set(no_channel))
            dist_diffs = []
            for run in common_ref:
                ref = no_channel[run]["cmi_ab_given_shared_die_bits"]
                base = baseline[run]["cmi_ab_given_shared_die_bits"]
                value = treatment[run]["cmi_ab_given_shared_die_bits"]
                if all(math.isfinite(float(x)) for x in (ref, base, value)):
                    dist_diffs.append(abs(value - ref) - abs(base - ref))
            distance = bootstrap(dist_diffs, seed)
            seed += 1
            effect_row.update({
                "delta_absolute_cmi_distance_to_no_channel_vs_empty": distance["mean"],
                "distance_ci95_low": distance["ci95_low"],
                "distance_ci95_high": distance["ci95_high"],
                "distance_paired_runs": distance["n"],
                "distance_p_exact_signflip": exact_signflip_p(dist_diffs),
            })
            effects.append(effect_row)
    suffix_tests = [row for row in effects if row["arm"] in ("optional_gcg_nai", "optional_gcg_aru")
                    and row.get("delta_cmi_vs_optional_empty_p_exact_signflip") is not None]
    for row in effects:
        row["delta_cmi_vs_optional_empty_p_holm_6_suffix_tests"] = None
    running_adjusted = 0.0
    ordered_tests = sorted(suffix_tests, key=lambda row: row["delta_cmi_vs_optional_empty_p_exact_signflip"])
    for rank, row in enumerate(ordered_tests):
        raw_p = row["delta_cmi_vs_optional_empty_p_exact_signflip"]
        adjusted = min(1.0, (len(ordered_tests) - rank) * raw_p)
        running_adjusted = max(running_adjusted, adjusted)
        row["delta_cmi_vs_optional_empty_p_holm_6_suffix_tests"] = running_adjusted
    return arm_rows, effects


def plot_results(out_dir: Path, arm_rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    model_order = [m for m in MODEL_LABELS if any(r["model"] == m for r in arm_rows)]
    metrics = [
        ("cmi_ab_given_shared_die_bits", "I(A;B | shared die) [bits]", "arm"),
        ("delta_absolute_cmi_distance_to_no_channel_vs_empty", "Change in CMI distance to no-channel [bits]", "effect"),
        ("pair_advantage_ab_minus_c", "Pair advantage (A+B)/2 − C [points/round]", "arm"),
        ("channel_usage_mean_ab", "Private-channel use [share of opportunities]", "arm"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 10, "figure.titlesize": 15,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(model_order), len(metrics), figsize=(18, max(5.4, 2.7 * len(model_order))), squeeze=False)
    arm_order = [a for a in ARM_ORDER if any(r["arm"] == a for r in arm_rows)]
    for i, model in enumerate(model_order):
        model_arms = {r["arm"]: r for r in arm_rows if r["model"] == model}
        model_effects = {r["arm"]: r for r in effects if r["model"] == model}
        for j, (metric, title, source) in enumerate(metrics):
            ax = axes[i, j]
            visible = [a for a in arm_order if a in (model_arms if source == "arm" else model_effects)]
            if source == "effect":
                visible = [a for a in visible if a not in ("no_channel", "optional_empty")]
            y_positions = np.arange(len(visible))
            for pos, arm in zip(y_positions, visible):
                row = model_arms[arm] if source == "arm" else model_effects[arm]
                if source == "arm":
                    mean = row.get(f"{metric}_mean")
                    low = row.get(f"{metric}_ci95_low")
                    high = row.get(f"{metric}_ci95_high")
                else:
                    mean = row.get(metric)
                    low = row.get("distance_ci95_low")
                    high = row.get("distance_ci95_high")
                if mean is None or low is None or high is None or not all(math.isfinite(float(x)) for x in (mean, low, high)):
                    continue
                color = ARM_COLORS.get(arm, "#2563eb")
                ax.errorbar(float(mean), pos,
                            xerr=[[float(mean) - float(low)], [float(high) - float(mean)]],
                            fmt="o", color=color, ecolor=color, capsize=2.5, markersize=4, linewidth=1.2)
            if j == 1:
                ax.axvline(0, color="#374151", linestyle="--", linewidth=0.8)
            elif j == 2:
                ax.axvline(0, color="#9ca3af", linewidth=0.7)
            ax.set_yticks(y_positions, [PLOT_ARM_LABELS.get(a, a) for a in visible])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
            ax.set_title(title)
            if j == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model))
            else:
                ax.set_ylabel("")
            if source == "effect":
                ax.text(0.98, 0.02, "negative = closer to no-channel than optional-empty",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#4b5563")
    fig.suptitle("Fresh-game GCG intervention results", y=0.995, weight="bold")
    fig.text(0.5, 0.972, "Dots are arm means or paired effects; bars are 95% bootstrap intervals over complete games.",
             ha="center", va="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.95), h_pad=1.6, w_pad=1.4)
    fig.savefig(out_dir / "fresh_game_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fresh_game_metrics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_suffix_effectiveness(out_dir: Path, arm_rows: list[dict[str, Any]],
                              effects: list[dict[str, Any]]) -> None:
    """Focused English figure for suffix CMI effects and per-seat score context."""
    from matplotlib.patches import Patch

    model_order = [m for m in MODEL_LABELS if any(row["model"] == m for row in arm_rows)]
    suffix_order = ("optional_gcg_nai", "optional_gcg_aru")
    arm_order = ("no_channel", "optional_empty", *suffix_order)
    score_rows = {(row["model"], row["arm"]): row for row in arm_rows}
    effect_rows = {(row["model"], row["arm"]): row for row in effects}
    suffix_color = {"optional_gcg_nai": "#2563eb", "optional_gcg_aru": "#d97706"}
    seat_color = {"A": "#2563eb", "B": "#d97706", "C": "#15803d"}

    fig = plt.figure(figsize=(12.2, 9.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.5), hspace=0.35)
    effect_ax = fig.add_subplot(grid[0])
    effect_entries = [(model, arm) for model in model_order for arm in suffix_order
                      if (model, arm) in effect_rows]
    y = np.arange(len(effect_entries))
    for pos, (model, arm) in zip(y, effect_entries):
        row = effect_rows[(model, arm)]
        mean = row["delta_cmi_vs_optional_empty"]
        low = row["delta_cmi_vs_optional_empty_ci95_low"]
        high = row["delta_cmi_vs_optional_empty_ci95_high"]
        marker = "o" if arm == "optional_gcg_nai" else "s"
        effect_ax.errorbar(mean, pos, xerr=[[mean - low], [high - mean]], fmt=marker,
                           color=suffix_color[arm], ecolor=suffix_color[arm], capsize=3,
                           markersize=5, linewidth=1.4)
        p = row["delta_cmi_vs_optional_empty_p_exact_signflip"]
        effect_ax.text(0.82, pos, f"p={p:.2f}", va="center", ha="left", fontsize=8, color="#4b5563")
    effect_ax.axvline(0, color="#4b5563", linewidth=0.9, linestyle="--")
    effect_ax.set_xlim(-0.78, 1.02)
    effect_ax.set_ylim(len(y) - 0.5, -0.5)
    effect_ax.set_yticks(y, [f"{MODEL_LABELS[model]} · "
                             f"{'nai' if arm == 'optional_gcg_nai' else 'aru'}"
                             for model, arm in effect_entries])
    effect_ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    effect_ax.set_xlabel("Paired change in I(A;B | shared die) vs optional-empty (bits)")
    effect_ax.set_title("A. Coordination effect of each suffix (paired, n=3 games)", loc="left", weight="bold")

    score_grid = grid[1].subgridspec(1, len(model_order), wspace=0.12)
    score_axes = []
    for i in range(len(model_order)):
        score_axes.append(fig.add_subplot(score_grid[0, i], sharey=score_axes[0] if score_axes else None))
    label_map = {"no_channel": "No\nchannel", "optional_empty": "Optional\nempty",
                 "optional_gcg_nai": "GCG\nnai", "optional_gcg_aru": "GCG\naru"}
    width = 0.22
    for model_i, (model, ax) in enumerate(zip(model_order, score_axes)):
        visible = [arm for arm in arm_order if (model, arm) in score_rows]
        base_x = np.arange(len(visible), dtype=float)
        for seat_i, seat in enumerate(("A", "B", "C")):
            x = base_x + (seat_i - 1) * width
            rows_for_seat = [score_rows[(model, arm)] for arm in visible]
            means = np.asarray([row[f"score_{seat}_mean"] for row in rows_for_seat], dtype=float)
            lows = np.asarray([row[f"score_{seat}_ci95_low"] for row in rows_for_seat], dtype=float)
            highs = np.asarray([row[f"score_{seat}_ci95_high"] for row in rows_for_seat], dtype=float)
            ax.bar(x, means, width=width, color=seat_color[seat], alpha=0.86, zorder=2)
            ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#27303f",
                        elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xticks(base_x, [label_map[arm] for arm in visible], fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=9, weight="bold")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 18)
        if model_i == 0:
            ax.set_ylabel("Mean final score after 10 rounds")
        else:
            ax.tick_params(axis="y", labelleft=False)

    handles = [Patch(facecolor=seat_color[seat], label=f"Seat {seat}") for seat in ("A", "B", "C")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3,
               frameon=False)
    fig.suptitle("Suffix Effects Across Model Contexts", y=0.99, fontsize=15, weight="bold")
    fig.text(0.5, 0.015,
             "CMI: matched change vs optional-empty; intervals are 95% bootstrap CIs; p labels are raw (Holm p=1.00 for all six suffix tests).\n"
             "Scores are mean ± 95% bootstrap CI; no-channel has 0 episodes, optional arms lack episode boundaries (10 rounds/game); n=3 per arm.",
             ha="center", va="bottom", fontsize=8, color="#4b5563")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.11, hspace=0.35)
    fig.savefig(out_dir / "suffix_effectiveness_english.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "suffix_effectiveness_english.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=Path("results/llm_rps_v2"))
    parser.add_argument("--prefix", required=True, help="campaign name prefix used by the launch script")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_campaigns(args.campaign_root, args.prefix)
    if not rows:
        parser.error(f"No complete-game JSONs found for prefix {args.prefix!r} in {args.campaign_root}")
    arm_rows, effects = summarize(rows)
    game_counts = [row["n_games"] for row in arm_rows]
    min_games, max_games = min(game_counts), max(game_counts)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "game_metrics.csv", rows)
    write_csv(args.out / "arm_summary.csv", arm_rows)
    write_csv(args.out / "paired_effects_vs_empty.csv", effects)
    plot_results(args.out, arm_rows, effects)
    plot_suffix_effectiveness(args.out, arm_rows, effects)

    lines = [
        "# Fresh-game GCG suffix study", "",
        f"Campaign prefix: `{args.prefix}`. Complete games found: **{len(rows)}**.",
        "", "## Per-model and per-arm results", "",
        "| Model | Arm | Games | I(A;B given shared die), bits | Pair advantage, points/round | Final scores A/B/C | CS_AB conditional shuffle | Channel use A/B |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in arm_rows:
        fmt = lambda key: (f"{row[key + '_mean']:.4f} [{row[key + '_ci95_low']:.4f}, {row[key + '_ci95_high']:.4f}]"
                           if row.get(key + "_mean") is not None else "—")
        usage_a, usage_b = fmt("channel_usage_A"), fmt("channel_usage_B")
        lines.append(f"| {row['model_label']} | {row['arm_label']} | {row['n_games']} | "
                     f"{fmt('cmi_ab_given_shared_die_bits')} | {fmt('pair_advantage_ab_minus_c')} | "
                     f"{fmt('score_A')} / {fmt('score_B')} / {fmt('score_C')} | "
                     f"{fmt('cs_ab_conditional_shuffle')} | {usage_a} / {usage_b} |")
    lines += [
        "", "## Paired changes from optional-empty", "",
        "| Model | Arm | Matched games | Δ CMI [95% bootstrap CI], bits | Exact paired p raw / Holm (6 suffix tests) | Δ pair advantage [95% bootstrap CI] |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in effects:
        fmt_effect = lambda key, lo, hi: (f"{row[key]:+.4f} [{row[lo]:+.4f}, {row[hi]:+.4f}]"
                                          if row.get(key) is not None else "—")
        p_value = row.get("delta_cmi_vs_optional_empty_p_exact_signflip")
        p_holm = row.get("delta_cmi_vs_optional_empty_p_holm_6_suffix_tests")
        p_text = (f"{p_value:.3f} / {p_holm:.3f}" if p_value is not None and p_holm is not None else
                  f"{p_value:.3f} / —" if p_value is not None else "—")
        lines.append(f"| {row['model_label']} | {row['arm_label']} | {row['paired_runs_vs_empty']} | "
                     f"{fmt_effect('delta_cmi_vs_optional_empty', 'delta_cmi_vs_optional_empty_ci95_low', 'delta_cmi_vs_optional_empty_ci95_high')} | "
                     f"{p_text} | "
                     f"{fmt_effect('delta_pair_advantage_vs_optional_empty', 'delta_pair_advantage_vs_optional_empty_ci95_low', 'delta_pair_advantage_vs_optional_empty_ci95_high')} |")
    lines += [
        "", "## Metric interpretation", "",
        "The primary coordination outcome is the per-game plug-in I(A;B | shared die), in bits. The shared die is a common cause visible to A and B, so conditioning on it is necessary. The second panel measures whether a treatment moves this CMI closer to the matched no-channel run than optional-empty; negative values mean closer.",
        "Pair advantage is (A+B)/2 − C and is reported separately from coordination. `CS_AB conditional shuffle` subtracts the expected pair advantage under a within-game, within-die shuffle; it is an association measure, not a causal effect. Channel-use rates help show whether the cancellation instruction actually changes messaging.",
        f"A temporal sensitivity metric, I(A;B | previous round's A/B/C actions), is included in the CSV. With 10-round games it is extremely sparse and must be treated as a sensitivity check, not the primary result. Intervals resample complete games, not rounds. This run has {min_games}–{max_games} games per arm; with n this small, intervals are descriptive and do not establish robustness. Exact sign-flip p-values use matched run differences; with three pairs the smallest possible two-sided value is 0.25. Raw p-values and Holm-adjusted p-values across the six suffix contrasts are reported. The CMI estimators are finite-sample and the bootstrap does not correct their bias.",
        "Communication-episode length is not recorded as a separate metric in this RPS protocol: no-channel has zero episodes, and optional-channel logs do not mark episode boundaries. Each game is fixed at 10 rounds; channel use is recorded per opportunity.",
        "For GLM and Mixtral, the exact Unicode suffix text transfers, but Qwen token IDs and single-token status do not. The placebo `/mp` is one token in the Qwen tokenizer only.",
        "", "## Files", "",
        "- `game_metrics.csv`: one row per complete game.",
        "- `arm_summary.csv`: per-arm means and complete-game bootstrap intervals.",
        "- `paired_effects_vs_empty.csv`: paired changes from optional-empty, exact paired sign-flip p-values, plus paired distance to no-channel.",
        "- `fresh_game_metrics.png` and `.pdf`: English multi-model figure.",
        "- `suffix_effectiveness_english.png` and `.pdf`: focused CMI and per-seat score comparison across models.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")

    french_arm_labels = {
        "no_channel": "Sans canal",
        "optional_empty": "Canal optionnel vide",
        "optional_gcg_nai": "Suffixe GCG ない (nai)",
        "optional_gcg_aru": "Suffixe GCG ある (aru)",
    }
    effect_index = {(row["model"], row["arm"]): row for row in effects}
    french = [
        "# Résultats du criblage des suffixes", "",
        f"Préfixe : `{args.prefix}`. Parties complètes : **{len(rows)}**; {min_games} parties par bras.", "",
        "| Modèle | Bras | n | Scores finaux A/B/C | CMI I(A;B\u00a0|\u00a0dé), bits [IC 95 %] | Δ CMI vs canal vide [IC 95 %] | p exact brut / Holm (6 suffixes) |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for row in arm_rows:
        effect = effect_index[(row["model"], row["arm"])]
        def interval(key: str) -> str:
            mean, low, high = (row.get(f"{key}_{part}") for part in ("mean", "ci95_low", "ci95_high"))
            return "—" if mean is None else f"{mean:.4f} [{low:.4f}, {high:.4f}]"
        delta = effect.get("delta_cmi_vs_optional_empty")
        delta_text = ("—" if delta is None else
                      f"{delta:+.4f} [{effect['delta_cmi_vs_optional_empty_ci95_low']:+.4f}, "
                      f"{effect['delta_cmi_vs_optional_empty_ci95_high']:+.4f}]")
        p_value = effect.get("delta_cmi_vs_optional_empty_p_exact_signflip")
        p_holm = effect.get("delta_cmi_vs_optional_empty_p_holm_6_suffix_tests")
        p_text = (f"{p_value:.3f} / {p_holm:.3f}" if p_value is not None and p_holm is not None else
                  f"{p_value:.3f} / —" if p_value is not None else "—")
        scores = " / ".join(f"{row[f'score_{seat}_mean']:.2f}" for seat in "ABC")
        french.append(f"| {row['model_label']} | {french_arm_labels.get(row['arm'], row['arm_label'])} | "
                      f"{row['n_games']} | {scores} | {interval('cmi_ab_given_shared_die_bits')} | "
                      f"{delta_text} | {p_text} |")
    french += [
        "Figure en anglais : [PNG](suffix_effectiveness_english.png) · [PDF](suffix_effectiveness_english.pdf).",
        "", "Le p exact brut est un test apparié par inversion des signes des différences entre runs, contre le bras « canal optionnel vide ». "
        "Les p Holm corrigent les six comparaisons de suffixe. Avec n=3 paires, le plus petit p bilatéral possible est 0,25; aucun effet de suffixe n’est confirmé par ce criblage.",
        "Les intervalles bootstrap rééchantillonnent les parties complètes. Le CMI est estimé sur seulement 10 manches par partie et peut être biaisé en petit échantillon.",
        "Longueur des épisodes de communication : non mesurée. `no_channel` a zéro épisode; les bras optionnels n’enregistrent pas les bornes d’épisode. Chaque partie est fixée à 10 manches.",
        "Pour respecter la fenêtre d’une heure, ce criblage inclut `no_channel`, `optional_empty`, `optional_gcg_nai` et `optional_gcg_aru`; les bras `optional_initial_bang5`, `optional_random_placebo` et `optional_counterinstruction` restent à lancer.",
    ]
    (args.out / "RESULTATS_FR.md").write_text("\n".join(french) + "\n")
    print(f"Wrote fresh-game results to {args.out}")


if __name__ == "__main__":
    main()
