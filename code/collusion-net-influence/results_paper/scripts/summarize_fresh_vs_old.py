#!/usr/bin/env python3
"""Summarize fresh suffix runs against the original v4 references."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from plot_mi_curves import cmi_bits, game_rows, row_counts  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "llm_rps_v2"
OUT_DIR = ROOT / "figures_to_keep"
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
SUFFIXES = (
    ("nai", "ない"),
    ("aru", "ある"),
)
MAX_ROUNDS = 10
N_PERM = 10_000
SEED = 20261026


def load_games(folder: Path) -> list[dict]:
    games = [json.loads(path.read_text()) for path in sorted(folder.glob("*_run*.json"))]
    if not games:
        raise SystemExit(f"No games found under {folder}")
    return games


def run_advantage(game: dict, max_rounds: int = MAX_ROUNDS) -> float:
    rewards = np.asarray(game["rewards"], dtype=float)[:max_rounds]
    return float(np.mean(rewards[:, :2].mean(axis=1) - rewards[:, 2]))


def run_mi(game: dict, max_rounds: int = MAX_ROUNDS) -> float:
    rows = game_rows(game)[:max_rounds]
    return float(cmi_bits(row_counts(rows)[None])[0])


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


def format_p(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []

    for model_key, model_label in MODELS:
        old_channel = load_games(RESULTS / V4_CAMPAIGNS[model_key] / "games")
        old_channel = [g for g in old_channel if g.get("condition") == "channel_optional"]
        old_no_channel = load_games(RESULTS / V4_CAMPAIGNS[model_key] / "games")
        old_no_channel = [g for g in old_no_channel if g.get("condition") == "no_channel"]

        ref_channel_adv = np.asarray([run_advantage(g) for g in old_channel])
        ref_channel_mi = np.asarray([run_mi(g) for g in old_channel])
        ref_no_adv = np.asarray([run_advantage(g) for g in old_no_channel])
        ref_no_mi = np.asarray([run_mi(g) for g in old_no_channel])

        for suffix_key, suffix_text in SUFFIXES:
            suffix_games = load_games(RESULTS / f"{FRESH_PREFIX}_{model_key}_optional_gcg_{suffix_key}" / "games")
            adv = np.asarray([run_advantage(g) for g in suffix_games])
            mi = np.asarray([run_mi(g) for g in suffix_games])

            p_channel = permutation_pvalue(adv, ref_channel_adv, rng, args.permutations)
            p_no_channel = permutation_pvalue(adv, ref_no_adv, rng, args.permutations)

            rows.append({
                "model": model_label,
                "suffix": suffix_text,
                "n_suffix": len(suffix_games),
                "pair_advantage": float(adv.mean()),
                "mi_prev_joint_bits": float(mi.mean()),
                "n_old_reference": len(old_channel),
                "old_channel_adv": float(ref_channel_adv.mean()),
                "p_adv_vs_old_channel": float(p_channel),
                "old_no_channel_adv": float(ref_no_adv.mean()),
                "p_adv_vs_old_no_channel": float(p_no_channel),
                "old_channel_mi": float(ref_channel_mi.mean()),
                "old_no_channel_mi": float(ref_no_mi.mean()),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "fresh_suffix_vs_old_references.csv"
    md_path = args.out_dir / "fresh_suffix_vs_old_references.md"

    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| Model | Suffix | n | Pair advantage (pts/round) | MI (bits) | "
        "Old channel adv (n=20) | p (adv vs old channel) | "
        "Old no-channel adv (n=20) | p (adv vs old no-channel) | "
        "Old channel MI (n=20) | Old no-channel MI (n=20) |"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        display = dict(row)
        display["p_adv_vs_old_channel"] = format_p(row["p_adv_vs_old_channel"])
        display["p_adv_vs_old_no_channel"] = format_p(row["p_adv_vs_old_no_channel"])
        lines.append(
            "| {model} | {suffix} | {n_suffix} | {pair_advantage:+.3f} | {mi_prev_joint_bits:.3f} | "
            "{old_channel_adv:+.3f} | {p_adv_vs_old_channel} | "
            "{old_no_channel_adv:+.3f} | {p_adv_vs_old_no_channel} | "
            "{old_channel_mi:.3f} | {old_no_channel_mi:.3f} |".format(**display)
        )

    md = "\n".join([
        "# Fresh suffix runs vs original v4 references",
        "",
        "Suffix rows: fresh GCG screen, 3 complete games × 10 rounds per arm. "
        "Old references: original v4 campaigns, 20 complete games, truncated to the first 10 rounds.",
        "",
        "Pair advantage is the mean per-round `(A+B)/2 − C`. MI is the mean per-run "
        "`I(A;B | previous joint action)` in bits over 10 rounds. "
        "Old references are not matched to the suffix runs (different runs and seed offsets).",
        "",
        "p-values are two-sided permutation tests on the difference of means "
        f"({args.permutations:,} permutations, no correction). No p-values are reported for MI by request. "
        "With only 3 suffix games, the tests are exploratory and have limited resolution.",
        "",
        *lines,
        "",
    ])
    md_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
