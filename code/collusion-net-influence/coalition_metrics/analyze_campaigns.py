"""Exploratory, context-conditioned metrics for the v2 RPS campaign JSON files.

Run from the repository root:
    python coalition_metrics/analyze_campaigns.py results/llm_rps_v2/<campaign> [...]

Permutation p-values are exploratory because the within-game histories make rounds
serially dependent. See coalition_metrics/README.md before interpreting them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("AB", "AC", "BC")
PLAYER_NAMES = ("A", "B", "C")


def conditional_mi(x: np.ndarray, y: np.ndarray, states: np.ndarray) -> float:
    """Plug-in I(X;Y|S), in nats."""
    n = len(x)
    if n == 0:
        return float("nan")
    n_states = int(states.max()) + 1
    counts = np.bincount(states * 9 + x * 3 + y, minlength=n_states * 9).reshape(n_states, 3, 3)
    totals = counts.sum(axis=(1, 2))
    valid = totals > 0
    pxy = counts[valid] / totals[valid, None, None]
    px = pxy.sum(axis=2, keepdims=True)
    py = pxy.sum(axis=1, keepdims=True)
    denom = px * py
    nz = pxy > 0
    terms = np.zeros_like(pxy)
    terms[nz] = pxy[nz] * np.log(pxy[nz] / denom[nz])
    return float(np.sum(terms.sum(axis=(1, 2)) * totals[valid] / n))


def strata(game_ids: np.ndarray, die: np.ndarray) -> list[np.ndarray]:
    keys = np.rec.fromarrays((game_ids, die), names="game,die")
    return [np.flatnonzero(keys == key) for key in np.unique(keys)]


def permute_columns(actions: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator,
                    columns: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    permuted = actions.copy()
    for group in groups:
        if len(group) < 2:
            continue
        for column in columns:
            permuted[group, column] = actions[rng.permutation(group), column]
    return permuted


def pair_mis(actions: np.ndarray, die: np.ndarray) -> np.ndarray:
    return np.asarray([conditional_mi(actions[:, i], actions[:, j], die) for i, j in PAIRS])


def payoff_advantage(rewards: np.ndarray) -> float:
    return float(np.mean(rewards[:, :2].mean(axis=1) - rewards[:, 2]))


def expected_advantage_after_conditional_shuffle(actions: np.ndarray, die: np.ndarray) -> float:
    """Exact expected advantage when B's moves are shuffled only within die strata."""
    total = 0.0
    n = len(actions)
    for value in np.unique(die):
        idx = np.flatnonzero(die == value)
        b_counts = np.bincount(actions[idx, 1], minlength=3) / len(idx)
        for t in idx:
            expected = 0.0
            for b, probability in enumerate(b_counts):
                payoff = pot_payoff(np.array([actions[t, 0], b, actions[t, 2]]))
                expected += probability * (payoff[:2].mean() - payoff[2])
            total += expected
    return total / n


def game_statistics(game: dict) -> dict:
    actions = np.asarray(game["actions"], dtype=int)
    rewards = np.asarray(game["rewards"], dtype=float)
    die = np.asarray([r["shared_die"] for r in game["rounds"]], dtype=int)
    mis = pair_mis(actions, die)
    observed_adv = payoff_advantage(rewards)
    transfer = observed_adv - expected_advantage_after_conditional_shuffle(actions, die)
    return {
        "campaign": game["campaign"], "condition": game["condition"], "run": game["run"],
        "n_rounds": len(actions), "final_A": game["final_scores"][0], "final_B": game["final_scores"][1],
        "final_C": game["final_scores"][2], "pair_advantage": observed_adv,
        "conditional_shuffle_transfer": transfer,
        **{f"cmi_{name}_given_die": float(mis[k]) for k, name in enumerate(PAIR_NAMES)},
    }


def pot_payoff(actions: np.ndarray) -> np.ndarray:
    """Three-player 3-point pot payoff, matching src/netinf/games/rps_pot.py."""
    actions = np.asarray(actions, dtype=int)
    rewards = np.zeros(3, dtype=float)
    distinct = np.unique(actions)
    if len(distinct) != 2:
        return rewards
    # R=0 beats S=2; P=1 beats R=0; S=2 beats P=1.
    beats = {0: 2, 1: 0, 2: 1}
    winner_move = int(distinct[0] if beats[int(distinct[0])] == int(distinct[1]) else distinct[1])
    winners = actions == winner_move
    rewards[winners] = 3.0 / int(winners.sum())
    return rewards


def mean_ci_by_game(values: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--out", type=Path, default=Path("coalition_metrics"))
    args = parser.parse_args()
    if args.permutations < 99:
        parser.error("use at least 99 permutations; 999 or more is recommended")
    rng = np.random.default_rng(args.seed)
    game_rows = []
    raw_games = []
    for campaign in args.campaigns:
        paths = sorted((campaign / "games").glob("*_run*.json"))
        if not paths:
            parser.error(f"no game JSON files found in {campaign / 'games'}")
        for path in paths:
            game = json.loads(path.read_text())
            game_rows.append(game_statistics(game))
            raw_games.append((game["campaign"], game["condition"], game))

    summary_rows = []
    keys = sorted({(row["campaign"], row["condition"]) for row in game_rows})
    for campaign, condition in keys:
        cell = [row for row in game_rows if row["campaign"] == campaign and row["condition"] == condition]
        result = {"campaign": campaign, "condition": condition, "games": len(cell)}
        for key in ("pair_advantage", "conditional_shuffle_transfer", "cmi_AB_given_die", "cmi_AC_given_die", "cmi_BC_given_die"):
            mean, lo, hi = mean_ci_by_game(np.asarray([row[key] for row in cell]), rng)
            result.update({f"{key}_mean": mean, f"{key}_ci95_low": lo, f"{key}_ci95_high": hi})
        selected = [g for camp, cond, g in raw_games if camp == campaign and cond == condition]
        acts = np.concatenate([np.asarray(g["actions"], dtype=int) for g in selected])
        dies = np.concatenate([[r["shared_die"] for r in g["rounds"]] for g in selected]).astype(int)
        game_ids = np.concatenate([np.full(len(g["actions"]), i, dtype=int) for i, g in enumerate(selected)])
        groups = strata(game_ids, dies)
        observed_mis = pair_mis(acts, dies)
        exceed = np.zeros(len(PAIRS), dtype=int)
        for _ in range(args.permutations):
            shuffled = permute_columns(acts, groups, rng)
            exceed += pair_mis(shuffled, dies) >= observed_mis
        pvals = (1 + exceed) / (args.permutations + 1)
        for pair in PAIR_NAMES:
            k = PAIR_NAMES.index(pair)
            result[f"cmi_{pair}_pooled_given_die"] = float(observed_mis[k])
            result[f"p_{pair}_pooled_raw"] = float(pvals[k])
            result[f"p_{pair}_pooled_bonferroni"] = float(min(1.0, pvals[k] * 3))
        summary_rows.append(result)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "game_metrics.csv", game_rows)
    write_csv(args.out / "condition_summary.csv", summary_rows)
    print(f"Wrote {args.out / 'game_metrics.csv'}")
    print(f"Wrote {args.out / 'condition_summary.csv'}")


if __name__ == "__main__":
    main()
