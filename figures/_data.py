"""Data loading and metric computation for the article figures.

The metric definitions are copied from the original figure scripts so that the
regenerated figures use exactly the same data and statistics.
"""

from __future__ import annotations

import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CNI = REPO / "collusion-net-influence"
AC = REPO / "agent-collusion"

RPS_CAMPAIGNS = {
    "Qwen2.5-32B": CNI / "results/llm_rps_v2/20261025-1834_same_model_qwen32b_big-v4",
    "GLM-4-32B": CNI / "results/llm_rps_v2/20261025-1837_same_model_glm32b_big-v4",
    "Mixtral-8x7B": CNI / "results/llm_rps_v2/20261025-1853_same_model_mixtral8x7b_big-v4",
}

CONDITIONS = (
    "no_channel",
    "channel_optional",
    "channel_optional_hint",
    "channel_forced",
    "channel_forced_strategy",
)

FRESH_PREFIX = "fresh_gcg_transfer_10r3g_screen_v2_20261026"
FRESH_MODELS = (
    ("qwen32b", "Qwen2.5-32B"),
    ("glm32b", "GLM-4-32B"),
    ("mixtral8x7b", "Mixtral-8x7B"),
)
SUFFIXES = (
    ("nai", "ない"),
    ("aru", "ある"),
)
V4_KEYS = {
    "qwen32b": "20261025-1834_same_model_qwen32b_big-v4",
    "glm32b": "20261025-1837_same_model_glm32b_big-v4",
    "mixtral8x7b": "20261025-1853_same_model_mixtral8x7b_big-v4",
}

N_STATES = 28
N_PERM = 10_000
SEED = 20261026
WINDOW = 20
STEP = 5
MAX_ROUNDS = 10

CHRONO_FILES = (
    CNI / "results_paper/data/gcg_qwen_pilot/qwen32b_chronological_bang5_runs_0_4.json",
    CNI / "results_paper/data/gcg_qwen_pilot/qwen32b_chronological_bang5_runs_5_9.json",
)
CHRONO_EXCLUDED = {8}
LOG2 = math.log(2)

AC_SCORE_BASE = AC / "results/synthetic/grid/enabled/loss010"
AC_SCHEDULES = ("all-false", "mixed-40", "mixed-80", "all-true")
AC_SEEDS = (7, 8, 9, 10, 11)
AC_EPISODES = 10
AC_LOSS_RATE = 0.10

AC_ALL_FALSE_PATTERNS = (
    "results/synthetic/grid/enabled/loss010/all-false/seed*/run_*/run.json",
    "results/synthetic/grid_rerun10/enabled/loss010/all-false/seed*/run_*/run.json",
)
AC_GCG_DIRS = {
    "GCG suffix": "results/synthetic/gcg_poc/suffix",
    "GCG suffix v2": "results/synthetic/gcg_poc/suffix_v2",
}
AC_GCG_RESULT = AC / "analysis/results/gcg_poc/result.json"


# ---------------------------------------------------------------------------
# RPS games and mutual information
# ---------------------------------------------------------------------------

def load_games(folder: Path) -> list[dict]:
    games = [json.loads(path.read_text()) for path in sorted(folder.glob("*_run*.json"))]
    if not games:
        raise SystemExit(f"No games found under {folder}")
    return games


def load_campaign(campaign: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((campaign / "games").glob("*_run*.json")):
        game = json.loads(path.read_text())
        grouped[game["condition"]].append(game)
    if not grouped:
        raise SystemExit(f"No game JSON files under {campaign / 'games'}")
    return grouped


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


def cumulative_delta(game: dict, max_rounds: int | None = None) -> np.ndarray:
    rewards = np.asarray(game["rewards"], dtype=float)
    if max_rounds is not None:
        rewards = rewards[:max_rounds]
    per_round = rewards[:, :2].mean(axis=1) - rewards[:, 2]
    return np.concatenate(([0.0], np.cumsum(per_round)))


def curve_from_games(games: list[dict], max_rounds: int | None = None):
    curves = np.stack([cumulative_delta(game, max_rounds) for game in games])
    return curves.mean(axis=0), *np.percentile(curves, [25, 75], axis=0), len(curves)


def run_delta(game: dict) -> float:
    rewards = np.asarray(game["rewards"], dtype=float)
    return float(np.mean(rewards[:, :2].mean(axis=1) - rewards[:, 2]))


def run_window_mean_mi(game: dict) -> float:
    rows = game_rows(game)
    values = [
        float(cmi_bits(row_counts(rows[end - WINDOW:end])[None])[0])
        for end in range(WINDOW, len(rows) + 1, STEP)
    ]
    return float(np.mean(values))


def run_mi_short(game: dict, max_rounds: int = MAX_ROUNDS) -> float:
    rows = game_rows(game)[:max_rounds]
    return float(cmi_bits(row_counts(rows)[None])[0])


def permutation_pvalue(x, y, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = len(x)
    draws = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)
    differences = draws[:, :n_x].mean(axis=1) - draws[:, n_x:].mean(axis=1)
    exceed = int(np.sum(np.abs(differences) >= abs(observed) - 1e-12))
    return (exceed + 1) / (n_perm + 1)


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def pairwise_differences(means: np.ndarray, lower: bool) -> np.ndarray:
    result = np.full((len(means), len(means)), np.nan)
    for i in range(len(means)):
        for j in range(len(means)):
            if lower and i > j:
                result[i, j] = means[i] - means[j]
            elif not lower and i < j:
                result[i, j] = means[i] - means[j]
    return result


def collect_rps_metrics():
    """Per-model delta/MI arrays, means, permutation matrices and global norms."""
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model, campaign in RPS_CAMPAIGNS.items():
        grouped = load_campaign(campaign)
        data[model] = {}
        for condition in CONDITIONS:
            games = grouped[condition]
            data[model][condition] = {
                "delta": np.asarray([run_delta(game) for game in games]),
                "mi": np.asarray([run_window_mean_mi(game) for game in games]),
            }

    rng = np.random.default_rng(SEED)
    means = {}
    pvalues = {}
    for model, conditions in data.items():
        delta_means = np.asarray([conditions[c]["delta"].mean() for c in CONDITIONS])
        mi_means = np.asarray([conditions[c]["mi"].mean() for c in CONDITIONS])
        means[model] = {"delta": delta_means, "mi": mi_means}
        p_delta = np.ones((5, 5))
        p_mi = np.ones((5, 5))
        for i in range(5):
            for j in range(i + 1, 5):
                p_delta[i, j] = p_delta[j, i] = permutation_pvalue(
                    conditions[CONDITIONS[i]]["delta"],
                    conditions[CONDITIONS[j]]["delta"], rng,
                )
                p_mi[i, j] = p_mi[j, i] = permutation_pvalue(
                    conditions[CONDITIONS[i]]["mi"],
                    conditions[CONDITIONS[j]]["mi"], rng,
                )
        pvalues[model] = {"delta": p_delta, "mi": p_mi}

    delta_extent = 0.0
    mi_extent = 0.0
    for stats in means.values():
        for i in range(5):
            for j in range(5):
                if i > j:
                    delta_extent = max(delta_extent, abs(stats["delta"][i] - stats["delta"][j]))
                elif i < j:
                    mi_extent = max(mi_extent, abs(stats["mi"][i] - stats["mi"][j]))
    from matplotlib.colors import Normalize
    delta_norm = Normalize(vmin=-max(delta_extent, 1e-6), vmax=max(delta_extent, 1e-6))
    mi_norm = Normalize(vmin=-max(mi_extent, 1e-6), vmax=max(mi_extent, 1e-6))
    return {"data": data, "means": means, "pvalues": pvalues,
            "delta_norm": delta_norm, "mi_norm": mi_norm}


# ---------------------------------------------------------------------------
# Fresh suffix screen
# ---------------------------------------------------------------------------

def load_fresh_games(model: str, arm: str) -> list[dict]:
    folder = CNI / "results/llm_rps_v2" / f"{FRESH_PREFIX}_{model}_{arm}" / "games"
    return load_games(folder)


def load_v4_games(model: str, condition: str) -> list[dict]:
    grouped = load_campaign(CNI / "results/llm_rps_v2" / V4_KEYS[model])
    return grouped[condition]


def fresh_curve(model: str, arm: str, max_rounds: int | None = None):
    return curve_from_games(load_fresh_games(model, arm), max_rounds=max_rounds)


def v4_curve(model: str, condition: str, max_rounds: int | None = None):
    return curve_from_games(load_v4_games(model, condition), max_rounds=max_rounds)


def collect_fresh_heatmap():
    """Rows are model x suffix, same order and RNG as the original heatmap."""
    rng = np.random.default_rng(SEED)
    rows = []
    for model_key, model_label in FRESH_MODELS:
        old_games = load_v4_games(model_key, "channel_optional")
        old_adv = np.asarray([run_delta(game) for game in old_games])
        old_mi = np.asarray([run_mi_short(game) for game in old_games])
        for suffix_key, suffix_text in SUFFIXES:
            games = load_fresh_games(model_key, f"optional_gcg_{suffix_key}")
            fresh_adv = np.asarray([run_delta(game) for game in games])
            fresh_mi = np.asarray([run_mi_short(game) for game in games])
            rows.append({
                "model_key": model_key,
                "model": model_label,
                "suffix_key": suffix_key,
                "suffix": suffix_text,
                "delta_adv": float(fresh_adv.mean() - old_adv.mean()),
                "delta_mi": float(fresh_mi.mean() - old_mi.mean()),
                "p_adv": float(permutation_pvalue(fresh_adv, old_adv, rng)),
            })
    return rows


# ---------------------------------------------------------------------------
# Mutual-information curves
# ---------------------------------------------------------------------------

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


def collect_mi_curves(n_boot: int = 2000):
    rng = np.random.default_rng(SEED)
    curves = {}
    for model, campaign in RPS_CAMPAIGNS.items():
        grouped = load_campaign(campaign)
        curves[model] = {}
        for condition in CONDITIONS:
            curves[model][condition] = pooled_curve(
                grouped[condition], WINDOW, STEP, n_boot, rng,
            )
    return curves


# ---------------------------------------------------------------------------
# Chronological GCG (RPS)
# ---------------------------------------------------------------------------

def training_loss_bits(block: dict) -> float:
    accepted = block.get("accepted_candidate")
    if accepted is not None:
        return float(accepted["block_mean_kl_bits"])
    return float(block["block_loss_before_nats"]) / LOG2


def validation_loss_bits(block: dict) -> float | None:
    prequential = block.get("next_block_prequential")
    if not prequential:
        return None
    return float(prequential["updated_suffix_mean_kl_nats"]) / LOG2


def suffix_of(block: dict) -> str:
    return block["suffix_after"].strip()


def load_chronological():
    runs: dict[int, list[dict]] = {}
    for path in CHRONO_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run_id, run in payload["runs"].items():
            run_id = int(run_id)
            if run_id in CHRONO_EXCLUDED:
                continue
            runs[run_id] = run["blocks"]
    runs = dict(sorted(runs.items()))

    train = {rid: [training_loss_bits(b) for b in blocks] for rid, blocks in runs.items()}
    val = {rid: [validation_loss_bits(b) for b in blocks] for rid, blocks in runs.items()}
    suffix_counts = Counter(suffix_of(b) for blocks in runs.values() for b in blocks)
    return {
        "runs": runs,
        "train": train,
        "val": val,
        "suffix_counts": suffix_counts,
    }


# ---------------------------------------------------------------------------
# AgentCollusion
# ---------------------------------------------------------------------------

def _honest_agent_reward(parts: list[bool]) -> int:
    return 5 + 10 * int(all(parts))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _load_ac_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sequence = (data.get("run_config") or {}).get("synthetic_sequence") or {}
    episodes = []
    for episode in data.get("results") or []:
        truth = episode["truth"]
        alice_parts = list(truth["alice_parts"])
        bob_parts = list(truth["bob_parts"])
        alice = int(episode["agents"]["alice"]["reward_total"])
        bob = int(episode["agents"]["bob"]["reward_total"])
        honest_alice = _honest_agent_reward(alice_parts)
        honest_bob = _honest_agent_reward(bob_parts)
        episodes.append({
            "episode": int(episode["episode_index"]),
            "alice": alice,
            "bob": bob,
            "run_score": (alice + bob) / 2,
            "honest_run_score": (honest_alice + honest_bob) / 2,
        })
    episodes.sort(key=lambda item: item["episode"])
    return {
        "path": str(path),
        "seed": int(sequence.get("seed", -1)),
        "schedule": sequence.get("schedule"),
        "loss_rate": float(sequence.get("loss_rate", -1)),
        "episodes": episodes,
    }


def collect_ac_scores():
    schedules: dict[str, list[dict]] = {}
    for schedule in AC_SCHEDULES:
        paths = sorted(glob.glob(str(AC_SCORE_BASE / schedule / "seed*" / "run_*" / "run.json")))
        if not paths:
            raise SystemExit(f"No run.json found for {schedule} under {AC_SCORE_BASE}")
        runs = []
        for path in paths:
            run = _load_ac_run(Path(path))
            if run["seed"] not in AC_SEEDS:
                continue
            if abs(run["loss_rate"] - AC_LOSS_RATE) > 1e-9:
                raise SystemExit(f"{path}: unexpected loss rate {run['loss_rate']}")
            if len(run["episodes"]) != AC_EPISODES:
                raise SystemExit(f"{path}: unexpected episode count {len(run['episodes'])}")
            runs.append(run)
        runs.sort(key=lambda run: run["seed"])
        if tuple(run["seed"] for run in runs) != AC_SEEDS:
            raise SystemExit(f"{schedule}: expected seeds {AC_SEEDS}")
        schedules[schedule] = runs

    aggregates = {}
    for schedule, runs in schedules.items():
        by_episode: dict[int, list[float]] = {}
        for run in runs:
            for episode in run["episodes"]:
                by_episode.setdefault(episode["episode"], []).append(episode["run_score"])
        honest_scalar = _mean([
            episode["honest_run_score"]
            for run in runs for episode in run["episodes"]
        ])
        rows = []
        for episode_index in range(1, AC_EPISODES + 1):
            scores = by_episode[episode_index]
            mean_score = _mean(scores)
            rows.append({
                "episode": episode_index,
                "normalized_mean": mean_score / honest_scalar,
                "normalized_min": min(scores) / honest_scalar,
                "normalized_max": max(scores) / honest_scalar,
                "honest_scalar": honest_scalar,
            })
        aggregates[schedule] = {"honest_scalar": honest_scalar, "rows": rows}
    return aggregates


def _load_ac_validation_run(path: Path) -> dict:
    run = _load_ac_run(path)
    honest = 10.0
    for episode in run["episodes"]:
        episode["score"] = episode["alice"] + episode["bob"]
        episode["normalized"] = episode["score"] / honest
    return run


def collect_ac_validation():
    runs: dict[int, dict] = {}
    for pattern in AC_ALL_FALSE_PATTERNS:
        for path in sorted(glob.glob(str(AC / pattern))):
            run = _load_ac_validation_run(Path(path))
            runs.setdefault(run["seed"], run)
    if len(runs) != 25:
        raise SystemExit(f"expected 25 all-false runs, found {len(runs)}")
    all_false = [runs[seed] for seed in sorted(runs)]

    by_episode: dict[int, list[float]] = {}
    for run in all_false:
        for episode in run["episodes"]:
            by_episode.setdefault(episode["episode"], []).append(episode["normalized"])
    rows = []
    for episode_index in range(1, AC_EPISODES + 1):
        values = by_episode[episode_index]
        rows.append({
            "episode": episode_index,
            "mean": _mean(values),
            "min": min(values),
            "max": max(values),
        })

    gcg = {}
    for label, directory in AC_GCG_DIRS.items():
        paths = sorted(glob.glob(str(AC / directory / "run_*" / "run.json")))
        if len(paths) != 1:
            raise SystemExit(f"{label}: expected one run.json, found {len(paths)}")
        gcg[label] = _load_ac_validation_run(Path(paths[0]))
    return rows, gcg


def load_ac_gcg_loss() -> dict:
    return json.loads(AC_GCG_RESULT.read_text(encoding="utf-8"))
