"""Individual and net influence (Bonjour, Aggarwal & Bhargava, UAI 2022).

Notation follows the paper:

    gamma(i; j) = sum_{s_ij} p(s_ij) sum_{a_i, a_j} p(a_i, a_j | s_ij)
                  * log[ p(a_i, a_j | s_ij) / (p(a_i | s_i) p(a_j | s_j)) ]      (Eq. 3)

    Gamma(i; j) = gamma(i; j) - max_{k != i, j} gamma(k; j)                       (Eq. 4)

    collusion between i and j  iff  Gamma(i; j) >= alpha and Gamma(j; i) >= alpha (Eq. 5)

All probabilities are plug-in (count-based) estimates, as in the paper (Eq. 12-13).
Only simultaneous-move games are supported: action order within a timestep is ignored.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _encode(x: np.ndarray | None, n: int) -> np.ndarray:
    """Map arbitrary hashable rows (or None = single state) to dense int codes 0..K-1."""
    if x is None:
        return np.zeros(n, dtype=np.int64)
    x = np.asarray(x)
    if len(x) != n:
        raise ValueError(f"expected {n} rows, got {len(x)}")
    axis = 0 if x.ndim > 1 else None
    return np.unique(x, axis=axis, return_inverse=True)[1].reshape(-1).astype(np.int64)


def individual_influence(
    a_i: np.ndarray,
    a_j: np.ndarray,
    s_i: np.ndarray | None = None,
    s_j: np.ndarray | None = None,
    base: float = np.e,
) -> float:
    """gamma(i; j) from Eq. 3, estimated from T paired samples.

    a_i, a_j: actions of agents i and j at each of T timesteps.
    s_i, s_j: each agent's observed state per timestep (None = one shared constant state).
    For fully observable games pass the same array for s_i and s_j; gamma then reduces to
    the conditional mutual information I(a_i; a_j | s) and is symmetric.
    """
    t = len(a_i)
    if len(a_j) != t:
        raise ValueError("a_i and a_j must have the same length")
    if t == 0:
        raise ValueError("need at least one sample")

    ai, aj = _encode(a_i, t), _encode(a_j, t)
    si, sj = _encode(s_i, t), _encode(s_j, t)
    n_ai, n_aj = ai.max() + 1, aj.max() + 1
    n_si, n_sj = si.max() + 1, sj.max() + 1

    # p(a_i | s_i) and p(a_j | s_j) from each agent's own marginal counts (Eq. 12).
    c_i = np.bincount(si * n_ai + ai, minlength=n_si * n_ai).reshape(n_si, n_ai)
    c_j = np.bincount(sj * n_aj + aj, minlength=n_sj * n_aj).reshape(n_sj, n_aj)
    p_i = c_i / c_i.sum(axis=1, keepdims=True)
    p_j = c_j / c_j.sum(axis=1, keepdims=True)

    # N(a_i, a_j, s_i, s_j) and N(s_i, s_j) (Eq. 13).
    joint_key = ((si * n_sj + sj) * n_ai + ai) * n_aj + aj
    size = n_si * n_sj * n_ai * n_aj
    if size <= 1_000_000:  # dense bincount is ~10x faster than unique for small spaces
        counts = np.bincount(joint_key, minlength=size)
        keys = np.flatnonzero(counts)
        counts = counts[keys]
    else:
        keys, counts = np.unique(joint_key, return_counts=True)
    k_aj = keys % n_aj
    k_ai = (keys // n_aj) % n_ai
    k_sij = keys // (n_aj * n_ai)
    k_sj = k_sij % n_sj
    k_si = k_sij // n_sj
    n_sij = np.bincount(si * n_sj + sj, minlength=n_si * n_sj)

    # p(s_ij) * p(a_i, a_j | s_ij) == N(a_i, a_j, s_ij) / T
    p_cond = counts / n_sij[k_sij]
    indep = p_i[k_si, k_ai] * p_j[k_sj, k_aj]
    gamma = np.sum((counts / t) * np.log(p_cond / indep)) / np.log(base)
    return float(max(gamma, 0.0))  # clip float noise; plug-in CMI is non-negative


def influence_matrix(
    actions: np.ndarray,
    states: np.ndarray | None = None,
    agent_states: list[np.ndarray] | None = None,
    base: float = np.e,
) -> np.ndarray:
    """gamma[i, j] for all agent pairs.

    actions: (T, n) array.
    states: shared, fully observed state per timestep, shape (T,) or (T, d). None = constant.
    agent_states: per-agent observed states (partially observable case); overrides `states`.
    """
    actions = np.asarray(actions)
    n = actions.shape[1]
    if agent_states is not None and len(agent_states) != n:
        raise ValueError("agent_states needs one entry per agent")

    def state_of(k):
        return agent_states[k] if agent_states is not None else states

    g = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        g[i, j] = individual_influence(actions[:, i], actions[:, j], state_of(i), state_of(j), base)
        if agent_states is None:
            g[j, i] = g[i, j]  # shared state: gamma is symmetric (Sec. 3.1)
        else:
            g[j, i] = individual_influence(actions[:, j], actions[:, i], state_of(j), state_of(i), base)
    return g


def net_influence(gamma: np.ndarray) -> np.ndarray:
    """Gamma[i, j] = gamma[i, j] - max_{k != i, j} gamma[k, j] (Eq. 4). Diagonal is NaN."""
    gamma = np.asarray(gamma, dtype=float)
    n = gamma.shape[0]
    if n < 3:
        raise ValueError("net influence needs at least 3 agents")
    net = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [k for k in range(n) if k not in (i, j)]
            net[i, j] = gamma[i, j] - gamma[others, j].max()
    return net


def detect(net: np.ndarray, alpha: float = 0.05) -> tuple[int, int] | None:
    """Return the colluding pair if exactly one pair has mutual net influence >= alpha (Eq. 5)."""
    n = net.shape[0]
    hits = [(i, j) for i, j in combinations(range(n), 2) if net[i, j] >= alpha and net[j, i] >= alpha]
    return hits[0] if len(hits) == 1 else None


def permutation_pvalues(
    actions: np.ndarray,
    states: np.ndarray | None = None,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Permutation p-value for every gamma[i, j] under the null "actions independent given state".

    Each permutation shuffles every agent's actions independently *within* timesteps sharing a
    state, which keeps each agent's p(a | s) but breaks cross-agent dependence. Shared-state
    (simultaneous, fully observed) games only.
    """
    rng = rng if rng is not None else np.random.default_rng()
    actions = np.asarray(actions)
    t, n = actions.shape
    n_pairs = n * (n - 1) // 2
    if 1 / (n_perm + 1) >= 0.05 / n_pairs:  # smallest p must clear detect_calibrated's Bonferroni cutoff
        raise ValueError(f"n_perm={n_perm} too small to reach p < 0.05/{n_pairs}; use > {int(20 * n_pairs)}")
    s = _encode(states, t)
    by_state = np.argsort(s, kind="stable")
    observed = influence_matrix(actions, s)
    exceed = np.zeros((n, n))
    for _ in range(n_perm):
        null = actions.copy()
        for k in range(n):
            null[by_state, k] = actions[np.lexsort((rng.random(t), s)), k]  # random order within each state
        exceed += influence_matrix(null, s) >= observed
    return (1 + exceed) / (1 + n_perm)


def detect_calibrated(net: np.ndarray, pvals: np.ndarray, level: float = 0.05) -> tuple[int, int] | None:
    """Eq. 5 with the fixed alpha replaced by a significance test.

    Flag (i, j) if both net influences are positive and gamma(i; j) is significant at `level`
    (Bonferroni-corrected over pairs). As in `detect`, exactly one pair must qualify.
    """
    n = net.shape[0]
    pairs = list(combinations(range(n), 2))
    hits = [(i, j) for i, j in pairs if net[i, j] > 0 and net[j, i] > 0 and pvals[i, j] < level / len(pairs)]
    return hits[0] if len(hits) == 1 else None


def pair_scores(net: np.ndarray) -> dict[tuple[int, int], float]:
    """Symmetric per-pair score min(Gamma(i;j), Gamma(j;i)): the margin by which a pair clears Eq. 5.

    Useful for ranking pairs (e.g. ROC/AUC) without fixing alpha.
    """
    n = net.shape[0]
    return {(i, j): float(min(net[i, j], net[j, i])) for i, j in combinations(range(n), 2)}
