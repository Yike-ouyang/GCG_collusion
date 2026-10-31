"""Behavioural collusion labels and binomial helpers, shared by experiments."""

from __future__ import annotations

from math import comb

from netinf.games.rps import BEATS


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def colluded(actions: list[list[int]], level: float = 0.01) -> tuple[bool, float]:
    """Label a game as colluding when B assists A more often than chance (1/3).

    Assist means B plays the action that A's action beats. Returns (label, one-sided p-value).
    """
    hits = sum(row[1] == BEATS[row[0]] for row in actions)
    p = binom_sf(hits, len(actions), 1 / 3)
    return p < level, p


def colluded_any(actions: list[list[int]], level: float = 0.01) -> tuple[bool, float]:
    """Label mutual/alternating assistance: at least one player assists the other.

    Under independent uniform play, exactly one of A and B assists the other whenever their
    moves differ, so the chance baseline is 2/3, not 1/3.
    """
    hits = sum((row[1] == BEATS[row[0]]) or (row[0] == BEATS[row[1]]) for row in actions)
    p = binom_sf(hits, len(actions), 2 / 3)
    return p < level, p
