"""N-player Rock-Paper-Scissors where every player stakes into a pot that the winners split.

Rules (stake 1, 3 players by default):
  1. every player puts `stake` into the pot (pot = stake * n_players)
  2. exactly two actions played -> players of the winning action split the pot equally
  3. all actions identical, or all three actions present -> no winner:
       tie_rule="refund": every stake is returned (net 0 for everyone; the game is zero-sum)
       tie_rule="burn":   every stake is lost (net -stake for everyone)

Net rewards with stake 1 and 3 players: a lone winner gets +2 and each loser -1;
two winners get +0.5 each and the loser -1.
"""

from __future__ import annotations

import numpy as np

from netinf.games.rps import BEATS

STAKE = 1.0
TIE_RULES = ("refund", "burn")


def payoff_stake(actions: np.ndarray | list[int], stake: float = STAKE, tie_rule: str = "refund") -> np.ndarray:
    """Return each player's net reward (gain minus stake) under the stake rule."""
    if tie_rule not in TIE_RULES:
        raise ValueError(f"tie_rule must be one of {TIE_RULES}")
    actions = np.asarray(actions)
    n = len(actions)
    distinct = set(actions.tolist())
    if len(distinct) != 2:  # identical moves or a three-way cycle: no winner
        return np.zeros(n) if tie_rule == "refund" else np.full(n, -stake)
    a, b = distinct
    winner = a if BEATS[a] == b else b
    winners = actions == winner
    rewards = np.full(n, -stake)
    rewards[winners] += stake * n / winners.sum()
    return rewards
