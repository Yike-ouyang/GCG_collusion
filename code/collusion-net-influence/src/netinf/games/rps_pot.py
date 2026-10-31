"""N-player Rock-Paper-Scissors with a fixed pot split among the winners only.

Rules (3 points per round by default):
  1. all actions identical   -> no winner, nobody scores (pot is lost)
  2. all three actions occur -> cycle, no winner, nobody scores (pot is lost)
  3. exactly two actions     -> players of the winning action split the pot equally
"""

from __future__ import annotations

import numpy as np

from netinf.games.rps import BEATS

POT = 3.0


def payoff_pot(actions: np.ndarray | list[int], pot: float = POT) -> np.ndarray:
    """Return each player's reward under the pot rule.

    `actions` holds one action per player (ROCK=0, PAPER=1, SCISSORS=2).
    Rewards are floats: a lone winner takes the whole pot, two winners split it.
    """
    actions = np.asarray(actions)
    rewards = np.zeros(len(actions), dtype=float)
    distinct = set(actions.tolist())
    if len(distinct) != 2:  # identical moves or a three-way cycle: no winner
        return rewards
    a, b = distinct
    winner = a if BEATS[a] == b else b
    winners = actions == winner
    rewards[winners] = pot / winners.sum()
    return rewards


def simulate_alternating(n_rounds: int, rng: np.random.Generator, pot: float = POT) -> tuple[np.ndarray, np.ndarray]:
    """Scripted A/B alternating assist: even rounds B sacrifices to A, odd rounds A to B.

    C plays uniformly at random. Useful as a positive control for the detector and for the
    alternation metrics.
    """
    actions = rng.integers(0, 3, size=(n_rounds, 3))
    for t in range(n_rounds):
        if t % 2 == 0:  # B plays the move A beats
            actions[t, 1] = BEATS[actions[t, 0]]
        else:  # A plays the move B beats
            actions[t, 0] = BEATS[actions[t, 1]]
    rewards = np.array([payoff_pot(row, pot) for row in actions])
    return actions, rewards
