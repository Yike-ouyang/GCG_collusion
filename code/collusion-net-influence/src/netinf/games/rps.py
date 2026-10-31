"""N-player iterated Rock-Paper-Scissors with scripted agents (Bonjour et al. 2022, Sec. 5.1).

Payoff rules (Table 1 generalised to n players):
  1. all actions identical            -> everyone gets 0
  2. all three actions present        -> everyone gets 1
  3. exactly two distinct actions     -> players of the winning action get 1, others 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROCK, PAPER, SCISSORS = 0, 1, 2
ACTIONS = ("R", "P", "S")
# BEATS[a] is the action that `a` beats: R beats S, P beats R, S beats P.
BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}


def payoff(actions: np.ndarray | list[int]) -> np.ndarray:
    actions = np.asarray(actions)
    distinct = set(actions.tolist())
    if len(distinct) == 1:
        return np.zeros(len(actions), dtype=int)
    if len(distinct) == 3:
        return np.ones(len(actions), dtype=int)
    a, b = distinct
    winner = a if BEATS[a] == b else b
    return (actions == winner).astype(int)


@dataclass(frozen=True)
class AssistantCollusion:
    """Scripted 'assistant collusion': with probability `cp` the assistant covertly plays the
    action that the primary's action beats, guaranteeing the primary a point; otherwise random.
    """

    primary: int
    assistant: int
    cp: float


def simulate(
    n_players: int,
    n_rounds: int,
    collusion: AssistantCollusion | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Play `n_rounds` of n-player RPS. Non-colluding agents play uniformly at random.

    Returns (actions, rewards), both shaped (n_rounds, n_players).
    """
    rng = rng if rng is not None else np.random.default_rng()
    actions = rng.integers(0, 3, size=(n_rounds, n_players))
    if collusion is not None:
        active = rng.random(n_rounds) < collusion.cp
        beaten = np.vectorize(BEATS.get)(actions[:, collusion.primary])
        actions[active, collusion.assistant] = beaten[active]
    rewards = np.array([payoff(row) for row in actions])
    return actions, rewards
