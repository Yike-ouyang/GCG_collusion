"""Synthetic episode state and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.protocol import COMMUNICATION_ATTEMPTS, VERDICT_ATTEMPTS
from experiments.protocol.state import create_channel_state
from experiments.synthetic.constants import (
    PROTOCOL_ID,
    SYNTHETIC_PHASES,
)
from experiments.synthetic.sequences import SyntheticTruth


@dataclass(frozen=True)
class SyntheticEpisodeConfig:
    communication_enabled: bool = True
    char_limit: int = 200
    max_rounds: int = 3
    loss_rate: float = 0.10
    reflection: bool = True
    verdict_suffix: str = ""
    verdict_suffix_agent: str = ""
    verbose: bool = True
    usage_journal_path: Path | None = None
    usage_invocation_id: str = ""


SYNTHETIC_PHASE_BUDGETS = {
    "communication": COMMUNICATION_ATTEMPTS,
    "verdict": VERDICT_ATTEMPTS,
}


def create_synthetic_state(
    *,
    truth: SyntheticTruth,
    char_limit: int = 200,
    communication_enabled: bool = True,
    loss_rate: float = 0.10,
) -> dict[str, Any]:
    """Create an episode state without reading tasks or calling a model."""
    state = create_channel_state(
        episode_id=f"synthetic-ep{truth.episode_index:03d}",
        throttled=True,
        char_limit=char_limit,
    )
    state.update(
        {
            "protocol": PROTOCOL_ID,
            "task_type": "synthetic_certification",
            "phase": "state",
            "truth": truth,
            "communication_enabled": communication_enabled,
            "loss_rate": loss_rate,
        }
    )
    return state


def set_synthetic_phase(state: dict[str, Any], phase: str) -> None:
    if phase not in SYNTHETIC_PHASES:
        raise ValueError(f"Unknown synthetic phase: {phase}")
    state["phase"] = phase


def truth_of(state: dict[str, Any]) -> SyntheticTruth:
    truth = state.get("truth")
    if not isinstance(truth, SyntheticTruth):
        raise ValueError("Synthetic state is missing its authoritative truth.")
    return truth


def set_synthetic_phase_checked(state: dict[str, Any], phase: str) -> None:
    set_synthetic_phase(state, phase)
