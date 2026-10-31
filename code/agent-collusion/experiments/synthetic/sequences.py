"""Reproducible synthetic groups of three task parts."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.synthetic.constants import (
    DEFAULT_LOSS_RATE,
    DEFAULT_SYNTHETIC_EPISODES,
    DEFAULT_SYNTHETIC_SCHEDULE,
    DEFAULT_SYNTHETIC_SEED,
    MIXED_FALSE_RATES,
    PARTS_PER_GROUP,
    SYNTHETIC_SCHEDULES,
)


@dataclass(frozen=True)
class SyntheticTruth:
    """One episode's immutable truth and transmission masks."""

    episode_index: int
    alice_parts: tuple[bool, ...]
    bob_parts: tuple[bool, ...]
    # True means the part reached the peer.
    alice_to_bob_received: tuple[bool, ...]
    bob_to_alice_received: tuple[bool, ...]

    def parts(self, agent_id: str) -> tuple[bool, ...]:
        if agent_id == "alice":
            return self.alice_parts
        if agent_id == "bob":
            return self.bob_parts
        raise ValueError(f"Unknown agent: {agent_id}")

    def peer_parts(self, agent_id: str) -> tuple[bool, ...]:
        return self.parts("bob" if agent_id == "alice" else "alice")

    def received_by(self, agent_id: str) -> tuple[bool, ...]:
        """Which of the peer's parts this agent received."""
        if agent_id == "alice":
            return self.bob_to_alice_received
        if agent_id == "bob":
            return self.alice_to_bob_received
        raise ValueError(f"Unknown agent: {agent_id}")

    def delivered_by(self, agent_id: str) -> tuple[bool, ...]:
        """Which of this agent's own parts reached the peer."""
        if agent_id == "alice":
            return self.alice_to_bob_received
        if agent_id == "bob":
            return self.bob_to_alice_received
        raise ValueError(f"Unknown agent: {agent_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "alice_parts": list(self.alice_parts),
            "bob_parts": list(self.bob_parts),
            "alice_to_bob_received": list(self.alice_to_bob_received),
            "bob_to_alice_received": list(self.bob_to_alice_received),
        }


@dataclass(frozen=True)
class SyntheticSequence:
    schedule: str
    source: str
    seed: int
    loss_rate: float
    p_false_per_part: float | None
    episodes: tuple[SyntheticTruth, ...]
    path: str = ""

    @property
    def sha256(self) -> str:
        return sequence_sha256(self.episodes)

    def to_config(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule,
            "source": self.source,
            "seed": self.seed,
            "loss_rate": self.loss_rate,
            "p_false_per_part": self.p_false_per_part,
            "parts_per_group": PARTS_PER_GROUP,
            "episode_count": len(self.episodes),
            "sha256": self.sha256,
            "path": self.path,
        }


def normalize_schedule(schedule: str) -> str:
    if schedule == "mixed":
        return "mixed-40"
    if schedule not in SYNTHETIC_SCHEDULES:
        raise ValueError(
            f"Unknown synthetic schedule {schedule!r}; expected one of "
            + ", ".join(SYNTHETIC_SCHEDULES)
        )
    return schedule


def _draw_parts(rng: random.Random, schedule: str) -> tuple[bool, ...]:
    if schedule == "all-false":
        return tuple(False for _ in range(PARTS_PER_GROUP))
    if schedule == "all-true":
        return tuple(True for _ in range(PARTS_PER_GROUP))
    p_false = MIXED_FALSE_RATES[schedule]
    return tuple(rng.random() >= p_false for _ in range(PARTS_PER_GROUP))


def generate_sequence(
    schedule: str,
    seed: int,
    *,
    loss_rate: float = DEFAULT_LOSS_RATE,
    episodes: int = DEFAULT_SYNTHETIC_EPISODES,
) -> SyntheticSequence:
    """Generate one group of three parts per agent and per episode."""
    if episodes < 1:
        raise ValueError("A synthetic sequence needs at least one episode.")
    if not 0 <= loss_rate <= 1:
        raise ValueError("loss_rate must be between 0 and 1.")
    normalized = normalize_schedule(schedule)
    rng = random.Random(seed)
    truth_rows = []
    for index in range(1, episodes + 1):
        alice_parts = _draw_parts(rng, normalized)
        bob_parts = _draw_parts(rng, normalized)
        alice_to_bob = tuple(
            rng.random() >= loss_rate for _ in range(PARTS_PER_GROUP)
        )
        bob_to_alice = tuple(
            rng.random() >= loss_rate for _ in range(PARTS_PER_GROUP)
        )
        truth_rows.append(
            SyntheticTruth(
                episode_index=index,
                alice_parts=alice_parts,
                bob_parts=bob_parts,
                alice_to_bob_received=alice_to_bob,
                bob_to_alice_received=bob_to_alice,
            )
        )
    p_false = MIXED_FALSE_RATES.get(normalized)
    return SyntheticSequence(
        schedule=normalized,
        source="generated",
        seed=seed,
        loss_rate=loss_rate,
        p_false_per_part=p_false,
        episodes=tuple(truth_rows),
    )


def _validate_parts(value: Any, label: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or len(value) != PARTS_PER_GROUP:
        raise ValueError(f"{label} must contain exactly {PARTS_PER_GROUP} booleans.")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{label} must contain booleans only.")
    return tuple(value)


def load_sequence(path: Path) -> SyntheticSequence:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        entries = document
        loss_rate = DEFAULT_LOSS_RATE
    elif isinstance(document, dict):
        unknown = sorted(set(document) - {"version", "episodes", "loss_rate"})
        if unknown:
            raise ValueError(f"Synthetic sequence JSON has unknown keys: {unknown}")
        version = document.get("version", 2)
        if version != 2:
            raise ValueError(f"Unsupported synthetic sequence version: {version!r}")
        entries = document.get("episodes")
        loss_rate = float(document.get("loss_rate", DEFAULT_LOSS_RATE))
    else:
        raise ValueError("Synthetic sequence JSON must be an object or a list.")
    if not isinstance(entries, list) or not entries:
        raise ValueError("An explicit synthetic sequence needs a non-empty episodes list.")
    rows = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Episode {index} must be an object.")
        allowed = {
            "alice_parts",
            "bob_parts",
            "alice_to_bob_received",
            "bob_to_alice_received",
            "episode_index",
        }
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(f"Episode {index} has unknown keys: {unknown}")
        declared = entry.get("episode_index", index)
        if declared != index:
            raise ValueError(
                f"Episode {index} declares episode_index={declared!r}; "
                "explicit sequences are indexed in file order."
            )
        rows.append(
            SyntheticTruth(
                episode_index=index,
                alice_parts=_validate_parts(entry.get("alice_parts"), "alice_parts"),
                bob_parts=_validate_parts(entry.get("bob_parts"), "bob_parts"),
                alice_to_bob_received=_validate_parts(
                    entry.get("alice_to_bob_received"), "alice_to_bob_received"
                ),
                bob_to_alice_received=_validate_parts(
                    entry.get("bob_to_alice_received"), "bob_to_alice_received"
                ),
            )
        )
    return SyntheticSequence(
        schedule="explicit",
        source="file",
        seed=0,
        loss_rate=loss_rate,
        p_false_per_part=None,
        episodes=tuple(rows),
        path=str(path),
    )


def sequence_sha256(episodes: tuple[SyntheticTruth, ...] | list[SyntheticTruth]) -> str:
    payload = [truth.to_dict() for truth in episodes]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_sequence(
    *,
    schedule: str = "",
    sequence_path: str = "",
    seed: int = DEFAULT_SYNTHETIC_SEED,
    loss_rate: float = DEFAULT_LOSS_RATE,
    episodes: int = DEFAULT_SYNTHETIC_EPISODES,
    repo_root: Path,
) -> SyntheticSequence:
    if schedule and sequence_path:
        raise ValueError(
            "--synthetic-schedule and --synthetic-sequence are mutually exclusive."
        )
    if sequence_path:
        path = Path(sequence_path)
        if not path.is_absolute():
            path = repo_root / path
        return load_sequence(path)
    return generate_sequence(
        schedule or DEFAULT_SYNTHETIC_SCHEDULE,
        seed,
        loss_rate=loss_rate,
        episodes=episodes,
    )


def sequence_to_dicts(sequence: SyntheticSequence) -> list[dict[str, Any]]:
    return [truth.to_dict() for truth in sequence.episodes]
