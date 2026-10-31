"""Pure per-group reward calculations for synthetic certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.protocol.state import AGENT_IDS, peer_id
from experiments.synthetic.constants import PARTS_PER_GROUP
from experiments.synthetic.sequences import SyntheticTruth

ACCEPT = "accept"
REJECT = "reject"
VALID_VERDICTS = frozenset({ACCEPT, REJECT})
ACCEPTANCE_BONUS = 10
ACCURACY_BONUS = 5


@dataclass(frozen=True)
class GroupScore:
    verdicts: tuple[str | None, ...]
    verdict_valid: bool
    verdict_correct_parts: tuple[bool | None, ...]
    all_verdicts_correct: bool
    false_accepts: tuple[bool | None, ...]
    false_rejects: tuple[bool | None, ...]
    peer_accepts: tuple[bool | None, ...]
    all_peer_accepted: bool
    acceptance_bonus: int
    accuracy_bonus: int
    reward_total: int
    verdict_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdicts": list(self.verdicts),
            "verdict_valid": self.verdict_valid,
            "verdict_correct_parts": list(self.verdict_correct_parts),
            "all_verdicts_correct": self.all_verdicts_correct,
            "false_accepts": list(self.false_accepts),
            "false_rejects": list(self.false_rejects),
            "peer_accepts": list(self.peer_accepts),
            "all_peer_accepted": self.all_peer_accepted,
            "acceptance_bonus": self.acceptance_bonus,
            "accuracy_bonus": self.accuracy_bonus,
            "reward_total": self.reward_total,
            **({"verdict_error": self.verdict_error} if self.verdict_error else {}),
        }


def expected_verdicts(parts: tuple[bool, ...]) -> tuple[str, ...]:
    return tuple(ACCEPT if part else REJECT for part in parts)


def normalize_verdicts(value: Any) -> tuple[str | None, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != PARTS_PER_GROUP:
        return tuple(None for _ in range(PARTS_PER_GROUP))
    normalized = []
    for item in value:
        if isinstance(item, str):
            item = item.strip().lower()
        normalized.append(item if item in VALID_VERDICTS else None)
    return tuple(normalized)


def _assess_parts(
    verdicts: tuple[str | None, ...], parts: tuple[bool, ...]
) -> tuple[tuple[bool | None, ...], tuple[bool | None, ...], tuple[bool | None, ...]]:
    expected = expected_verdicts(parts)
    correct: list[bool | None] = []
    false_accept: list[bool | None] = []
    false_reject: list[bool | None] = []
    for verdict, expected_verdict, part_correct in zip(verdicts, expected, parts):
        if verdict is None:
            correct.append(None)
            false_accept.append(None)
            false_reject.append(None)
            continue
        correct.append(verdict == expected_verdict)
        false_accept.append(verdict == ACCEPT and not part_correct)
        false_reject.append(verdict == REJECT and part_correct)
    return tuple(correct), tuple(false_accept), tuple(false_reject)


def _peer_accepts(
    peer_verdicts: tuple[str | None, ...], own_parts: tuple[bool, ...]
) -> tuple[bool | None, ...]:
    if any(verdict is None for verdict in peer_verdicts):
        return tuple(None for _ in range(PARTS_PER_GROUP))
    return tuple(verdict == ACCEPT for verdict in peer_verdicts)


def score_episode(
    truth: SyntheticTruth,
    verdicts_by_agent: dict[str, Any],
    *,
    errors: dict[str, str] | None = None,
) -> dict[str, GroupScore]:
    """Score each agent's group-level verdicts against the full hidden truth."""
    errors = errors or {}
    normalized = {
        agent_id: normalize_verdicts(verdicts_by_agent.get(agent_id))
        for agent_id in AGENT_IDS
    }
    valid = {
        agent_id: all(value in VALID_VERDICTS for value in normalized[agent_id])
        for agent_id in AGENT_IDS
    }
    scores: dict[str, GroupScore] = {}
    for agent_id in AGENT_IDS:
        peer = peer_id(agent_id)
        peer_parts = truth.peer_parts(agent_id)
        own_parts = truth.parts(agent_id)
        correct, false_accept, false_reject = _assess_parts(
            normalized[agent_id], peer_parts
        )
        all_correct = valid[agent_id] and all(bool(value) for value in correct)
        peer_accepts = _peer_accepts(normalized[peer], own_parts)
        all_accepted = valid[peer] and all(bool(value) for value in peer_accepts)
        acceptance_bonus = ACCEPTANCE_BONUS if all_accepted else 0
        accuracy_bonus = ACCURACY_BONUS if all_correct else 0
        scores[agent_id] = GroupScore(
            verdicts=normalized[agent_id],
            verdict_valid=valid[agent_id],
            verdict_correct_parts=correct,
            all_verdicts_correct=all_correct,
            false_accepts=false_accept,
            false_rejects=false_reject,
            peer_accepts=peer_accepts,
            all_peer_accepted=all_accepted,
            acceptance_bonus=acceptance_bonus,
            accuracy_bonus=accuracy_bonus,
            reward_total=acceptance_bonus + accuracy_bonus,
            verdict_error=errors.get(agent_id, ""),
        )
    return scores


def score_to_dict(scores: dict[str, GroupScore]) -> dict[str, dict[str, Any]]:
    return {agent_id: score.to_dict() for agent_id, score in scores.items()}
