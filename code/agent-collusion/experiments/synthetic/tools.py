"""Synthetic tool dispatch and model-turn wrappers."""

from __future__ import annotations

from typing import Any, Callable

from experiments.agents import run_agent_turn, run_reflection_turn
from experiments.protocol.errors import error_string
from experiments.protocol.messaging import _send_message
from experiments.protocol.state import _log_event, agent_state, peer_id
from experiments.synthetic.constants import PARTS_PER_GROUP
from experiments.synthetic.rewards import VALID_VERDICTS
from experiments.synthetic.schemas import (
    synthetic_reflection_tool_schemas,
    synthetic_tool_schemas,
)

COMMUNICATION_TOOL = "send_message"
VERDICT_TOOL = "submit_verdict"

_ALLOWED_TOOLS = {
    "communication": frozenset({COMMUNICATION_TOOL}),
    "verdict": frozenset({VERDICT_TOOL}),
}


def synthetic_forced_tool_choice(phase: str) -> dict[str, Any]:
    if phase == "communication":
        name = COMMUNICATION_TOOL
    elif phase == "verdict":
        name = VERDICT_TOOL
    else:
        raise ValueError(f"Phase {phase!r} does not force a tool.")
    return {"type": "function", "function": {"name": name}}


def _reject(
    state: dict[str, Any],
    actor: str,
    tool_name: str,
    error: str,
) -> dict[str, Any]:
    _log_event(
        state,
        {
            "actor": actor,
            "tool": tool_name,
            "success": False,
            "error": error,
        },
    )
    return {"success": False, "error": error}


def synthetic_execute_tool_call(
    *,
    state: dict[str, Any],
    actor: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    phase = str(state.get("phase", ""))
    allowed = _ALLOWED_TOOLS.get(phase, frozenset())
    if tool_name not in allowed:
        return _reject(
            state,
            actor,
            tool_name,
            error_string(
                "PhaseError",
                f"{tool_name} is not available during the {phase} phase.",
            ),
        )
    if tool_name == COMMUNICATION_TOOL:
        return _send_message(
            state=state,
            actor=actor,
            content=str(arguments.get("content", "")),
            message_type=str(arguments.get("message_type", "other")),
        )
    if tool_name == VERDICT_TOOL:
        return _submit_group_verdict(
            state=state,
            actor=actor,
            verdicts=arguments.get("verdicts"),
            reasoning=str(arguments.get("reasoning", "")),
        )
    return _reject(
        state,
        actor,
        tool_name,
        error_string("ToolUnavailableError", f"{tool_name} is not available."),
    )


def _submit_group_verdict(
    *,
    state: dict[str, Any],
    actor: str,
    verdicts: Any,
    reasoning: str,
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    target = peer_id(actor)
    if slot.get("verdict_submission") is not None:
        return _reject(
            state,
            actor,
            VERDICT_TOOL,
            error_string("VerdictError", f"{actor} already submitted verdicts"),
        )
    if (
        not isinstance(verdicts, list)
        or len(verdicts) != PARTS_PER_GROUP
        or any(str(value).lower() not in VALID_VERDICTS for value in verdicts)
    ):
        return _reject(
            state,
            actor,
            VERDICT_TOOL,
            error_string(
                "VerdictError",
                f"verdicts must contain exactly {PARTS_PER_GROUP} accept/reject values",
            ),
        )
    normalized = tuple(str(value).strip().lower() for value in verdicts)
    slot["verdict_submission"] = {
        "verdicts": list(normalized),
        "reasoning": reasoning,
        "target": target,
    }
    _log_event(
        state,
        {
            "actor": actor,
            "target": target,
            "tool": VERDICT_TOOL,
            "success": True,
            "verdicts": list(normalized),
        },
    )
    return {"recorded": True, "verdicts": list(normalized), "target": target}


def synthetic_reject_undecodable_arguments(
    *,
    state: dict[str, Any],
    actor: str,
    tool_name: str,
) -> dict[str, Any]:
    phase = str(state.get("phase", ""))
    allowed = _ALLOWED_TOOLS.get(phase, frozenset())
    if tool_name not in allowed:
        return _reject(
            state,
            actor,
            tool_name,
            error_string(
                "PhaseError",
                f"{tool_name} is not available during the {phase} phase.",
            ),
        )
    return _reject(
        state,
        actor,
        tool_name,
        error_string("InvalidArgumentError", "arguments are not a JSON object"),
    )


def run_synthetic_agent_turn(
    *,
    runtime: Any,
    state: dict[str, Any],
    phase: str,
    round_index: int | None,
    attempt_index: int,
    usage_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    actor = runtime.agent_id
    return run_agent_turn(
        actor=actor,
        model=runtime.config.model,
        messages=runtime.messages,
        state=state,
        tool_choice=synthetic_forced_tool_choice(phase),
        reasoning_effort=runtime.config.reasoning_effort,
        temperature=runtime.config.temperature,
        max_output_tokens=runtime.config.max_output_tokens,
        usage_recorder=usage_recorder,
        tools=synthetic_tool_schemas(agent_id=actor, phase=phase),
        tool_executor=synthetic_execute_tool_call,
        undecodable_handler=synthetic_reject_undecodable_arguments,
    )


def run_synthetic_reflection_turn(
    *,
    runtime: Any,
    state: dict[str, Any],
    prompt: str,
    usage_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return run_reflection_turn(
        model=runtime.config.model,
        messages=runtime.messages,
        state=state,
        actor=runtime.agent_id,
        reasoning_effort=runtime.config.reasoning_effort,
        temperature=runtime.config.temperature,
        max_output_tokens=runtime.config.max_output_tokens,
        usage_recorder=usage_recorder,
        tools=synthetic_reflection_tool_schemas(runtime.agent_id),
    )


def current_verdicts(state: dict[str, Any], agent_id: str) -> list[str] | None:
    submission = agent_state(state, agent_id).get("verdict_submission")
    if not isinstance(submission, dict):
        return None
    value = submission.get("verdicts")
    return value if isinstance(value, list) else None
