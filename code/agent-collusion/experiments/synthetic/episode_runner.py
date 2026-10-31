"""Orchestrate one synthetic-certification episode."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from experiments.memory.cross_episode import append_reflection
from experiments.memory.messages import (
    active_episode_start_message,
    communication_phase_start_message,
    outcome_feedback_start_message,
    snapshot_visible_messages,
)
from experiments.models import AgentConfig, AgentRuntime
from experiments.prompts.messages import (
    communication_round_instruction,
    communication_round_tag,
    incoming_channel_message,
    peer_delivery_failure_notice,
)
from experiments.protocol.state import (
    AGENT_IDS,
    agent_state,
    deliver_runner_notice,
    peer_id,
    pop_incoming_messages,
    reset_message_delivery,
)
from experiments.synthetic.constants import PROTOCOL_ID, SYNTHETIC_PHASES
from experiments.synthetic.prompts import (
    communication_phase_message,
    environment_state_message,
    feedback_message,
    initial_agent_messages,
    reflection_prompt,
    verdict_phase_message,
)
from experiments.synthetic.rewards import score_episode, score_to_dict
from experiments.synthetic.sequences import SyntheticTruth
from experiments.synthetic.state import (
    SYNTHETIC_PHASE_BUDGETS,
    SyntheticEpisodeConfig,
    create_synthetic_state,
    set_synthetic_phase,
)
from experiments.synthetic.tools import (
    VERDICT_TOOL,
    run_synthetic_agent_turn,
    run_synthetic_reflection_turn,
)
from experiments.usage import make_llm_usage_recorder, summarize_llm_usage

TurnRunner = Callable[..., dict[str, Any]]
ReflectionRunner = Callable[..., dict[str, Any]]


def _phase_completed(state: dict[str, Any], agent_id: str, phase: str) -> bool:
    slot = agent_state(state, agent_id)
    if phase == "communication":
        return bool(slot["delivered"])
    if phase == "verdict":
        return slot["verdict_submission"] is not None
    raise ValueError(f"Unknown phase: {phase}")


def _attempt_errors(turn: dict[str, Any]) -> list[str]:
    errors = []
    for call in turn.get("tool_calls") or []:
        result = call.get("result") or {}
        if isinstance(result, dict) and result.get("success") is False:
            errors.append(str(result.get("error") or "tool call failed"))
    return errors


def _run_attempts(
    *,
    runtime: AgentRuntime,
    state: dict[str, Any],
    phase: str,
    round_index: int | None,
    config: SyntheticEpisodeConfig,
    turn_runner: TurnRunner,
    usage_records: list[dict[str, Any]],
    reasoning_traces: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    budget = SYNTHETIC_PHASE_BUDGETS[phase]
    attempt_index = 0
    attempts: list[dict[str, Any]] = []
    agent_id = runtime.agent_id
    while attempt_index < budget:
        reset_message_delivery(state, agent_id)
        recorder = make_llm_usage_recorder(
            records=usage_records,
            journal_path=config.usage_journal_path,
            invocation_id=config.usage_invocation_id,
            episode_id=str(state["episode_id"]),
            episode_index=int(state["truth"].episode_index) - 1,
            actor=agent_id,
            phase=phase,
            round_idx=round_index,
            attempt_idx=attempt_index,
        )
        try:
            turn = turn_runner(
                runtime=runtime,
                state=state,
                phase=phase,
                round_index=round_index,
                attempt_index=attempt_index,
                usage_recorder=recorder,
            )
        except Exception as exc:  # keep technical failures separate from decisions
            record = {
                "agent": agent_id,
                "phase": phase,
                "round": None if round_index is None else round_index + 1,
                "attempt": attempt_index + 1,
                "category": "technical",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(record)
            attempts.append(record)
            attempt_index += 1
            continue

        trace = turn.get("reasoning_trace")
        if trace:
            reasoning_traces.append(
                {
                    "agent": agent_id,
                    "phase": phase,
                    "round": None if round_index is None else round_index + 1,
                    "attempt": attempt_index + 1,
                    "trace": trace,
                }
            )
        record = {
            "agent": agent_id,
            "phase": phase,
            "round": None if round_index is None else round_index + 1,
            "attempt": attempt_index + 1,
            "tool_calls": deepcopy(turn.get("tool_calls") or []),
            "errors": _attempt_errors(turn),
        }
        attempts.append(record)
        for error in record["errors"]:
            errors.append(
                {
                    **{k: v for k, v in record.items() if k != "tool_calls"},
                    "category": "submission",
                    "error": error,
                }
            )
        if _phase_completed(state, agent_id, phase):
            return True, attempts
        attempt_index += 1
    return False, attempts


def _inject_incoming(
    messages: list[dict[str, Any]],
    *,
    incoming: list[dict[str, Any]],
    max_rounds: int,
) -> None:
    for message in incoming:
        sender = str(message.get("from", "runner"))
        message_round = int(message.get("round", 0))
        messages.append(
            {
                "role": "user",
                "content": incoming_channel_message(
                    sender=sender,
                    message_type=str(message.get("message_type", "other")),
                    content=str(message.get("content", "")),
                    round_tag=communication_round_tag(message_round + 1, max_rounds),
                ),
            }
        )


def _run_communication_phase(
    *,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    config: SyntheticEpisodeConfig,
    turn_runner: TurnRunner,
    usage_records: list[dict[str, Any]],
    reasoning_traces: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    turns: dict[str, list[dict[str, Any]]],
) -> None:
    set_synthetic_phase(state, "communication")
    for agent_id in AGENT_IDS:
        runtime = runtimes[agent_id]
        runtime.messages.append(communication_phase_start_message())
        runtime.messages.append(
            {
                "role": "user",
                "content": communication_phase_message(
                    agent_id=agent_id,
                    episode_number=int(state["truth"].episode_index),
                    max_rounds=config.max_rounds,
                    char_limit=config.char_limit,
                ),
            }
        )
    for round_index in range(config.max_rounds):
        state["round"] = round_index
        for agent_id in AGENT_IDS:
            runtime = runtimes[agent_id]
            _inject_incoming(
                runtime.messages,
                incoming=pop_incoming_messages(state, agent_id),
                max_rounds=config.max_rounds,
            )
            runtime.messages.append(
                {
                    "role": "user",
                    "content": communication_round_instruction(
                        round_index + 1, config.max_rounds
                    ),
                }
            )
            delivered, attempts = _run_attempts(
                runtime=runtime,
                state=state,
                phase="communication",
                round_index=round_index,
                config=config,
                turn_runner=turn_runner,
                usage_records=usage_records,
                reasoning_traces=reasoning_traces,
                errors=errors,
            )
            turns[agent_id].extend(attempts)
            if not delivered:
                deliver_runner_notice(
                    state,
                    peer_id(agent_id),
                    peer_delivery_failure_notice(
                        peer=agent_id,
                        round_number=round_index + 1,
                    ),
                )
    # Deliver the final queued message before the verdict phase.
    for agent_id in AGENT_IDS:
        _inject_incoming(
            runtimes[agent_id].messages,
            incoming=pop_incoming_messages(state, agent_id),
            max_rounds=config.max_rounds,
        )


def _agent_verdict_raw(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw: dict[str, Any] | None = None
    for attempt in attempts:
        for call in attempt.get("tool_calls") or []:
            if call.get("tool_name") == VERDICT_TOOL:
                raw = {
                    "arguments": deepcopy(call.get("arguments") or {}),
                    "result": deepcopy(call.get("result") or {}),
                }
    return raw


def _run_verdict_phase(
    *,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    config: SyntheticEpisodeConfig,
    turn_runner: TurnRunner,
    usage_records: list[dict[str, Any]],
    reasoning_traces: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    turns: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any] | None]:
    set_synthetic_phase(state, "verdict")
    for agent_id in AGENT_IDS:
        suffix = (
            config.verdict_suffix
            if config.verdict_suffix_agent in ("", agent_id)
            else ""
        )
        runtimes[agent_id].messages.append(
            {
                "role": "user",
                "content": verdict_phase_message(
                    agent_id=agent_id,
                    episode_number=int(state["truth"].episode_index),
                    suffix=suffix,
                ),
            }
        )
    verdicts: dict[str, dict[str, Any] | None] = {}
    for agent_id in AGENT_IDS:
        delivered, attempts = _run_attempts(
            runtime=runtimes[agent_id],
            state=state,
            phase="verdict",
            round_index=None,
            config=config,
            turn_runner=turn_runner,
            usage_records=usage_records,
            reasoning_traces=reasoning_traces,
            errors=errors,
        )
        turns[agent_id].extend(attempts)
        submission = agent_state(state, agent_id)["verdict_submission"]
        if not delivered or submission is None:
            errors.append(
                {
                    "agent": agent_id,
                    "phase": "verdict",
                    "round": None,
                    "attempt": len(attempts),
                    "category": "submission",
                    "error_type": "MissingVerdict",
                    "error": "no valid verdict after the attempt budget",
                }
            )
        verdicts[agent_id] = deepcopy(submission)
    return verdicts


def _channel_transcript(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transcript = []
    for event in events:
        if event.get("tool") != "send_message" or not event.get("success"):
            continue
        sender = str(event.get("actor", ""))
        transcript.append(
            {
                "event_id": event.get("event_id"),
                "round": event.get("round"),
                "sender": sender,
                "receiver": str(event.get("recipient") or peer_id(sender)),
                "message_type": str(event.get("message_type", "other")),
                "content": str(event.get("content", "")),
            }
        )
    return transcript


def run_synthetic_episode(
    *,
    truth: SyntheticTruth,
    agent_configs: dict[str, AgentConfig],
    config: SyntheticEpisodeConfig,
    agent_messages: dict[str, list[dict[str, Any]]] | None = None,
    previous_cumulative: dict[str, int] | None = None,
    turn_runner: TurnRunner | None = None,
    reflection_runner: ReflectionRunner | None = None,
) -> dict[str, Any]:
    """Run one synthetic episode with no task phase or initialization LLM call."""
    if config.max_rounds < 1:
        raise ValueError("Synthetic communication needs at least one round.")
    if config.char_limit < 1:
        raise ValueError("Synthetic communication needs a positive character limit.")
    turn_runner = turn_runner or run_synthetic_agent_turn
    reflection_runner = reflection_runner or run_synthetic_reflection_turn

    if agent_messages is None:
        agent_messages = {
            agent_id: initial_agent_messages(
                agent_id=agent_id,
                communication_enabled=config.communication_enabled,
                max_rounds=config.max_rounds,
                char_limit=config.char_limit,
            )
            for agent_id in AGENT_IDS
        }
    state = create_synthetic_state(
        truth=truth,
        char_limit=config.char_limit,
        communication_enabled=config.communication_enabled,
        loss_rate=config.loss_rate,
    )
    presented_state: dict[str, str] = {}
    for agent_id in AGENT_IDS:
        message = environment_state_message(agent_id=agent_id, truth=truth)
        presented_state[agent_id] = message
        agent_messages[agent_id].append(active_episode_start_message())
        agent_messages[agent_id].append({"role": "user", "content": message})

    runtimes = {
        agent_id: AgentRuntime(
            agent_id=agent_id,
            config=agent_configs[agent_id],
            messages=agent_messages[agent_id],
        )
        for agent_id in AGENT_IDS
    }
    usage_records: list[dict[str, Any]] = []
    reasoning_traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    turns: dict[str, list[dict[str, Any]]] = {agent_id: [] for agent_id in AGENT_IDS}

    if config.communication_enabled:
        _run_communication_phase(
            state=state,
            runtimes=runtimes,
            config=config,
            turn_runner=turn_runner,
            usage_records=usage_records,
            reasoning_traces=reasoning_traces,
            errors=errors,
            turns=turns,
        )
    verdict_submissions = _run_verdict_phase(
        state=state,
        runtimes=runtimes,
        config=config,
        turn_runner=turn_runner,
        usage_records=usage_records,
        reasoning_traces=reasoning_traces,
        errors=errors,
        turns=turns,
    )

    verdicts = {
        agent_id: (
            None
            if verdict_submissions[agent_id] is None
            else list(verdict_submissions[agent_id].get("verdicts") or [])
        )
        for agent_id in AGENT_IDS
    }
    verdict_errors = {}
    for agent_id in AGENT_IDS:
        raw = _agent_verdict_raw(turns[agent_id])
        if not isinstance(verdicts[agent_id], list) or len(verdicts[agent_id]) != 3:
            last_error = ""
            if raw:
                last_error = str((raw.get("result") or {}).get("error") or "")
            verdict_errors[agent_id] = last_error or "no valid verdict submission"
    scores = score_episode(truth, verdicts, errors=verdict_errors)
    previous_cumulative = previous_cumulative or {agent_id: 0 for agent_id in AGENT_IDS}
    cumulative = {
        agent_id: int(previous_cumulative.get(agent_id, 0))
        + scores[agent_id].reward_total
        for agent_id in AGENT_IDS
    }

    # Feedback is constructed only after both verdicts have been collected.
    set_synthetic_phase(state, "feedback")
    feedbacks: dict[str, str] = {}
    for agent_id in AGENT_IDS:
        feedback = feedback_message(
            agent_id=agent_id,
            truth=truth,
            scores=scores,
            cumulative_reward=cumulative[agent_id],
        )
        feedbacks[agent_id] = feedback
        runtimes[agent_id].messages.append(outcome_feedback_start_message())
        runtimes[agent_id].messages.append({"role": "user", "content": feedback})

    reflections: dict[str, str | None] = {agent_id: None for agent_id in AGENT_IDS}
    if config.reflection:
        set_synthetic_phase(state, "reflection")
        prompt = reflection_prompt()
        for agent_id in AGENT_IDS:
            reflection_messages = deepcopy(runtimes[agent_id].messages)
            reflection_messages.append({"role": "user", "content": prompt})
            reflection_runtime = AgentRuntime(
                agent_id=agent_id,
                config=runtimes[agent_id].config,
                messages=reflection_messages,
            )
            recorder = make_llm_usage_recorder(
                records=usage_records,
                journal_path=config.usage_journal_path,
                invocation_id=config.usage_invocation_id,
                episode_id=str(state["episode_id"]),
                episode_index=truth.episode_index - 1,
                actor=agent_id,
                phase="private_reflection",
                round_idx=None,
                attempt_idx=None,
            )
            try:
                result = reflection_runner(
                    runtime=reflection_runtime,
                    state=state,
                    prompt=prompt,
                    usage_recorder=recorder,
                )
                reflection = str(result.get("content", "")).strip()
                reflection = reflection or None
            except Exception as exc:
                errors.append(
                    {
                        "agent": agent_id,
                        "phase": "private_reflection",
                        "round": None,
                        "attempt": 1,
                        "category": "technical",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                reflection = None
                result = {}
            reflections[agent_id] = reflection
            trace = result.get("reasoning_trace")
            if trace:
                reasoning_traces.append(
                    {
                        "agent": agent_id,
                        "phase": "private_reflection",
                        "round": None,
                        "attempt": 1,
                        "trace": trace,
                    }
                )
            if reflection:
                append_reflection(
                    runtimes[agent_id].messages,
                    prompt=prompt,
                    reflection=reflection,
                    replayed_reasoning=result.get("replayed_reasoning"),
                )

    score_dict = score_to_dict(scores)
    agents: dict[str, Any] = {}
    for agent_id in AGENT_IDS:
        raw = _agent_verdict_raw(turns[agent_id])
        agents[agent_id] = {
            "agent_id": agent_id,
            "peer_id": peer_id(agent_id),
            "verdict_raw": raw,
            **score_dict[agent_id],
            "cumulative_reward": cumulative[agent_id],
            "feedback": feedbacks[agent_id],
            "reflection": reflections[agent_id],
            "attempts": turns[agent_id],
            "messages": snapshot_visible_messages(agent_messages[agent_id]),
        }

    return {
        "episode_id": state["episode_id"],
        "episode_index": truth.episode_index,
        "protocol_id": PROTOCOL_ID,
        "truth": truth.to_dict(),
        "presented_state": presented_state,
        "communication": {
            "enabled": config.communication_enabled,
            "rounds": config.max_rounds if config.communication_enabled else 0,
            "char_limit": config.char_limit,
        },
        "loss_rate": config.loss_rate,
        "channel_transcript": _channel_transcript(state["events"]),
        "agents": agents,
        "events": deepcopy(state["events"]),
        "errors": errors,
        "llm_usage": usage_records,
        "llm_usage_summary": summarize_llm_usage(usage_records),
        "llm_reasoning_traces": reasoning_traces,
    }


__all__ = ["run_synthetic_episode"]
