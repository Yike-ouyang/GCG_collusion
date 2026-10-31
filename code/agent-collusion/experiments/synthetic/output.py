"""Build and persist synthetic-certification run output."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.llm import get_endpoint_metadata, get_litellm_version
from experiments.models import AgentConfig
from experiments.output import write_run_output
from experiments.protocol.state import AGENT_IDS
from experiments.synthetic.constants import (
    PROMPT_VERSION,
    PROTOCOL_ID,
    SCHEMA_VERSION,
)
from experiments.synthetic.prompts import (
    communication_phase_message,
    environment_state_message,
    initial_agent_messages,
    reflection_prompt,
    verdict_phase_message,
)
from experiments.synthetic.sequences import SyntheticSequence, sequence_to_dicts
from experiments.usage import summarize_llm_usage


@lru_cache(maxsize=8)
def _code_version(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unknown"


def _prompt_texts(
    *,
    sequence: SyntheticSequence,
    communication_enabled: bool,
    max_rounds: int,
    char_limit: int,
    verdict_suffix: str = "",
    verdict_suffix_agent: str = "",
) -> dict[str, Any]:
    first = sequence.episodes[0]
    return {
        "system_prompts": {
            agent_id: initial_agent_messages(
                agent_id=agent_id,
                communication_enabled=communication_enabled,
                max_rounds=max_rounds,
                char_limit=char_limit,
            )[0]["content"]
            for agent_id in AGENT_IDS
        },
        "first_environment_state": {
            agent_id: environment_state_message(agent_id=agent_id, truth=first)
            for agent_id in AGENT_IDS
        },
        "communication_phase": {
            agent_id: communication_phase_message(
                agent_id=agent_id,
                episode_number=1,
                max_rounds=max_rounds,
                char_limit=char_limit,
            )
            for agent_id in AGENT_IDS
        }
        if communication_enabled
        else {},
        "verdict_phase": {
            agent_id: verdict_phase_message(
                agent_id=agent_id,
                episode_number=1,
                suffix=(
                    verdict_suffix
                    if verdict_suffix_agent in ("", agent_id)
                    else ""
                ),
            )
            for agent_id in AGENT_IDS
        },
        "verdict_suffix": verdict_suffix,
        "verdict_suffix_agent": verdict_suffix_agent,
        "reflection_prompt": reflection_prompt(),
        "feedback_fields": [
            "own work status",
            "peer verdict",
            "peer work status",
            "own verdict",
            "peer acceptance bonus",
            "verdict accuracy bonus",
            "total",
            "cumulative",
            "honesty mismatch or invalid-submission notice",
        ],
    }


def _prompt_sha256(prompt_texts: dict[str, Any]) -> str:
    payload = json.dumps(prompt_texts, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def build_synthetic_run_output(
    *,
    repo_root: Path,
    run_label: str,
    sequence: SyntheticSequence,
    communication_enabled: bool,
    max_rounds: int,
    char_limit: int,
    verdict_suffix: str = "",
    verdict_suffix_agent: str = "",
    agent_configs: dict[str, AgentConfig],
    memory_scope: str,
    memory_length_by_agent: dict[str, int],
    results: list[dict[str, Any]],
    llm_usage_records: list[dict[str, Any]],
    usage_journal_path: Path | None,
) -> dict[str, Any]:
    prompts = _prompt_texts(
        sequence=sequence,
        communication_enabled=communication_enabled,
        max_rounds=max_rounds,
        char_limit=char_limit,
        verdict_suffix=verdict_suffix,
        verdict_suffix_agent=verdict_suffix_agent,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": PROTOCOL_ID,
        "run_config": {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_label": run_label,
            "code_version": _code_version(repo_root),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _prompt_sha256(prompts),
            "prompt_texts": prompts,
            "synthetic_sequence": {
                **sequence.to_config(),
                "episodes": sequence_to_dicts(sequence),
            },
            "synthetic_loss_rate": sequence.loss_rate,
            "communication": (
                "enabled" if communication_enabled else "disabled"
            ),
            "max_rounds": max_rounds,
            "char_limit": char_limit,
            "verdict_suffix": verdict_suffix,
            "verdict_suffix_agent": verdict_suffix_agent,
            "reflection": True,
            "memory_scope": memory_scope,
            "cross_episode_memory_length": memory_length_by_agent,
            "models": {
                agent_id: agent_configs[agent_id].model for agent_id in AGENT_IDS
            },
            "endpoints": {
                agent_id: get_endpoint_metadata(agent_configs[agent_id].model)
                for agent_id in AGENT_IDS
            },
            "llm_request_parameters": {
                agent_id: {
                    "temperature": agent_configs[agent_id].temperature,
                    "max_output_tokens": agent_configs[agent_id].max_output_tokens,
                    "reasoning_effort": agent_configs[agent_id].reasoning_effort,
                }
                for agent_id in AGENT_IDS
            },
            "litellm_version": get_litellm_version(),
            "llm_usage_journal": (
                None
                if usage_journal_path is None
                else {
                    "path": str(usage_journal_path),
                    "format": "jsonl",
                }
            ),
            "episode_count": len(results),
            "configured_episode_count": len(sequence.episodes),
        },
        "llm_usage_summary": summarize_llm_usage(llm_usage_records),
        "results": results,
    }


def persist_synthetic_run(output_path: Path, run_output: dict[str, Any]) -> None:
    write_run_output(output_path, run_output)
