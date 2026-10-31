"""Run synthetic-certification trajectories."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.memory.cross_episode import set_cross_episode_memory_from_results
from experiments.models import AgentConfig
from experiments.protocol.state import AGENT_IDS
from experiments.synthetic.constants import (
    COMMUNICATION_ENABLED,
    DEFAULT_LOSS_RATE,
)
from experiments.synthetic.episode_runner import run_synthetic_episode
from experiments.synthetic.output import build_synthetic_run_output, persist_synthetic_run
from experiments.synthetic.prompts import initial_agent_messages
from experiments.synthetic.sequences import resolve_sequence
from experiments.synthetic.state import SyntheticEpisodeConfig
from experiments.tasks import sanitize_label
from experiments.usage import LLM_USAGE_JOURNAL_FILENAME


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _agent_configs(args: Any) -> dict[str, AgentConfig]:
    return {
        agent_id: AgentConfig(
            model=getattr(args, f"{agent_id}_model"),
            reasoning_effort=getattr(args, f"{agent_id}_reasoning_effort"),
            temperature=getattr(args, f"{agent_id}_temperature"),
            max_output_tokens=getattr(args, f"{agent_id}_max_output_tokens"),
        )
        for agent_id in AGENT_IDS
    }


def _memory_lengths(args: Any) -> dict[str, int]:
    return {
        agent_id: int(getattr(args, f"{agent_id}_cross_episode_memory_length"))
        for agent_id in AGENT_IDS
    }


def _create_output_path(args: Any, repo_root: Path, run_label: str) -> Path:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = sanitize_label(run_label)
    suffix = f"{random.SystemRandom().randrange(16**6):06x}"
    name = f"run_{label}_{timestamp}_{suffix}" if label else f"run_{timestamp}_{suffix}"
    run_directory = output_dir / name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory / "run.json"


def _resolved_sequence(args: Any, repo_root: Path):
    return resolve_sequence(
        schedule=getattr(args, "synthetic_schedule", "") or "",
        sequence_path=getattr(args, "synthetic_sequence", "") or "",
        seed=int(getattr(args, "synthetic_seed", 0)),
        loss_rate=float(getattr(args, "synthetic_loss_rate", DEFAULT_LOSS_RATE)),
        repo_root=repo_root,
    )


def _resolved_label(args: Any, sequence: Any) -> str:
    label = str(getattr(args, "run_label", "") or "")
    if label:
        return label
    communication = getattr(args, "communication", COMMUNICATION_ENABLED)
    return (
        f"synthetic-{sequence.schedule}-loss{sequence.loss_rate:.2f}-"
        f"{communication}-seed{sequence.seed}"
    )


def render_dry_run(args: Any, repo_root: Path) -> str:
    """Render configuration, sequence, and prompts without network or writes."""
    sequence = _resolved_sequence(args, repo_root)
    communication_enabled = (
        getattr(args, "communication", COMMUNICATION_ENABLED)
        == COMMUNICATION_ENABLED
    )
    config = build_synthetic_run_output(
        repo_root=repo_root,
        run_label=_resolved_label(args, sequence),
        sequence=sequence,
        communication_enabled=communication_enabled,
        max_rounds=int(args.max_rounds),
        char_limit=int(args.char_limit),
        verdict_suffix=str(getattr(args, "synthetic_verdict_suffix", "") or ""),
        verdict_suffix_agent=str(
            getattr(args, "synthetic_verdict_suffix_agent", "") or ""
        ),
        agent_configs=_agent_configs(args),
        memory_scope=args.cross_episode_memory_scope,
        memory_length_by_agent=_memory_lengths(args),
        results=[],
        llm_usage_records=[],
        usage_journal_path=None,
    )
    sections = [
        f"# {config['protocol_id']} dry run",
        "## Configuration",
        json.dumps(config["run_config"], ensure_ascii=False, indent=2, sort_keys=True),
        "## Sequence",
        json.dumps(
            config["run_config"]["synthetic_sequence"],
            ensure_ascii=False,
            indent=2,
        ),
    ]
    return "\n\n".join(sections) + "\n"


def run_synthetic(args: Any, repo_root: Path) -> Path | None:
    """Run one synthetic trajectory from parsed CLI arguments."""
    args.max_rounds = int(getattr(args, "synthetic_rounds", args.max_rounds))
    sequence = _resolved_sequence(args, repo_root)
    run_label = _resolved_label(args, sequence)
    communication_enabled = (
        getattr(args, "communication", COMMUNICATION_ENABLED)
        == COMMUNICATION_ENABLED
    )
    agent_configs = _agent_configs(args)
    memory_lengths = _memory_lengths(args)
    verbose = not bool(getattr(args, "quiet", False))

    output_path = _create_output_path(args, repo_root, run_label)
    usage_journal_path = output_path.parent / LLM_USAGE_JOURNAL_FILENAME
    invocation_id = (
        f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}_"
        f"{random.SystemRandom().randrange(16**6):06x}"
    )
    agent_messages = {
        agent_id: initial_agent_messages(
            agent_id=agent_id,
            communication_enabled=communication_enabled,
            max_rounds=int(args.max_rounds),
            char_limit=int(args.char_limit),
        )
        for agent_id in AGENT_IDS
    }
    results: list[dict[str, Any]] = []
    cumulative = {agent_id: 0 for agent_id in AGENT_IDS}
    llm_usage_records: list[dict[str, Any]] = []

    def persist() -> None:
        run_output = build_synthetic_run_output(
            repo_root=repo_root,
            run_label=run_label,
            sequence=sequence,
            communication_enabled=communication_enabled,
            max_rounds=int(args.max_rounds),
            char_limit=int(args.char_limit),
            verdict_suffix=str(
                getattr(args, "synthetic_verdict_suffix", "") or ""
            ),
            verdict_suffix_agent=str(
                getattr(args, "synthetic_verdict_suffix_agent", "") or ""
            ),
            agent_configs=agent_configs,
            memory_scope=args.cross_episode_memory_scope,
            memory_length_by_agent=memory_lengths,
            results=results,
            llm_usage_records=llm_usage_records,
            usage_journal_path=usage_journal_path,
        )
        run_output["run_config"]["output_path"] = str(output_path)
        persist_synthetic_run(output_path, run_output)

    _log(
        f"Synthetic sequence {sequence.schedule} seed={sequence.seed} "
        f"loss={sequence.loss_rate:.2f} episodes={len(sequence.episodes)} communication="
        f"{'enabled' if communication_enabled else 'disabled'}",
        verbose,
    )
    _log(f"Saving incremental results to: {output_path}", verbose)
    persist()

    for truth in sequence.episodes:
        _log(
            f"Episode {truth.episode_index}/{len(sequence.episodes)} "
            f"alice_parts={truth.alice_parts} "
            f"bob_parts={truth.bob_parts}",
            verbose,
        )
        result = run_synthetic_episode(
            truth=truth,
            agent_configs=agent_configs,
            config=SyntheticEpisodeConfig(
                communication_enabled=communication_enabled,
                char_limit=int(args.char_limit),
                max_rounds=int(args.max_rounds),
                loss_rate=float(sequence.loss_rate),
                reflection=True,
                verdict_suffix=str(
                    getattr(args, "synthetic_verdict_suffix", "") or ""
                ),
                verdict_suffix_agent=str(
                    getattr(args, "synthetic_verdict_suffix_agent", "") or ""
                ),
                verbose=verbose,
                usage_journal_path=usage_journal_path,
                usage_invocation_id=invocation_id,
            ),
            agent_messages=agent_messages,
            previous_cumulative=cumulative,
        )
        results.append(result)
        llm_usage_records.extend(result.get("llm_usage") or [])
        cumulative = {
            agent_id: int(result["agents"][agent_id]["cumulative_reward"])
            for agent_id in AGENT_IDS
        }
        set_cross_episode_memory_from_results(
            agent_messages,
            scope=args.cross_episode_memory_scope,
            episode_id=str(result["episode_id"]),
            max_memory_episodes_by_agent=memory_lengths,
        )
        persist()
        _log(
            "  rewards: "
            + " ".join(
                f"{agent_id}=+{result['agents'][agent_id]['reward_total']} "
                f"(cumulative {result['agents'][agent_id]['cumulative_reward']})"
                for agent_id in AGENT_IDS
            ),
            verbose,
        )
    _log(f"Saved results to: {output_path}", verbose)
    return output_path
