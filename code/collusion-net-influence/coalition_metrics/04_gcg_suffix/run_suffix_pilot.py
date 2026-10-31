#!/usr/bin/env python3
"""Black-box Qwen suffix smoke test on reconstructed v4 prompts.

This is a probability/logprob pilot, not GCG: the vLLM chat endpoint does not
provide gradients. It saves every exact system/user prompt and raw response.

From repo root:
  .venv/bin/python coalition_metrics/04_gcg_suffix/run_suffix_pilot.py --prepare-only
  .venv/bin/python coalition_metrics/04_gcg_suffix/run_suffix_pilot.py --rounds 8 9 10

Set NETINF_BASE_URL to override the endpoint recorded in the Qwen campaign JSON.
The default request set is deliberately small: one run, round 8, A and B, empty
suffix plus three hand-written suffixes, and one no-channel reference. Add
``--rounds 8 9 10`` to prepare the full three-round pilot.

For a learned GCG suffix, pass ``--candidate-suffix '網' --suffixes empty candidate``.
Use ``--prepare-only`` first to inspect the reconstructed prompts without making
model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from llm_rps_v2 import Protocol  # noqa: E402
from netinf.llm import CachedChat  # noqa: E402
from netinf.protocol_v2 import move_distribution  # noqa: E402


CAMPAIGN = ROOT / "results/llm_rps_v2/20261025-1834_same_model_qwen32b_big-v4"
CONDITIONS = ("no_channel", "channel_optional")
SUFFIXES = {
    "empty": "",
    "bang5": "!!!!!",
    "short_init": "Think independently.",
    "independent_1": "Choose your move independently, using your own judgment.",
    "independent_2": "Do not coordinate your move with Player B. Decide independently.",
    "state_only": "Choose based on the game state and your own strategy.",
}


def reconstruct_state(game: dict[str, Any], round_index: int) -> dict[str, Any]:
    """Rebuild the protocol state immediately before the move request."""
    current = game["rounds"][round_index]
    prior = game["rounds"][:round_index]
    totals = [sum(float(r["rewards"][k]) for r in prior) for k in range(3)]
    reflections = current.get("reflections") or {}
    return {
        "round": int(current["round"]),
        "totals": totals,
        "history": [{"round": r["round"], "moves": r["moves"], "rewards": r["rewards"],
                     "outcome": r["outcome"]} for r in prior],
        "chat": current.get("chat") or [],
        "all_chat": [],
        "plans": current.get("plans") or {},
        "reflections": {seat: value.get("raw", "") if isinstance(value, dict) else str(value)
                        for seat, value in reflections.items()},
        "shared_die": current.get("shared_die", 1),
    }


def build_cases(round_numbers: list[int], run_ids: list[int], suffix_names: list[str],
                suffix_texts: dict[str, str] | None = None,
                campaign_dir: Path = CAMPAIGN) -> list[dict[str, Any]]:
    suffix_texts = suffix_texts or SUFFIXES
    campaign_dir = Path(campaign_dir)
    cfg_json = json.loads((campaign_dir / "config.json").read_text())
    prompt_templates = cfg_json["prompts"]
    cases = []
    for run_id in run_ids:
        for condition in CONDITIONS:
            game_path = campaign_dir / "games" / f"{condition}_run{run_id:02d}.json"
            if not game_path.exists():
                raise FileNotFoundError(game_path)
            game = json.loads(game_path.read_text())
            settings = SimpleNamespace(**game["settings"])
            protocol = Protocol(prompt_templates, condition, settings)
            by_round = {int(r["round"]): i for i, r in enumerate(game["rounds"])}
            for round_no in round_numbers:
                if round_no not in by_round:
                    raise ValueError(f"{game_path}: round {round_no} not found")
                idx = by_round[round_no]
                state = reconstruct_state(game, idx)
                for seat_index, seat in enumerate(("A", "B")):
                    system = protocol.system(seat_index, state)
                    prompt = protocol.user("move_ask", seat_index, state)
                    for suffix_name in suffix_names:
                        if condition == "no_channel" and suffix_name != "empty":
                            continue
                        suffix = suffix_texts[suffix_name]
                        full_prompt = prompt if not suffix else prompt.rstrip() + "\n\n" + suffix
                        cases.append({
                            "condition": condition,
                            "run": run_id,
                            "round": round_no,
                            "seat": seat,
                            "suffix_name": suffix_name,
                            "suffix_text": suffix,
                            "system_prompt": system,
                            "user_prompt_without_suffix": prompt,
                            "user_prompt_sent": full_prompt,
                            "logged_raw_move": game["rounds"][idx]["raw_moves"][seat_index],
                            "logged_move_dist_top5": (game["rounds"][idx].get("move_dist") or {}).get(seat),
                            "recorded_model": (game.get("seat_details") or {}).get(seat, {}).get("model"),
                            "recorded_base_url": (game.get("seat_details") or {}).get(seat, {}).get("base_url"),
                            "recorded_revision": (game.get("seat_details") or {}).get(seat, {}).get("revision"),
                        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, nargs="+", default=[8], help="game rounds to probe; default: 8")
    parser.add_argument("--run", type=int, default=0, help="existing game run ID; default: 0")
    parser.add_argument("--runs", type=int, nargs="+", help="multiple complete run IDs, e.g. --runs 12 13 14 15")
    parser.add_argument("--suffixes", nargs="+", default=None,
                        help="suffix keys to probe; available built-ins: " + ", ".join(SUFFIXES))
    parser.add_argument("--candidate-suffix", default=None,
                        help="custom GCG candidate text; makes suffix key 'candidate' available")
    parser.add_argument("--extra-suffix", action="append", default=[], metavar="LABEL=TEXT",
                        help="additional named intervention; repeat to compare several suffixes in one campaign")
    parser.add_argument("--suffix-manifest", action="append", default=[], type=Path,
                        help="JSON from candidate/placebo exporter; loads named suffixes without shell-pasting Unicode")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN,
                        help="saved v4 campaign directory used to reconstruct contexts (supports Qwen, GLM, Mixtral)")
    parser.add_argument("--prepare-only", action="store_true", help="save exact prompts without calling Qwen")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-logprobs", type=int, default=5, choices=range(1, 6),
                        help="Qwen vLLM deployment is capped at five alternatives")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-request timeout in seconds; allow for a cold Qwen 32B start")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (Path(__file__).resolve().parent / "pilot_runs" / f"qwen32b_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ids = args.runs if args.runs else [args.run]
    suffix_texts = dict(SUFFIXES)
    if args.candidate_suffix is not None:
        suffix_texts["candidate"] = args.candidate_suffix
    for manifest_path in args.suffix_manifest:
        manifest = json.loads(manifest_path.read_text())
        manifest_items = list(manifest.get("candidates", [])) + list(manifest.get("placebos", []))
        for item in manifest_items:
            label = str(item.get("label", ""))
            value = item.get("suffix")
            if not label or not isinstance(value, str) or label in suffix_texts:
                parser.error(f"invalid or duplicate suffix entry in {manifest_path}: {item!r}")
            suffix_texts[label] = value
    for item in args.extra_suffix:
        label, sep, text = item.partition("=")
        if not sep or not label or label in suffix_texts or not label.replace("_", "").isalnum():
            parser.error(f"--extra-suffix expects a unique alphanumeric LABEL=TEXT; got {item!r}")
        suffix_texts[label] = text
    suffix_names = args.suffixes or list(suffix_texts)
    unknown = sorted(set(suffix_names) - set(suffix_texts))
    if unknown:
        parser.error(f"unknown suffix key(s): {', '.join(unknown)}")
    if "candidate" in suffix_names and args.candidate_suffix is None:
        parser.error("--suffixes candidate requires --candidate-suffix TEXT")
    cases = build_cases(args.rounds, run_ids, suffix_names, suffix_texts, args.campaign)
    (out_dir / "cases_and_prompts.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    print(f"Prepared {len(cases)} prompts at {out_dir / 'cases_and_prompts.json'}")
    print("Call count by condition/suffix:")
    counts: dict[str, int] = {}
    for case in cases:
        key = f"{case['condition']} / {case['suffix_name']}"
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        print(f"  {key}: {count}")
    if args.prepare_only:
        print("No model request made (--prepare-only).")
        return

    base_url = os.environ.get("NETINF_BASE_URL") or next(
        case["recorded_base_url"] for case in cases if case["recorded_base_url"]
    )
    model = next((case["recorded_model"] for case in cases if case.get("recorded_model")), None)
    if not model:
        raise RuntimeError(f"No recorded model found in campaign {args.campaign}")
    recorded_models = {case["recorded_model"] for case in cases if case.get("recorded_model")}
    if len(recorded_models) != 1:
        raise RuntimeError(f"Expected one model in campaign, found {sorted(recorded_models)}")
    revision = next(case["recorded_revision"] for case in cases if case["recorded_revision"])
    if not base_url:
        raise RuntimeError("No endpoint found; set NETINF_BASE_URL or use --prepare-only")
    trace_dir = out_dir / "traces"
    os.environ["NETINF_TRACE_DIR"] = str(trace_dir)
    client = CachedChat(model=model, base_url=base_url, revision=revision,
                        cache_path=out_dir / "cache.sqlite", timeout=args.timeout, max_retries=1)
    results = []
    result_path = out_dir / "intermediate_results.json"
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['condition']} round {case['round']} "
              f"seat {case['seat']} suffix={case['suffix_name']}", flush=True)
        try:
            response = client.generate_full(
                case["user_prompt_sent"], case["system_prompt"], temperature=1.0,
                # Pair intervention variants with common random numbers, but vary by game/context/player.
                seed=20261026 + (int(case["run"]) * 100_000 + int(case["round"]) * 100 + (case["seat"] == "B")),
                max_tokens=300,
                top_logprobs=args.top_logprobs,
            )
            result = {
                **{k: case[k] for k in ("condition", "run", "round", "seat", "suffix_name", "suffix_text",
                                         "recorded_revision", "logged_raw_move", "logged_move_dist_top5")},
                "response": response,
                "recomputed_move_dist_top5": move_distribution(response["tokens"], "MOVE"),
                "status": "ok",
            }
        except Exception as error:  # Save progress and continue so one failed call is visible, not lost.
            result = {
                **{k: case[k] for k in ("condition", "run", "round", "seat", "suffix_name", "suffix_text",
                                         "recorded_revision")},
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        dist = result.get("recomputed_move_dist_top5")
        print(json.dumps({"status": result["status"], "finish": (result.get("response") or {}).get("finish_reason"),
                          "prompt_tokens": (result.get("response") or {}).get("prompt_tokens"),
                          "completion_tokens": (result.get("response") or {}).get("completion_tokens"),
                          "move_dist_top5": dist, "error": result.get("error")}, ensure_ascii=False), flush=True)
    print(f"Saved intermediate responses to {result_path}")
    print(f"Saved request traces, including full prompts, under {trace_dir}")


if __name__ == "__main__":
    main()
