"""Run the minimal GCG proof of concept entirely on Modal.

Usage:
    .venv/bin/modal run deploy/gcg_poc_modal.py

Optional:
    .venv/bin/modal run deploy/gcg_poc_modal.py \
        --run-json results/synthetic/grid/enabled/loss010/all-false/seed10/<run>/run.json \
        --episode 3 --agent alice --target reject
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RESULT_PATH = REPO_ROOT / "analysis/results/gcg_poc/result.json"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.44",
    "accelerate",
    "nanogcg",
    "sentencepiece",
)

app = modal.App("gcg-collusion-poc", image=image)
hf_cache = modal.Volume.from_name("gcg-hf-cache", create_if_missing=True)


def _build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
        prompt = prompt.replace(tokenizer.bos_token, "", 1)
    return prompt


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool-call metadata if a tokenizer cannot render the rich history."""
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            continue
        item = deepcopy(message)
        item.pop("tool_calls", None)
        item.pop("tool_call_id", None)
        item.pop("name", None)
        item.pop("reasoning_content", None)
        item.pop("reasoning", None)
        if role == "assistant" and not str(item.get("content") or "").strip():
            continue
        cleaned.append(item)
    return cleaned


def _with_suffix(messages: list[dict[str, Any]], suffix: str) -> list[dict[str, Any]]:
    rendered = deepcopy(messages)
    rendered[-1]["content"] = str(rendered[-1].get("content") or "").replace(
        "{optim_str}", suffix
    )
    return rendered


def _score_target(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    import torch

    prompt = _build_prompt(tokenizer, messages)
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(model.device)
    target_ids = tokenizer(
        target, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    target_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
    log_probs = torch.log_softmax(target_logits, dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    total = float(gathered.sum().item())
    tokens = int(target_ids.shape[1])
    return {
        "logprob_sum": total,
        "prob": math.exp(total),
        "mean_nll": -total / tokens,
        "target_token_ids": target_ids[0].tolist(),
        "target_token_count": tokens,
    }


def _parse_decision(text: str) -> str | None:
    match = re.search(r"\b(accept|reject)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    num_samples: int = 5,
) -> dict[str, Any]:
    import torch

    prompt = _build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        sampled = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
        )
        greedy = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    sampled_text = tokenizer.batch_decode(
        sampled[:, prompt_len:], skip_special_tokens=True
    )
    greedy_text = tokenizer.decode(greedy[0, prompt_len:], skip_special_tokens=True)
    return {
        "samples": sampled_text,
        "decisions": [_parse_decision(text) for text in sampled_text],
        "greedy": greedy_text,
        "greedy_decision": _parse_decision(greedy_text),
    }


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_gcg(payload: dict[str, Any]) -> dict[str, Any]:
    import nanogcg
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_id = str(payload.get("model_id") or MODEL_ID)
    target = str(payload.get("target") or "reject")
    messages = payload["messages"]

    if "{optim_str}" not in str(messages[-1].get("content") or ""):
        raise ValueError("the last message must contain the {optim_str} placeholder")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    try:
        _build_prompt(tokenizer, messages)
    except Exception:
        messages = _sanitize_messages(messages)
        if "{optim_str}" not in str(messages[-1].get("content") or ""):
            messages[-1]["content"] = str(messages[-1].get("content") or "") + (
                "\n\nFor this PoC probe, answer with exactly one word: "
                "\"accept\" or \"reject\".\n\n{optim_str}"
            )

    # Pick a 10-token initialization when possible.
    init_candidates = [
        "x x x x x x x x x x",
        "! ! ! ! ! ! ! ! ! !",
        "a a a a a a a a a a",
        "the the the the the the the the the the",
    ]
    optim_str_init = init_candidates[0]
    init_token_count = None
    for candidate in init_candidates:
        count = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if count == 10:
            optim_str_init = candidate
            init_token_count = count
            break
    if init_token_count is None:
        init_token_count = len(
            tokenizer(optim_str_init, add_special_tokens=False).input_ids
        )

    config = nanogcg.GCGConfig(
        num_steps=30,
        search_width=32,
        topk=64,
        batch_size=16,
        optim_str_init=optim_str_init,
        n_replace=1,
        buffer_size=0,
        use_mellowmax=False,
        early_stop=False,
        use_prefix_cache=True,
        allow_non_ascii=False,
        filter_ids=False,
        add_space_before_target=False,
        seed=0,
        verbosity="INFO",
    )

    baseline_messages = _with_suffix(messages, "")
    initial = _score_target(model, tokenizer, baseline_messages, target)
    before_generation = _generate(model, tokenizer, baseline_messages)

    result = nanogcg.run(
        model,
        tokenizer,
        messages,
        target,
        config=config,
    )

    final_messages = _with_suffix(messages, result.best_string)
    final = _score_target(model, tokenizer, final_messages, target)
    after_generation = _generate(model, tokenizer, final_messages)

    suffix_token_ids = tokenizer.encode(result.best_string, add_special_tokens=False)
    return {
        "model_id": model_id,
        "target": target,
        "gpu": torch.cuda.get_device_name(0),
        "optim_str_init": optim_str_init,
        "init_token_count": init_token_count,
        "suffix": result.best_string,
        "suffix_token_ids": suffix_token_ids,
        "suffix_tokens": tokenizer.convert_ids_to_tokens(suffix_token_ids),
        "suffix_token_count": len(suffix_token_ids),
        "initial": initial,
        "final": final,
        "losses": result.losses,
        "best_loss": result.best_loss,
        "before": before_generation,
        "after": after_generation,
        "metadata": payload.get("metadata", {}),
        "max_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 1e9, 3
        ),
    }


@app.local_entrypoint()
def main(
    run_json: str = "",
    episode: int = 3,
    agent: str = "alice",
    target: str = "reject",
    model_id: str = MODEL_ID,
) -> None:
    from analysis.gcg_poc.extract_context import extract_context

    if not run_json:
        matches = sorted(
            glob.glob(
                "results/synthetic/grid/enabled/loss010/all-false/seed10/*/run.json"
            )
        )
        if not matches:
            raise SystemExit(
                "No run.json found. Pass --run-json explicitly."
            )
        run_json = matches[0]

    payload = extract_context(
        run_json,
        episode_index=episode,
        agent=agent,
        target=target,
    )
    payload["model_id"] = model_id
    print(
        "Submitting to Modal H100: "
        f"{payload['metadata']['message_count']} messages, "
        f"{payload['metadata']['context_chars']} chars, target={target!r}"
    )
    result = run_gcg.remote(payload)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    before_p = result["before"]["greedy_decision"]
    after_p = result["after"]["greedy_decision"]
    print(f"Wrote {RESULT_PATH}")
    print(f"Suffix: {result['suffix']!r}")
    print(f"Suffix tokens: {result['suffix_token_count']}")
    print(
        f"Loss: {result['losses'][0]:.4f} -> {result['best_loss']:.4f} "
        f"({len(result['losses'])} steps)"
    )
    print(
        f"P({target}): {result['initial']['prob']:.6g} -> "
        f"{result['final']['prob']:.6g}"
    )
    print(
        "Greedy decision: "
        f"{before_p!r} -> {after_p!r}; "
        f"sampled before={result['before']['decisions']}, "
        f"after={result['after']['decisions']}"
    )
