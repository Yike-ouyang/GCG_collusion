"""Qwen GCG for v4 decision prompts.

The multicontext mode optimizes the mean context-level KL for rounds 8-10.
The chronological mode optimizes one suffix independently per complete run,
walking rounds 1-60 in blocks and scoring each update on the next block.
Both match optional-channel move-policy distributions to the no-channel
reference; neither directly estimates observed action CMI. Move probabilities
are computed per prompt before losses are averaged; raw logits are never
averaged across prompts.

Prepare bundles with run_suffix_pilot.py --prepare-only, then launch via Modal.
The default starting suffix is five exclamation marks ("!!!!!").
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import modal


MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
MOVES = ("Rock", "Paper", "Scissors")
APP = modal.App("netinf-gcg-qwen-multicontext")
RESULTS = modal.Volume.from_name("netinf-v2", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("netinf-hf-cache", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.9.1",
        setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"],
    )
    .entrypoint([])
    .pip_install("bitsandbytes>=0.46.1,<0.47", "accelerate>=1.2")
)


def _find_case(cases: list[dict[str, Any]], condition: str, run: int, round_no: int,
               seat: str, suffix_name: str = "empty") -> dict[str, Any]:
    for case in cases:
        if (case["condition"] == condition and int(case["run"]) == run
                and int(case["round"]) == round_no and case["seat"] == seat
                and case["suffix_name"] == suffix_name):
            return case
    raise KeyError(
        f"missing {condition} run={run} round={round_no} seat={seat} suffix={suffix_name}"
    )


def _render(tokenizer, case: dict[str, Any], suffix_text: str):
    messages = [
        {"role": "system", "content": case["system_prompt"]},
        {"role": "user", "content": case["user_prompt_without_suffix"].rstrip() +
         ("\n\n" + suffix_text if suffix_text else "")},
        {"role": "assistant", "content": "MOVE: "},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    enc = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    positions: list[int] = []
    if suffix_text:
        start = rendered.rfind(suffix_text)
        if start < 0:
            raise ValueError("suffix text not found in rendered prompt")
        end = start + len(suffix_text)
        positions = [i for i, (a, b) in enumerate(offsets)
                     if b > a and a < end and b > start]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("suffix token span is missing or non-contiguous")
    return {
        "input_ids": enc["input_ids"].to("cuda"),
        "suffix_positions": positions,
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "rendered_prompt": rendered,
    }


def _move_token_sequences(tokenizer) -> list[list[int]]:
    sequences = [tokenizer.encode(move, add_special_tokens=False) for move in MOVES]
    if any(not ids for ids in sequences):
        raise RuntimeError(f"empty move token sequence: {sequences}")
    return sequences


def _next_logits(model, encoded: dict[str, Any], suffix_embeds=None,
                 replacement_ids: list[int] | None = None,
                 append_ids: list[int] | None = None):
    import torch

    ids = encoded["input_ids"]
    if replacement_ids is not None:
        if len(replacement_ids) != len(encoded["suffix_positions"]):
            raise ValueError("replacement token count differs from this suffix span")
        ids = ids.clone()
        ids[0, encoded["suffix_positions"]] = torch.tensor(replacement_ids, device=ids.device)
    embedding = model.get_input_embeddings()
    embeds = embedding(ids)
    if suffix_embeds is not None:
        positions = encoded["suffix_positions"]
        if not positions or len(positions) != suffix_embeds.shape[0]:
            raise ValueError("trainable suffix embedding does not match encoded suffix span")
        first, last = positions[0], positions[-1] + 1
        embeds = torch.cat((embeds[:, :first], suffix_embeds.unsqueeze(0), embeds[:, last:]), dim=1)
    if append_ids:
        extra = embedding(torch.tensor(append_ids, device=ids.device).unsqueeze(0))
        embeds = torch.cat((embeds, extra), dim=1)
    mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
    out = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
    return model.lm_head(out.last_hidden_state[:, -1, :]).float()[0]


def _move_probs(model, encoded: dict[str, Any], move_sequences: list[list[int]],
                suffix_embeds=None, replacement_ids: list[int] | None = None):
    import torch
    import torch.nn.functional as F

    logits = _next_logits(model, encoded, suffix_embeds, replacement_ids)
    first_logps = F.log_softmax(logits, dim=-1)
    scores = []
    for sequence in move_sequences:
        score = first_logps[sequence[0]]
        prefix = [sequence[0]]
        for token_id in sequence[1:]:
            cont_logits = _next_logits(model, encoded, suffix_embeds, replacement_ids,
                                       append_ids=prefix)
            score = score + F.log_softmax(cont_logits, dim=-1)[token_id]
            prefix.append(token_id)
        scores.append(score)
    logps = torch.stack(scores)
    return torch.softmax(logps, dim=0), logps


def _kl(target, candidate):
    target = target.clamp_min(1e-30)
    candidate = candidate.clamp_min(1e-30)
    return (target * (target.log() - candidate.log())).sum()


def _pair_context_loss(target_probs, candidate_probs, individual_weight: float):
    import torch

    ta, tb = target_probs
    pa, pb = candidate_probs
    q0 = torch.outer(ta, tb).reshape(-1)
    q = torch.outer(pa, pb).reshape(-1)
    pair_kl = _kl(q0, q)
    indiv = _kl(ta, pa) + _kl(tb, pb)
    return pair_kl + individual_weight * indiv, pair_kl, indiv


def _float_probs(p) -> list[float]:
    return [float(x) for x in p.detach().float().cpu().tolist()]


def _loss_summary(context_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_run: dict[int, list[float]] = {}
    by_round: dict[int, list[float]] = {}
    for row in context_rows:
        by_run.setdefault(int(row["run"]), []).append(float(row[f"{label}_loss_nats"]))
        by_round.setdefault(int(row["round"]), []).append(float(row[f"{label}_loss_nats"]))
    values = [float(row[f"{label}_loss_nats"]) for row in context_rows]
    mean_nats = sum(values) / len(values) if values else float("nan")
    return {
        "mean_context_kl_nats": mean_nats,
        "mean_context_kl_bits": mean_nats / math.log(2),
        "per_run_mean_kl_nats": {str(k): sum(v) / len(v) for k, v in sorted(by_run.items())},
        "per_round_mean_kl_nats": {str(k): sum(v) / len(v) for k, v in sorted(by_round.items())},
        "context_count": len(values),
    }


def _individual_kl_summary(context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seat_index, seat in enumerate(("A", "B")):
        values = [float(row[f"individual_kl_{seat}_nats"]) for row in context_rows]
        result[seat] = {
            "mean_context_kl_nats": sum(values) / len(values),
            "mean_context_kl_bits": sum(values) / len(values) / math.log(2),
            "per_round_mean_kl_nats": {
                str(rnd): sum(float(row[f"individual_kl_{seat}_nats"]) for row in context_rows
                              if int(row["round"]) == rnd) /
                sum(1 for row in context_rows if int(row["round"]) == rnd)
                for rnd in sorted({int(row["round"]) for row in context_rows})
            },
        }
    return result


def _bootstrap_run_gain(empty_rows, candidate_rows, n_boot: int = 2000, seed: int = 20261026):
    """Percentile interval over complete game IDs, keeping rounds clustered."""
    import random

    def means(rows, key):
        by_run: dict[int, list[float]] = {}
        for row in rows:
            by_run.setdefault(int(row["run"]), []).append(float(row[key]))
        return {r: sum(v) / len(v) for r, v in by_run.items()}

    a = means(empty_rows, "loss_nats")
    b = means(candidate_rows, "loss_nats")
    runs = sorted(set(a) & set(b))
    gains = [a[r] - b[r] for r in runs]
    if len(gains) < 2:
        return {"run_count": len(gains), "mean_gain_nats": sum(gains) / max(1, len(gains)),
                "ci95_nats": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [gains[rng.randrange(len(gains))] for _ in gains]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return {
        "run_count": len(gains),
        "mean_gain_nats": sum(gains) / len(gains),
        "mean_gain_bits": sum(gains) / len(gains) / math.log(2),
        "ci95_nats": [samples[int(0.025 * n_boot)], samples[min(n_boot - 1, int(0.975 * n_boot))]],
        "per_run_gain_nats": {str(r): a[r] - b[r] for r in runs},
    }


def _pooled_soft_mi(context_probs: list[dict[str, Any]]) -> float:
    """Secondary pooled MI in bits; context mixing can induce apparent dependence."""
    import torch

    qbar = torch.stack([torch.outer(x["A"], x["B"]).reshape(-1) for x in context_probs]).mean(0)
    pbar_a = torch.stack([x["A"] for x in context_probs]).mean(0)
    pbar_b = torch.stack([x["B"] for x in context_probs]).mean(0)
    product = torch.outer(pbar_a, pbar_b).reshape(-1)
    return float(_kl(qbar, product).detach().cpu() / math.log(2))


def _encode_contexts(tokenizer, cases, run_ids, round_ids, suffix_text):
    contexts = []
    for run_id in run_ids:
        for round_no in round_ids:
            target_cases = [_find_case(cases, "no_channel", run_id, round_no, s) for s in ("A", "B")]
            optional_cases = [_find_case(cases, "channel_optional", run_id, round_no, s) for s in ("A", "B")]
            contexts.append({
                "run": run_id,
                "round": round_no,
                "target_cases": target_cases,
                "optional_cases": optional_cases,
                "target_encoded": [_render(tokenizer, x, "") for x in target_cases],
                "optional_encoded": [_render(tokenizer, x, suffix_text) for x in optional_cases],
            })
    return contexts


def _target_probs(model, move_sequences, contexts):
    import torch

    rows = []
    with torch.no_grad():
        for ctx in contexts:
            probs = [_move_probs(model, enc, move_sequences)[0].detach()
                     for enc in ctx["target_encoded"]]
            rows.append(probs)
    return rows


def _evaluate(model, tokenizer, move_sequences, contexts, target_rows, suffix_text, individual_weight):
    import torch

    context_rows = []
    pooled = []
    with torch.no_grad():
        for ctx, targets in zip(contexts, target_rows):
            encoded = [_render(tokenizer, case, suffix_text) for case in ctx["optional_cases"]]
            probs = [_move_probs(model, enc, move_sequences)[0].detach() for enc in encoded]
            total, pair_kl, indiv = _pair_context_loss(targets, probs, individual_weight)
            context_rows.append({
                "run": ctx["run"], "round": ctx["round"],
                "loss_nats": float(total.cpu()), "pair_kl_nats": float(pair_kl.cpu()),
                "individual_kl_nats": float(indiv.cpu()),
                "individual_kl_A_nats": float(_kl(targets[0], probs[0]).cpu()),
                "individual_kl_B_nats": float(_kl(targets[1], probs[1]).cpu()),
                "target_probabilities_A_B": [_float_probs(x) for x in targets],
                "candidate_probabilities_A_B": [_float_probs(x) for x in probs],
            })
            pooled.append({"A": probs[0], "B": probs[1]})
    return {
        "context_rows": context_rows,
        "summary": _loss_summary(
            [{**r, "candidate_loss_nats": r["loss_nats"]} for r in context_rows],
            "candidate",
        ),
        "individual_kl_summary": _individual_kl_summary(context_rows),
        "pooled_soft_mi_bits_secondary": _pooled_soft_mi(pooled),
    }


def _gradient(model, tokenizer, move_sequences, contexts, target_rows, suffix_text,
              suffix_ids, individual_weight):
    import torch

    emb = model.get_input_embeddings()
    parameter = torch.nn.Parameter(
        emb(torch.tensor(suffix_ids, device="cuda")).detach().clone()
    )
    parameter.grad = None
    # Recompute current partner distributions without graphs, then accumulate
    # each player's partial derivative one at a time to fit long 32B prompts.
    for ctx, targets in zip(contexts, target_rows):
        enc_a, enc_b = [_render(tokenizer, c, suffix_text) for c in ctx["optional_cases"]]
        with torch.no_grad():
            p_a = _move_probs(model, enc_a, move_sequences)[0].detach()
            p_b = _move_probs(model, enc_b, move_sequences)[0].detach()
        for seat_index, enc in enumerate((enc_a, enc_b)):
            own, _ = _move_probs(model, enc, move_sequences, suffix_embeds=parameter)
            partner = p_b if seat_index == 0 else p_a
            target_own = targets[seat_index]
            target_partner = targets[1 - seat_index]
            if seat_index == 0:
                total, _, _ = _pair_context_loss((target_own, target_partner), (own, partner), individual_weight)
            else:
                # Restore A,B order for the joint law.
                total, _, _ = _pair_context_loss((target_partner, target_own), (partner, own), individual_weight)
            (total / len(contexts)).backward()
    grad = parameter.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0.0:
        raise RuntimeError("suffix gradient was absent, non-finite, or zero")
    return parameter.detach(), grad.detach()


def _propose_candidates(tokenizer, embedding, suffix_ids, grad, candidate_count):
    import torch

    position = int(grad.float().norm(dim=1).argmax().item())
    scores = embedding.weight.detach().float() @ grad[position].float()
    scores[list(tokenizer.all_special_ids)] = float("inf")
    # A conservative punctuation/word shortlist. The exact decoded text is
    # still rescored on every training context before an update is accepted.
    proposed = torch.topk(scores, k=min(64, scores.numel()), largest=False).indices.cpu().tolist()
    rows = []
    for token_id in proposed:
        if token_id == suffix_ids[position]:
            continue
        ids = list(suffix_ids)
        ids[position] = int(token_id)
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "<|" in text or "|>" in text or not text.strip():
            continue
        if tokenizer.encode(text, add_special_tokens=False) == suffix_ids:
            continue
        rows.append({"suffix": text, "token_ids": tokenizer.encode(text, add_special_tokens=False),
                     "changed_position": position, "changed_token_id": int(token_id)})
        if len(rows) >= candidate_count:
            break
    return position, rows


def _gain(empty_loss, loss):
    nats = empty_loss - loss
    return {"gain_nats": nats, "gain_bits": nats / math.log(2),
            "relative_gain_percent": (100 * nats / empty_loss) if empty_loss > 0 else None}


@APP.function(
    image=IMAGE,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/vol": RESULTS, "/root/.cache/huggingface": HF_CACHE},
)
def run_multicontext_gcg(train_json: str, validation_json: str, output_name: str,
                         initial_suffix: str = "!!!!!", max_steps: int = 0,
                         candidate_count: int = 4, patience: int = 2,
                         individual_weight: float = 0.25) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.time()
    out_dir = Path("/vol/results/gcg_qwen_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{output_name}.json"
    report_path = out_dir / f"{output_name}.md"
    record: dict[str, Any] = {
        "status": "running", "model": MODEL_ID, "revision": REVISION,
        "weight_mode": "frozen 8-bit Qwen weights; gradient only at suffix input embeddings",
        "initial_suffix": initial_suffix,
        "objective": "mean per-context KL(no-channel A×B || optional-channel A×B) + alpha times mean individual KLs",
        "alpha_individual_kl": individual_weight,
        "aggregation": "Compute each prompt's probability distribution and each matched context's KL separately; average context losses. Never average raw logits.",
        "hypotheses": [
            "H1: a GCG token replacement lowers the mean matched-context KL on training games.",
            "H2: the KL reduction persists on held-out complete games and rounds 8-10.",
            "H3: both players' move distributions move toward the no-channel reference; the pair-law KL does not hide a one-player-only change.",
            "H4: the selected suffix remains useful after round and run variation; this prompt-level result still needs full-game rollout validation.",
        ],
        "stages": [], "warnings": [
            "No-channel and optional-channel contexts share run IDs and round numbers but come from separate trajectories, not identical public states.",
            "The pair law is a product of the two independently queried policies given the same context; this is a distribution-matching surrogate, not observed CMI or a causal effect.",
            "Validation is used for early stopping/model selection. Reserve fresh game IDs for a final unbiased experiment.",
            "Qwen legal-move probabilities are normalized over Rock/Paper/Scissors sequence probabilities; the assistant prefix is MOVE: .",
        ],
    }

    def save():
        record["elapsed_seconds"] = time.time() - started
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        report_path.write_text(_make_report(record))
        RESULTS.commit()

    try:
        train_cases = json.loads(train_json)
        validation_cases = json.loads(validation_json)
        train_runs = sorted({int(x["run"]) for x in train_cases})
        validation_runs = sorted({int(x["run"]) for x in validation_cases})
        train_rounds = sorted({int(x["round"]) for x in train_cases})
        validation_rounds = sorted({int(x["round"]) for x in validation_cases})
        if not train_runs or not validation_runs or not train_rounds:
            raise ValueError("train and validation bundles must contain complete paired contexts")
        if set(train_runs) & set(validation_runs):
            raise ValueError("training and validation run IDs must be disjoint")
        if train_rounds != [8, 9, 10] or validation_rounds != [8, 9, 10]:
            raise ValueError(f"expected rounds 8,9,10 in both bundles; got {train_rounds}, {validation_rounds}")
        record["split"] = {"training_run_ids": train_runs, "validation_run_ids": validation_runs,
                           "training_rounds": train_rounds, "validation_rounds": validation_rounds,
                           "training_contexts": len(train_runs) * len(train_rounds),
                           "validation_contexts": len(validation_runs) * len(validation_rounds)}
        record["search"] = {"max_coordinate_steps": max_steps or None,
                            "candidate_count_per_step": candidate_count,
                            "validation_patience": patience,
                            "stop_rule": "stop after patience steps without a new best validation KL or a training plateau; no internal wall-clock cap"}

        print("stage=load tokenizer", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
        move_sequences = _move_token_sequences(tokenizer)
        print(f"stage=move_tokens ids={move_sequences} moves={MOVES}", flush=True)
        print("stage=load frozen 8-bit Qwen weights", flush=True)
        quant = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, quantization_config=quant,
            torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # configured dropout is zero; required for checkpointed backward
        torch.cuda.reset_peak_memory_stats()
        record["gpu"] = torch.cuda.get_device_name(0)
        record["model_load_seconds"] = time.time() - started
        print(f"stage=model_ready gpu={record['gpu']} load_seconds={record['model_load_seconds']:.1f}", flush=True)

        train_contexts = _encode_contexts(tokenizer, train_cases, train_runs, train_rounds, initial_suffix)
        val_contexts = _encode_contexts(tokenizer, validation_cases, validation_runs, validation_rounds, initial_suffix)
        targets_train = _target_probs(model, move_sequences, train_contexts)
        targets_val = _target_probs(model, move_sequences, val_contexts)
        initial_ids = tokenizer.encode(initial_suffix, add_special_tokens=False)
        if not initial_ids:
            raise ValueError("initial suffix tokenized to zero tokens")
        record["move_token_sequences"] = {m: ids for m, ids in zip(MOVES, move_sequences)}
        record["suffix_initial_token_ids"] = initial_ids
        record["initial_suffix_token_count"] = len(initial_ids)
        record["train_prompt_tokens"] = [
            {"run": c["run"], "round": c["round"],
             "no_channel_A_B": [x["prompt_tokens"] for x in c["target_encoded"]],
             "optional_A_B": [x["prompt_tokens"] for x in c["optional_encoded"]]}
            for c in train_contexts]

        # Fixed reference: optional_channel with empty suffix.
        def make_empty_contexts(cases, runs, rounds):
            empty = _encode_contexts(tokenizer, cases, runs, rounds, "")
            return empty
        train_empty_contexts = make_empty_contexts(train_cases, train_runs, train_rounds)
        val_empty_contexts = make_empty_contexts(validation_cases, validation_runs, validation_rounds)
        train_empty = _evaluate(model, tokenizer, move_sequences, train_empty_contexts, targets_train, "", individual_weight)
        val_empty = _evaluate(model, tokenizer, move_sequences, val_empty_contexts, targets_val, "", individual_weight)
        start_train = _evaluate(model, tokenizer, move_sequences, train_contexts, targets_train, initial_suffix, individual_weight)
        start_val = _evaluate(model, tokenizer, move_sequences, val_contexts, targets_val, initial_suffix, individual_weight)
        record["stage0_baselines"] = {
            "hypothesis": "H0: the empty optional-channel prompt is the fixed reference for measuring suffix gain.",
            "empty_suffix_train": train_empty,
            "empty_suffix_validation": val_empty,
            "initial_suffix_train": start_train,
            "initial_suffix_validation": start_val,
            "initial_suffix_gain_vs_empty_train": _gain(train_empty["summary"]["mean_context_kl_nats"], start_train["summary"]["mean_context_kl_nats"]),
            "initial_suffix_gain_vs_empty_validation": _gain(val_empty["summary"]["mean_context_kl_nats"], start_val["summary"]["mean_context_kl_nats"]),
        }
        save()

        current_suffix = initial_suffix
        current_ids = initial_ids
        current_train_loss = start_train["summary"]["mean_context_kl_nats"]
        best_suffix = ""  # only replace this if the held-out KL beats its empty baseline
        best_val_loss = val_empty["summary"]["mean_context_kl_nats"]
        best_val_eval = val_empty
        no_val_improve = 0
        no_train_improve = 0

        step = 1
        while True:
            if max_steps and step > max_steps:
                record["stop_reason"] = "max_coordinate_steps"
                break
            embeds, grad = _gradient(
                model, tokenizer, move_sequences, train_contexts, targets_train,
                current_suffix, current_ids, individual_weight,
            )
            coordinate, candidates = _propose_candidates(
                tokenizer, model.get_input_embeddings(), current_ids, grad, candidate_count
            )
            stage = {
                "step": step,
                "hypothesis": "Choose a gradient-sensitive suffix token; rank discrete replacements by mean KL over all training contexts, then inspect held-out behavior.",
                "suffix_before": current_suffix,
                "suffix_token_ids_before": current_ids,
                "training_loss_before_nats": current_train_loss,
                "training_loss_before_bits": current_train_loss / math.log(2),
                "gradient_norm": float(grad.norm().cpu()),
                "gradient_norm_by_position": [float(x) for x in grad.float().norm(dim=1).cpu().tolist()],
                "coordinate": coordinate,
                "candidate_scores": [],
            }
            if not candidates:
                stage["status"] = "no_printable_candidates"
                record["stages"].append(stage)
                record["stop_reason"] = "no_printable_candidates"
                save()
                break

            for candidate in candidates:
                train_eval = _evaluate(model, tokenizer, move_sequences, train_empty_contexts, targets_train,
                                       candidate["suffix"], individual_weight)
                val_eval = _evaluate(model, tokenizer, move_sequences, val_empty_contexts, targets_val,
                                     candidate["suffix"], individual_weight)
                train_loss = train_eval["summary"]["mean_context_kl_nats"]
                val_loss = val_eval["summary"]["mean_context_kl_nats"]
                candidate["train_mean_kl_nats"] = train_loss
                candidate["train_mean_kl_bits"] = train_loss / math.log(2)
                candidate["train_gain_vs_empty"] = _gain(train_empty["summary"]["mean_context_kl_nats"], train_loss)
                candidate["validation_mean_kl_nats"] = val_loss
                candidate["validation_mean_kl_bits"] = val_loss / math.log(2)
                candidate["validation_gain_vs_empty"] = _gain(val_empty["summary"]["mean_context_kl_nats"], val_loss)
                candidate["validation_gain_bootstrap_by_game"] = _bootstrap_run_gain(
                    [{"run": r["run"], "loss_nats": r["loss_nats"]} for r in val_empty["context_rows"]],
                    [{"run": r["run"], "loss_nats": r["loss_nats"]} for r in val_eval["context_rows"]],
                )
                stage["candidate_scores"].append(candidate.copy())
                print(f"stage=candidate step={step} suffix={candidate['suffix']!r} "
                      f"train_kl={train_loss:.6f} val_kl={val_loss:.6f}", flush=True)

            if not stage["candidate_scores"]:
                raise RuntimeError("GCG produced no candidate scores")
            chosen = min(stage["candidate_scores"], key=lambda x: x["train_mean_kl_nats"])
            stage["chosen_by_training_loss"] = chosen
            train_delta = current_train_loss - chosen["train_mean_kl_nats"]
            if train_delta > 1e-8:
                current_suffix = chosen["suffix"]
                current_ids = tokenizer.encode(current_suffix, add_special_tokens=False)
                current_train_loss = chosen["train_mean_kl_nats"]
                no_train_improve = 0
            else:
                no_train_improve += 1
            stage["training_gain_this_step_nats"] = train_delta
            stage["training_gain_this_step_bits"] = train_delta / math.log(2)
            stage["suffix_after"] = current_suffix
            stage["suffix_token_ids_after"] = current_ids

            # Track validation-best suffix, including only a real improvement
            # over empty. A no-suffix result remains available if candidates do not generalize.
            if chosen["validation_mean_kl_nats"] < best_val_loss - 1e-8:
                best_suffix = chosen["suffix"]
                best_val_loss = chosen["validation_mean_kl_nats"]
                best_val_eval = _evaluate(model, tokenizer, move_sequences, val_empty_contexts, targets_val,
                                          best_suffix, individual_weight)
                no_val_improve = 0
                stage["new_validation_best"] = True
            else:
                no_val_improve += 1
                stage["new_validation_best"] = False
            stage["best_validation_suffix_so_far"] = best_suffix
            stage["best_validation_kl_nats_so_far"] = best_val_loss
            stage["best_validation_gain_vs_empty"] = _gain(
                val_empty["summary"]["mean_context_kl_nats"], best_val_loss
            )
            stage["status"] = "completed"
            record["stages"].append(stage)
            record["current_suffix"] = current_suffix
            record["best_validation_suffix"] = best_suffix
            record["best_validation_loss_nats"] = best_val_loss
            save()
            print(f"stage=step_done step={step} train_gain_nats={train_delta:.6f} "
                  f"best_val_suffix={best_suffix!r} best_val_kl={best_val_loss:.6f} "
                  f"patience={no_val_improve}/{patience}", flush=True)
            if no_val_improve >= patience:
                record["stop_reason"] = "validation_patience"
                break
            if no_train_improve >= patience:
                record["stop_reason"] = "training_plateau"
                break
            step += 1

        final_suffix = best_suffix
        final_val = _evaluate(model, tokenizer, move_sequences, val_empty_contexts, targets_val,
                              final_suffix, individual_weight)
        final_train = _evaluate(model, tokenizer, move_sequences, train_empty_contexts, targets_train,
                                final_suffix, individual_weight)
        record["final"] = {
            "selected_suffix": final_suffix,
            "selected_suffix_token_ids": tokenizer.encode(final_suffix, add_special_tokens=False),
            "selection_rule": "lowest validation mean-context KL among evaluated candidates; empty suffix retained unless a candidate improves validation KL",
            "training": final_train,
            "validation": final_val,
            "training_gain_vs_empty": _gain(train_empty["summary"]["mean_context_kl_nats"], final_train["summary"]["mean_context_kl_nats"]),
            "validation_gain_vs_empty": _gain(val_empty["summary"]["mean_context_kl_nats"], final_val["summary"]["mean_context_kl_nats"]),
            "validation_gain_bootstrap_by_complete_run": _bootstrap_run_gain(
                [{"run": r["run"], "loss_nats": r["loss_nats"]} for r in val_empty["context_rows"]],
                [{"run": r["run"], "loss_nats": r["loss_nats"]} for r in final_val["context_rows"]],
            ),
            "pooled_soft_mi_bits_secondary": final_val["pooled_soft_mi_bits_secondary"],
            "pooled_soft_mi_empty_bits_secondary": val_empty["pooled_soft_mi_bits_secondary"],
            "cuda_peak_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
        record["status"] = "completed"
        save()
        return {"status": record["status"], "output_json": str(result_path),
                "output_report": str(report_path), "selected_suffix": final_suffix,
                "validation_gain_bits": record["final"]["validation_gain_vs_empty"]["gain_bits"],
                "stop_reason": record.get("stop_reason"), "elapsed_seconds": record["elapsed_seconds"],
                "cuda_peak_gb": record["final"]["cuda_peak_gb"]}
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        print(f"gcg_error={record['error']}", flush=True)
        raise


@APP.function(
    image=IMAGE,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={"/vol": RESULTS, "/root/.cache/huggingface": HF_CACHE},
)
def run_per_run_chronological_gcg(train_json: str, output_name: str,
                                  initial_suffix: str = "!!!!!", block_size: int = 10,
                                  candidate_count: int = 4,
                                  individual_weight: float = 0.25) -> dict[str, Any]:
    """Run an independent online GCG trajectory per complete game.

    Blocks are processed from round 1 to 60. Each update uses one contiguous
    chronological block, then the selected suffix is measured on the next block
    before that block receives its own update.
    """
    import torch
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.time()
    out_dir = Path("/vol/results/gcg_qwen_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{output_name}.json"
    report_path = out_dir / f"{output_name}.md"
    record: dict[str, Any] = {
        "status": "running", "model": MODEL_ID, "revision": REVISION,
        "method": "independent per-run chronological GCG; contiguous block objective; next-block prequential diagnostic",
        "initial_suffix": initial_suffix, "block_size": block_size,
        "candidate_count_per_block": candidate_count,
        "individual_kl_weight": individual_weight,
        "objective": "mean per-context KL(no-channel A×B || optional-channel A×B) plus weighted individual-policy KLs",
        "hypotheses": [
            "H1: chronological block GCG lowers no-channel-to-optional KL within individual runs.",
            "H2: suffix updates learned from earlier rounds also improve the next, not-yet-trained chronological block.",
            "H3: independently optimized runs converge toward repeated token IDs or text fragments more often than expected by chance.",
            "H4: any repeated suffix pattern is a candidate for a separate fresh-game evaluation, not proof of anti-collusion.",
        ],
        "warnings": [
            "The GCG loss matches optional-channel policy distributions to the no-channel reference; it does not directly minimize observed I(A;B|S).",
            "The pair law is a product of two separately queried move policies, not an empirically sampled joint action law.",
            "Within a run, each suffix update is selected using one contiguous current block; the next block is measured before it is used for its own update.",
            "No-channel and optional-channel states share run/round labels but are separate saved trajectories with different public histories.",
            "This prompt-level procedure is not a full game rollout. Confirm a consensus suffix on fresh complete games and game-level metrics.",
        ],
        "runs": {}, "elapsed_seconds": 0.0,
    }

    def save():
        record["elapsed_seconds"] = time.time() - started
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        report_path.write_text(_make_chronological_report(record))
        RESULTS.commit()

    try:
        cases = json.loads(train_json)
        run_ids = sorted({int(x["run"]) for x in cases})
        rounds = sorted({int(x["round"]) for x in cases})
        if not run_ids:
            raise ValueError("bundle contains no run IDs")
        if rounds != list(range(1, 61)):
            raise ValueError(f"chronological mode requires every round 1-60; found {rounds[:3]}...{rounds[-3:]}")
        if block_size < 1 or candidate_count < 1:
            raise ValueError("block_size and candidate_count must be positive")
        record["split"] = {"run_ids": run_ids, "rounds": rounds,
                           "complete_runs": len(run_ids),
                           "contexts_per_run": len(rounds),
                           "chronological_blocks_per_run": (len(rounds) + block_size - 1) // block_size}
        print("stage=load tokenizer", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
        move_sequences = _move_token_sequences(tokenizer)
        print(f"stage=move_tokens ids={move_sequences} moves={MOVES}", flush=True)
        print("stage=load frozen 8-bit Qwen weights", flush=True)
        quant = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, quantization_config=quant,
            torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        torch.cuda.reset_peak_memory_stats()
        record["gpu"] = torch.cuda.get_device_name(0)
        record["model_load_seconds"] = time.time() - started
        save()

        for run_id in run_ids:
            print(f"stage=run_begin run={run_id} rounds=1-60", flush=True)
            contexts = _encode_contexts(tokenizer, cases, [run_id], rounds, initial_suffix)
            targets = _target_probs(model, move_sequences, contexts)
            empty_contexts = _encode_contexts(tokenizer, cases, [run_id], rounds, "")
            empty_all = _evaluate(model, tokenizer, move_sequences, empty_contexts,
                                  targets, "", individual_weight)
            current_suffix = initial_suffix
            current_ids = tokenizer.encode(initial_suffix, add_special_tokens=False)
            if not current_ids:
                raise ValueError("initial suffix has no tokens")
            initial_all = _evaluate(model, tokenizer, move_sequences, empty_contexts,
                                    targets, current_suffix, individual_weight)
            run_record: dict[str, Any] = {
                "run": run_id,
                "initial_suffix_token_ids": current_ids,
                "empty_suffix_all_rounds": empty_all["summary"],
                "initial_suffix_all_rounds": initial_all["summary"],
                "blocks": [],
            }
            block_starts = list(range(0, len(rounds), block_size))
            for block_index, start_index in enumerate(block_starts, start=1):
                end_index = min(start_index + block_size, len(rounds))
                block_contexts = contexts[start_index:end_index]
                block_targets = targets[start_index:end_index]
                block_count = end_index - start_index
                block_empty_loss = sum(
                    float(row["loss_nats"]) for row in empty_all["context_rows"][start_index:end_index]
                ) / block_count
                before_eval = _evaluate(model, tokenizer, move_sequences, block_contexts,
                                        block_targets, current_suffix, individual_weight)
                before_loss = before_eval["summary"]["mean_context_kl_nats"]
                _, grad = _gradient(model, tokenizer, move_sequences, block_contexts,
                                    block_targets, current_suffix, current_ids, individual_weight)
                coordinate, candidates = _propose_candidates(
                    tokenizer, model.get_input_embeddings(), current_ids, grad, candidate_count
                )
                stage = {
                    "block_index": block_index,
                    "rounds_in_new_block": [rounds[start_index], rounds[end_index - 1]],
                    "objective_rounds": [rounds[start_index], rounds[end_index - 1]],
                    "suffix_before": current_suffix,
                    "suffix_token_ids_before": current_ids,
                    "block_loss_before_nats": before_loss,
                    "block_empty_loss_nats": block_empty_loss,
                    "gradient_norm": float(grad.norm().cpu()),
                    "coordinate": coordinate,
                    "candidates": [],
                }
                for candidate in candidates:
                    candidate_eval = _evaluate(model, tokenizer, move_sequences, block_contexts,
                                               block_targets, candidate["suffix"], individual_weight)
                    candidate_loss = candidate_eval["summary"]["mean_context_kl_nats"]
                    stage["candidates"].append({
                        "suffix": candidate["suffix"],
                        "token_ids": candidate["token_ids"],
                        "changed_token_id": candidate["changed_token_id"],
                        "block_mean_kl_nats": candidate_loss,
                        "block_mean_kl_bits": candidate_loss / math.log(2),
                        "gain_vs_empty_block_bits": (block_empty_loss - candidate_loss) / math.log(2),
                    })
                    print(f"stage=chrono_candidate run={run_id} block={block_index} "
                          f"suffix={candidate['suffix']!r} block_kl={candidate_loss:.6f}", flush=True)

                accepted = None
                if stage["candidates"]:
                    best = min(stage["candidates"], key=lambda x: x["block_mean_kl_nats"])
                    if best["block_mean_kl_nats"] < before_loss - 1e-8:
                        accepted = best
                        current_suffix = best["suffix"]
                        current_ids = tokenizer.encode(current_suffix, add_special_tokens=False)
                stage["accepted_candidate"] = accepted
                stage["suffix_after"] = current_suffix
                stage["suffix_token_ids_after"] = current_ids
                stage["block_score_before_update"] = {
                    "suffix": stage["suffix_before"],
                    "mean_kl_nats": before_loss,
                    "gain_vs_empty_bits": (block_empty_loss - before_loss) / math.log(2),
                }
                stage["block_gain_this_update_nats"] = before_loss - (
                    accepted["block_mean_kl_nats"] if accepted else before_loss
                )
                stage["block_gain_vs_empty_bits_after"] = (
                    block_empty_loss - (accepted["block_mean_kl_nats"] if accepted else before_loss)
                ) / math.log(2)

                # The next block is measured after selecting from this block;
                # it is not used to select the current update.
                if end_index < len(rounds):
                    future_end = min(end_index + block_size, len(rounds))
                    future_contexts = contexts[end_index:future_end]
                    future_targets = targets[end_index:future_end]
                    future_empty = _evaluate(model, tokenizer, move_sequences, future_contexts,
                                             future_targets, "", individual_weight)
                    future_candidate = _evaluate(model, tokenizer, move_sequences, future_contexts,
                                                 future_targets, current_suffix, individual_weight)
                    f0 = future_empty["summary"]["mean_context_kl_nats"]
                    f1 = future_candidate["summary"]["mean_context_kl_nats"]
                    stage["next_block_prequential"] = {
                        "rounds": [rounds[end_index], rounds[future_end - 1]],
                        "empty_suffix_mean_kl_nats": f0,
                        "updated_suffix_mean_kl_nats": f1,
                        "gain_bits_vs_empty": (f0 - f1) / math.log(2),
                        "empty_suffix_per_round_kl_nats": future_empty["summary"]["per_round_mean_kl_nats"],
                        "updated_suffix_per_round_kl_nats": future_candidate["summary"]["per_round_mean_kl_nats"],
                    }
                else:
                    stage["next_block_prequential"] = None
                run_record["blocks"].append(stage)
                record["runs"][str(run_id)] = run_record
                save()
                print(f"stage=chrono_block_done run={run_id} block={block_index}/"
                      f"{len(block_starts)} suffix={current_suffix!r} "
                      f"block_gain_bits={stage['block_gain_this_update_nats']/math.log(2):.4f}", flush=True)

            run_record["final_suffix"] = current_suffix
            run_record["final_suffix_token_ids"] = tokenizer.encode(current_suffix, add_special_tokens=False)
            run_record["final_suffix_after_round_60"] = current_suffix
            record["runs"][str(run_id)] = run_record
            save()
            print(f"stage=run_done run={run_id} suffix_after_round_60={current_suffix!r}", flush=True)

        suffix_counts = Counter(r["final_suffix"] for r in record["runs"].values())
        token_run_counts = Counter()
        for run_record in record["runs"].values():
            token_run_counts.update(set(run_record.get("final_suffix_token_ids", [])))
        block_patterns = {}
        for block_index in range(1, (len(rounds) + block_size - 1) // block_size + 1):
            block_suffixes = Counter()
            block_tokens = Counter()
            for run_record in record["runs"].values():
                stage = next((x for x in run_record["blocks"] if x["block_index"] == block_index), None)
                if stage is not None:
                    block_suffixes[stage["suffix_after"]] += 1
                    block_tokens.update(set(stage["suffix_token_ids_after"]))
            block_patterns[str(block_index)] = {
                "suffix_frequency": dict(block_suffixes.most_common()),
                "token_id_run_frequency": {str(k): v for k, v in block_tokens.most_common()},
            }
        record["cross_run_pattern_summary"] = {
            "exact_suffix_frequency": dict(suffix_counts.most_common()),
            "token_id_run_frequency": {str(k): v for k, v in token_run_counts.most_common()},
            "token_pieces": {str(token_id): tokenizer.convert_ids_to_tokens(int(token_id))
                             for token_id, _ in token_run_counts.most_common()},
            "blockwise_suffix_patterns": block_patterns,
            "run_count": len(record["runs"]),
            "note": "Token/text frequency is descriptive. A repeated token is a candidate pattern, not proof of semantic anti-collusion or statistical significance.",
        }
        record["status"] = "completed"
        record["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        save()
        return {"status": "completed", "output_json": str(result_path),
                "output_report": str(report_path),
                "run_suffixes": {k: v["final_suffix"] for k, v in record["runs"].items()},
                "cross_run_pattern_summary": record["cross_run_pattern_summary"],
                "elapsed_seconds": record["elapsed_seconds"], "cuda_peak_gb": record["cuda_peak_gb"]}
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        print(f"chronological_gcg_error={record['error']}", flush=True)
        raise


def _make_chronological_report(record: dict[str, Any]) -> str:
    lines = [
        "# Per-run chronological Qwen GCG",
        "",
        f"Status: **{record.get('status')}**; start suffix `{record.get('initial_suffix')}`; "
        f"block size {record.get('block_size')} rounds.",
        "",
        "## Method and hypotheses",
        *[f"- {x}" for x in record.get("hypotheses", [])],
        "",
        "The objective averages per-context forward KLs within each contiguous chronological block. The candidate is selected on the current block and scored on the next block before that future block receives its own update. Rounds remain grouped in chronological run trajectories.",
        "",
        "## Per-run suffix path",
        "",
        "| Run | Block | Rounds | Suffix used → suffix after update | Block gain (bits) | Next-block gain (bits) |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for run_id, run in sorted(record.get("runs", {}).items(), key=lambda kv: int(kv[0])):
        for block in run.get("blocks", []):
            nxt = block.get("next_block_prequential")
            next_gain = f"{nxt['gain_bits_vs_empty']:.4f}" if nxt else "—"
            round_range = block["rounds_in_new_block"]
            lines.append(f"| {run_id} | {block['block_index']} | {round_range[0]}–{round_range[1]} | "
                         f"`{block['suffix_before']}` → `{block['suffix_after']}` | "
                         f"{block['block_gain_this_update_nats']/math.log(2):.4f} | {next_gain} |")
    lines += ["", "## Final per-run candidates", "",
              "| Run | Suffix after round 60 | Token IDs |",
              "|---:|---|---|"]
    for run_id, run in sorted(record.get("runs", {}).items(), key=lambda kv: int(kv[0])):
        if "final_suffix" in run:
            lines.append(f"| {run_id} | `{run['final_suffix']}` | `{run['final_suffix_token_ids']}` |")
    patterns = record.get("cross_run_pattern_summary", {})
    if patterns:
        lines += ["", "## Patterns across runs", "",
                  f"- Exact suffix frequency: `{patterns.get('exact_suffix_frequency')}`.",
                  f"- Token ID frequency by run: `{patterns.get('token_id_run_frequency')}`.",
                  f"- Suffix frequency at each chronological block: `{patterns.get('blockwise_suffix_patterns')}`.",
                  f"- Token pieces: `{patterns.get('token_pieces')}`.",
                  f"- {patterns.get('note')}"]
    lines += ["", "## Limits", "", *[f"- {w}" for w in record.get("warnings", [])]]
    if record.get("error"):
        lines += ["", "## Error", "", f"`{record['error']}`"]
    return "\n".join(lines) + "\n"


def _make_report(record: dict[str, Any]) -> str:
    lines = [
        "# Multi-context Qwen GCG pilot",
        "",
        f"Status: **{record.get('status')}**  ",
        f"Initial suffix: `{record.get('initial_suffix')}`  ",
        f"Stop reason: `{record.get('stop_reason', 'running')}`",
        "",
        "## Hypotheses",
    ]
    lines += [f"- {x}" for x in record.get("hypotheses", [])]
    lines += ["", "## Split and method", "",
              f"- Training: `{record.get('split', {}).get('training_run_ids', [])}`; rounds 8-10; "
              f"{record.get('split', {}).get('training_contexts', 0)} paired contexts.",
              f"- Validation: `{record.get('split', {}).get('validation_run_ids', [])}`; rounds 8-10; "
              f"{record.get('split', {}).get('validation_contexts', 0)} paired contexts.",
              "- Loss: mean of context-level forward KLs from the no-channel joint product law to the optional-channel candidate law, plus the configured individual-policy KL weight.",
              "- KL gain in bits is `(empty-suffix KL - candidate KL) / ln(2)`. Positive values mean the candidate is closer to the no-channel reference.",
              "- The secondary pooled soft MI can include dependence induced by mixing contexts; it is not state-conditioned CMI.",
              "", "## Baselines", ""]
    baselines = record.get("stage0_baselines", {})
    for key in ("empty_suffix_train", "empty_suffix_validation", "initial_suffix_train", "initial_suffix_validation"):
        if key in baselines:
            lines.append(f"- {key}: {baselines[key]['summary']['mean_context_kl_nats']:.6f} nats "
                         f"({baselines[key]['summary']['mean_context_kl_bits']:.6f} bits).")
    if baselines:
        empty_individual = baselines.get("empty_suffix_validation", {}).get("individual_kl_summary", {})
        initial_individual = baselines.get("initial_suffix_validation", {}).get("individual_kl_summary", {})
        for seat in ("A", "B"):
            if seat in empty_individual and seat in initial_individual:
                lines.append(f"- Validation individual KL {seat}: empty "
                             f"{empty_individual[seat]['mean_context_kl_nats']:.6f} nats; initial suffix "
                             f"{initial_individual[seat]['mean_context_kl_nats']:.6f} nats.")
    lines += ["", "## Coordinate search", ""]
    for stage in record.get("stages", []):
        lines += [f"### Step {stage['step']}", "",
                  f"- Hypothesis/action: {stage['hypothesis']}",
                  f"- Start suffix: `{stage['suffix_before']}`; token IDs `{stage['suffix_token_ids_before']}`.",
                  f"- Gradient norm: {stage.get('gradient_norm', float('nan')):.6g}; coordinate: {stage.get('coordinate')}."]
        if stage.get("candidate_scores"):
            lines += ["- Candidate scores (train and held-out mean KL):"]
            for c in stage["candidate_scores"]:
                lines.append(f"  - `{c['suffix']}` IDs `{c['token_ids']}`: train "
                             f"{c['train_mean_kl_nats']:.6f} nats, validation "
                             f"{c['validation_mean_kl_nats']:.6f} nats; validation gain "
                             f"{c['validation_gain_vs_empty']['gain_bits']:.6f} bits.")
            chosen = stage.get("chosen_by_training_loss", {})
            lines.append(f"- Train-selected: `{chosen.get('suffix')}`; accepted suffix now `{stage.get('suffix_after')}`.")
            lines.append(f"- Best validation suffix so far: `{stage.get('best_validation_suffix_so_far')}`; "
                         f"gain `{stage.get('best_validation_gain_vs_empty', {}).get('gain_bits', 0):.6f}` bits.")
        lines.append("")
    final = record.get("final")
    if final:
        gain = final["validation_gain_vs_empty"]
        lines += ["## Selected result", "",
                  f"- Suffix: `{final['selected_suffix']}`; token IDs `{final['selected_suffix_token_ids']}`.",
                  f"- Training gain: {final['training_gain_vs_empty']['gain_bits']:.6f} bits "
                  f"({final['training_gain_vs_empty']['relative_gain_percent'] or 0:.3f}%).",
                  f"- Validation gain: {gain['gain_bits']:.6f} bits "
                  f"({gain['relative_gain_percent'] or 0:.3f}%).",
                  f"- Validation individual KL A/B: "
                  f"{final['validation']['individual_kl_summary']['A']['mean_context_kl_nats']:.6f} / "
                  f"{final['validation']['individual_kl_summary']['B']['mean_context_kl_nats']:.6f} nats.",
                  f"- Validation run-cluster bootstrap: {final['validation_gain_bootstrap_by_complete_run']}.",
                  f"- Validation pooled soft MI, secondary only: empty "
                  f"{final['pooled_soft_mi_empty_bits_secondary']:.6f} bits; selected "
                  f"{final['pooled_soft_mi_bits_secondary']:.6f} bits.",
                  f"- Peak allocated GPU memory: {final['cuda_peak_gb']:.2f} GiB.",
                  "", "## Interpretation limits", "",
                  *[f"- {x}" for x in record.get("warnings", [])],
                  "- This is a prompt-level distribution-matching pilot, not a full-game performance or coalition result.",
                  "- If validation gain is non-positive, the selected suffix is empty and the search found no suffix that generalized in this split."]
    if record.get("error"):
        lines += ["", "## Error", "", f"`{record['error']}`"]
    return "\n".join(lines) + "\n"


@APP.local_entrypoint()
def main(train_bundle: str, validation_bundle: str = "",
         output_name: str = "qwen32b_gcg_multicontext",
         mode: str = "multicontext",
         initial_suffix: str = "!!!!!", max_steps: int = 0,
         candidate_count: int = 4, patience: int = 2,
         individual_weight: float = 0.25, block_size: int = 10):
    train_text = Path(train_bundle).read_text()
    if mode == "chronological":
        result = run_per_run_chronological_gcg.remote(
            train_text, output_name, initial_suffix, block_size,
            candidate_count, individual_weight,
        )
    elif mode == "multicontext":
        if not validation_bundle:
            raise ValueError("--validation-bundle is required for multicontext mode")
        validation_text = Path(validation_bundle).read_text()
        result = run_multicontext_gcg.remote(
            train_text, validation_text, output_name, initial_suffix, max_steps,
            candidate_count, patience, individual_weight,
        )
    else:
        raise ValueError("--mode must be multicontext or chronological")
    print(json.dumps(result, indent=2, ensure_ascii=False))
