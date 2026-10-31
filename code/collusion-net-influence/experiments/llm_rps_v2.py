"""v2 protocol: 3 LLM players, pot-3 RPS, five channel conditions, reflection before each move, post-game debrief.

Spec: PROMPT_agent_5_experiences.md. Payoff: each round a pot of 3 goes to the winning move (3 alone,
1.5 each when shared, 0 when all moves are equal or all different). The environment computes every
outcome and shows it as a scoreboard on top of every request. Every text comes from
prompts/pot_plan_v2.json and goes through `Protocol`, which `--render` also uses, so the rendered
prompts in EXPERIMENT_PLAN.md are exactly what the agents receive.

Round t (1-indexed):
  1. chat (conditions with a channel): A writes first in odd rounds, B in even rounds; each turn both may
     write (optional conditions: or reply SKIP; if both skip, the chat ends for the round)
  2. plan (forced conditions, or --plan): A and B each answer 'PLAN: A=<move>, B=<move>'
  3. reflection (--reflection, default on): every LLM player answers LAST_ROUND / STANDING / WORKING /
     OTHERS / INTENDED_MOVE
  4. move: every LLM player may think briefly, then ends with 'MOVE: <move>'
After the last round each LLM player answers a debrief (separate call, cannot affect the game).

Static commands (no LLM call):
    python experiments/llm_rps_v2.py --check
    python experiments/llm_rps_v2.py --render > rendered_prompts.md
    python experiments/llm_rps_v2.py --estimate --runs 3 --rounds 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from netinf.games.rps_pot import payoff_pot
from netinf.llm import CachedChat
from netinf.metrics import (alternation_score, assist_rate_sym, channel_usage, conditional_entropy_bits, cycle_adherence,
                            die_adherence, die_metrics, entropy_bits, net_transfer, pair_advantage, plan_adherence, same_move_rate, shuffle_pvalue)
from netinf.protocol_v2 import (MOVE_NAMES, NAMES, chat_block, drop_empty_lines, first_speaker, history_block,
                                is_skip, move_distribution, outcome_line, parse_plan, parse_tagged_move,
                                unknown_fields)
from netinf.seeding import stable_seed
from netinf.understanding import debrief_accuracy, read_debrief, read_reflection, reflection_metrics
from llm_rps import parse_move, score

CONDITIONS = {
    "no_channel": {"channel": None, "hint": False, "strategy": False},
    "channel_optional": {"channel": "optional", "hint": False, "strategy": False},
    "channel_optional_hint": {"channel": "optional", "hint": True, "strategy": False},
    "channel_forced": {"channel": "forced", "hint": False, "strategy": False},
    "channel_forced_strategy": {"channel": "forced", "hint": False, "strategy": True},
}
# chat templates that reject a system message: the system prompt is sent at the top of the user message instead
NO_SYSTEM_ROLE = {"mistralai/Mixtral-8x7B-Instruct-v0.1", "RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8"}
SEATINGS = list(itertools.permutations(range(3)))  # the 6 ways to seat a 3-model line-up in A, B, C
DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "pot_plan_v2.json"
STATE_VARS = {"round", "score_A", "score_B", "score_C", "history", "scoreboard", "chat", "chat_section", "your_plan_line",
              "your_reflection_line", "speaking_order", "skip_clause", "debrief_history", "debrief_chat", "move_suffix"}
BASE_VARS = {"me", "others", "partner", "outsider", "msg_tokens", "role_line", "shared_die"}


def show(m: int | None) -> str:
    return "?" if m is None else MOVE_NAMES[m]


class Protocol:
    """Renders every text an agent receives, for one condition. play() and --render both use it."""

    def __init__(self, prompts: dict, condition: str, cfg):
        self.p, self.condition, self.c, self.cfg = prompts, condition, CONDITIONS[condition], cfg

    @property
    def plans_on(self) -> bool:
        return self.c["channel"] == "forced" or (self.c["channel"] == "optional" and self.cfg.plan)

    def base_vars(self, k: int) -> dict:
        return {"me": NAMES[k], "others": " and ".join(n for n in NAMES if n != NAMES[k]),
                "partner": NAMES[1 - k] if k < 2 else "", "outsider": "C" if k < 2 else "",
                "msg_tokens": self.cfg.msg_tokens}

    def system(self, k: int, st: dict | None = None) -> str:
        """System prompt of seat k; {shared_die} is this round's private number of A and B (v4 hint)."""
        v = self.base_vars(k)
        v["shared_die"] = (st or {}).get("shared_die", "?")
        if k == 2 or self.c["channel"] is None:
            role = self.p["role_all"]
        else:
            role = self.p["role_partner_hint" if self.c["hint"] else "role_partner"]
        parts = [self.p["rules_base"].format(role_line=role.format(**v), **v)]
        if k < 2 and self.c["channel"]:
            parts.append(self.p[f"channel_{self.c['channel']}"].format(**v))
            if self.c["strategy"]:
                parts.append(self.p["strategy_hint"].format(**v))
        return "\n\n".join(parts)

    def state_vars(self, k: int, st: dict) -> dict:
        v = self.base_vars(k)
        t = st["totals"]
        v.update(round=st["round"], score_A=f"{t[0]:g}", score_B=f"{t[1]:g}", score_C=f"{t[2]:g}",
                 history=history_block(st["history"], self.cfg.history))
        v["scoreboard"] = self.p["scoreboard"].format(**v)
        talks = k < 2 and self.c["channel"] is not None
        v["chat"] = chat_block(st["chat"]) if talks else ""
        # this round's chat with its header for A and B, empty for C and without a channel (pot_plan_v2_chat.json)
        v["chat_section"] = f"Private chat with Player {v['partner']} this round:\n{v['chat']}" if talks else ""
        own = st["plans"].get(NAMES[k]) if k < 2 else None
        v["your_plan_line"] = f"Your plan: A={show(own['A'])}, B={show(own['B'])}" if own else ""
        refl = st["reflections"].get(NAMES[k])
        v["your_reflection_line"] = f"Your reflection: {refl}" if refl else ""
        v["speaking_order"] = st.get("speaking_order", "")
        v["skip_clause"] = self.p["skip_clause"] if self.c["channel"] == "optional" else ""
        # Research treatment suffixes are appended to the move request for partners A/B only.
        # The counter-player C and all no-channel conditions receive an empty value.
        v["move_suffix"] = self.p.get("_move_suffix_ab", "") if k < 2 and self.c["channel"] else ""
        return v

    def user(self, key: str, k: int, st: dict) -> str:
        return drop_empty_lines(self.p[key].format(**self.state_vars(k, st)))

    def debrief(self, k: int, st: dict) -> str:
        v = self.state_vars(k, st)
        v["debrief_history"] = "\n".join(h["outcome"] for h in st["history"]) or "(no round played)"
        chat = "\n".join(f"[round {m['round']}] {m['from']}: {m['text']}" for m in st["all_chat"]) or "(no messages)"
        if len(chat) > self.cfg.debrief_chat_chars:
            chat = "(earlier messages omitted)\n" + chat[-self.cfg.debrief_chat_chars:]
        v["debrief_chat"] = chat
        key = "debrief_c" if k == 2 else "debrief_ab" if self.c["channel"] else "debrief_ab_no_channel"
        return drop_empty_lines(self.p[key].format(**v))


def new_state() -> dict:
    return {"round": 1, "totals": np.zeros(3), "history": [], "chat": [], "all_chat": [], "plans": {},
            "reflections": {}, "shared_die": 1}


def shared_die(run: int, seed_offset: int, round_no: int) -> int:
    """The round's private number (1-6) of A and B: drawn by the environment, reproducible, never shown to C."""
    return 1 + stable_seed(run, seed_offset, round_no, "shared_die") % 6


def call(llm, prompt: str, system: str, cfg, seed: int, max_tokens: int, top_logprobs: int = 0) -> dict:
    """One LLM request; the full record when the client supports it (CachedChat), plain text otherwise."""
    if hasattr(llm, "generate_full"):
        return llm.generate_full(prompt, system, cfg.temperature, seed=seed, max_tokens=max_tokens,
                                 top_logprobs=top_logprobs)
    text = llm.generate(prompt, system, cfg.temperature, seed=seed, max_tokens=max_tokens)
    return {"text": text, "finish_reason": None, "tokens": [], "prompt_tokens": 0, "completion_tokens": 0}


def play(llms: list, proto: Protocol, run: int, cfg, rng) -> dict:
    """One game. llms[k] is the client of seat k, or None for a scripted uniform-random C."""
    st, rounds, failures, max_prompt, tokens = new_state(), [], Counter(), Counter(), Counter()
    systems = first_systems = [proto.system(k) for k in range(3)]
    seats = [k for k in range(3) if llms[k] is not None]

    def ask(k: int, key: str, kind: str, max_tokens: int, *extra, top_logprobs: int = 0) -> dict:
        out = call(llms[k], proto.user(key, k, st), systems[k], cfg,
                   stable_seed(run, cfg.seed_offset, st["round"], k, kind, *extra), max_tokens, top_logprobs)
        failures[f"truncated_{kind}"] += out["finish_reason"] == "length"
        max_prompt[kind] = max(max_prompt[kind], out["prompt_tokens"])
        tokens.update(prompt=out["prompt_tokens"], completion=out["completion_tokens"])
        return out

    for t in range(cfg.rounds):
        r = t + 1
        st.update(round=r, chat=[], plans={}, reflections={}, shared_die=shared_die(run, cfg.seed_offset, r))
        systems = [proto.system(k, st) for k in range(3)]  # changes each round only if a prompt uses {shared_die}
        if t == 0:
            first_systems = systems
        rec = {"round": r, "chat": [], "channel": None, "plans": None, "reflections": None,
               "shared_die": st["shared_die"]}
        if proto.c["channel"]:
            first = first_speaker(r)
            usage = {n: {"asked": 0, "sent": 0, "skipped": 0} for n in ("A", "B")}
            for turn in range(cfg.negotiation_turns):
                skipped = 0
                for k in (first, 1 - first):
                    st["speaking_order"] = "first" if k == first else "second"
                    text = ask(k, "negotiate_ask", "msg", cfg.msg_tokens, turn)["text"]
                    usage[NAMES[k]]["asked"] += 1
                    if is_skip(text) or not text.strip():
                        usage[NAMES[k]]["skipped"] += 1
                        skipped += 1
                        failures["forced_skip"] += proto.c["channel"] == "forced"
                        continue
                    # some models open with a speaker label ("A: Sounds good"), which reads as if A wrote it
                    clean = re.sub(r"^\W*(player\s+)?[ABC]\s*:\s*", "", " ".join(text.split()), flags=re.I)
                    msg = {"round": r, "turn": turn, "from": NAMES[k], "text": clean or " ".join(text.split())}
                    st["chat"].append(msg)
                    st["all_chat"].append(msg)
                    usage[NAMES[k]]["sent"] += 1
                if skipped == 2:  # both partners skipped this turn: the chat is over for this round
                    break
            st.pop("speaking_order", None)
            rec.update(chat=list(st["chat"]), channel=usage, first_speaker=NAMES[first])
            if proto.plans_on:
                for k in (0, 1):
                    text = ask(k, "plan_ask", "plan", cfg.plan_tokens)["text"]
                    parsed = parse_plan(text)
                    failures["plan"] += None in parsed.values()
                    st["plans"][NAMES[k]] = {**parsed, "raw": text.strip()}
                rec["plans"] = dict(st["plans"])
        if cfg.reflection:
            refl = {}
            for k in seats:
                out = ask(k, "reflection_ask", "reflection", cfg.reflection_tokens, top_logprobs=cfg.top_logprobs)
                refl[NAMES[k]] = {"raw": out["text"].strip(), "parsed": read_reflection(out["text"], k),
                                  "intended_dist": move_distribution(out["tokens"], "INTENDED_MOVE")}
            st["reflections"] = {n: x["raw"] for n, x in refl.items()}
            rec["reflections"] = refl
        moves, raw, dists = [], [], {}
        for k in range(3):
            if llms[k] is None:
                moves.append(int(rng.integers(3)))
                raw.append(None)
                continue
            out = ask(k, "move_ask", "move", cfg.move_tokens, top_logprobs=cfg.top_logprobs)
            text = out["text"]
            dists[NAMES[k]] = move_distribution(out["tokens"], "MOVE")
            move = parse_tagged_move(text)
            if move is None:  # some models (Mixtral) end the move answer with the reflection's INTENDED_MOVE: tag
                move = parse_tagged_move(text, "INTENDED_MOVE")
                if move is not None:
                    dists[NAMES[k]] = move_distribution(out["tokens"], "INTENDED_MOVE")
                    failures["move_intended_tag"] += 1
            if move is None:  # no tag at all; accept a bare move on the last line, and count it
                lines = [x for x in text.strip().split("\n") if x.strip()]
                move = parse_move(lines[-1]) if lines else None
                failures["move_untagged"] += move is not None
            if move is None:
                failures["move"] += 1
                move = int(rng.integers(3))  # random fallback; the rate is reported per game
            moves.append(move)
            raw.append(text.strip())
        rewards = payoff_pot(moves).tolist()
        st["totals"] = st["totals"] + rewards
        outcome = outcome_line(r, moves, rewards)
        st["history"].append({"round": r, "moves": moves, "rewards": rewards, "outcome": outcome})
        rec.update(moves=moves, raw_moves=raw, move_dist=dists, rewards=rewards, outcome=outcome)
        rounds.append(rec)

    debriefs = {}
    if cfg.debrief:  # separate calls after the game: cannot change any move
        st.update(round=cfg.rounds + 1, chat=[], plans={}, reflections={})
        for k in seats:
            prompt = proto.debrief(k, st)
            try:
                out = call(llms[k], prompt, systems[k], cfg, stable_seed(run, cfg.seed_offset, "debrief", k),
                           cfg.debrief_tokens)
            except Exception as err:  # e.g. a prompt over max_model_len: keep the game, record the failed debrief
                failures["debrief_error"] += 1
                debriefs[NAMES[k]] = {"raw": "", "parsed": read_debrief("", k), "error": repr(err)[:300],
                                      "prompt_chars": len(prompt) + len(systems[k]), "prompt_tokens": 0}
                continue
            failures["truncated_debrief"] += out["finish_reason"] == "length"
            max_prompt["debrief"] = max(max_prompt["debrief"], out["prompt_tokens"])
            tokens.update(prompt=out["prompt_tokens"], completion=out["completion_tokens"])
            debriefs[NAMES[k]] = {"raw": out["text"].strip(), "parsed": read_debrief(out["text"], k),
                                  # sizes let the GPU benchmark project the longest prompt of a longer game
                                  "prompt_chars": len(prompt) + len(systems[k]), "prompt_tokens": out["prompt_tokens"]}
    return {"rounds": rounds, "actions": [x["moves"] for x in rounds], "rewards": [x["rewards"] for x in rounds],
            "final_scores": st["totals"].tolist(), "debriefs": debriefs,
            "failures": {k: v for k, v in failures.items() if v}, "max_prompt_tokens": dict(max_prompt),
            "token_usage": dict(tokens),
            "temperature": cfg.temperature, "system_prompts": first_systems}


def game_metrics(game: dict, n_perm: int, rng) -> dict:
    acts = np.array(game["actions"])
    adv = lambda a: pair_advantage(np.array([payoff_pot(row) for row in a]))  # noqa: E731
    plans = [x["plans"] for x in game["rounds"]]
    m = {"pair_advantage": pair_advantage(np.array(game["rewards"])),
         "pair_advantage_p": shuffle_pvalue(adv, acts, n_perm, rng),
         "net_transfer": net_transfer(acts, payoff_pot, n_perm, rng),
         "assist_rate_sym": assist_rate_sym(acts), "alternation": alternation_score(acts),
         "same_move_rate": same_move_rate(acts), "cycle_adherence": cycle_adherence(acts),
         "die_adherence": die_adherence(acts, [x["shared_die"] for x in game["rounds"]]),
         # meaningful in channel_forced_strategy with v4; elsewhere a chance-level placebo of the same numbers
         "die": die_metrics(acts, [x["shared_die"] for x in game["rounds"]], np.array(game["rewards"])),
         "entropy": [entropy_bits(acts[:, k]) for k in range(3)],
         "cond_entropy": [conditional_entropy_bits(acts[1:, k], acts[:-1]) for k in range(3)],
         "channel_usage": channel_usage(game["rounds"]), **plan_adherence(plans, acts)}
    return m


def parse_spec(spec: str) -> tuple[str, str, str]:
    """'model@base_url' or 'model@base_url#revision' -> (model, base_url, revision)."""
    model, sep, rest = spec.strip().partition("@")
    if not sep:
        raise SystemExit(f"expected model@base_url[#revision], got {spec!r}")
    url, _, revision = rest.partition("#")
    return model, url, revision


def seat_models(cfg, run: int) -> list[tuple[str, str, str] | None]:
    """(model, base_url, revision) per seat; None for a scripted C. Mixed designs rotate the line-up over the
    6 seatings (game r uses seating r mod 6), so 6 games per condition put every model in every seat twice."""
    if cfg.agents:
        specs = [parse_spec(a) for a in cfg.agents.split(",")]
    elif cfg.design == "mixed_models":
        lineup = [parse_spec(a) for a in cfg.lineup.split(",")]
        if len(lineup) != 3:
            raise SystemExit("--lineup needs exactly three model@base_url entries")
        specs = [lineup[i] for i in SEATINGS[run % len(SEATINGS)]]
    else:
        specs = [(cfg.model, cfg.base_url, cfg.revision)] * 3
    if len(specs) != 3:
        raise SystemExit("--agents needs exactly three model@base_url entries")
    return [specs[0], specs[1], None if cfg.c_player == "uniform" else specs[2]]


# ---------- static tools: check, render, estimate ----------

def sample_state(round_no: int, with_chat: bool) -> dict:
    """A synthetic mid-game state for rendering examples (real renderer, made-up history)."""
    st = new_state()
    script = [(0, 0, 1), (0, 2, 2), (1, 1, 0), (2, 1, 2), (0, 2, 1), (1, 0, 0), (2, 2, 1), (0, 1, 2)]
    for r in range(1, round_no):
        moves = list(script[(r - 1) % len(script)])
        rewards = payoff_pot(moves).tolist()
        st["totals"] = st["totals"] + rewards
        st["history"].append({"round": r, "moves": moves, "rewards": rewards, "outcome": outcome_line(r, moves, rewards)})
        if with_chat:
            st["all_chat"] += [{"round": r, "from": "A", "text": "Let's both play Rock again."},
                               {"round": r, "from": "B", "text": "OK, Rock."}]
    st["round"] = round_no
    return st


def check_prompts(prompts: dict, cfg) -> None:
    allowed = BASE_VARS | STATE_VARS
    bad = unknown_fields(prompts, allowed)
    if bad:
        raise SystemExit(f"unknown placeholders in {cfg.prompts}: {bad}. Available: {sorted(allowed)}")
    for cond in CONDITIONS:  # render every request once, so a stray brace fails now and not mid-campaign
        proto = Protocol(prompts, cond, cfg)
        st = sample_state(6, True)
        st["chat"] = st["all_chat"][-2:]
        st["plans"] = {"A": {"A": 0, "B": 2}, "B": {"A": 0, "B": 2}}
        st["reflections"] = {"A": "INTENDED_MOVE: Rock", "B": "INTENDED_MOVE: Scissors", "C": "INTENDED_MOVE: Paper"}
        st["speaking_order"] = "first"
        for k in range(3):
            proto.system(k)
            proto.debrief(k, st)
            for key in ("negotiate_ask", "plan_ask", "reflection_ask", "move_ask"):
                proto.user(key, k, st)


def render_all(prompts: dict, cfg, conditions: list[str]) -> str:
    """Markdown with every distinct text, and which (condition, player, request) receives it."""
    texts: dict[str, list[str]] = {}

    def add(label, text):
        texts.setdefault(text, []).append(label)

    for cond in conditions:
        proto = Protocol(prompts, cond, cfg)
        llm_seats = [0, 1] + ([2] if cfg.c_player == "llm" else [])
        for k in llm_seats:
            add(f"{cond} · {NAMES[k]} · system (round 6 example)", proto.system(k, sample_state(6, False)))
        for round_no in (1, 6):
            st = sample_state(round_no, proto.c["channel"] is not None)
            for k in llm_seats:
                tag = f"{cond} · {NAMES[k]} · round {round_no}"
                if k < 2 and proto.c["channel"]:
                    st["speaking_order"] = "first" if k == first_speaker(round_no) else "second"
                    st["chat"] = [] if k == first_speaker(round_no) else [
                        {"round": round_no, "from": NAMES[1 - k], "text": "Let's both play Rock this round."}]
                    add(f"{tag} · negotiate (turn 1)", proto.user("negotiate_ask", k, st))
                    st["chat"] = [{"round": round_no, "from": "A", "text": "Let's both play Rock this round."},
                                  {"round": round_no, "from": "B", "text": "OK, Rock."}]
                    st.pop("speaking_order")
                    if proto.plans_on:
                        add(f"{tag} · plan", proto.user("plan_ask", k, st))
                        st["plans"][NAMES[k]] = {"A": 0, "B": 0}
                if cfg.reflection:
                    add(f"{tag} · reflection", proto.user("reflection_ask", k, st))
                    st["reflections"][NAMES[k]] = ("LAST_ROUND: ... STANDING: behind. WORKING: no. "
                                                   "OTHERS: ... INTENDED_MOVE: Rock")
                add(f"{tag} · move", proto.user("move_ask", k, st))
        st = sample_state(31, proto.c["channel"] is not None)
        st.update(round=31, chat=[], plans={}, reflections={})
        if cfg.debrief:
            for k in llm_seats:
                text = proto.debrief(k, st)
                lines = text.split("\n")
                if len(lines) > 30:  # 30 outcome lines and 60 messages: keep the example readable
                    text = "\n".join(lines[:6] + ["[... full history and chat of the game ...]"] + lines[-12:])
                add(f"{cond} · {NAMES[k]} · debrief (shortened)", text)
    out = []
    for i, (text, labels) in enumerate(texts.items(), 1):
        out.append(f"#### Text {i}\nUsed by: " + "; ".join(labels) + f"\n\n```text\n{text}\n```\n")
    return "\n".join(out)


def estimate(prompts: dict, cfg, conditions: list[str]) -> str:
    """LLM calls and tokens per game and in total, from rendered prompts (chars / 4 per token)."""
    fill = {"msg": 0.6, "plan": 1.0, "reflection": 0.8, "move": 0.5, "debrief": 0.8}
    rows, total_in, total_out, total_calls = [], 0.0, 0.0, 0
    n_llm = 3 if cfg.c_player == "llm" else 2
    for cond in conditions:
        proto = Protocol(prompts, cond, cfg)
        st = sample_state(min(6, cfg.rounds), proto.c["channel"] is not None)
        st["chat"] = st["all_chat"][-2:]
        sys_tok = np.mean([len(proto.system(k)) for k in range(3)]) / 4
        tok = lambda key: sys_tok + len(proto.user(key, 0, st)) / 4  # noqa: E731
        calls = {"msg": cfg.rounds * 2 * cfg.negotiation_turns if proto.c["channel"] else 0,
                 "plan": cfg.rounds * 2 if proto.plans_on else 0,
                 "reflection": cfg.rounds * n_llm if cfg.reflection else 0,
                 "move": cfg.rounds * n_llm, "debrief": n_llm if cfg.debrief else 0}
        max_out = {"msg": cfg.msg_tokens, "plan": cfg.plan_tokens, "reflection": cfg.reflection_tokens,
                   "move": cfg.move_tokens, "debrief": cfg.debrief_tokens}
        debrief_in = sys_tok + (cfg.rounds * 110 + calls["msg"] * 60 * 4 / 2) / 4
        tin = {"msg": tok("negotiate_ask"), "plan": tok("plan_ask"), "reflection": tok("reflection_ask"),
               "move": tok("move_ask") + cfg.reflection_tokens * fill["reflection"] * cfg.reflection,
               "debrief": debrief_in}
        g_in = sum(calls[x] * tin[x] for x in calls)
        g_out = sum(calls[x] * max_out[x] * fill[x] for x in calls)
        n = sum(calls.values())
        rows.append(f"| {cond} | " + " | ".join(str(calls[x]) for x in calls) + f" | **{n}** | {g_in / 1e3:.0f}k | {g_out / 1e3:.0f}k |")
        total_in += g_in * cfg.runs
        total_out += g_out * cfg.runs
        total_calls += n * cfg.runs
    head = ("| condition | messages | plans | reflections | moves | debriefs | calls / game | input tokens / game "
            "| output tokens / game |\n|---|---|---|---|---|---|---|---|---|")
    api = total_in / 1e6 * cfg.price_in + total_out / 1e6 * cfg.price_out
    n_gpu = cfg.gpu_count or (3 if cfg.design == "mixed_models" else 1)
    busy = total_out / cfg.tokens_per_sec / 3600  # GPU-hours spent decoding, summed over the models
    n_games = cfg.runs * len(conditions)
    # rounds within a game are sequential: a game lasts at least its own tokens at single-stream speed
    latency = math.ceil(n_games / cfg.workers) * (total_out / n_games) / cfg.stream_tokens_per_sec / 3600
    wall = max(busy / n_gpu, latency)  # throughput-bound or latency-bound, whichever is longer
    billed = n_gpu * (wall + (cfg.cold_start_min + 5) / 60)  # warm for the whole run, + cold start + 5 min scale-down
    return (head + "\n" + "\n".join(rows) + f"\n\nTotal for {cfg.runs} games per condition: **{total_calls} calls**, "
            f"{total_in / 1e6:.1f}M input and {total_out / 1e6:.1f}M output tokens.\n"
            f"- GPU: {n_gpu} model container(s), {cfg.tokens_per_sec:g} output tokens/s each, ${cfg.gpu_price:g}/h: "
            f"**about {wall:.1f} h wall time, {billed:.1f} GPU-hours billed, ${billed * cfg.gpu_price:.2f}** "
            f"(cold start {cfg.cold_start_min:g} min per container; {cfg.workers} games at once, "
            f"{cfg.stream_tokens_per_sec:g} tokens/s per game)\n"
            f"- API alternative at ${cfg.price_in:g} / ${cfg.price_out:g} per M tokens: about ${api:.2f}\n"
            "Upper bound: assumes no SKIP. chars/4 per token; outputs at a fixed share of each max_tokens; "
            "tokens/s is a placeholder until the GPU benchmark measures it.")


# ---------- main ----------

def load_prompts(path: Path, overrides: list[str]) -> dict:
    prompts = json.loads(Path(path).read_text())
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in prompts:
            raise SystemExit(f"--prompt expects KEY=TEXT or KEY=@file with KEY in {sorted(prompts)}; got {item!r}")
        prompts[key] = Path(value[1:]).read_text().strip() if value.startswith("@") else value
    return prompts


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("design")
    g.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated, from: " + ", ".join(CONDITIONS))
    g.add_argument("--runs", type=int, default=3, help="games per condition")
    g.add_argument("--rounds", type=int, default=30)
    g.add_argument("--negotiation-turns", type=int, default=2, help="chat turns per round (each partner writes once per turn)")
    g.add_argument("--msg-tokens", type=int, default=200)
    g.add_argument("--plan-tokens", type=int, default=30)
    g.add_argument("--plan", action=argparse.BooleanOptionalAction, default=False,
                   help="also ask for a plan in the optional-channel conditions (always asked in forced ones)")
    g.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--debrief", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--history", type=int, default=5, help="rounds listed in the scoreboard")
    g.add_argument("--c-player", choices=("llm", "uniform"), default="llm", help="uniform = scripted random C")
    m = ap.add_argument_group("models")
    m.add_argument("--design", choices=("same_model", "mixed_models"), default="same_model")
    m.add_argument("--model", default=None, help="same_model: model name (default NETINF_MODEL)")
    m.add_argument("--base-url", default=None, help="same_model: server URL (default NETINF_BASE_URL)")
    m.add_argument("--revision", default="", help="same_model: HF revision hash of the served weights (logged)")
    m.add_argument("--lineup", default="", help="mixed_models: 'm1@url[#rev],m2@url[#rev],m3@url[#rev]', rotated over the 6 seatings")
    m.add_argument("--agents", default="", help="fixed seats 'mA@url[#rev],mB@url[#rev],mC@url[#rev]' (overrides --design)")
    m.add_argument("--temperature", type=float, default=1.0)
    m.add_argument("--move-tokens", type=int, default=300)
    m.add_argument("--reflection-tokens", type=int, default=400)
    m.add_argument("--debrief-tokens", type=int, default=600)
    m.add_argument("--top-logprobs", type=int, default=5, help="alternatives logged at MOVE: and INTENDED_MOVE: (0 = off)")
    m.add_argument("--debrief-chat-chars", type=int, default=24000, help="keep the most recent chat in the debrief")
    m.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    m.add_argument("--prompt", action="append", default=[], metavar="KEY=TEXT", help="override one prompt (repeatable)")
    r = ap.add_argument_group("run")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--perms", type=int, default=300, help="permutations for p-values and net transfer (> 60)")
    r.add_argument("--out", type=Path, default=Path("results/llm_rps_v2"))
    r.add_argument("--campaign", help="folder name; reuse it to resume (finished games are skipped)")
    r.add_argument("--tag", default="")
    r.add_argument("--run-offset", type=int, default=0)
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--cache", type=Path, default=Path("cache/llm_v2.sqlite"))
    s = ap.add_argument_group("static tools (no LLM call)")
    s.add_argument("--check", action="store_true", help="check prompt placeholders and render every request")
    s.add_argument("--render", action="store_true", help="print every prompt, rendered by the real code, as markdown")
    s.add_argument("--estimate", action="store_true", help="print calls, tokens and cost, then exit")
    s.add_argument("--price-in", type=float, default=0.10, help="$ per M input tokens (API)")
    s.add_argument("--price-out", type=float, default=0.20, help="$ per M output tokens (API)")
    s.add_argument("--tokens-per-sec", type=float, default=300.0, help="aggregate decode throughput per GPU (benchmark it)")
    s.add_argument("--gpu-price", type=float, default=1.10, help="$ per GPU hour (Modal A10G: 1.10, L4: 0.80, A100-40GB: 2.10)")
    s.add_argument("--stream-tokens-per-sec", type=float, default=30.0, help="decode speed seen by one game (benchmark it)")
    s.add_argument("--gpu-count", type=int, default=0, help="model containers (default: 3 for mixed_models, else 1)")
    s.add_argument("--cold-start-min", type=float, default=4.0, help="minutes for a container to load its model")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if unknown := set(conditions) - set(CONDITIONS):
        raise SystemExit(f"unknown conditions {sorted(unknown)}; choose from {list(CONDITIONS)}")
    if args.negotiation_turns < 1 and any(CONDITIONS[c]["channel"] for c in conditions):
        raise SystemExit("--negotiation-turns must be >= 1 when a condition has a channel")
    if args.design == "mixed_models" and not (args.lineup or args.agents):
        raise SystemExit("mixed_models needs --lineup")
    args.model = args.model or os.environ.get("NETINF_MODEL", "")
    args.base_url = args.base_url or os.environ.get("NETINF_BASE_URL", "http://localhost:11434/v1")
    prompts = load_prompts(args.prompts, args.prompt)
    check_prompts(prompts, args)
    if args.check:
        print(f"OK: {args.prompts} renders for every condition, player and request")
        return
    if args.render:
        print(render_all(prompts, args, conditions))
        return
    print(estimate(prompts, args, conditions))
    if args.estimate:
        return
    if not (args.agents or args.design == "mixed_models" or args.model):
        raise SystemExit("same_model needs --model or NETINF_MODEL")

    short = lambda name: name.split("/")[-1].replace(":", "_")  # noqa: E731
    label = "mixed" if args.design == "mixed_models" else short(args.model or "model")
    campaign = args.campaign or f"{datetime.now():%Y%m%d-%H%M}_{args.design}_{label}" + (f"_{args.tag}" if args.tag else "")
    cdir = args.out / campaign
    (cdir / "games").mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (cdir / "config.json").write_text(json.dumps({"settings": settings, "prompts": prompts, "git_commit": git_commit(),
                                                  "written": datetime.now().isoformat(timespec="seconds")}, indent=2))
    clients: dict[tuple[str, str, str], CachedChat] = {}

    lock = threading.Lock()

    def client(spec):
        if spec is None:
            return None
        with lock:  # games start in parallel: without the lock each thread made its own client and usage was undercounted
            return _client(spec)

    def _client(spec):
        if spec not in clients:
            clients[spec] = CachedChat(model=spec[0], base_url=spec[1], revision=spec[2], cache_path=args.cache,
                                       system_as_user=spec[0] in NO_SYSTEM_ROLE)
        return clients[spec]

    jobs = [(c, run) for c in conditions for run in range(args.run_offset, args.run_offset + args.runs)]
    todo = [(c, run) for c, run in jobs if not (cdir / "games" / f"{c}_run{run:02d}.json").exists()]
    if len(todo) < len(jobs):
        print(f"skipping {len(jobs) - len(todo)} games already in {cdir}")

    def one(job) -> str:
        cond, run = job
        start = time.time()
        try:
            specs = seat_models(args, run)
            llms = [client(s) for s in specs]
            rng = np.random.default_rng(stable_seed(run, args.seed_offset, cond))
            proto = Protocol(prompts, cond, args)
            game = play(llms, proto, run, args, rng)
            game.update(condition=cond, run=run, campaign=campaign, design=args.design, settings=settings,
                        seats={n: (s[0] if s else "uniform") for n, s in zip(NAMES, specs)},
                        seat_details={n: ({"model": s[0], "base_url": s[1], "revision": s[2]} if s else None)
                                      for n, s in zip(NAMES, specs)})
            game["metrics"] = game_metrics(game, args.perms, rng)
            game["detection"] = score(np.array(game["actions"]), args.perms, rng)
            game["reflection_metrics"] = reflection_metrics(game)
            m = game["metrics"]
            game["debrief_accuracy"] = debrief_accuracy(game, m["pair_advantage"] > 0 and m["pair_advantage_p"] < 0.05)
            game["elapsed_s"] = round(time.time() - start, 1)
            (cdir / "games" / f"{cond}_run{run:02d}.json").write_text(json.dumps(game))
            s, f = game["final_scores"], game["failures"]
            return (f"[{cond:<24} run {run:02d}] A {s[0]:6.1f} B {s[1]:6.1f} C {s[2]:6.1f} | "
                    f"A+B vs C {m['pair_advantage']:+.2f}/round, CS_AB {m['net_transfer']:+.2f} | "
                    f"plan executed {m['executed_agreement']:.0%} | failures {f} | {game['elapsed_s']:.0f}s")
        except Exception as e:  # one broken game must not stop the campaign; it is retried on the next launch
            return f"[{cond:<24} run {run:02d}] FAILED after {time.time() - start:.0f}s: {e!r}"

    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)
    elapsed = time.time() - start
    for c in clients.values():
        u = c.usage
        print(f"{c.model}: {u.calls} calls ({u.cache_hits} cached), {u.input_tokens} in / {u.output_tokens} out tokens, "
              f"{u.calls / max(elapsed, 1e-9):.1f} calls/s over {elapsed / 60:.1f} min")
    print(f"\nResults: {cdir}\nNext:    python experiments/analyze_v2.py {cdir}")


if __name__ == "__main__":
    main()
