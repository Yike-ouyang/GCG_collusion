"""Text side of the v2 protocol: what the environment tells the agents, and how their replies are read.

The environment computes every outcome and shows it as one line per round, so agents never compute
payoffs themselves. Agents answer in free text with tagged lines (MOVE:, PLAN:, HEADING:), parsed here.
"""

from __future__ import annotations

import re
import string

import numpy as np

NAMES = ("A", "B", "C")
MOVE_NAMES = ("Rock", "Paper", "Scissors")
BEATS = {0: 2, 1: 0, 2: 1}  # BEATS[a] is the move that a beats
_MOVE = r"(ROCK|PAPER|SCISSORS|R|P|S)\b"


def move_index(token: str) -> int:
    return "RPS".index(token.strip().upper()[0])


def outcome_line(round_no: int, moves: list[int], rewards: list[float]) -> str:
    """'Round 7: A=Rock, B=Scissors, C=Rock -> Rock wins, shared by A and C. Points: A 1.5, B 0, C 1.5'."""
    played = ", ".join(f"{n}={MOVE_NAMES[m]}" for n, m in zip(NAMES, moves))
    distinct = set(moves)
    if len(distinct) == 1:
        result = "all the same, nobody scores"
    elif len(distinct) == 3:
        result = "all different, nobody scores"
    else:
        a, b = distinct
        winner = a if BEATS[a] == b else b
        who = [n for n, m in zip(NAMES, moves) if m == winner]
        result = f"{MOVE_NAMES[winner]} wins, " + (f"{who[0]} alone" if len(who) == 1 else f"shared by {who[0]} and {who[1]}")
    points = ", ".join(f"{n} {r:g}" for n, r in zip(NAMES, rewards))
    return f"Round {round_no}: {played} -> {result}. Points: {points}"


def history_block(history: list[dict], last: int) -> str:
    """The last `last` outcome lines (all rounds if last < 0), or a placeholder before round 1."""
    shown = history if last < 0 else history[len(history) - last:] if last else []
    return "\n".join(h["outcome"] for h in shown) if shown else "(no round played yet)"


def chat_block(messages: list[dict]) -> str:
    if not messages:
        return "(no messages yet)"
    return "\n".join(f"{m['from']}: {m['text']}" for m in messages)


def first_speaker(round_no: int) -> int:
    """A (0) writes first in odd rounds, B (1) in even rounds."""
    return 0 if round_no % 2 == 1 else 1


def is_skip(text: str) -> bool:
    return text.strip().strip(".!\"'*` ").upper() == "SKIP"


def parse_tagged_move(text: str, tag: str = "MOVE") -> int | None:
    """The move on the last 'TAG: <move>' line; None if there is none."""
    found = re.findall(rf"(?<![A-Z_]){tag}\s*[:=]\s*\**\s*{_MOVE}", text.upper())  # MOVE: is not INTENDED_MOVE:
    return move_index(found[-1]) if found else None


def parse_plan(text: str) -> dict:
    """{'A': move|None, 'B': move|None} from 'PLAN: A=<move>, B=<move>' (first value per player)."""
    body = text.upper().split("PLAN", 1)[-1]
    found = {}
    for who, move in re.findall(rf"\b([AB])\s*[=:]\s*\**\s*{_MOVE}", body):
        found.setdefault(who, move_index(move))
    return {"A": found.get("A"), "B": found.get("B")}


def parse_headings(text: str, headings: list[str]) -> dict[str, str]:
    """Split 'HEADING: text' sections; missing headings map to ''."""
    pattern = r"(?im)^\W*(" + "|".join(map(re.escape, headings)) + r")\W*:\s*"
    parts = re.split(pattern, text)
    out = {h: "" for h in headings}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].upper()
        if key in out and not out[key]:
            out[key] = parts[i + 1].strip().strip("*").strip()
    return out


def template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def unknown_fields(prompts: dict[str, str], allowed: set[str]) -> dict[str, set[str]]:
    """Placeholders used in each prompt that the renderer does not provide (typos)."""
    bad = {}
    for key, text in prompts.items():
        if key.startswith("_"):
            continue
        extra = template_fields(text) - allowed
        if extra:
            bad[key] = extra
    return bad


def drop_empty_lines(text: str) -> str:
    """User prompts have no intentional blank lines; empty optional variables leave some behind."""
    return "\n".join(line for line in text.split("\n") if line.strip())


def move_distribution(tokens: list[dict], tag: str = "MOVE") -> dict | None:
    """Probability of Rock / Paper / Scissors at the first word token after the last 'TAG:' in the answer.

    `tokens` is CachedChat.generate_full()["tokens"]: each generated token with its top-k alternatives as
    [token, logprob]. An alternative counts for a move when its letters are a prefix of the move name
    ('R', 'Ro', 'Rock', ' rock'...). Returns the three probabilities renormalised over the alternatives
    that name a move, plus `mass`: their total probability before renormalising (how much of the top-k
    is a move at all). None if the tag or its logprobs are missing.
    """
    if not tokens:
        return None
    text, starts = "", []
    for t in tokens:
        starts.append(len(text))
        text += t["token"]
    found = list(re.finditer(rf"(?<![A-Z_]){tag}\s*[:=]", text.upper()))
    if not found:
        return None
    end = found[-1].end()
    for i, t in enumerate(tokens):
        letters = re.sub(r"[^a-z]", "", t["token"].lower())
        if starts[i] + len(t["token"]) <= end or not letters:
            continue  # still inside the tag, or punctuation/spaces/'**' before the move word
        probs = [0.0, 0.0, 0.0]
        for alt, logprob in t["top"]:
            a = re.sub(r"[^a-z]", "", alt.lower())
            for m, name in enumerate(("rock", "paper", "scissors")):
                if a and name.startswith(a):
                    probs[m] += float(np.exp(logprob))
        mass = sum(probs)
        if mass == 0:
            return None
        return {"Rock": probs[0] / mass, "Paper": probs[1] / mass, "Scissors": probs[2] / mass, "mass": mass}
    return None
