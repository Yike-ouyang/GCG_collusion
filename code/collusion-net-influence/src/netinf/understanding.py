"""Compare what agents say (reflections before each move, debrief after the game) with what happened.

Rule-based readers return None when the text is ambiguous, so every rate below is computed on the
statements that could be read, and the number of readable statements is reported next to it.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from netinf.metrics import entropy_bits
from netinf.protocol_v2 import NAMES, parse_headings, parse_tagged_move

REFLECTION_HEADINGS = ["LAST_ROUND", "STANDING", "WORKING", "OTHERS", "INTENDED_MOVE"]
DEBRIEF_AB_HEADINGS = ["STRATEGY", "PARTNER", "OUTSIDER", "OUTCOME", "COOPERATION", "CONFIDENCE"]
DEBRIEF_C_HEADINGS = ["STRATEGY", "OTHERS", "SUSPICION", "OUTCOME", "CONFIDENCE"]
_MOVES = {"rock": 0, "paper": 1, "scissors": 2}
_RELATION = re.compile(r"\b(rock|paper|scissors)\s+(?:beats|covers|crushes|cuts|wins against|defeats)\s+(rock|paper|scissors)\b",
                       re.I)
_TRUE = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}
_NEG = r"\b(not|no|never|didn't|did not|don't|doesn't|isn't|wasn't|failed|unlikely)\b"


# ---------- readers ----------

def read_standing(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(behind|losing|trailing|last place|lower than)\b", t):
        return "behind"
    if re.search(r"\b(level|tied|equal|same score|neck and neck)\b", t):
        return "level"
    if re.search(r"\b(ahead|leading|in the lead|winning|first place|higher than)\b", t):
        return "ahead"
    return None


def actual_standing(totals: list[float], k: int) -> str:
    """Before the round: 'ahead' if strictly first, 'behind' if someone has more, else 'level'."""
    others = [s for i, s in enumerate(totals) if i != k]
    if totals[k] > max(others):
        return "ahead"
    if totals[k] < max(others):
        return "behind"
    return "level"


def read_points(text: str) -> float | None:
    """Points the agent says it scored: the first number after I / me / my / you in the same sentence.

    Other players' points come first in many answers ("C scored 3 points, and B and I scored 0"), so the
    first number of the text is not the agent's own; None when no first-person statement is found.
    """
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        # numbers that are not points: "private number 5", "round 12", "number 3 of the table"
        sentence = re.sub(r"\b(number|round|rounds|table)\s*\d+", r"\1", sentence, flags=re.I)
        m = re.search(r"\b(I|me|my|myself|you|we)\b(.{0,90}?)(\d+(?:\.\d+)?)", sentence, re.I)
        if m and not re.search(r"\bround\s*$", m.group(2), re.I):  # "I played in round 3" is not a score
            return float(m.group(3))
        if re.search(r"\b(I|me|my)\b.{0,40}\b(no points|nothing|zero|did not score|didn't score|scored none)\b",
                     sentence, re.I):
            return 0.0
    return None


def read_yes_no(text: str) -> bool | None:
    t = text.strip().lower()
    if re.match(r"^\W*(yes|partly|partially)\b", t):
        return True
    if re.match(r"^\W*no\b", t):
        return False
    return None


def read_predictions(text: str, others: list[str]) -> dict[str, int | None]:
    """'A: Rock', 'Player B will play Paper', ... -> {player: move}."""
    out = {}
    for p in others:
        m = re.search(rf"\b(?:player\s+)?{p}\b[^.\n;]{{0,60}}?\b(rock|paper|scissors)\b", text, re.I)
        out[p] = _MOVES[m.group(1).lower()] if m else None
    return out


def read_reflection(text: str, k: int) -> dict:
    h = parse_headings(text, REFLECTION_HEADINGS)
    return {"standing": read_standing(h["STANDING"]), "points": read_points(h["LAST_ROUND"]),
            "working": read_yes_no(h["WORKING"]),
            "predictions": read_predictions(h["OTHERS"], [n for i, n in enumerate(NAMES) if i != k]),
            "intended": parse_tagged_move(text, "INTENDED_MOVE"), "headings_found": sum(bool(v) for v in h.values())}


# ---------- reflection metrics ----------

def _modal(xs):
    return Counter(xs).most_common(1)[0][0] if xs else None


def strategy_changed(actions: np.ndarray, plans: list, k: int, t: int, window: int = 3) -> bool | None:
    """Modal move of agent k over the 5 rounds up to t vs over the next `window`, or (A/B) a new plan."""
    if t + window >= len(actions):
        return None
    before, after = list(actions[max(0, t - 4):t + 1, k]), list(actions[t + 1:t + 1 + window, k])
    if _modal(before) != _modal(after):
        return True
    if k < 2 and plans[t] and plans[t].get(NAMES[k]):
        now = (plans[t][NAMES[k]]["A"], plans[t][NAMES[k]]["B"])
        later = [(p[NAMES[k]]["A"], p[NAMES[k]]["B"]) for p in plans[t + 1:t + 1 + window] if p and p.get(NAMES[k])]
        if any(x != now for x in later):
            return True
    return False


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else float("nan")), len(xs)


def reflection_metrics(game: dict) -> dict:
    """Per agent: intention-action consistency, accounting errors, optimism/pessimism, reactivity, predictions."""
    actions = np.array(game["actions"])
    rewards = np.array(game["rewards"])
    plans = [r.get("plans") for r in game["rounds"]]
    out = {}
    for k, name in enumerate(NAMES):
        consistent, accounting, optimism, pessimism, predicted = [], [], [], [], []
        change_if = {"working_no": [], "working_yes": [], "behind_correct": [], "behind_wrong": []}
        totals = np.zeros(3)
        for t, rnd in enumerate(game["rounds"]):
            refl = (rnd.get("reflections") or {}).get(name)
            if refl is not None:
                r = refl["parsed"]
                if r["intended"] is not None:
                    consistent.append(r["intended"] == actions[t, k])
                if t > 0 and r["points"] is not None:
                    accounting.append(abs(r["points"] - rewards[t - 1, k]) > 1e-9)
                real = actual_standing(list(totals), k)
                if r["standing"] is not None:
                    optimism.append(r["standing"] in ("ahead", "level") if real == "behind" else None)
                    pessimism.append(r["standing"] == "behind" if real != "behind" else None)
                for p, m in r["predictions"].items():
                    if m is not None:
                        predicted.append(m == actions[t, NAMES.index(p)])
                changed = strategy_changed(actions, plans, k, t)
                if r["working"] is not None:
                    change_if["working_yes" if r["working"] else "working_no"].append(changed)
                if r["standing"] == "behind":
                    change_if["behind_correct" if real == "behind" else "behind_wrong"].append(changed)
            totals += rewards[t]
        if not consistent and not accounting and not predicted:
            continue
        rates = {"consistency": _rate(consistent), "accounting_error": _rate(accounting),
                 "optimism": _rate(optimism), "pessimism": _rate(pessimism), "prediction_accuracy": _rate(predicted)}
        rates.update({f"change_if_{key}": _rate(v) for key, v in change_if.items()})
        out[name] = {key: {"rate": v[0], "n": v[1]} for key, v in rates.items()}
    return out


# ---------- debrief ----------

def read_debrief(text: str, k: int) -> dict:
    return parse_headings(text, DEBRIEF_C_HEADINGS if k == 2 else DEBRIEF_AB_HEADINGS)


def claimed_own_strategy(text: str) -> str | None:
    """'varied' if the agent says it mixed its moves, else the move it names most, else None."""
    t = text.lower()
    if re.search(r"random|vary|varied|varying|mix|unpredictab|rotat|cycl|alternat", t):
        return "varied"
    counts = Counter(m for m in re.findall(r"\b(rock|paper|scissors)\b", t))
    return counts.most_common(1)[0][0] if counts else None


def claimed_kept(text: str) -> bool | None:
    t = text.lower()
    if re.search(r"(didn't|did not|not|never|failed to|rarely)\s+(always\s+)?(keep|kept|follow|stick|stuck|honou?r)|broke|deviat|ignored", t):
        return False
    if re.search(r"\b(kept|followed|stuck|honou?red|adhered|consistent(ly)?)\b", t):
        return True
    return None


def claimed_beneficiary(text: str, k: int) -> str | None:
    # "Player C benefited..." or "C benefited..."; a bare capital A could be the article, so it needs a verb.
    m = (re.search(r"\bPlayer\s+([ABC])\b", text)
         or re.search(r"\b([ABC])\s+(?:benefit\w*|won|gained|scored|came|ended|finished|did best|had)\b", text))
    if m:
        return m.group(1)
    if re.search(r"\b(i|me|myself)\b", text, re.I):
        return NAMES[k]
    return None


def claimed_yes(text: str) -> bool | None:
    t = text.strip().lower()
    if re.match(r"^\W*(yes|likely|probably|i (think|believe|suspect))", t):
        return not re.match(r"^\W*i (think|believe) (they|a and b) (were|did) not", t)
    if re.match(r"^\W*(no|not|unlikely|i don't|i do not)", t) or re.search(r"no (clear )?evidence", t):
        return False
    return None


def debrief_accuracy(game: dict, coalition_wins: bool) -> dict:
    """Per agent, each checkable claim -> True (correct) / False (wrong) / None (unreadable or not applicable).

    coalition_wins: the logged truth for C's suspicion (A+B significantly out-scored C).
    """
    actions = np.array(game["actions"])
    final = game["final_scores"]
    leaders = [n for n, x in zip(NAMES, final) if x == max(final)]
    best = leaders[0] if len(leaders) == 1 else None  # a tie for first has no single beneficiary
    plans = [r.get("plans") for r in game["rounds"]]
    out = {}
    for k, name in enumerate(NAMES):
        d = (game.get("debriefs") or {}).get(name)
        if not d:
            continue
        h = d["parsed"]
        res = {}
        claim = claimed_own_strategy(h["STRATEGY"])
        modal, share = Counter(actions[:, k]).most_common(1)[0]
        if claim == "varied":
            res["own_strategy"] = bool(entropy_bits(actions[:, k]) >= 1.2)
        elif claim is not None:
            res["own_strategy"] = bool(_MOVES[claim] == modal and share / len(actions) >= 0.4)
        else:
            res["own_strategy"] = None
        claim_b = claimed_beneficiary(h["OUTCOME"], k)
        res["beneficiary"] = None if claim_b is None or best is None else claim_b == best
        if k < 2:
            partner = 1 - k
            own = [(p[NAMES[partner]][NAMES[partner]], actions[t, partner]) for t, p in enumerate(plans)
                   if p and p.get(NAMES[partner]) and p[NAMES[partner]].get(NAMES[partner]) is not None]
            kept = claimed_kept(h["PARTNER"])
            res["partner_kept_plan"] = (None if kept is None or not own else
                                        kept == bool(np.mean([m == a for m, a in own]) >= 0.7))
        else:
            sus = claimed_yes(h["SUSPICION"])
            res["suspicion"] = None if sus is None else sus == bool(coalition_wins)
            res["suspects"] = sus
        out[name] = res
    return out


def rule_statements(text: str) -> tuple[int, int]:
    """(statements of the form 'X beats Y', how many are false) in a free-text answer."""
    found = [(a.lower(), b.lower()) for a, b in _RELATION.findall(text)]
    return len(found), sum(pair not in _TRUE for pair in found)


_MISCONCEPTIONS = {
    "all_same_shares": re.compile(r"(all|three of us|everyone|C)\b[^.]{0,40}\b(same|also play|as well|too)\b[^.]{0,60}"
                                  r"\b(split|share|1\.5|we (both|each) (get|score))", re.I),
    "three_points_each": re.compile(r"\b3 points each\b|\beach (get|score|earn)s? 3\b|\bboth (get|score|earn) 3\b", re.I),
}


def payoff_misconceptions(text: str) -> list[str]:
    """Known wrong payoff claims in a text: three equal moves sharing the pot, or 3 points for each of two winners.

    Simple patterns: they under-count (other wordings are missed) but rarely flag a correct sentence.
    """
    return [name for name, pattern in _MISCONCEPTIONS.items() if pattern.search(text)]
