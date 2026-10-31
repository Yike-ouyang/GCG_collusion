"""Summarise llm_rps_v2.py campaigns: tables per design x condition, reflection and debrief metrics,
results per model and per seat, a figure, report.md, and one history.csv row per campaign x condition.

Usage:
    python experiments/analyze_v2.py results/llm_rps_v2/<same_model campaign> [results/llm_rps_v2/<mixed campaign> ...]
Writes into the first campaign folder (or --out): summary.csv, games.csv, by_model.csv, report.md,
coalition_by_condition.png; appends to <out>/../history.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_llm import mean_ci, wilson
from llm_rps_v2 import CONDITIONS
from netinf.understanding import debrief_accuracy, read_debrief, read_reflection, reflection_metrics

NAMES = ("A", "B", "C")
DESIGN_COLORS = {"same_model": "#2a78d6", "mixed_models": "#eb6834"}  # categorical slots 1-2 of the dataviz palette
MUTED = "#6b6a63"
REFL_KEYS = ("consistency", "accounting_error", "optimism", "pessimism", "prediction_accuracy",
             "change_if_working_no", "change_if_working_yes", "change_if_behind_correct", "change_if_behind_wrong")


def reparse(g: dict) -> None:
    """Re-read reflections and debriefs from their raw text with the current readers, so games played with an
    older reader (e.g. one that took the v4 private number for a score) are analysed like new ones."""
    for r in g["rounds"]:
        for name, refl in (r.get("reflections") or {}).items():
            refl["parsed"] = read_reflection(refl["raw"], NAMES.index(name))
    for name, d in (g.get("debriefs") or {}).items():
        d["parsed"] = read_debrief(d.get("raw", ""), NAMES.index(name))
    g["reflection_metrics"] = reflection_metrics(g)
    m = g["metrics"]
    g["debrief_accuracy"] = debrief_accuracy(g, m["pair_advantage"] > 0 and m["pair_advantage_p"] < 0.05)


def game_row(g: dict) -> dict:
    m, d = g["metrics"], g["detection"]["prev"]
    row: dict = {"campaign": g["campaign"], "design": g["design"], "condition": g["condition"], "run": g["run"],
           **{f"seat_{n}": g["seats"][n] for n in NAMES},
           **{f"final_{n}": g["final_scores"][i] for i, n in enumerate(NAMES)},
           "pair_adv": m["pair_advantage"], "pair_adv_p": m["pair_advantage_p"], "cs_ab": m["net_transfer"],
           "assist": m["assist_rate_sym"], "alternation": m["alternation"], "same_move": m["same_move_rate"], "cycle": m.get("cycle_adherence", np.nan), "die": m.get("die_adherence", np.nan),
           "comply_A": m.get("die", {}).get("compliance_A", np.nan), "comply_B": m.get("die", {}).get("compliance_B", np.nan),
           "betrayal": m.get("die", {}).get("betrayal", np.nan), "c_per_round": m.get("die", {}).get("c_per_round", np.nan),
           "plan_executed": m["executed_agreement"], "follow_A": m["follow_A"], "follow_B": m["follow_B"],
           "usage_A": m["channel_usage"]["A"], "usage_B": m["channel_usage"]["B"],
           **{f"H_{n}": m["entropy"][i] for i, n in enumerate(NAMES)},
           **{f"Hs_{n}": m["cond_entropy"][i] for i, n in enumerate(NAMES)},
           "flag_fixed": d["fixed"] == [0, 1], "flag_calib": d["calibrated"] == [0, 1],
           "any_flag_calib": d["calibrated"] is not None,
           "move_fail": g["failures"].get("move", 0), "forced_skip": g["failures"].get("forced_skip", 0),
           "truncated": sum(v for k, v in g["failures"].items() if k.startswith("truncated")),
           "max_prompt": max(g.get("max_prompt_tokens", {}).values(), default=0)}
    moves = ("Rock", "Paper", "Scissors")
    for k, n in enumerate(NAMES):  # logprobs: confidence in the move played; intended move = top-1 at MOVE:
        conf, top1 = [], []
        for r in g["rounds"]:
            d = (r.get("move_dist") or {}).get(n)
            if not d:
                continue
            conf.append(d[moves[r["moves"][k]]])
            intended = (((r.get("reflections") or {}).get(n) or {}).get("parsed") or {}).get("intended")
            if intended is not None:
                top1.append(max(moves, key=d.get) == moves[intended])
        row[f"lp_conf_{n}"] = float(np.mean(conf)) if conf else np.nan
        row[f"lp_intent_top1_{n}"] = float(np.mean(top1)) if top1 else np.nan
    for n in NAMES:
        r = g.get("reflection_metrics", {}).get(n, {})
        for k in REFL_KEYS:
            row[f"refl_{n}_{k}"] = r.get(k, {}).get("rate", np.nan)
            row[f"refl_{n}_{k}_n"] = r.get(k, {}).get("n", 0)
        a = g.get("debrief_accuracy", {}).get(n, {})
        for k in ("own_strategy", "beneficiary", "partner_kept_plan", "suspicion", "suspects"):
            row[f"deb_{n}_{k}"] = a.get(k)
    return row


def nanmean(xs) -> float:
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    return float(np.nanmean(xs)) if xs.size and np.any(~np.isnan(xs)) else float("nan")


def pooled(rows, players: str, key: str) -> float:
    """Reflection rate pooled over games and players, weighted by the number of readable statements."""
    num = sum(r[f"refl_{n}_{key}"] * r[f"refl_{n}_{key}_n"] for r in rows for n in players if r[f"refl_{n}_{key}_n"])
    den = sum(r[f"refl_{n}_{key}_n"] for r in rows for n in players)
    return num / den if den else float("nan")


def share_true(rows, key: str) -> str:
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"


def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


def ci(xs, d: int = 2) -> str:
    m, lo, hi = mean_ci(xs)
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="output folder (default: the first campaign)")
    args = ap.parse_args()
    out = args.out or args.campaigns[0]
    games = [json.loads(f.read_text()) for c in args.campaigns for f in sorted((c / "games").glob("*_run*.json"))]
    if not games:
        raise SystemExit("no games found")
    for g in games:
        reparse(g)
    rows = [game_row(g) for g in games]
    cells = [(d, c) for d in ("same_model", "mixed_models") for c in CONDITIONS
             if any(r["design"] == d and r["condition"] == c for r in rows)]
    sel = lambda d, c: [r for r in rows if r["design"] == d and r["condition"] == c]  # noqa: E731

    # ---- main table ----
    h_main = ["design", "condition", "games", "final A / B / C", "A+B vs C /round", "CS_AB", "plan executed",
              "channel use A / B", "same move", "assist", "6-cycle played", "die table played", "table followed A / B", "betrayal",
              "C per round (0.5 if table followed)", "H(a) A/B/C", "H(a|s) A/B/C", "net-inf fixed / calib.",
              "P(move played) A/B/C", "intended = top-1 at MOVE", "truncated / parse fails", "longest prompt"]
    t_main, summary = [], []
    for d, c in cells:
        rs = sel(d, c)
        row = [d, c, len(rs), " / ".join(f"{nanmean(r[f'final_{n}'] for r in rs):.1f}" for n in NAMES),
               ci([r["pair_adv"] for r in rs]), ci([r["cs_ab"] for r in rs]), pct(nanmean(r["plan_executed"] for r in rs)),
               f"{pct(nanmean(r['usage_A'] for r in rs))} / {pct(nanmean(r['usage_B'] for r in rs))}",
               pct(nanmean(r["same_move"] for r in rs)), pct(nanmean(r["assist"] for r in rs)), pct(nanmean(r["cycle"] for r in rs)), pct(nanmean(r["die"] for r in rs)),
               f"{pct(nanmean(r['comply_A'] for r in rs))} / {pct(nanmean(r['comply_B'] for r in rs))}",
               pct(nanmean(r["betrayal"] for r in rs)), f"{nanmean(r['c_per_round'] for r in rs):.2f}",
               " / ".join(f"{nanmean(r[f'H_{n}'] for r in rs):.2f}" for n in NAMES),
               " / ".join(f"{nanmean(r[f'Hs_{n}'] for r in rs):.2f}" for n in NAMES),
               f"{sum(r['flag_fixed'] for r in rs)} / {sum(r['flag_calib'] for r in rs)} of {len(rs)}",
               " / ".join(pct(nanmean(r[f"lp_conf_{n}"] for r in rs)) for n in NAMES),
               pct(nanmean(r[f"lp_intent_top1_{n}"] for r in rs for n in NAMES)),
               f"{sum(r['truncated'] for r in rs)} / {sum(r['move_fail'] for r in rs)}", max(r["max_prompt"] for r in rs)]
        t_main.append(row)
        summary.append(dict(zip(h_main, row)))

    # ---- reflection ----
    h_refl = ["design", "condition", "player", "intention = action", "accounting errors", "optimism (says ahead/level, is behind)",
              "pessimism", "predicts others", "change if 'not working'", "change if 'working'",
              "change if correctly behind", "change if wrongly behind"]
    t_refl = []
    for d, c in cells:
        rs = sel(d, c)
        for group, players in (("A+B", "AB"), ("C", "C")):
            dens = [sum(r[f"refl_{n}_{k}_n"] for r in rs for n in players) for k in REFL_KEYS]
            vals = [f"{pct(pooled(rs, players, k))} ({den})" if den else "-" for k, den in zip(REFL_KEYS, dens)]
            if any(v != "-" for v in vals):
                t_refl.append([d, c, group, *vals])

    # ---- debrief ----
    h_deb = ["design", "condition", "own strategy right (A+B / C)", "who benefited right (A+B / C)",
             "'partner kept plan' right", "C suspicion right", "C suspects A-B"]
    t_deb = []
    for d, c in cells:
        rs = sel(d, c)
        ab = lambda key: [r[f"deb_{n}_{key}"] for r in rs for n in "AB"]  # noqa: E731
        frac = lambda vals: (f"{sum(v for v in vals if v is not None)}/{sum(v is not None for v in vals)}"  # noqa: E731
                             if any(v is not None for v in vals) else "-")
        t_deb.append([d, c, f"{frac(ab('own_strategy'))} / {share_true(rs, 'deb_C_own_strategy')}",
                      f"{frac(ab('beneficiary'))} / {share_true(rs, 'deb_C_beneficiary')}", frac(ab("partner_kept_plan")),
                      share_true(rs, "deb_C_suspicion"), share_true(rs, "deb_C_suspects")])

    # ---- per model and seat ----
    by_model = []
    for model in sorted({r[f"seat_{n}"] for r in rows for n in NAMES}):
        for n in NAMES:
            rs = [r for r in rows if r[f"seat_{n}"] == model]
            if not rs:
                continue
            by_model.append({"model": model, "seat": n, "games": len(rs),
                             "mean final score": f"{nanmean(r[f'final_{n}'] for r in rs):.1f}",
                             "H(a)": f"{nanmean(r[f'H_{n}'] for r in rs):.2f}",
                             "intention = action": pct(pooled(rs, n, "consistency")),
                             "optimism": pct(pooled(rs, n, "optimism")),
                             "move parse failures": sum(r["move_fail"] for r in rs)})

    # ---- false alarms: same vs mixed, no_channel ----
    fa = []
    for d in ("same_model", "mixed_models"):
        rs = sel(d, "no_channel")
        if rs:
            k = sum(r["any_flag_calib"] for r in rs)
            p, lo, hi = wilson(k, len(rs))
            fa.append([d, f"{k}/{len(rs)}", f"{p:.0f}% [{lo:.0f}, {hi:.0f}]"])

    # ---- output ----
    sections = [("COALITION, PLANS AND DETECTION", h_main, t_main), ("REFLECTION (pooled statements; n in brackets)", h_refl, t_refl),
                ("DEBRIEF UNDERSTANDING (correct / readable claims)", h_deb, t_deb)]
    for title, h, t in sections:
        print(f"\n{title}\n" + table(h, t))
    if by_model:
        print("\nPER MODEL AND SEAT\n" + table(list(by_model[0]), [list(r.values()) for r in by_model]))
    if fa:
        print("\nFALSE ALARMS IN no_channel (any pair flagged, calibrated)\n" + table(["design", "games", "rate [95% CI]"], fa))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=h_main)
        w.writeheader()
        w.writerows(summary)
    if by_model:
        with open(out / "by_model.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(by_model[0]))
            w.writeheader()
            w.writerows(by_model)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    for ax, key, title in ((axes[0], "pair_adv", "A+B vs C, points per round"), (axes[1], "cs_ab", "net transfer CS_AB")):
        for j, d in enumerate(("same_model", "mixed_models")):
            for x, c in enumerate(conds):
                v = [r[key] for r in sel(d, c)]
                if v:
                    xs = x + (j - 0.5) * 0.3 + np.random.default_rng(x).uniform(-0.05, 0.05, len(v))
                    ax.scatter(xs, v, s=30, color=DESIGN_COLORS[d], edgecolor="white", linewidth=1,
                               label=d if x == 0 else None, zorder=3)
                    ax.hlines(np.mean(v), x + (j - 0.5) * 0.3 - 0.12, x + (j - 0.5) * 0.3 + 0.12, color=MUTED, lw=2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(conds)), [c.replace("channel_", "ch_\n") for c in conds], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "coalition_by_condition.png", dpi=160)

    report = [f"# v2 campaigns: {', '.join(c.name for c in args.campaigns)}", f"Analysed {datetime.now():%Y-%m-%d %H:%M}, {len(games)} games."]
    report += [f"## {title.capitalize()}\n\n{table(h, t)}" for title, h, t in sections]
    if by_model:
        report.append("## Per model and seat\n\n" + table(list(by_model[0]), [list(r.values()) for r in by_model]))
    if fa:
        report.append("## False alarms in no_channel\n\n" + table(["design", "games", "rate [95% CI]"], fa))
    report.append("![coalition](coalition_by_condition.png)")
    (out / "report.md").write_text("\n\n".join(report) + "\n")

    hist = out.parent / "history.csv"
    old = list(csv.DictReader(hist.open())) if hist.exists() else []
    names = {c.name for c in args.campaigns}
    old = [h for h in old if h["campaign"] not in names]
    new = []
    for (d, c), s in zip(cells, summary):
        rs = sel(d, c)
        new.append({"campaign": "+".join(sorted({r["campaign"] for r in rs})), "design": d, "condition": c, "games": len(rs),
                    "seats": "; ".join(sorted({f"{r['seat_A']}|{r['seat_B']}|{r['seat_C']}" for r in rs})),
                    "pair_adv": s["A+B vs C /round"], "cs_ab": s["CS_AB"], "plan_executed": s["plan executed"],
                    "flags": s["net-inf fixed / calib."], "analysed": datetime.now().isoformat(timespec="seconds")})
    fields = list(dict.fromkeys(list(new[0]) + [k for h in old for k in h]))
    with open(hist, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old + new)
    print(f"\nWrote {out}/report.md, summary.csv, games.csv, coalition_by_condition.png; history in {hist}")


if __name__ == "__main__":
    main()
