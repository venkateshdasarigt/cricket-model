"""
ask_ai.py — Natural-language Q&A on live match state
=====================================================
No LLM — uses intent matching + parameter extraction against the simulator.
Handles questions like:
  • "What's the probability of 10+ runs in the next over?"
  • "Chance of a wicket in the next 6 balls?"
  • "Will the chase succeed?"
  • "Required run rate?"
  • "What does the model think the final score will be?"
  • "Who's more likely to win?"
  • "Best bowler to bowl this over?"
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# Intent patterns — ordered from most-specific to most-general
# ------------------------------------------------------------------

def answer_question(question: str, state, sim, intel,
                    venue: str = "") -> dict:
    """
    Parse `question` and return {"answer": str, "confidence": "high|mid|low",
    "details": dict|None}. Never raises.
    """
    if state is None:
        return {
            "answer": "No match state yet — start the replay or pick a live match.",
            "confidence": "low",
            "details": None,
        }

    q = question.lower().strip()
    if not q:
        return {"answer": _help_text(), "confidence": "low", "details": None}

    # 1. Probability of N+ runs in next/current over
    m = re.search(r'(?:prob|chance|odds?|likelihood)[\s\w]*?(\d+)\s*\+?\s*runs?', q)
    if m and ("over" in q or "ov" in q):
        threshold = int(m.group(1))
        ov = sim.realistic_next_over(state, intel=intel, n_sims=200)
        dist = ov["distribution"]
        prob = float(np.mean(dist >= threshold))
        return {
            "answer": (f"Probability of **{threshold}+ runs** in the next over: "
                       f"**{prob*100:.1f}%**\n\n"
                       f"(model expects ~{ov['mean']:.1f} runs; "
                       f"bad day {ov['percentiles'][10]:.0f}, "
                       f"great day {ov['percentiles'][90]:.0f})"),
            "confidence": "high",
            "details": {"threshold": threshold, "prob": prob, "over": ov},
        }

    # 2. Next over exact runs
    if re.search(r'(next|this)\s+over\s+(runs?|score)', q) or \
       re.search(r'how\s+many\s+runs.*next\s+over', q):
        ov = sim.realistic_next_over(state, intel=intel, n_sims=150)
        return {
            "answer": (f"Model predicts **{ov['mean']:.1f} runs** in the next over.\n\n"
                       f"• Most likely: {ov['median']:.0f} runs\n"
                       f"• Bad day: {ov['percentiles'][10]:.0f}\n"
                       f"• Great day: {ov['percentiles'][90]:.0f}\n"
                       f"• P(≥ 10 runs): {float(np.mean(ov['distribution'] >= 10))*100:.0f}%"),
            "confidence": "high",
            "details": {"over": ov},
        }

    # 3. Wicket probability in next N balls
    if "wicket" in q and ("probab" in q or "chance" in q or "odds" in q or "likelihood" in q):
        m2 = re.search(r'(?:next|in)\s+(\d+)\s*balls?', q)
        balls = int(m2.group(1)) if m2 else 6
        nb = sim.predict_next_ball_outcomes(state, intel=intel)
        # P(at least one wicket in N balls) = 1 - (1-p)^N
        prob = 1 - (1 - nb["wicket"]) ** balls
        return {
            "answer": (f"Probability of a **wicket in the next {balls} balls**: "
                       f"**{prob*100:.1f}%**\n\n"
                       f"(per-ball wicket probability: {nb['wicket']*100:.1f}%)"),
            "confidence": "high",
            "details": {"balls": balls, "prob": prob, "per_ball": nb["wicket"]},
        }

    # 4. Win probability / who wins
    if ("win" in q or "winning" in q or "wins" in q) and state.innings == 2:
        tot = sim.realistic_projected_total(state, intel=intel)
        wp = tot.get("win_prob", 0) * 100
        need = max(0, state.target - state.runs_scored)
        balls_left = 120 - state.balls_bowled
        return {
            "answer": (f"**Win probability for the batting team: {wp:.1f}%**\n\n"
                       f"• Need {need} runs off {balls_left} balls "
                       f"(RRR {state.required_rr:.2f})\n"
                       f"• Projected final: {tot['median']:.0f} "
                       f"(target: {state.target})"),
            "confidence": "high" if abs(wp - 50) > 15 else "mid",
            "details": {"win_prob": wp, "need": need, "balls_left": balls_left},
        }

    # 5. Boundary probability
    if "boundary" in q or (("four" in q or "4s" in q) and "prob" in q) or \
       (("six" in q or "6s" in q) and "prob" in q):
        m3 = re.search(r'(?:next|in)\s+(\d+)\s*balls?', q)
        balls = int(m3.group(1)) if m3 else 6
        nb = sim.predict_next_ball_outcomes(state, intel=intel)
        boundary_per = nb["boundary"] + nb["six"]
        prob = 1 - (1 - boundary_per) ** balls
        return {
            "answer": (f"Probability of at least **one boundary** in next {balls} balls: "
                       f"**{prob*100:.1f}%**\n\n"
                       f"(per-ball: 4s {nb['boundary']*100:.1f}%, "
                       f"6s {nb['six']*100:.1f}%)"),
            "confidence": "high",
            "details": {"balls": balls, "prob": prob},
        }

    # 6. Required run rate
    if ("required" in q and ("rate" in q or "rrr" in q)) or \
       q.strip() in ("rrr", "rr", "req rate"):
        if state.innings == 2 and state.target:
            return {
                "answer": (f"**Required RR: {state.required_rr:.2f}**\n\n"
                           f"Current RR: {state.current_rr:.2f}\n"
                           f"Pressure index: {state.pressure_index:+.2f}"),
                "confidence": "high",
                "details": {"rrr": state.required_rr, "crr": state.current_rr},
            }
        return {
            "answer": "Required run rate only applies to the chasing team (innings 2).",
            "confidence": "high",
            "details": None,
        }

    # 7. Current run rate / score
    if ("current" in q and ("rate" in q or "rr" in q or "score" in q)) or \
       "what's the score" in q or "whats the score" in q:
        return {
            "answer": (f"**{state.runs_scored}/{state.wickets_fallen}** "
                       f"in {state.balls_bowled//6}.{state.balls_bowled%6} overs\n"
                       f"CRR: {state.current_rr:.2f}"
                       + (f" · RRR {state.required_rr:.2f}" if state.innings == 2 else "")),
            "confidence": "high",
            "details": None,
        }

    # 8. Projected / final score
    if ("projected" in q or "final score" in q or "final total" in q or
        "end score" in q or "end up" in q):
        tot = sim.realistic_projected_total(state, intel=intel)
        return {
            "answer": (f"Projected final score: **{tot['median']:.0f}**\n\n"
                       f"• Bad day: {tot['percentiles'][10]:.0f}\n"
                       f"• Great day: {tot['percentiles'][90]:.0f}"),
            "confidence": "mid",
            "details": {"total": tot},
        }

    # 9. Venue stats
    if "venue" in q or ("pitch" in q and "like" in q):
        if not venue or not intel:
            return {"answer": "No venue info available.", "confidence": "low",
                    "details": None}
        v = intel.venue(venue)
        vc = intel.venue_chase(venue)
        chase_pct = (vc.get("chase_wins", 0) /
                     max(vc.get("matches", 1), 1) * 100)
        return {
            "answer": (f"**{venue}** — {v.get('matches', 0)} IPL matches\n\n"
                       f"• Avg 1st innings: {v.get('avg_first_innings', 0):.0f} runs\n"
                       f"• Chase wins: {chase_pct:.0f}% of matches\n"
                       f"• Phase RPO: PP {v.get('phase_rpo', {}).get('powerplay', 0):.1f} · "
                       f"Mid {v.get('phase_rpo', {}).get('middle', 0):.1f} · "
                       f"Death {v.get('phase_rpo', {}).get('death', 0):.1f}"),
            "confidence": "high",
            "details": {"venue": v, "chase": vc},
        }

    # 10. How many balls left / overs left
    if ("balls" in q and ("left" in q or "remaining" in q)) or \
       ("overs" in q and ("left" in q or "remaining" in q)):
        bl = 120 - state.balls_bowled
        return {
            "answer": f"**{bl} balls** ({bl//6} overs {bl%6} balls) remaining",
            "confidence": "high",
            "details": {"balls_left": bl},
        }

    # Default — show help
    return {
        "answer": _help_text(),
        "confidence": "low",
        "details": None,
    }


def _help_text() -> str:
    return (
        "I can answer questions like:\n\n"
        "• *What's the probability of 10+ runs in the next over?*\n"
        "• *Chance of a wicket in the next 6 balls?*\n"
        "• *What's the projected final score?*\n"
        "• *What's the win probability?*\n"
        "• *Probability of a boundary in next 12 balls?*\n"
        "• *Required run rate?*\n"
        "• *How many balls left?*\n"
        "• *Tell me about this venue*"
    )
