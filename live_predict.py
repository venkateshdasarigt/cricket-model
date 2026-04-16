"""
live_predict.py
---------------
Run live (or replay) AI predictions on a cricket match.

For every new ball that arrives from a fetcher, this script:
  1. Updates an in-memory MatchState
  2. Runs Monte Carlo simulations to project:
       - Runs in the current/next over
       - Final innings total
       - Win probability (innings 2 only)
  3. Prints a rolling, terminal-friendly snapshot

Usage
-----
# Replay any IPL match from your CricSheet data folder (no API needed):
python live_predict.py --replay data/1370352.csv --speed 0.05

# Use real Cricbuzz live (after RAPIDAPI_KEY env var is set):
python live_predict.py --cricbuzz <matchId>

# Use cricketdata.org (after CRICAPI_KEY env var is set):
python live_predict.py --cricapi <matchId>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from feature_engine import FeatureEngine
from model import BallOutcomeModel
from simulator import MonteCarloSimulator, MatchState
from live_fetcher import (
    LiveFetcher,
    CricSheetReplayFetcher,
    CricAPIFetcher,
    CricbuzzRapidAPIFetcher,
    BallEvent,
)


MODEL_PATH = "models/ball_outcome_model.joblib"

# How often to re-run the simulator. Running it after every single
# ball is overkill; once per over is informative and fast.
SIM_EVERY_N_BALLS = 6


# ---------------------------------------------------------------------------
# State updater — converts BallEvents into the MatchState the simulator wants
# ---------------------------------------------------------------------------

class LiveStateTracker:
    """Maintains a MatchState that mirrors what's actually happening on field."""

    def __init__(self):
        self.state: Optional[MatchState] = None
        self.current_innings = 0
        # Per-innings recent history (used by simulator for momentum features)
        self.recent_balls: list[int] = []
        # Partnership tracking
        self.partnership_runs: int = 0
        self.partnership_balls: int = 0
        self.partnership_wickets_at_start: int = 0
        # Batter/bowler running counts (per innings)
        self.batter_runs: dict[str, int] = {}
        self.batter_balls: dict[str, int] = {}
        self.bowler_runs: dict[str, int] = {}
        self.bowler_balls: dict[str, int] = {}
        self.bowler_wickets: dict[str, int] = {}
        # Order of dismissed batters (for "remaining batters" UI)
        self.dismissed: list[str] = []
        self.batters_seen: list[str] = []   # appearance order

    def _reset_partnership(self):
        self.partnership_runs = 0
        self.partnership_balls = 0
        self.partnership_wickets_at_start = self.state.wickets_fallen if self.state else 0

    def apply(self, e: BallEvent) -> MatchState:
        # New innings → full reset
        if e.innings != self.current_innings:
            self.current_innings = e.innings
            self.recent_balls = []
            self.batter_runs.clear()
            self.batter_balls.clear()
            self.bowler_runs.clear()
            self.bowler_balls.clear()
            self.bowler_wickets.clear()
            self.dismissed = []
            self.batters_seen = []
            self.state = MatchState(
                innings=e.innings,
                over_num=e.over,
                ball_in_over=e.ball_in_over,
                runs_scored=0,
                wickets_fallen=0,
                balls_bowled=0,
                target=e.target,
                striker=e.striker,
                non_striker=e.non_striker,
                bowler=e.bowler,
            )
            self.partnership_runs = 0
            self.partnership_balls = 0
            self.partnership_wickets_at_start = 0

        s = self.state
        s.runs_scored += e.total_runs
        s.recent_balls = self.recent_balls  # share list

        is_legal = (e.wides == 0 and e.noballs == 0)
        if is_legal:
            s.balls_bowled += 1
            s.over_num = e.over
            s.ball_in_over = e.ball_in_over

        # Wicket — partnership ends, dismissed batter recorded
        if e.is_wicket:
            s.wickets_fallen += 1
            if e.player_dismissed:
                self.dismissed.append(e.player_dismissed)
            elif s.striker:
                self.dismissed.append(s.striker)
            self._reset_partnership()
        else:
            # Add to current partnership
            self.partnership_runs += e.total_runs
            if is_legal:
                self.partnership_balls += 1

        # Per-batter / per-bowler counters
        if e.striker:
            self.batter_runs[e.striker] = self.batter_runs.get(e.striker, 0) + e.runs_off_bat
            if is_legal:
                self.batter_balls[e.striker] = self.batter_balls.get(e.striker, 0) + 1
            if e.striker not in self.batters_seen:
                self.batters_seen.append(e.striker)
        if e.bowler:
            self.bowler_runs[e.bowler] = self.bowler_runs.get(e.bowler, 0) + e.total_runs
            if is_legal:
                self.bowler_balls[e.bowler] = self.bowler_balls.get(e.bowler, 0) + 1
            if e.is_wicket:
                self.bowler_wickets[e.bowler] = self.bowler_wickets.get(e.bowler, 0) + 1

        # Update on-field players from feed
        if e.striker:
            s.striker = e.striker
        if e.non_striker:
            s.non_striker = e.non_striker
        if e.bowler:
            s.bowler = e.bowler

        # Track last ~30 balls for momentum
        self.recent_balls.append(e.total_runs)
        if len(self.recent_balls) > 30:
            self.recent_balls.pop(0)

        if e.target is not None:
            s.target = e.target

        return s

    # ------------------------------------------------------------------
    # Convenience getters used by the dashboard
    # ------------------------------------------------------------------

    def partnership_summary(self) -> dict:
        balls = max(self.partnership_balls, 1)
        return {
            "runs": self.partnership_runs,
            "balls": self.partnership_balls,
            "rr": self.partnership_runs * 6 / balls,
        }

    def current_batter_card(self, name: str) -> dict:
        balls = self.batter_balls.get(name, 0)
        runs = self.batter_runs.get(name, 0)
        return {
            "runs": runs,
            "balls": balls,
            "sr": (runs * 100 / balls) if balls else 0.0,
        }

    def current_bowler_card(self, name: str) -> dict:
        balls = self.bowler_balls.get(name, 0)
        runs = self.bowler_runs.get(name, 0)
        wkts = self.bowler_wickets.get(name, 0)
        return {
            "overs": f"{balls//6}.{balls%6}",
            "runs": runs,
            "wickets": wkts,
            "economy": (runs * 6 / balls) if balls else 0.0,
        }


# ---------------------------------------------------------------------------
# Snapshot printer
# ---------------------------------------------------------------------------

def print_snapshot(e: BallEvent, state: MatchState,
                   over_pred: dict | None,
                   inn_pred: dict | None) -> None:
    """Print a compact terminal status line + (occasionally) full predictions."""

    delivery = (
        f"  Inn{e.innings} {e.over}.{e.ball_in_over}"
        f"  {e.bowler[:14]:>14}  ->  {e.striker[:14]:<14}"
        f"  {e.runs_off_bat}r"
        f"{'  W' if e.is_wicket else '   '}"
        f"{'  +'+str(e.extras) if e.extras else ''}"
    )
    score = (
        f"   {state.runs_scored}/{state.wickets_fallen}"
        f" ({state.balls_bowled//6}.{state.balls_bowled%6})"
        f"  CRR {state.current_rr:5.2f}"
    )
    if state.innings == 2 and state.target:
        score += f"  TGT {state.target}  RRR {state.required_rr:5.2f}"

    print(delivery + score)

    if over_pred is None or inn_pred is None:
        return

    print()
    print("    " + "─" * 72)
    print(f"    🤖 AI PREDICTION  (after over {state.balls_bowled//6})")
    print(f"       Next over runs:    {over_pred['mean']:5.2f}  "
          f"(median {over_pred['median']:.0f}, σ {over_pred['std']:.1f})")
    print(f"       P(over ≥ 7.5):     {over_pred['prob_over_7.5']*100:5.1f}%")
    print(f"       P(over ≥ 9.5):     {over_pred['prob_over_9.5']*100:5.1f}%")
    print(f"       Projected total:   {inn_pred['mean']:5.1f}  "
          f"(p10={inn_pred['percentiles'][10]:.0f}, "
          f"p90={inn_pred['percentiles'][90]:.0f})")
    if state.innings == 2 and 'win_prob' in inn_pred:
        print(f"       Win probability:   {inn_pred['win_prob']*100:5.1f}%")
    print("    " + "─" * 72)
    print()


# ---------------------------------------------------------------------------
# Innings simulation wrapper that also computes win prob (innings 2)
# ---------------------------------------------------------------------------

def project_innings(sim: MonteCarloSimulator, state: MatchState,
                    next_over_pred: dict | None = None) -> dict:
    """
    Hybrid forward projection:
      • AI model predicts the NEXT over (already-validated path)
      • Remaining overs extrapolated at a phase-appropriate rate that
        blends model momentum with realistic IPL averages.

    This avoids the momentum-feedback runaway in the original
    `simulate_innings`, which compounds high-runs streaks.
    """
    import numpy as np

    balls_done = state.balls_bowled
    overs_done = balls_done // 6
    overs_left = max(0, 20 - overs_done)

    # Realistic IPL per-over averages by phase (computed offline from
    # CricSheet IPL data: powerplay ~8.4, middle ~7.9, death ~10.6)
    PHASE_AVG = {"powerplay": 8.4, "middle": 7.9, "death": 10.6}

    if overs_left == 0:
        d = np.array([state.runs_scored])
        return {"mean": float(state.runs_scored), "median": float(state.runs_scored),
                "std": 0.0,
                "percentiles": {p: float(state.runs_scored) for p in [10, 25, 50, 75, 90]},
                "distribution": d}

    # Use the AI model for the immediate next over (this is the part it's good at)
    if next_over_pred is None:
        next_over_pred = sim.simulate_over(state)
    next_over_dist = next_over_pred["distribution"]

    # Extrapolate remaining overs using phase averages, weighted by current momentum
    extra_overs = overs_left - 1
    if extra_overs > 0:
        # Damp the model's hot prediction toward phase average (50/50 blend)
        ai_signal = float(next_over_pred["mean"])
        rest_per_over = []
        for o in range(overs_done + 1, 20):
            phase = ("powerplay" if o < 6
                     else "middle" if o < 15
                     else "death")
            blended = 0.5 * PHASE_AVG[phase] + 0.5 * ai_signal
            # Wickets-down penalty: each fallen wicket trims 5%
            blended *= (1 - 0.05 * min(state.wickets_fallen, 8))
            rest_per_over.append(blended)
        extrapolated = sum(rest_per_over)
    else:
        extrapolated = 0.0

    final_dist = state.runs_scored + next_over_dist + extrapolated

    result = {
        "mean": float(np.mean(final_dist)),
        "median": float(np.median(final_dist)),
        "std": float(np.std(final_dist)),
        "percentiles": {p: float(np.percentile(final_dist, p))
                        for p in [10, 25, 50, 75, 90]},
        "distribution": final_dist,
    }
    if state.innings == 2 and state.target is not None:
        result["win_prob"] = float(np.mean(final_dist >= state.target))
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(fetcher: LiveFetcher, n_sims: int = 200) -> None:
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"Model not found at {MODEL_PATH}. "
            "Train one first with: python main.py --data-dir data"
        )

    print("Loading model…", flush=True)
    model = BallOutcomeModel(model_dir="models")
    model.load("ball_outcome_model.joblib")

    fe = FeatureEngine()
    sim = MonteCarloSimulator(model=model, feature_engine=fe, n_simulations=n_sims)
    tracker = LiveStateTracker()

    print(f"Streaming balls…  (sims/ball={n_sims}, "
          f"predict every {SIM_EVERY_N_BALLS} balls)\n", flush=True)
    print("=" * 84)

    ball_counter = 0
    last_inn = 0

    for event in fetcher.iter_balls():
        if event.innings != last_inn:
            last_inn = event.innings
            print(f"\n{'='*84}\n  INNINGS {event.innings}  "
                  f"{event.batting_team} batting"
                  f"{' (target ' + str(event.target) + ')' if event.target else ''}"
                  f"\n{'='*84}\n")

        state = tracker.apply(event)
        ball_counter += 1

        run_predict = (
            ball_counter % SIM_EVERY_N_BALLS == 0
            or event.is_wicket
        )

        over_pred = inn_pred = None
        if run_predict:
            try:
                over_pred = sim.simulate_over(state)
                inn_pred = project_innings(sim, state, next_over_pred=over_pred)
            except Exception as exc:
                print(f"  [sim error: {exc}]")

        print_snapshot(event, state, over_pred, inn_pred)

    print("\n" + "=" * 84)
    print("  Match feed ended.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Live AI cricket predictor")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="Path to a CricSheet match CSV to replay")
    src.add_argument("--cricapi", help="cricketdata.org match id (needs CRICAPI_KEY)")
    src.add_argument("--cricbuzz", help="Cricbuzz match id (needs RAPIDAPI_KEY)")

    p.add_argument("--speed", type=float, default=0.05,
                   help="Delay between balls in replay mode (seconds). "
                        "0 = instant, 1.0 = realistic. Default 0.05.")
    p.add_argument("--sims", type=int, default=200,
                   help="Monte Carlo iterations per prediction. Default 200.")
    args = p.parse_args()

    if args.replay:
        fetcher = CricSheetReplayFetcher(args.replay, ball_delay_s=args.speed)
    elif args.cricapi:
        fetcher = CricAPIFetcher(args.cricapi)
    else:
        fetcher = CricbuzzRapidAPIFetcher(args.cricbuzz)

    try:
        run(fetcher, n_sims=args.sims)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
