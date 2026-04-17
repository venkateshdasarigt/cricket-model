"""
evaluate.py — Backtest predictions against real matches
========================================================
Replays historical IPL matches, predicting at each over boundary,
and measures how close our predictions are to what actually happened.

Metrics:
  • Next-over MAE (mean absolute error in runs)
  • Next-over accuracy within ±2 runs
  • Innings-total MAE (from various game states)
  • Match-winner accuracy (from midway point)

Usage:
    python evaluate.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

from data_loader import CricketDataLoader
from feature_engine import FeatureEngine
from model import BallOutcomeModel
from simulator import MonteCarloSimulator, MatchState
from match_intel import MatchIntelligence


def evaluate(n_test_matches: int = 30, n_sims: int = 50):
    """Run the backtest on the last N matches in the dataset."""
    print("Loading model + intel…")
    m = BallOutcomeModel(model_dir="models")
    m.load("ball_outcome_model.joblib")
    sim = MonteCarloSimulator(model=m, feature_engine=FeatureEngine(),
                               n_simulations=n_sims)
    intel = MatchIntelligence.load_or_build("data")

    # Load the last N match CSVs (by date)
    files = sorted(
        [f for f in glob.glob("data/*.csv") if "_info" not in f],
        key=os.path.getmtime, reverse=True,
    )[:n_test_matches * 2]  # some may be non-ball-by-ball

    loader = CricketDataLoader()
    over_errors = []        # |predicted - actual| per over
    over_within_2 = []      # 1 if within ±2, else 0
    total_errors = []       # |predicted_total - actual_total| at over 10
    winner_correct = []     # 1 if predicted winner matches at over 10

    # Baseline: simple CRR extrapolation for comparison
    baseline_total_errors = []
    baseline_over_errors = []

    matches_evaluated = 0

    for csv_path in files:
        if matches_evaluated >= n_test_matches:
            break
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if "ball" not in df.columns:
                continue
            df = loader.clean_data(df)
        except Exception:
            continue

        match_id = df["match_id"].iloc[0]
        venue = str(df["venue"].iloc[0]) if "venue" in df.columns else ""
        innings_data = {}

        for inn in (1, 2):
            inn_df = df[df["innings"] == inn].copy()
            if inn_df.empty:
                continue

            # Build over-by-over actual runs
            inn_df["over_idx"] = inn_df["over_num"]
            over_runs = inn_df.groupby("over_idx")["total_runs"].sum()
            actual_total = int(inn_df["total_runs"].sum())
            innings_data[inn] = {
                "over_runs": over_runs.to_dict(),
                "actual_total": actual_total,
            }

        if 1 not in innings_data or 2 not in innings_data:
            continue

        target = innings_data[1]["actual_total"] + 1

        # Evaluate predictions at over boundaries for INNINGS 1
        for eval_over in range(2, 18, 2):  # predict at over 2, 4, 6, ..., 16
            # Build state from actual data up to this over
            overs_done = {k: v for k, v in innings_data[1]["over_runs"].items()
                          if k < eval_over}
            if not overs_done:
                continue
            runs_so_far = sum(overs_done.values())
            wkts_so_far = int(
                df[(df["innings"] == 1) & (df["over_num"] < eval_over)
                   ]["is_wicket"].sum())
            balls_done = eval_over * 6

            state = MatchState(
                innings=1, over_num=eval_over, ball_in_over=1,
                runs_scored=runs_so_far, wickets_fallen=wkts_so_far,
                balls_bowled=balls_done,
                striker="?", non_striker="?", bowler="?",
            )

            # Predict next over
            try:
                ov_pred = sim.realistic_next_over(state, intel=intel, n_sims=n_sims)
                actual_next = innings_data[1]["over_runs"].get(eval_over, 0)
                err = abs(ov_pred["mean"] - actual_next)
                over_errors.append(err)
                over_within_2.append(1 if err <= 2 else 0)
            except Exception:
                pass

            # Baseline: CRR * remaining overs
            crr = runs_so_far * 6 / max(balls_done, 1)
            baseline_next = crr  # simple: predict CRR for next over
            actual_next_base = innings_data[1]["over_runs"].get(eval_over, 0)
            baseline_over_errors.append(abs(baseline_next - actual_next_base))

            # Predict total (only at over 10)
            if eval_over == 10:
                try:
                    tot_pred = sim.realistic_projected_total(state, intel=intel)
                    total_err = abs(tot_pred["mean"] - innings_data[1]["actual_total"])
                    total_errors.append(total_err)
                    # Baseline total = CRR * 20
                    baseline_total = crr * 20
                    baseline_total_errors.append(
                        abs(baseline_total - innings_data[1]["actual_total"]))
                except Exception:
                    pass

        # Evaluate MATCH WINNER prediction at over 10 of innings 2
        runs_inn2_at_10 = sum(v for k, v in innings_data[2]["over_runs"].items()
                               if k < 10)
        wkts_inn2_at_10 = int(
            df[(df["innings"] == 2) & (df["over_num"] < 10)]["is_wicket"].sum())

        state2 = MatchState(
            innings=2, over_num=10, ball_in_over=1,
            runs_scored=runs_inn2_at_10, wickets_fallen=wkts_inn2_at_10,
            balls_bowled=60, target=target,
            striker="?", non_striker="?", bowler="?",
        )
        try:
            chase_pred = sim.realistic_projected_total(state2, intel=intel)
            predicted_chase_wins = chase_pred.get("win_prob", 0) > 0.5
            actual_chase_wins = (innings_data[2]["actual_total"] >= target)
            winner_correct.append(1 if predicted_chase_wins == actual_chase_wins else 0)
        except Exception:
            pass

        matches_evaluated += 1
        if matches_evaluated % 5 == 0:
            print(f"  Evaluated {matches_evaluated}/{n_test_matches} matches…")

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Matches evaluated: {matches_evaluated}")
    print(f"  Over predictions tested: {len(over_errors)}")
    print()

    if over_errors:
        mae = np.mean(over_errors)
        within_2 = np.mean(over_within_2) * 100
        within_3 = np.mean([1 if e <= 3 else 0 for e in over_errors]) * 100
        print(f"  NEXT-OVER PREDICTION:")
        print(f"    Mean Absolute Error: {mae:.2f} runs")
        print(f"    Accuracy within ±2 runs: {within_2:.1f}%")
        print(f"    Accuracy within ±3 runs: {within_3:.1f}%")
        print(f"    Median error: {np.median(over_errors):.2f} runs")

    if total_errors:
        print(f"\n  INNINGS TOTAL PREDICTION (at over 10):")
        print(f"    Mean Absolute Error: {np.mean(total_errors):.1f} runs")
        print(f"    Accuracy within ±15 runs: "
              f"{np.mean([1 if e <= 15 else 0 for e in total_errors]) * 100:.1f}%")
        print(f"    Accuracy within ±20 runs: "
              f"{np.mean([1 if e <= 20 else 0 for e in total_errors]) * 100:.1f}%")

    if winner_correct:
        print(f"\n  MATCH WINNER PREDICTION (at over 10 of chase):")
        print(f"    Accuracy: {np.mean(winner_correct) * 100:.1f}%")
        print(f"    Correct: {sum(winner_correct)}/{len(winner_correct)}")

    if baseline_over_errors:
        print(f"\n  BASELINE COMPARISON (simple CRR extrapolation):")
        print(f"    Baseline over MAE: {np.mean(baseline_over_errors):.2f} "
              f"vs Model: {np.mean(over_errors):.2f} runs "
              f"({'better' if np.mean(over_errors) < np.mean(baseline_over_errors) else 'worse'})")
    if baseline_total_errors:
        print(f"    Baseline total MAE: {np.mean(baseline_total_errors):.1f} "
              f"vs Model: {np.mean(total_errors):.1f} runs "
              f"({'better' if np.mean(total_errors) < np.mean(baseline_total_errors) else 'worse'})")

    print()
    print("=" * 70)
    return {
        "over_mae": float(np.mean(over_errors)) if over_errors else None,
        "over_within_2": float(np.mean(over_within_2) * 100) if over_within_2 else None,
        "total_mae": float(np.mean(total_errors)) if total_errors else None,
        "winner_accuracy": float(np.mean(winner_correct) * 100) if winner_correct else None,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    evaluate(n_test_matches=n, n_sims=50)
