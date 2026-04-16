"""
main.py - Cricket Betting Model: Full Pipeline
================================================
Run this to:
1. Load/generate ball-by-ball data
2. Engineer features
3. Train the ball-outcome model
4. Run Monte Carlo simulations
5. Detect value vs bookmaker lines

Usage:
    python main.py              # Full pipeline with synthetic data
    python main.py --data-dir /path/to/cricsheet/csvs   # Use real data

For real data, download from https://cricsheet.org/downloads/
- T20 internationals: t20s_csv2.zip
- IPL: ipl_csv2.zip
- BBL: bbl_csv2.zip
Extract CSVs into your data directory.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import CricketDataLoader
from feature_engine import FeatureEngine
from model import BallOutcomeModel
from simulator import MonteCarloSimulator, MatchState


def print_banner(text: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_pipeline(data_dir: str = "data"):
    """Execute the full pipeline."""

    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print_banner("STEP 1: LOADING DATA")

    loader = CricketDataLoader(data_dir=data_dir)
    raw = loader.load_all_matches()
    clean = loader.clean_data(raw)
    context = loader.compute_match_context(clean)
    player_stats = loader.get_player_stats(context)

    # Quick data summary
    print(f"\nData Summary:")
    print(f"  Total deliveries: {len(context):,}")
    print(f"  Total matches:    {context['match_id'].nunique()}")
    print(f"  Venues:           {context['venue'].nunique()}")
    print(f"  Batters:          {context['striker'].nunique()}")
    print(f"  Bowlers:          {context['bowler'].nunique()}")
    print(f"  Date range:       {context['start_date'].min()} to {context['start_date'].max()}")

    # Outcome distribution
    print(f"\n  Outcome distribution (per ball):")
    print(f"    Dot balls:  {context['is_dot'].mean():.1%}")
    print(f"    Singles:    {(context['runs_off_bat'] == 1).mean():.1%}")
    print(f"    Fours:      {context['is_four'].mean():.1%}")
    print(f"    Sixes:      {context['is_six'].mean():.1%}")
    print(f"    Wickets:    {context['is_wicket'].mean():.1%}")

    # Phase-wise run rates
    print(f"\n  Average runs per over by phase:")
    for phase in ['powerplay', 'middle', 'death']:
        phase_data = context[context['phase'] == phase]
        rpo = phase_data['total_runs'].sum() / (len(phase_data) / 6)
        print(f"    {phase:>10s}: {rpo:.1f}")

    # ============================================================
    # STEP 2: Feature Engineering
    # ============================================================
    print_banner("STEP 2: FEATURE ENGINEERING")

    engine = FeatureEngine()
    X, y = engine.build_features(context)

    print(f"\nFeature matrix:  {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Target classes:  {sorted(y.unique().tolist())}")
    print(f"Target distribution:")
    outcome_labels = ['dot', 'single', 'double', 'triple',
                      'four', 'six', 'wicket', 'extra']
    for code, label in enumerate(outcome_labels):
        count = (y == code).sum()
        pct = count / len(y) * 100
        bar = '█' * int(pct * 2)
        print(f"  {label:>8s} ({code}): {count:>7,} ({pct:5.1f}%) {bar}")

    # ============================================================
    # STEP 3: Train Model
    # ============================================================
    print_banner("STEP 3: TRAINING BALL-OUTCOME MODEL")

    ball_model = BallOutcomeModel(model_dir="models")
    ball_model.train(X, y)

    # ============================================================
    # STEP 4: Monte Carlo Simulations
    # ============================================================
    print_banner("STEP 4: MONTE CARLO SIMULATIONS")

    n_sims = 300
    sim = MonteCarloSimulator(ball_model, engine, n_simulations=n_sims)
    print(f"Running {n_sims} simulations per scenario...\n")

    # --- Scenario A: Powerplay, strong start ---
    print("-" * 50)
    print("SCENARIO A: Powerplay — Strong start (52/0 in 6 overs)")
    print("-" * 50)

    state_a = MatchState(
        innings=1, over_num=6, ball_in_over=1,
        runs_scored=52, wickets_fallen=0, balls_bowled=36,
        venue_avg_rpb=1.35,
        striker="India_Bat1", non_striker="India_Bat2",
        bowler="Australia_Bowl3",
        batters_remaining=[f"India_Bat{i}" for i in range(3, 12)],
        recent_balls=[1, 4, 0, 6, 1, 2, 0, 1, 4, 1, 0, 1]
    )

    over_a = sim.simulate_over(state_a)
    print(f"  Over 7 projection: {over_a['mean']:.1f} runs "
          f"(median: {over_a['median']:.0f}, std: {over_a['std']:.1f})")

    # --- Scenario B: Middle overs, under pressure ---
    print(f"\n{'-' * 50}")
    print("SCENARIO B: Middle overs — Struggling (68/4 in 12 overs)")
    print("-" * 50)

    state_b = MatchState(
        innings=1, over_num=12, ball_in_over=1,
        runs_scored=68, wickets_fallen=4, balls_bowled=72,
        venue_avg_rpb=1.25,
        striker="India_Bat5", non_striker="India_Bat6",
        bowler="Australia_Bowl4",
        batters_remaining=[f"India_Bat{i}" for i in range(7, 12)],
        recent_balls=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    )

    over_b = sim.simulate_over(state_b)
    print(f"  Over 13 projection: {over_b['mean']:.1f} runs "
          f"(median: {over_b['median']:.0f})")

    # --- Scenario C: Death overs chase ---
    print(f"\n{'-' * 50}")
    print("SCENARIO C: Chase — Need 42 off 18 balls, 3 wickets down")
    print("-" * 50)

    state_c = MatchState(
        innings=2, over_num=17, ball_in_over=1,
        runs_scored=139, wickets_fallen=3, balls_bowled=102,
        target=181, venue_avg_rpb=1.35,
        striker="England_Bat4", non_striker="England_Bat5",
        bowler="India_Bowl5",
        batters_remaining=[f"England_Bat{i}" for i in range(6, 12)],
        recent_balls=[4, 1, 6, 0, 2, 1, 1, 4, 0, 1, 2, 1]
    )

    over_c = sim.simulate_over(state_c)
    print(f"  Over 18 projection: {over_c['mean']:.1f} runs")

    # ============================================================
    # STEP 5: Value Detection
    # ============================================================
    print_banner("STEP 5: VALUE DETECTION VS BOOKMAKER")

    # Simulated bookmaker lines for each scenario
    test_cases = [
        ("Scenario A - Over 7", state_a, 7.5, 1.85, 1.95),
        ("Scenario B - Over 13", state_b, 5.5, 1.90, 1.90),
        ("Scenario C - Over 18", state_c, 9.5, 1.80, 2.00),
    ]

    for label, state, line, odds_over, odds_under in test_cases:
        value = sim.find_value_vs_bookmaker(
            state, line, odds_over, odds_under, "over_runs"
        )
        print(f"\n  {label}")
        print(f"    Bookmaker: O/U {line} | Over@{odds_over} Under@{odds_under}")
        print(f"    Model mean: {value['model_mean']:.1f} runs")
        print(f"    Model P(over):  {value['model_prob_over']:.1%} vs "
              f"implied {value['implied_prob_over']:.1%} "
              f"| edge: {value['edge_over']:+.1%}")
        print(f"    Model P(under): {value['model_prob_under']:.1%} vs "
              f"implied {value['implied_prob_under']:.1%} "
              f"| edge: {value['edge_under']:+.1%}")
        print(f"    >>> {value['recommendation']} "
              f"(confidence: {value['confidence']})")
        if value['quarter_kelly_stake'] > 0:
            print(f"    >>> Stake: {value['quarter_kelly_stake']:.2%} of bankroll")

    # ============================================================
    # STEP 6: Generate Visualizations
    # ============================================================
    print_banner("STEP 6: GENERATING VISUALIZATIONS")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Over runs distribution (Scenario A - Powerplay)
    axes[0].hist(over_a['distribution'], bins=range(0, 25),
                     color='steelblue', edgecolor='white', alpha=0.8, density=True)
    axes[0].axvline(over_a['mean'], color='red', linestyle='--',
                        label=f'Mean: {over_a["mean"]:.1f}')
    axes[0].set_title('Over Runs: Powerplay\n52/0 after 6 overs')
    axes[0].set_xlabel('Runs in Over')
    axes[0].set_ylabel('Probability')
    axes[0].legend()

    # Plot 2: Over runs distribution (Scenario B - Middle)
    axes[1].hist(over_b['distribution'], bins=range(0, 25),
                     color='forestgreen', edgecolor='white', alpha=0.8, density=True)
    axes[1].axvline(over_b['mean'], color='red', linestyle='--',
                        label=f'Mean: {over_b["mean"]:.1f}')
    axes[1].set_title('Over Runs: Middle Overs\n68/4 after 12 overs')
    axes[1].set_xlabel('Runs in Over')
    axes[1].legend()

    # Plot 3: Over runs distribution (Scenario C - Death chase)
    axes[2].hist(over_c['distribution'], bins=range(0, 30),
                     color='darkorange', edgecolor='white', alpha=0.8, density=True)
    axes[2].axvline(over_c['mean'], color='red', linestyle='--',
                        label=f'Mean: {over_c["mean"]:.1f}')
    axes[2].set_title('Over Runs: Death Chase\nNeed 42 off 18, 3 down')
    axes[2].set_xlabel('Runs in Over')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('models/simulation_results.png', dpi=150)
    plt.close()
    print("  Saved: models/simulation_results.png")

    # ============================================================
    # SUMMARY
    # ============================================================
    print_banner("PIPELINE COMPLETE")
    print("""
    Files created:
      models/ball_outcome_model.joblib  - Trained XGBoost model
      models/feature_importance.png     - Feature importance chart
      models/confusion_matrix.png       - Model confusion matrix
      models/simulation_results.png     - Simulation visualizations

    Next steps:
      1. Download real CricSheet data and retrain
      2. Add live data feed integration
      3. Build odds comparison engine
      4. Paper-trade against live bookmaker lines
      5. Track model accuracy over time (calibration)

    To use with real CricSheet data:
      python main.py --data-dir /path/to/cricsheet/csvs
    """)


if __name__ == "__main__":
    data_dir = "data"

    # Check for command-line arg
    if len(sys.argv) > 2 and sys.argv[1] == "--data-dir":
        data_dir = sys.argv[2]

    start = time.time()
    run_pipeline(data_dir=data_dir)
    elapsed = time.time() - start
    print(f"Total runtime: {elapsed:.1f} seconds")
