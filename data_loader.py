"""
data_loader.py - CricSheet Ball-by-Ball Data Loader & Parser
=============================================================
Handles loading, parsing, and cleaning ball-by-ball cricket data
from CricSheet CSV format (https://cricsheet.org/downloads/).

Download T20 CSVs from: https://cricsheet.org/downloads/t20s_csv2.zip
Extract into the /data folder.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional


class CricketDataLoader:
    """Load and parse CricSheet ball-by-ball CSV data."""

    # Standard column mapping for CricSheet CSV2 format
    REQUIRED_COLS = [
        'match_id', 'season', 'start_date', 'venue', 'innings',
        'ball', 'batting_team', 'bowling_team', 'striker', 'non_striker',
        'bowler', 'runs_off_bat', 'extras', 'wides', 'noballs',
        'byes', 'legbyes', 'penalty', 'wicket_type', 'player_dismissed'
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_data: Optional[pd.DataFrame] = None
        self.match_info: Optional[pd.DataFrame] = None

    def load_all_matches(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """Load all CSV match files from data directory."""
        csv_files = glob.glob(os.path.join(self.data_dir, file_pattern))

        if not csv_files:
            print(f"No CSV files found in {self.data_dir}/")
            print("Download from: https://cricsheet.org/downloads/t20s_csv2.zip")
            print("\nGenerating synthetic data for demo purposes...\n")
            return self._generate_synthetic_data()

        print(f"Found {len(csv_files)} match files. Loading...")

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"  Skipping {f}: {e}")

        self.raw_data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.raw_data):,} deliveries from {self.raw_data['match_id'].nunique()} matches")
        return self.raw_data

    def _generate_synthetic_data(self, n_matches: int = 500) -> pd.DataFrame:
        """
        Generate realistic synthetic T20 ball-by-ball data for demo/testing.
        This lets you run the full pipeline without real data.
        """
        np.random.seed(42)
        rows = []

        teams = ['India', 'Australia', 'England', 'South Africa',
                 'New Zealand', 'Pakistan', 'West Indies', 'Sri Lanka']
        venues = ['Mumbai', 'Melbourne', 'Lords', 'Dubai', 'Colombo',
                  'Johannesburg', 'Wellington', 'Lahore', 'Sydney', 'Delhi']

        # Player pools per team (synthetic names)
        batters_pool = {t: [f"{t}_Bat{i}" for i in range(1, 12)] for t in teams}
        bowlers_pool = {t: [f"{t}_Bowl{i}" for i in range(1, 8)] for t in teams}

        for match_id in range(1, n_matches + 1):
            t1, t2 = np.random.choice(teams, 2, replace=False)
            venue = np.random.choice(venues)
            season = np.random.choice(range(2018, 2025))
            date = f"{season}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"

            for innings in [1, 2]:
                bat_team = t1 if innings == 1 else t2
                bowl_team = t2 if innings == 1 else t1

                batting_order = list(batters_pool[bat_team])
                bowlers = list(bowlers_pool[bowl_team])
                np.random.shuffle(batting_order)

                bat_idx = 0  # next batter index
                striker = batting_order[0]
                non_striker = batting_order[1]
                bat_idx = 2
                wickets_fallen = 0

                for over in range(20):
                    bowler = bowlers[over % len(bowlers)]

                    # Phase-based run distribution
                    if over < 6:        # powerplay
                        run_probs = [0.28, 0.28, 0.08, 0.02, 0.20, 0.00, 0.14]
                        wicket_prob = 0.04
                    elif over < 15:     # middle overs
                        run_probs = [0.35, 0.30, 0.08, 0.02, 0.15, 0.00, 0.10]
                        wicket_prob = 0.05
                    else:               # death overs
                        run_probs = [0.22, 0.20, 0.08, 0.03, 0.22, 0.02, 0.23]
                        wicket_prob = 0.06

                    for ball_num in range(1, 7):
                        if wickets_fallen >= 10:
                            break

                        ball_label = float(f"{over}.{ball_num}")

                        # Check for extras (wide/noball)
                        wide = 1 if np.random.random() < 0.04 else 0
                        noball = 1 if (not wide and np.random.random() < 0.02) else 0

                        # Determine runs off bat
                        run_outcomes = [0, 1, 2, 3, 4, 5, 6]
                        runs = np.random.choice(run_outcomes, p=run_probs)

                        # Check for wicket (not on wide)
                        wicket_type = None
                        dismissed = None
                        if not wide and np.random.random() < wicket_prob:
                            wicket_type = np.random.choice(
                                ['bowled', 'caught', 'lbw', 'run out',
                                 'stumped', 'caught and bowled'],
                                p=[0.18, 0.45, 0.15, 0.10, 0.07, 0.05]
                            )
                            dismissed = striker
                            wickets_fallen += 1
                            runs = 0  # typically 0 on dismissal

                            if bat_idx < len(batting_order):
                                striker = batting_order[bat_idx]
                                bat_idx += 1

                        extras = wide + noball
                        total_runs = runs + extras

                        rows.append({
                            'match_id': match_id,
                            'season': str(season),
                            'start_date': date,
                            'venue': venue,
                            'innings': innings,
                            'ball': ball_label,
                            'batting_team': bat_team,
                            'bowling_team': bowl_team,
                            'striker': striker,
                            'non_striker': non_striker,
                            'bowler': bowler,
                            'runs_off_bat': runs,
                            'extras': extras,
                            'wides': wide,
                            'noballs': noball,
                            'byes': 0,
                            'legbyes': 0,
                            'penalty': 0,
                            'wicket_type': wicket_type if wicket_type else '',
                            'player_dismissed': dismissed if dismissed else ''
                        })

                        # Rotate strike on odd runs
                        if runs % 2 == 1:
                            striker, non_striker = non_striker, striker

                    # Rotate strike at end of over
                    striker, non_striker = non_striker, striker

                    if wickets_fallen >= 10:
                        break

        self.raw_data = pd.DataFrame(rows)
        print(f"Generated {len(self.raw_data):,} synthetic deliveries "
              f"from {n_matches} matches")
        return self.raw_data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and standardize the loaded data."""
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()

        # Fill NaN in numeric columns with 0
        numeric_cols = ['runs_off_bat', 'extras', 'wides', 'noballs',
                        'byes', 'legbyes', 'penalty']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Total runs for each delivery
        df['total_runs'] = df['runs_off_bat'] + df['extras']

        # Is wicket flag
        df['is_wicket'] = (df['wicket_type'].notna() &
                           (df['wicket_type'] != '')).astype(int)

        # Extract over number (integer part of ball column)
        df['over_num'] = df['ball'].apply(lambda x: int(float(x)))

        # Ball within over
        df['ball_in_over'] = df['ball'].apply(
            lambda x: int(round((float(x) - int(float(x))) * 10))
        )

        # Is boundary
        df['is_four'] = (df['runs_off_bat'] == 4).astype(int)
        df['is_six'] = (df['runs_off_bat'] == 6).astype(int)
        df['is_boundary'] = ((df['runs_off_bat'] == 4) |
                              (df['runs_off_bat'] == 6)).astype(int)

        # Is dot ball
        df['is_dot'] = ((df['runs_off_bat'] == 0) &
                         (df['wides'] == 0) &
                         (df['noballs'] == 0) &
                         (df['is_wicket'] == 0)).astype(int)

        # Phase of innings
        df['phase'] = pd.cut(
            df['over_num'],
            bins=[-1, 5, 14, 20],
            labels=['powerplay', 'middle', 'death']
        )

        print(f"Cleaned data: {len(df):,} deliveries, "
              f"{df['match_id'].nunique()} matches")
        return df

    def compute_match_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cumulative match context features:
        - Cumulative runs, wickets, run rate at each ball
        - Partnership runs
        - Required run rate (for 2nd innings)
        """
        df = df.sort_values(['match_id', 'innings', 'ball']).copy()

        # Cumulative runs in innings
        df['cum_runs'] = df.groupby(['match_id', 'innings'])['total_runs'].cumsum()

        # Cumulative wickets
        df['cum_wickets'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()

        # Balls bowled so far (approximate from ball column)
        df['balls_bowled'] = df.groupby(['match_id', 'innings']).cumcount() + 1

        # Current run rate
        df['current_rr'] = df['cum_runs'] / (df['balls_bowled'] / 6)
        df['current_rr'] = df['current_rr'].replace([np.inf, -np.inf], 0)

        # First innings total (for second innings context)
        first_innings_totals = (
            df[df['innings'] == 1]
            .groupby('match_id')['total_runs']
            .sum()
            .reset_index()
            .rename(columns={'total_runs': 'target'})
        )
        first_innings_totals['target'] += 1  # need to beat it

        df = df.merge(first_innings_totals, on='match_id', how='left')

        # Runs still needed (2nd innings only)
        df['runs_needed'] = np.where(
            df['innings'] == 2,
            df['target'] - df['cum_runs'],
            np.nan
        )

        # Balls remaining
        df['balls_remaining'] = np.where(
            df['innings'] == 2,
            120 - df['balls_bowled'],
            120 - df['balls_bowled']
        )
        df['balls_remaining'] = df['balls_remaining'].clip(lower=1)

        # Required run rate (2nd innings)
        df['required_rr'] = np.where(
            df['innings'] == 2,
            (df['runs_needed'] / (df['balls_remaining'] / 6)),
            np.nan
        )

        # Pressure index = required_rr - current_rr (higher = more pressure)
        df['pressure_index'] = np.where(
            df['innings'] == 2,
            df['required_rr'] - df['current_rr'],
            0
        )

        print(f"Added match context features. Shape: {df.shape}")
        return df

    def get_player_stats(self, df: pd.DataFrame) -> dict:
        """
        Compute per-player career stats broken down by phase.
        Returns dict of DataFrames for batters and bowlers.
        """
        # --- Batter stats by phase ---
        batter_stats = (
            df.groupby(['striker', 'phase'])
            .agg(
                balls_faced=('runs_off_bat', 'count'),
                runs_scored=('runs_off_bat', 'sum'),
                dismissals=('is_wicket', 'sum'),
                fours=('is_four', 'sum'),
                sixes=('is_six', 'sum'),
                dots=('is_dot', 'sum')
            )
            .reset_index()
        )
        batter_stats['strike_rate'] = (
            batter_stats['runs_scored'] / batter_stats['balls_faced'] * 100
        )
        batter_stats['dot_pct'] = (
            batter_stats['dots'] / batter_stats['balls_faced'] * 100
        )
        batter_stats['boundary_pct'] = (
            (batter_stats['fours'] + batter_stats['sixes'])
            / batter_stats['balls_faced'] * 100
        )
        batter_stats['dismissal_rate'] = (
            batter_stats['dismissals'] / batter_stats['balls_faced']
        )

        # --- Bowler stats by phase ---
        bowler_stats = (
            df.groupby(['bowler', 'phase'])
            .agg(
                balls_bowled=('runs_off_bat', 'count'),
                runs_conceded=('total_runs', 'sum'),
                wickets_taken=('is_wicket', 'sum'),
                dots_bowled=('is_dot', 'sum'),
                fours_conceded=('is_four', 'sum'),
                sixes_conceded=('is_six', 'sum')
            )
            .reset_index()
        )
        bowler_stats['economy'] = (
            bowler_stats['runs_conceded'] / (bowler_stats['balls_bowled'] / 6)
        )
        bowler_stats['dot_pct'] = (
            bowler_stats['dots_bowled'] / bowler_stats['balls_bowled'] * 100
        )
        bowler_stats['wicket_rate'] = (
            bowler_stats['wickets_taken'] / bowler_stats['balls_bowled']
        )

        print(f"Computed stats for {batter_stats['striker'].nunique()} batters "
              f"and {bowler_stats['bowler'].nunique()} bowlers")

        return {
            'batter_stats': batter_stats,
            'bowler_stats': bowler_stats
        }


# ----- Quick test -----
if __name__ == "__main__":
    loader = CricketDataLoader(data_dir="data")
    raw = loader.load_all_matches()
    clean = loader.clean_data(raw)
    with_context = loader.compute_match_context(clean)
    stats = loader.get_player_stats(with_context)

    print("\n--- Sample Data ---")
    print(with_context[['match_id', 'innings', 'ball', 'striker', 'bowler',
                         'runs_off_bat', 'total_runs', 'is_wicket', 'phase',
                         'cum_runs', 'cum_wickets', 'current_rr']].head(20))

    print("\n--- Top Batters (Powerplay SR) ---")
    bs = stats['batter_stats']
    top = bs[(bs['phase'] == 'powerplay') & (bs['balls_faced'] >= 50)]
    print(top.nlargest(10, 'strike_rate')[
        ['striker', 'balls_faced', 'strike_rate', 'boundary_pct']
    ].to_string(index=False))
