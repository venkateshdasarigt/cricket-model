"""
feature_engine.py - Feature Engineering for Ball-Outcome Prediction
====================================================================
Transforms raw ball-by-ball data into ML-ready feature vectors.

Each row = one delivery. The target = outcome category (0,1,2,3,4,6,wicket,extra).
Features capture: phase, match situation, player ability, pressure, momentum.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class FeatureEngine:
    """Build ML features from cleaned ball-by-ball data."""

    # Outcome encoding for the ball-outcome model
    OUTCOME_MAP = {
        'dot': 0,       # 0 runs, no wicket, no extra
        'single': 1,    # 1 run
        'double': 2,    # 2 runs
        'triple': 3,    # 3 runs
        'four': 4,      # boundary 4
        'six': 5,       # boundary 6
        'wicket': 6,    # dismissal
        'extra': 7      # wide or noball
    }

    def __init__(self, recency_weight_half_life: int = 15):
        """
        Args:
            recency_weight_half_life: number of innings for exponential
                decay half-life when computing recent form.
        """
        self.half_life = recency_weight_half_life
        self.batter_career = None
        self.bowler_career = None

    def encode_ball_outcome(self, row: pd.Series) -> str:
        """Classify a delivery into an outcome category."""
        if row['wides'] > 0 or row['noballs'] > 0:
            return 'extra'
        if row['is_wicket'] == 1:
            return 'wicket'
        runs = row['runs_off_bat']
        if runs == 0:
            return 'dot'
        elif runs == 1:
            return 'single'
        elif runs == 2:
            return 'double'
        elif runs == 3:
            return 'triple'
        elif runs == 4:
            return 'four'
        elif runs >= 6:
            return 'six'
        return 'dot'

    def build_career_baselines(self, df: pd.DataFrame):
        """
        Pre-compute career baseline stats for all batters and bowlers.
        These serve as priors when recent form data is sparse.
        """
        # --- Batter baselines (overall, not phase-specific) ---
        self.batter_career = (
            df.groupby('striker')
            .agg(
                career_balls=('runs_off_bat', 'count'),
                career_runs=('runs_off_bat', 'sum'),
                career_sr=('runs_off_bat', lambda x: x.sum() / len(x) * 100),
                career_dot_pct=('is_dot', 'mean'),
                career_boundary_pct=('is_boundary', 'mean'),
                career_dismissal_rate=('is_wicket', 'mean')
            )
            .reset_index()
            .rename(columns={'striker': 'player'})
        )

        # --- Bowler baselines ---
        self.bowler_career = (
            df.groupby('bowler')
            .agg(
                career_balls_bowled=('total_runs', 'count'),
                career_runs_conceded=('total_runs', 'sum'),
                career_economy=('total_runs', lambda x: x.sum() / (len(x) / 6)),
                career_wicket_rate=('is_wicket', 'mean'),
                career_dot_rate=('is_dot', 'mean')
            )
            .reset_index()
            .rename(columns={'bowler': 'player'})
        )

        print(f"Career baselines: {len(self.batter_career)} batters, "
              f"{len(self.bowler_career)} bowlers")

    def _get_batter_features(self, player: str, phase: str,
                              df_history: pd.DataFrame) -> dict:
        """Get feature dict for a specific batter in a specific phase."""
        # Phase-specific stats
        if len(df_history) > 0 and 'striker' in df_history.columns:
            phase_data = df_history[
                (df_history['striker'] == player) &
                (df_history['phase'] == phase)
            ]
        else:
            phase_data = pd.DataFrame()

        # Career baseline (fallback)
        if self.batter_career is None:
            return {
                'bat_sr': 125.0, 'bat_dot_pct': 0.35,
                'bat_boundary_pct': 0.15, 'bat_dismissal_rate': 0.05,
                'bat_experience': 0
            }

        career = self.batter_career[
            self.batter_career['player'] == player
        ]

        if len(career) == 0:
            # Unknown player - use global averages
            return {
                'bat_sr': 125.0,
                'bat_dot_pct': 0.35,
                'bat_boundary_pct': 0.15,
                'bat_dismissal_rate': 0.05,
                'bat_experience': 0
            }

        career = career.iloc[0]

        if len(phase_data) >= 30:
            # Enough phase-specific data
            sr = phase_data['runs_off_bat'].sum() / len(phase_data) * 100
            dot_pct = phase_data['is_dot'].mean()
            bound_pct = phase_data['is_boundary'].mean()
            dismiss_rate = phase_data['is_wicket'].mean()
        else:
            # Bayesian shrinkage: blend phase data with career baseline
            n = len(phase_data)
            w = n / (n + 30)  # weight for phase data (shrinks toward career)
            if n > 0:
                phase_sr = phase_data['runs_off_bat'].sum() / n * 100
                sr = w * phase_sr + (1 - w) * career['career_sr']
                dot_pct = w * phase_data['is_dot'].mean() + (1 - w) * career['career_dot_pct']
                bound_pct = w * phase_data['is_boundary'].mean() + (1 - w) * career['career_boundary_pct']
                dismiss_rate = w * phase_data['is_wicket'].mean() + (1 - w) * career['career_dismissal_rate']
            else:
                sr = career['career_sr']
                dot_pct = career['career_dot_pct']
                bound_pct = career['career_boundary_pct']
                dismiss_rate = career['career_dismissal_rate']

        return {
            'bat_sr': sr,
            'bat_dot_pct': dot_pct,
            'bat_boundary_pct': bound_pct,
            'bat_dismissal_rate': dismiss_rate,
            'bat_experience': int(career['career_balls'])
        }

    def _get_bowler_features(self, player: str, phase: str,
                              df_history: pd.DataFrame) -> dict:
        """Get feature dict for a specific bowler in a specific phase."""
        if len(df_history) > 0 and 'bowler' in df_history.columns:
            phase_data = df_history[
                (df_history['bowler'] == player) &
                (df_history['phase'] == phase)
            ]
        else:
            phase_data = pd.DataFrame()

        if self.bowler_career is None:
            return {
                'bowl_economy': 8.0, 'bowl_wicket_rate': 0.04,
                'bowl_dot_rate': 0.33, 'bowl_experience': 0
            }

        career = self.bowler_career[
            self.bowler_career['player'] == player
        ]

        if len(career) == 0:
            return {
                'bowl_economy': 8.0,
                'bowl_wicket_rate': 0.04,
                'bowl_dot_rate': 0.33,
                'bowl_experience': 0
            }

        career = career.iloc[0]

        if len(phase_data) >= 30:
            econ = phase_data['total_runs'].sum() / (len(phase_data) / 6)
            wkt_rate = phase_data['is_wicket'].mean()
            dot_rate = phase_data['is_dot'].mean()
        else:
            n = len(phase_data)
            w = n / (n + 30)
            if n > 0:
                phase_econ = phase_data['total_runs'].sum() / (n / 6)
                econ = w * phase_econ + (1 - w) * career['career_economy']
                wkt_rate = w * phase_data['is_wicket'].mean() + (1 - w) * career['career_wicket_rate']
                dot_rate = w * phase_data['is_dot'].mean() + (1 - w) * career['career_dot_rate']
            else:
                econ = career['career_economy']
                wkt_rate = career['career_wicket_rate']
                dot_rate = career['career_dot_rate']

        return {
            'bowl_economy': econ,
            'bowl_wicket_rate': wkt_rate,
            'bowl_dot_rate': dot_rate,
            'bowl_experience': int(career['career_balls_bowled'])
        }

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build full feature matrix and target vector from ball-by-ball data.

        Returns:
            X: DataFrame of features (one row per delivery)
            y: Series of outcome labels (encoded as integers)
        """
        print("Building features... (this may take a minute)")

        # Pre-compute career baselines if not done
        if self.batter_career is None:
            self.build_career_baselines(df)

        # --- Encode target ---
        df = df.copy()
        df['outcome'] = df.apply(self.encode_ball_outcome, axis=1)
        df['outcome_code'] = df['outcome'].map(self.OUTCOME_MAP)

        # ============================================================
        # FEATURE BLOCK 1: Match Situation (available for every ball)
        # ============================================================
        features = pd.DataFrame(index=df.index)

        # Phase encoding (one-hot)
        features['is_powerplay'] = (df['phase'] == 'powerplay').astype(int)
        features['is_middle'] = (df['phase'] == 'middle').astype(int)
        features['is_death'] = (df['phase'] == 'death').astype(int)

        # Over number (normalized 0-1)
        features['over_normalized'] = df['over_num'] / 19.0

        # Ball within over (1-6)
        features['ball_in_over'] = df['ball_in_over']

        # Innings (1st or 2nd)
        features['is_second_innings'] = (df['innings'] == 2).astype(int)

        # Wickets fallen so far (0-10)
        features['wickets_fallen'] = df['cum_wickets']

        # Current run rate
        features['current_rr'] = df['current_rr'].clip(0, 36)

        # Balls remaining in innings
        features['balls_remaining'] = df['balls_remaining']

        # ============================================================
        # FEATURE BLOCK 2: Pressure / Chase Context
        # ============================================================
        features['runs_needed'] = df['runs_needed'].fillna(0)
        features['required_rr'] = df['required_rr'].fillna(0).clip(0, 36)
        features['pressure_index'] = df['pressure_index'].fillna(0).clip(-15, 25)

        # Is the required rate "gettable" (< 10 in T20)?
        features['chase_gettable'] = np.where(
            df['innings'] == 2,
            (df['required_rr'] < 10).astype(int),
            0
        )

        # ============================================================
        # FEATURE BLOCK 3: Recent Momentum (last 6, 12, 18 balls)
        # ============================================================
        for window in [6, 12, 18]:
            col_runs = f'momentum_runs_{window}'
            col_wkts = f'momentum_wkts_{window}'
            col_dots = f'momentum_dots_{window}'

            features[col_runs] = (
                df.groupby(['match_id', 'innings'])['total_runs']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            features[col_wkts] = (
                df.groupby(['match_id', 'innings'])['is_wicket']
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            features[col_dots] = (
                df.groupby(['match_id', 'innings'])['is_dot']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

        # ============================================================
        # FEATURE BLOCK 4: Venue/Pitch Proxy
        # ============================================================
        # Average runs per over at this venue (computed from training data)
        venue_avg = (
            df.groupby('venue')['total_runs']
            .mean()
            .reset_index()
            .rename(columns={'total_runs': 'venue_avg_runs_per_ball'})
        )
        df = df.merge(venue_avg, on='venue', how='left')
        features['venue_avg_rpb'] = df['venue_avg_runs_per_ball'].fillna(
            df['total_runs'].mean()
        )

        # ============================================================
        # FEATURE BLOCK 5: Batter & Bowler Ability Proxies
        # ============================================================
        # For training, we use career stats as static features.
        # In live prediction, these get updated with recent form.

        bat_features = []
        bowl_features = []

        # Vectorized approach: merge career stats
        bat_merged = df[['striker']].merge(
            self.batter_career.rename(columns={'player': 'striker'}),
            on='striker', how='left'
        )
        features['bat_sr'] = bat_merged['career_sr'].fillna(125.0)
        features['bat_dot_pct'] = bat_merged['career_dot_pct'].fillna(0.35)
        features['bat_boundary_pct'] = bat_merged['career_boundary_pct'].fillna(0.15)
        features['bat_dismissal_rate'] = bat_merged['career_dismissal_rate'].fillna(0.05)

        bowl_merged = df[['bowler']].merge(
            self.bowler_career.rename(columns={'player': 'bowler'}),
            on='bowler', how='left'
        )
        features['bowl_economy'] = bowl_merged['career_economy'].fillna(8.0)
        features['bowl_wicket_rate'] = bowl_merged['career_wicket_rate'].fillna(0.04)
        features['bowl_dot_rate'] = bowl_merged['career_dot_rate'].fillna(0.33)

        # ============================================================
        # FEATURE BLOCK 6: Interaction Features
        # ============================================================
        # Batter aggression vs bowler economy (key matchup signal)
        features['bat_vs_bowl_diff'] = (
            features['bat_sr'] / 100 * 6 - features['bowl_economy']
        )

        # Phase-adjusted expected runs
        phase_means = df.groupby('phase')['total_runs'].mean()
        features['phase_expected_runs'] = df['phase'].map(phase_means).fillna(
            df['total_runs'].mean()
        ).astype(float)

        # Wickets * Phase interaction
        features['wickets_x_death'] = (
            features['wickets_fallen'] * features['is_death']
        )

        # ============================================================
        # Clean up
        # ============================================================
        features = features.fillna(0)
        target = df['outcome_code']

        feature_cols = features.columns.tolist()
        print(f"Built {len(feature_cols)} features for {len(features):,} deliveries")
        print(f"Features: {feature_cols}")

        return features, target

    def get_feature_names(self) -> list:
        """Return list of feature names in order."""
        return [
            'is_powerplay', 'is_middle', 'is_death', 'over_normalized',
            'ball_in_over', 'is_second_innings', 'wickets_fallen',
            'current_rr', 'balls_remaining', 'runs_needed', 'required_rr',
            'pressure_index', 'chase_gettable',
            'momentum_runs_6', 'momentum_wkts_6', 'momentum_dots_6',
            'momentum_runs_12', 'momentum_wkts_12', 'momentum_dots_12',
            'momentum_runs_18', 'momentum_wkts_18', 'momentum_dots_18',
            'venue_avg_rpb',
            'bat_sr', 'bat_dot_pct', 'bat_boundary_pct', 'bat_dismissal_rate',
            'bowl_economy', 'bowl_wicket_rate', 'bowl_dot_rate',
            'bat_vs_bowl_diff', 'phase_expected_runs', 'wickets_x_death'
        ]


# ----- Quick test -----
if __name__ == "__main__":
    from data_loader import CricketDataLoader

    loader = CricketDataLoader(data_dir="data")
    raw = loader.load_all_matches()
    clean = loader.clean_data(raw)
    context = loader.compute_match_context(clean)

    engine = FeatureEngine()
    X, y = engine.build_features(context)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print(f"\nSample features:\n{X.head()}")
