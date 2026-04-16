"""
simulator.py - Monte Carlo Match Simulator
=============================================
Uses the trained ball-outcome model to simulate:
- Next over runs (probability distribution)
- Innings total projection
- Match win probability
- Value detection vs bookmaker lines

This is the ENGINE that turns model predictions into betting signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MatchState:
    """Current state of a live match."""
    innings: int                        # 1 or 2
    over_num: int                       # current over (0-19)
    ball_in_over: int                   # ball within current over (1-6)
    runs_scored: int                    # total runs in this innings
    wickets_fallen: int                 # wickets lost
    balls_bowled: int                   # total balls bowled
    target: Optional[int] = None       # target to chase (innings 2 only)
    venue_avg_rpb: float = 1.3         # venue average runs per ball

    # Current players
    striker: str = "Unknown_Bat"
    non_striker: str = "Unknown_Bat"
    bowler: str = "Unknown_Bowl"

    # Available batters remaining (in order)
    batters_remaining: List[str] = field(default_factory=list)

    # Available bowlers and their overs bowled
    bowlers_available: Dict[str, int] = field(default_factory=dict)

    # Recent momentum (last N balls: list of run values)
    recent_balls: List[int] = field(default_factory=list)

    @property
    def current_rr(self) -> float:
        if self.balls_bowled == 0:
            return 0.0
        return self.runs_scored / (self.balls_bowled / 6)

    @property
    def balls_remaining(self) -> int:
        return max(1, 120 - self.balls_bowled)

    @property
    def phase(self) -> str:
        if self.over_num < 6:
            return 'powerplay'
        elif self.over_num < 15:
            return 'middle'
        return 'death'

    @property
    def required_rr(self) -> float:
        if self.innings != 2 or self.target is None:
            return 0.0
        runs_needed = self.target - self.runs_scored
        if self.balls_remaining <= 0:
            return 99.0
        return runs_needed / (self.balls_remaining / 6)

    @property
    def pressure_index(self) -> float:
        if self.innings != 2:
            return 0.0
        return self.required_rr - self.current_rr


class MonteCarloSimulator:
    """
    Simulate cricket match outcomes using the ball-outcome model.

    Core idea: for each ball, the model gives P(outcome).
    We sample from this distribution thousands of times to build
    probability distributions for overs, innings, and match results.
    """

    # Run values for each outcome category
    OUTCOME_RUNS = {
        0: 0,   # dot
        1: 1,   # single
        2: 2,   # double
        3: 3,   # triple
        4: 4,   # four
        5: 6,   # six
        6: 0,   # wicket (0 runs + dismissal)
        7: 1,   # extra (1 run, ball doesn't count)
    }

    def __init__(self, model, feature_engine, n_simulations: int = 10000):
        """
        Args:
            model: trained BallOutcomeModel instance
            feature_engine: FeatureEngine instance (for player stats)
            n_simulations: number of Monte Carlo iterations
        """
        self.model = model
        self.fe = feature_engine
        self.n_sims = n_simulations

    def _build_ball_features(self, state: MatchState) -> dict:
        """Convert current match state into a feature dict for the model."""
        # Momentum features from recent balls
        recent = state.recent_balls
        mom_6 = sum(recent[-6:]) if len(recent) >= 6 else sum(recent)
        mom_12 = sum(recent[-12:]) if len(recent) >= 12 else sum(recent)
        mom_18 = sum(recent[-18:]) if len(recent) >= 18 else sum(recent)

        # Count wickets in recent windows (approximate)
        wkt_6 = 0   # simplified; in production, track wicket balls
        wkt_12 = 0
        wkt_18 = 0

        dot_6 = sum(1 for r in recent[-6:] if r == 0) / max(len(recent[-6:]), 1)
        dot_12 = sum(1 for r in recent[-12:] if r == 0) / max(len(recent[-12:]), 1)
        dot_18 = sum(1 for r in recent[-18:] if r == 0) / max(len(recent[-18:]), 1)

        # Batter/bowler stats (from career baselines)
        bat_stats = self.fe._get_batter_features(
            state.striker, state.phase, pd.DataFrame()  # empty = use career
        )
        bowl_stats = self.fe._get_bowler_features(
            state.bowler, state.phase, pd.DataFrame()
        )

        features = {
            'is_powerplay': int(state.phase == 'powerplay'),
            'is_middle': int(state.phase == 'middle'),
            'is_death': int(state.phase == 'death'),
            'over_normalized': state.over_num / 19.0,
            'ball_in_over': state.ball_in_over,
            'is_second_innings': int(state.innings == 2),
            'wickets_fallen': state.wickets_fallen,
            'current_rr': min(state.current_rr, 36),
            'balls_remaining': state.balls_remaining,
            'runs_needed': max(0, (state.target or 0) - state.runs_scored) if state.innings == 2 else 0,
            'required_rr': min(state.required_rr, 36),
            'pressure_index': max(-15, min(25, state.pressure_index)),
            'chase_gettable': int(state.innings == 2 and state.required_rr < 10),

            'momentum_runs_6': mom_6,
            'momentum_wkts_6': wkt_6,
            'momentum_dots_6': dot_6,
            'momentum_runs_12': mom_12,
            'momentum_wkts_12': wkt_12,
            'momentum_dots_12': dot_12,
            'momentum_runs_18': mom_18,
            'momentum_wkts_18': wkt_18,
            'momentum_dots_18': dot_18,

            'venue_avg_rpb': state.venue_avg_rpb,

            'bat_sr': bat_stats['bat_sr'],
            'bat_dot_pct': bat_stats['bat_dot_pct'],
            'bat_boundary_pct': bat_stats['bat_boundary_pct'],
            'bat_dismissal_rate': bat_stats['bat_dismissal_rate'],
            'bowl_economy': bowl_stats['bowl_economy'],
            'bowl_wicket_rate': bowl_stats['bowl_wicket_rate'],
            'bowl_dot_rate': bowl_stats['bowl_dot_rate'],

            'bat_vs_bowl_diff': bat_stats['bat_sr'] / 100 * 6 - bowl_stats['bowl_economy'],
            'phase_expected_runs': {
                'powerplay': 1.4, 'middle': 1.2, 'death': 1.6
            }.get(state.phase, 1.3),
            'wickets_x_death': state.wickets_fallen * int(state.phase == 'death')
        }

        return features

    def simulate_over(self, state: MatchState) -> dict:
        """
        Simulate the current over N times.

        Returns:
            dict with:
                'distribution': array of simulated over totals
                'mean': expected runs in the over
                'median': median runs
                'percentiles': {10, 25, 50, 75, 90}
                'prob_under_X': probability of scoring under X runs
        """
        balls_left_in_over = 6 - (state.ball_in_over - 1)
        over_totals = np.zeros(self.n_sims)

        for sim in range(self.n_sims):
            sim_state = self._copy_state(state)
            over_runs = 0

            for ball in range(balls_left_in_over):
                if sim_state.wickets_fallen >= 10:
                    break

                features = self._build_ball_features(sim_state)
                proba_array = self.model.predict_fast(features)

                # Normalize probabilities
                proba_array = proba_array / proba_array.sum()

                # Sample outcome
                outcome = np.random.choice(8, p=proba_array)
                runs = self.OUTCOME_RUNS[outcome]
                over_runs += runs

                # Update state for next ball
                sim_state.runs_scored += runs
                sim_state.recent_balls.append(runs)

                if outcome == 7:  # extra - ball doesn't count
                    pass  # don't increment balls_bowled
                else:
                    sim_state.balls_bowled += 1
                    sim_state.ball_in_over += 1

                if outcome == 6:  # wicket
                    sim_state.wickets_fallen += 1
                    if sim_state.batters_remaining:
                        sim_state.striker = sim_state.batters_remaining.pop(0)

                # Rotate strike on odd runs
                if runs % 2 == 1:
                    sim_state.striker, sim_state.non_striker = (
                        sim_state.non_striker, sim_state.striker
                    )

            over_totals[sim] = over_runs

        # Build result distribution
        result = {
            'distribution': over_totals,
            'mean': float(np.mean(over_totals)),
            'median': float(np.median(over_totals)),
            'std': float(np.std(over_totals)),
            'percentiles': {
                p: float(np.percentile(over_totals, p))
                for p in [10, 25, 50, 75, 90]
            }
        }

        # Probability of scoring under/over various lines
        for line in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
            result[f'prob_under_{line}'] = float(
                np.mean(over_totals < line)
            )
            result[f'prob_over_{line}'] = float(
                np.mean(over_totals >= line)
            )

        return result

    def simulate_innings(self, state: MatchState) -> dict:
        """
        Simulate the remainder of the innings N times.

        Returns:
            dict with projected final total distribution.
        """
        final_totals = np.zeros(self.n_sims)

        for sim in range(self.n_sims):
            sim_state = self._copy_state(state)

            while (sim_state.balls_bowled < 120 and
                   sim_state.wickets_fallen < 10):

                # Update phase based on current over
                sim_state.over_num = sim_state.balls_bowled // 6

                features = self._build_ball_features(sim_state)
                proba_array = self.model.predict_fast(features)
                proba_array = proba_array / proba_array.sum()

                outcome = np.random.choice(8, p=proba_array)
                runs = self.OUTCOME_RUNS[outcome]

                sim_state.runs_scored += runs
                sim_state.recent_balls.append(runs)

                if outcome != 7:  # not an extra
                    sim_state.balls_bowled += 1

                if outcome == 6:
                    sim_state.wickets_fallen += 1
                    if sim_state.batters_remaining:
                        sim_state.striker = sim_state.batters_remaining.pop(0)

                if runs % 2 == 1:
                    sim_state.striker, sim_state.non_striker = (
                        sim_state.non_striker, sim_state.striker
                    )

                # End of over: rotate strike, change bowler
                if sim_state.balls_bowled % 6 == 0 and sim_state.balls_bowled > 0:
                    sim_state.striker, sim_state.non_striker = (
                        sim_state.non_striker, sim_state.striker
                    )

            final_totals[sim] = sim_state.runs_scored

        return {
            'distribution': final_totals,
            'mean': float(np.mean(final_totals)),
            'median': float(np.median(final_totals)),
            'std': float(np.std(final_totals)),
            'percentiles': {
                p: float(np.percentile(final_totals, p))
                for p in [10, 25, 50, 75, 90]
            },
            'prob_ranges': {
                f'{lo}-{hi}': float(np.mean((final_totals >= lo) & (final_totals < hi)))
                for lo, hi in [(100, 130), (130, 150), (150, 170),
                               (170, 190), (190, 210), (210, 240)]
            }
        }

    def simulate_win_probability(self, state: MatchState) -> dict:
        """
        For 2nd innings: simulate remaining balls to estimate
        probability of chasing team winning.

        Returns:
            dict with win_prob, lose_prob, tie_prob
        """
        if state.innings != 2 or state.target is None:
            raise ValueError("Win probability requires 2nd innings with a target")

        wins = 0
        losses = 0
        ties = 0

        for sim in range(self.n_sims):
            sim_state = self._copy_state(state)

            while (sim_state.balls_bowled < 120 and
                   sim_state.wickets_fallen < 10 and
                   sim_state.runs_scored < sim_state.target):

                sim_state.over_num = sim_state.balls_bowled // 6

                features = self._build_ball_features(sim_state)
                proba_array = self.model.predict_fast(features)
                proba_array = proba_array / proba_array.sum()

                outcome = np.random.choice(8, p=proba_array)
                runs = self.OUTCOME_RUNS[outcome]
                sim_state.runs_scored += runs
                sim_state.recent_balls.append(runs)

                if outcome != 7:
                    sim_state.balls_bowled += 1
                if outcome == 6:
                    sim_state.wickets_fallen += 1
                    if sim_state.batters_remaining:
                        sim_state.striker = sim_state.batters_remaining.pop(0)
                if runs % 2 == 1:
                    sim_state.striker, sim_state.non_striker = (
                        sim_state.non_striker, sim_state.striker
                    )
                if sim_state.balls_bowled % 6 == 0:
                    sim_state.striker, sim_state.non_striker = (
                        sim_state.non_striker, sim_state.striker
                    )

            if sim_state.runs_scored >= sim_state.target:
                wins += 1
            elif sim_state.runs_scored == sim_state.target - 1:
                ties += 1
            else:
                losses += 1

        return {
            'win_prob': wins / self.n_sims,
            'lose_prob': losses / self.n_sims,
            'tie_prob': ties / self.n_sims,
            'simulations': self.n_sims
        }

    def find_value_vs_bookmaker(self, state: MatchState,
                                 bookmaker_line: float,
                                 bookmaker_odds_over: float,
                                 bookmaker_odds_under: float,
                                 market_type: str = "over_runs") -> dict:
        """
        Compare model's projection against a bookmaker line
        to identify value bets.

        Args:
            state: current match state
            bookmaker_line: e.g. 7.5 for over/under 7.5 runs
            bookmaker_odds_over: decimal odds for "over" (e.g. 1.85)
            bookmaker_odds_under: decimal odds for "under" (e.g. 1.95)
            market_type: "over_runs" or "innings_total"

        Returns:
            dict with model probability, implied probability,
            edge, and recommendation
        """
        if market_type == "over_runs":
            sim_result = self.simulate_over(state)
            distribution = sim_result['distribution']
        else:
            sim_result = self.simulate_innings(state)
            distribution = sim_result['distribution']

        # Model's probability
        model_prob_over = float(np.mean(distribution >= bookmaker_line))
        model_prob_under = float(np.mean(distribution < bookmaker_line))

        # Bookmaker's implied probability (includes margin)
        implied_prob_over = 1.0 / bookmaker_odds_over
        implied_prob_under = 1.0 / bookmaker_odds_under

        # Edge = model_prob - implied_prob
        edge_over = model_prob_over - implied_prob_over
        edge_under = model_prob_under - implied_prob_under

        # Kelly criterion for recommended stake (fraction of bankroll)
        kelly_over = max(0, (model_prob_over * bookmaker_odds_over - 1) /
                        (bookmaker_odds_over - 1))
        kelly_under = max(0, (model_prob_under * bookmaker_odds_under - 1) /
                         (bookmaker_odds_under - 1))

        # Quarter Kelly for safety
        quarter_kelly_over = kelly_over / 4
        quarter_kelly_under = kelly_under / 4

        # Recommendation
        min_edge = 0.03  # require at least 3% edge
        if edge_over > min_edge and edge_over > edge_under:
            recommendation = "BET OVER"
            edge = edge_over
            kelly = quarter_kelly_over
        elif edge_under > min_edge and edge_under > edge_over:
            recommendation = "BET UNDER"
            edge = edge_under
            kelly = quarter_kelly_under
        else:
            recommendation = "NO BET"
            edge = max(edge_over, edge_under)
            kelly = 0

        return {
            'bookmaker_line': bookmaker_line,
            'model_mean': sim_result['mean'],
            'model_median': sim_result['median'],
            'model_prob_over': model_prob_over,
            'model_prob_under': model_prob_under,
            'implied_prob_over': implied_prob_over,
            'implied_prob_under': implied_prob_under,
            'edge_over': edge_over,
            'edge_under': edge_under,
            'recommendation': recommendation,
            'edge': edge,
            'quarter_kelly_stake': kelly,
            'confidence': 'HIGH' if abs(edge) > 0.08 else
                         'MEDIUM' if abs(edge) > 0.05 else 'LOW'
        }

    def _copy_state(self, state: MatchState) -> MatchState:
        """Deep copy a MatchState for simulation."""
        return MatchState(
            innings=state.innings,
            over_num=state.over_num,
            ball_in_over=state.ball_in_over,
            runs_scored=state.runs_scored,
            wickets_fallen=state.wickets_fallen,
            balls_bowled=state.balls_bowled,
            target=state.target,
            venue_avg_rpb=state.venue_avg_rpb,
            striker=state.striker,
            non_striker=state.non_striker,
            bowler=state.bowler,
            batters_remaining=list(state.batters_remaining),
            bowlers_available=dict(state.bowlers_available),
            recent_balls=list(state.recent_balls)
        )

    # ------------------------------------------------------------------
    # NEW: clearer, tab-friendly prediction methods
    # ------------------------------------------------------------------

    def predict_next_ball_outcomes(self, state: 'MatchState') -> dict:
        """
        Returns calibrated probabilities for the *very next ball*:
            {dot, single, boundary, six, wicket, extra}
        Built directly from the model's class probabilities — no sampling.
        """
        features = self._build_ball_features(state)
        proba = self.model.predict_fast(features)
        proba = proba / proba.sum()
        # OUTCOME_MAP: 0=dot 1=single 2=double 3=triple 4=four 5=six 6=wicket 7=extra
        return {
            "dot":      float(proba[0]),
            "single":   float(proba[1]),
            "double":   float(proba[2] + proba[3]),
            "boundary": float(proba[4]),
            "six":      float(proba[5]),
            "wicket":   float(proba[6]),
            "extra":    float(proba[7]),
        }

    def predict_phase_segments(self, state: 'MatchState',
                               phases: list[tuple[str, int, int]] | None = None
                               ) -> list[dict]:
        """
        Predict runs scored in each remaining phase of the innings.

        Default phase boundaries (over numbers, 1-indexed inclusive):
          • Powerplay      : 1-6
          • Middle Phase 1 : 7-10
          • Middle Phase 2 : 11-15
          • Death          : 16-20
        """
        if phases is None:
            phases = [
                ("Powerplay (1-6)",   1,  6),
                ("Middle 1 (7-10)",   7, 10),
                ("Middle 2 (11-15)", 11, 15),
                ("Death (16-20)",    16, 20),
            ]
        current_over = state.balls_bowled // 6 + 1  # 1-indexed
        results: list[dict] = []
        rolling = self._copy_state(state)

        for label, lo, hi in phases:
            if hi < current_over:
                # Phase already completed
                results.append({
                    "label": label,
                    "status": "completed",
                    "predicted_runs": None,
                    "p10": None, "p90": None,
                    "predicted_wickets": None,
                })
                continue

            overs_in_phase = hi - max(lo, current_over) + 1
            phase_run_dist = np.zeros(self.n_sims)
            phase_wkt_dist = np.zeros(self.n_sims)

            for _ in range(overs_in_phase):
                rolling.ball_in_over = 1
                over_result = self.simulate_over(rolling)
                phase_run_dist += over_result["distribution"]
                # Track wickets via mean prediction (cheap proxy)
                ob = self.predict_next_ball_outcomes(rolling)
                phase_wkt_dist += ob["wicket"] * 6  # rough: P × 6 balls
                # Advance state by mean over outcome
                rolling.runs_scored += int(round(over_result["mean"]))
                rolling.balls_bowled += 6
                rolling.over_num = rolling.balls_bowled // 6

            results.append({
                "label": label,
                "status": "current" if lo <= current_over <= hi else "future",
                "predicted_runs": float(np.mean(phase_run_dist)),
                "p10": float(np.percentile(phase_run_dist, 10)),
                "p90": float(np.percentile(phase_run_dist, 90)),
                "predicted_wickets": float(np.mean(phase_wkt_dist)),
                "overs": overs_in_phase,
            })

        return results

    def pre_match_simulate(self, batting_team_str_avg: float = 8.0,
                           batting_team_wkt_rate: float = 0.04,
                           bowling_team_eco: float = 8.0,
                           venue_phase_rpo: dict | None = None,
                           n_sims: int | None = None) -> dict:
        """
        Simulate the FULL T20 innings before the match starts using
        team-average stats + venue context. Returns score distribution
        + key probabilities (P(<=140), P(140-170), P(170-200), P(>200)).

        Pass venue_phase_rpo like {'powerplay': 8.4, 'middle': 7.9, 'death': 10.6}
        for venue-aware simulation; falls back to neutral averages otherwise.
        """
        n_sims = n_sims or self.n_sims
        venue_phase_rpo = venue_phase_rpo or {
            "powerplay": 8.4, "middle": 7.9, "death": 10.6,
        }

        totals = np.zeros(n_sims)
        wkts = np.zeros(n_sims)

        # Build a synthetic ball-0 state. The model's momentum features
        # will be neutral (empty recent history).
        for sim in range(n_sims):
            state = MatchState(
                innings=1, over_num=0, ball_in_over=1,
                runs_scored=0, wickets_fallen=0, balls_bowled=0,
                target=None,
                venue_avg_rpb=(venue_phase_rpo.get("powerplay", 8.4) / 6),
                striker=f"AvgBat_{batting_team_str_avg:.0f}",
                non_striker=f"AvgBat_{batting_team_str_avg:.0f}",
                bowler=f"AvgBowl_{bowling_team_eco:.0f}",
            )
            # Walk over by over, using simulate_over (which already runs n_sims internally — be careful)
            # To keep cost reasonable, use predict_next_ball_outcomes 120 times instead.
            for ball in range(120):
                if state.wickets_fallen >= 10:
                    break
                phase = ("powerplay" if state.over_num < 6
                         else "middle" if state.over_num < 15
                         else "death")
                state.venue_avg_rpb = venue_phase_rpo.get(phase, 8.0) / 6
                ob = self.predict_next_ball_outcomes(state)
                # Sample from the calibrated outcomes
                outcomes = list(ob.keys())
                probs = np.array(list(ob.values()))
                probs = probs / probs.sum()
                outcome = np.random.choice(outcomes, p=probs)
                runs_map = {"dot": 0, "single": 1, "double": 2,
                            "boundary": 4, "six": 6, "wicket": 0, "extra": 1}
                runs = runs_map[outcome]
                state.runs_scored += runs
                state.recent_balls.append(runs)
                if outcome != "extra":
                    state.balls_bowled += 1
                    state.ball_in_over += 1
                if state.ball_in_over > 6:
                    state.ball_in_over = 1
                    state.striker, state.non_striker = state.non_striker, state.striker
                state.over_num = state.balls_bowled // 6
                if outcome == "wicket":
                    state.wickets_fallen += 1
                if runs in (1, 3):
                    state.striker, state.non_striker = state.non_striker, state.striker

            totals[sim] = state.runs_scored
            wkts[sim] = state.wickets_fallen

        return {
            "mean": float(np.mean(totals)),
            "median": float(np.median(totals)),
            "std": float(np.std(totals)),
            "p10": float(np.percentile(totals, 10)),
            "p25": float(np.percentile(totals, 25)),
            "p75": float(np.percentile(totals, 75)),
            "p90": float(np.percentile(totals, 90)),
            "wickets_mean": float(np.mean(wkts)),
            "distribution": totals,
            "score_buckets": {
                "≤140":     float(np.mean(totals <= 140)),
                "141-170":  float(np.mean((totals > 140) & (totals <= 170))),
                "171-200":  float(np.mean((totals > 170) & (totals <= 200))),
                ">200":     float(np.mean(totals > 200)),
            },
            "n_sims": n_sims,
        }

    # ------------------------------------------------------------------
    # NEW: Heuristic simulator that doesn't suffer from momentum-feedback.
    # Anchored to real IPL averages. Uses player stats when provided.
    # ------------------------------------------------------------------

    def realistic_innings(self,
                          batting_team_stats: dict,
                          bowling_team_stats: dict,
                          venue_phase_rpo: dict | None = None,
                          target: int | None = None,
                          n_sims: int = 200,
                          intel=None) -> dict:
        """
        Simulate ONE innings n_sims times using **real conditional probability
        distributions** measured from 1,191 past IPL matches.

        For each ball, the outcome (dot / single / double / triple / four / six /
        wicket / extra) is sampled from the actual empirical distribution at
        that (phase, wickets-fallen, chase-pressure) bucket — not from any
        hand-tuned formula.

        Team-specific signals (boundary %, RPO) are then applied as a
        small multiplicative tilt on top of the base distribution.
        """
        import numpy as np

        # Load real distributions if available
        if intel is None:
            try:
                from match_intel import MatchIntelligence
                intel = MatchIntelligence.load_or_build("data")
            except Exception:
                intel = None

        # Team multipliers — relative to league-average team
        bat_rpo  = float(batting_team_stats.get("batting_rpo", 8.4))
        bowl_eco = float(bowling_team_stats.get("bowling_eco", 8.2))
        bat_boundary  = float(batting_team_stats.get("boundary_pct", 14.0)) / 100.0

        # Compute boundary multiplier (team's boundary % / league avg ~14%)
        boundary_mult = max(0.7, min(1.5, bat_boundary / 0.14))
        # RPO multiplier — used to scale single/double rates if team scores faster
        rpo_mult = max(0.85, min(1.20, bat_rpo / 8.4))
        # Bowling team bonus to wickets
        # (lower ECO → tighter bowling → slight wicket boost)
        wkt_mult = max(0.85, min(1.20, 8.4 / max(bowl_eco, 6.0)))

        outcomes_order = ["dot", "single", "double", "triple",
                          "four", "six", "wicket", "extra"]
        run_per_outcome = [0, 1, 2, 3, 4, 6, 0, 1]
        is_legal_outcome = [True, True, True, True, True, True, True, False]

        totals = np.zeros(n_sims, dtype=int)
        wkts_arr = np.zeros(n_sims, dtype=int)
        per_over_runs = np.zeros((n_sims, 20), dtype=int)
        per_over_wkts = np.zeros((n_sims, 20), dtype=int)

        for sim in range(n_sims):
            runs = 0
            wkts = 0
            balls = 0
            cur_over_runs = 0
            cur_over_wkts = 0
            cur_over_idx = 0

            while balls < 120 and wkts < 10:
                if target and runs >= target:
                    break

                over_idx = balls // 6
                phase = ("powerplay" if over_idx < 6
                         else "middle" if over_idx < 15
                         else "death")
                # Compute chase pressure bucket
                chase_bucket = "none"
                if target and balls >= 36:
                    needed = max(0, target - runs)
                    balls_left = max(1, 120 - balls)
                    rrr = needed * 6 / balls_left
                    if needed <= 0:
                        chase_bucket = "none"
                    elif rrr <= 8:    chase_bucket = "easy"
                    elif rrr <= 11:   chase_bucket = "medium"
                    else:             chase_bucket = "hard"

                # ---- REAL distribution lookup ----
                base = None
                if intel:
                    bucket = intel.ball_outcome_dist(phase, wkts, chase_bucket)
                    if bucket and bucket.get("n", 0) >= 50:
                        base = dict(bucket["dist"])
                if base is None:
                    # Fallback to a sensible default (shouldn't happen often)
                    base = {"dot": 0.40, "single": 0.30, "double": 0.06,
                            "triple": 0.012, "four": 0.10, "six": 0.04,
                            "wicket": 0.045, "extra": 0.043}

                # ---- Apply team-specific multipliers ----
                # Boost boundaries for big-hitting teams
                base["four"] *= boundary_mult
                base["six"]  *= boundary_mult
                # Adjust singles/doubles by RPO (slower teams take fewer 1s)
                base["single"] *= rpo_mult
                base["double"] *= rpo_mult
                # Stronger bowling = more wickets
                base["wicket"] *= wkt_mult
                # Renormalise
                s = sum(base.values())
                if s > 0:
                    base = {k: v / s for k, v in base.items()}

                probs = np.array([base[o] for o in outcomes_order])
                probs = probs / probs.sum()
                idx = int(np.random.choice(8, p=probs))

                r = run_per_outcome[idx]
                runs += r
                cur_over_runs += r
                if outcomes_order[idx] == "wicket":
                    wkts += 1
                    cur_over_wkts += 1
                if is_legal_outcome[idx]:
                    balls += 1

                # End of over
                if balls > 0 and balls % 6 == 0 and balls > cur_over_idx * 6:
                    if cur_over_idx < 20:
                        per_over_runs[sim, cur_over_idx] = cur_over_runs
                        per_over_wkts[sim, cur_over_idx] = cur_over_wkts
                    cur_over_runs = 0
                    cur_over_wkts = 0
                    cur_over_idx += 1

            # Sanity cap at 290 (highest IPL = 287)
            runs = min(runs, 290)
            totals[sim] = runs
            wkts_arr[sim] = wkts

        return {
            "mean": float(np.mean(totals)),
            "median": float(np.median(totals)),
            "std": float(np.std(totals)),
            "p10": float(np.percentile(totals, 10)),
            "p25": float(np.percentile(totals, 25)),
            "p75": float(np.percentile(totals, 75)),
            "p90": float(np.percentile(totals, 90)),
            "wickets_mean": float(np.mean(wkts_arr)),
            "distribution": totals,
            "per_over_runs_mean": per_over_runs.mean(axis=0).tolist(),
            "per_over_wkts_mean": per_over_wkts.mean(axis=0).tolist(),
            "n_sims": n_sims,
            "data_source": "real_conditional_distributions" if intel else "fallback",
        }

    # ------------------------------------------------------------------
    # PLAYER-AWARE SIMULATOR — uses actual XI from past data
    # ------------------------------------------------------------------

    def player_aware_innings(self,
                             batting_xi: list[str],
                             bowling_attack: list[tuple[str, float]],
                             venue_phase_rpo: dict | None = None,
                             target: int | None = None,
                             intel=None) -> dict:
        """
        Simulate ONE innings with REAL player stats.

        For every ball:
          • Sample base distribution from the (phase, wkts, chase) bucket
            measured from past IPL data
          • Tilt boundary % using THIS striker's career boundary %
            vs league average
          • Tilt wicket prob using THIS bowler's career wicket rate
            vs league average
          • Tilt singles using THIS striker's career SR vs league average
          • If a striker × bowler historical H2H exists in intel
            (≥6 prior balls), bias outcomes toward those proportions

        Returns the runs scored, wickets, balls bowled, and per-ball log.
        """
        import numpy as np
        if intel is None:
            try:
                from match_intel import MatchIntelligence
                intel = MatchIntelligence.load_or_build("data")
            except Exception:
                intel = None

        # Pre-fetch player stats
        bat_stats = {name: (intel.batter(name) or {}) for name in batting_xi} if intel else {}
        bowl_names = [b[0] for b in bowling_attack]
        bowl_stats = {name: (intel.bowler(name) or {}) for name in bowl_names} if intel else {}

        # League averages for normalisation
        LEAGUE_BOUNDARY_PCT = 14.0
        LEAGUE_SR = 132.0
        LEAGUE_ECO = 8.4
        LEAGUE_WICKET_RATE = 0.045

        outcomes_order = ["dot", "single", "double", "triple",
                          "four", "six", "wicket", "extra"]
        runs_per = [0, 1, 2, 3, 4, 6, 0, 1]
        is_legal = [True, True, True, True, True, True, True, False]

        runs = 0
        wkts = 0
        balls = 0
        per_over_runs = [0] * 20
        per_over_wkts = [0] * 20

        # Bowler rotation: spread bowlers across 20 overs based on typical_overs
        # Build an ordered list of (over_idx → bowler) that obeys "no 2 in a row"
        bowler_quota = {name: int(round(ov)) for name, ov in bowling_attack}
        # Ensure 20 overs covered
        total = sum(bowler_quota.values())
        if total < 20:
            # Distribute remaining to first bowler
            for name, _ in bowling_attack:
                if total >= 20: break
                bowler_quota[name] += 1
                total += 1
        # Build over schedule (alternating, no back-to-back same bowler)
        bowler_schedule = []
        last_bowler = None
        candidates = sorted(bowling_attack, key=lambda x: -x[1])
        for over_idx in range(20):
            picked = None
            for name, _ in candidates:
                if bowler_quota.get(name, 0) > 0 and name != last_bowler:
                    picked = name
                    break
            if picked is None:
                # Fall back to any with quota
                for name, _ in candidates:
                    if bowler_quota.get(name, 0) > 0:
                        picked = name
                        break
            if picked is None:
                picked = candidates[0][0]   # last-resort
            bowler_schedule.append(picked)
            bowler_quota[picked] = bowler_quota.get(picked, 0) - 1
            last_bowler = picked

        # Initialise on-field batters
        if len(batting_xi) >= 2:
            striker = batting_xi[0]
            non_striker = batting_xi[1]
            next_bat_idx = 2
        else:
            striker = "?"; non_striker = "?"; next_bat_idx = 0

        while balls < 120 and wkts < 10:
            if target and runs >= target:
                break

            over_idx = balls // 6
            phase = ("powerplay" if over_idx < 6
                     else "middle" if over_idx < 15
                     else "death")
            chase_bucket = "none"
            if target and balls >= 36:
                needed = max(0, target - runs)
                balls_left = max(1, 120 - balls)
                rrr = needed * 6 / balls_left
                if needed <= 0:
                    chase_bucket = "none"
                elif rrr <= 8:    chase_bucket = "easy"
                elif rrr <= 11:   chase_bucket = "medium"
                else:             chase_bucket = "hard"

            # Base distribution from real data
            base = None
            if intel:
                bucket = intel.ball_outcome_dist(phase, wkts, chase_bucket)
                if bucket and bucket.get("n", 0) >= 50:
                    base = dict(bucket["dist"])
            if base is None:
                base = {"dot": 0.40, "single": 0.30, "double": 0.06,
                        "triple": 0.012, "four": 0.10, "six": 0.04,
                        "wicket": 0.045, "extra": 0.043}

            # ---- PLAYER TILTS ----
            current_bowler = bowler_schedule[over_idx] if over_idx < len(bowler_schedule) else bowling_attack[0][0]
            sb = bat_stats.get(striker, {})
            bb = bowl_stats.get(current_bowler, {})

            # PREFER recent (2025+2026) form over all-time career — it's
            # more predictive of current performance.
            striker_boundary = sb.get("recent_boundary_pct",
                                       sb.get("boundary_pct", LEAGUE_BOUNDARY_PCT))
            striker_sr       = sb.get("recent_sr",
                                       sb.get("sr", LEAGUE_SR))
            bowler_eco       = bb.get("recent_economy",
                                       bb.get("economy", LEAGUE_ECO))
            # Wicket rate from recent stats if we have ≥30 balls
            if bb.get("recent_balls", 0) >= 30:
                bowler_wkt_rate = bb["recent_wickets"] / bb["recent_balls"]
            elif bb.get("balls", 0) > 50:
                bowler_wkt_rate = bb["wickets"] / bb["balls"]
            else:
                bowler_wkt_rate = LEAGUE_WICKET_RATE

            # Boundary tilt — striker-driven
            boundary_mult = max(0.5, min(2.0, striker_boundary / LEAGUE_BOUNDARY_PCT))
            base["four"] *= boundary_mult
            base["six"]  *= boundary_mult
            # SR tilt — striker rotates more = more singles
            sr_mult = max(0.7, min(1.5, striker_sr / LEAGUE_SR))
            base["single"] *= sr_mult
            base["double"] *= sr_mult
            # Wicket tilt — bowler-driven
            wkt_mult = max(0.5, min(2.5, bowler_wkt_rate / LEAGUE_WICKET_RATE))
            base["wicket"] *= wkt_mult
            # Bowling economy tilt — tight bowlers reduce boundaries
            eco_mult = max(0.6, min(1.6, bowler_eco / LEAGUE_ECO))
            base["four"] *= eco_mult
            base["six"]  *= eco_mult

            # ---- HEAD-TO-HEAD bias ----
            if intel:
                h2h = intel.matchup(striker, current_bowler)
                if h2h and h2h.get("balls", 0) >= 6:
                    h2h_boundary = h2h["boundary_pct"] / 100
                    h2h_sr = h2h["sr"]
                    h2h_dismissals = h2h["dismissals"]
                    h2h_balls = h2h["balls"]
                    # Blend H2H boundary % into the boundary outcomes
                    # Weight = min(0.5, h2h_balls / 30) — capped influence
                    weight = min(0.4, h2h_balls / 30)
                    target_boundary = h2h_boundary  # 0..1 prob per ball
                    cur_boundary = base["four"] + base["six"]
                    if cur_boundary > 0 and target_boundary > 0:
                        ratio = (target_boundary * (1 + weight)) / cur_boundary
                        ratio = max(0.5, min(2.0, ratio))
                        base["four"] *= ratio
                        base["six"]  *= ratio
                    # Bias wicket rate toward H2H dismissal rate
                    h2h_wkt_rate = h2h_dismissals / h2h_balls
                    target_wkt = (1 - weight) * base["wicket"] + weight * h2h_wkt_rate
                    base["wicket"] = max(0.005, min(0.30, target_wkt))

            # Renormalise
            s = sum(base.values())
            if s > 0:
                base = {k: v / s for k, v in base.items()}

            probs = np.array([base[o] for o in outcomes_order])
            probs = probs / probs.sum()
            idx = int(np.random.choice(8, p=probs))
            r = runs_per[idx]
            outcome = outcomes_order[idx]

            runs += r
            cur_over_idx = balls // 6
            if cur_over_idx < 20:
                per_over_runs[cur_over_idx] += r
            if outcome == "wicket":
                wkts += 1
                if cur_over_idx < 20:
                    per_over_wkts[cur_over_idx] += 1
                # Promote next batter
                if next_bat_idx < len(batting_xi):
                    striker = batting_xi[next_bat_idx]
                    next_bat_idx += 1
                else:
                    striker = "?"
            if is_legal[idx]:
                balls += 1
                # Strike rotation on odd runs
                if r in (1, 3):
                    striker, non_striker = non_striker, striker
                # End of over → swap
                if balls % 6 == 0 and outcome != "wicket":
                    striker, non_striker = non_striker, striker

        return {
            "runs": min(runs, 290),
            "wickets": wkts,
            "balls": balls,
            "per_over_runs": per_over_runs,
            "per_over_wkts": per_over_wkts,
        }

    def player_aware_pre_match(self,
                               team_a_name: str,
                               team_a_xi: dict,
                               team_b_name: str,
                               team_b_xi: dict,
                               venue_phase_rpo: dict | None = None,
                               n_sims: int = 100,
                               team_a_bats_first: bool = True,
                               intel=None) -> dict:
        """
        Run player-aware full match N times.

        team_a_xi/b_xi shape: {"batters": [11 names], "bowlers": [(name, ov), ...]}
        """
        import numpy as np

        if team_a_bats_first:
            f_name, f_xi, f_atk = team_a_name, team_a_xi, team_b_xi.get("bowlers", [])
            s_name, s_xi, s_atk = team_b_name, team_b_xi, team_a_xi.get("bowlers", [])
        else:
            f_name, f_xi, f_atk = team_b_name, team_b_xi, team_a_xi.get("bowlers", [])
            s_name, s_xi, s_atk = team_a_name, team_a_xi, team_b_xi.get("bowlers", [])

        first_dist = np.zeros(n_sims, dtype=int)
        second_dist = np.zeros(n_sims, dtype=int)
        first_wkts = np.zeros(n_sims, dtype=int)
        second_wkts = np.zeros(n_sims, dtype=int)
        first_per_over_runs = np.zeros((n_sims, 20))
        first_per_over_wkts = np.zeros((n_sims, 20))
        second_per_over_runs = np.zeros((n_sims, 20))
        second_per_over_wkts = np.zeros((n_sims, 20))
        wins_first = wins_second = ties = 0
        margins_first = []
        margins_second = []

        for i in range(n_sims):
            inn1 = self.player_aware_innings(
                batting_xi=f_xi.get("batters", []),
                bowling_attack=f_atk,
                venue_phase_rpo=venue_phase_rpo,
                intel=intel,
            )
            target = inn1["runs"] + 1
            inn2 = self.player_aware_innings(
                batting_xi=s_xi.get("batters", []),
                bowling_attack=s_atk,
                venue_phase_rpo=venue_phase_rpo,
                target=target,
                intel=intel,
            )

            first_dist[i] = inn1["runs"]
            second_dist[i] = inn2["runs"]
            first_wkts[i] = inn1["wickets"]
            second_wkts[i] = inn2["wickets"]
            first_per_over_runs[i] = inn1["per_over_runs"]
            first_per_over_wkts[i] = inn1["per_over_wkts"]
            second_per_over_runs[i] = inn2["per_over_runs"]
            second_per_over_wkts[i] = inn2["per_over_wkts"]

            if inn2["runs"] >= target:
                wins_second += 1
                margins_second.append(10 - inn2["wickets"])
            elif inn2["runs"] == target - 1:
                ties += 1
            else:
                wins_first += 1
                margins_first.append(target - 1 - inn2["runs"])

        wp = {f_name: wins_first / n_sims,
              s_name: wins_second / n_sims,
              "Tie":  ties / n_sims}
        if wp[f_name] >= wp[s_name]:
            mr = float(np.mean(margins_first)) if margins_first else 0
            expected = {"winner": f_name, "by": "runs", "value": round(mr, 1)}
        else:
            mw = float(np.mean(margins_second)) if margins_second else 0
            expected = {"winner": s_name, "by": "wickets", "value": round(mw, 1)}

        return {
            "first_team": {
                "name": f_name, "role": "bat-first",
                "distribution": first_dist,
                "mean": float(np.mean(first_dist)),
                "median": float(np.median(first_dist)),
                "p10": float(np.percentile(first_dist, 10)),
                "p90": float(np.percentile(first_dist, 90)),
                "wickets_mean": float(np.mean(first_wkts)),
                "per_over_runs_mean": first_per_over_runs.mean(axis=0).tolist(),
                "per_over_wkts_mean": first_per_over_wkts.mean(axis=0).tolist(),
                "xi": f_xi.get("batters", []),
                "bowling_attack": f_atk,
            },
            "second_team": {
                "name": s_name, "role": "chasing",
                "distribution": second_dist,
                "mean": float(np.mean(second_dist)),
                "median": float(np.median(second_dist)),
                "p10": float(np.percentile(second_dist, 10)),
                "p90": float(np.percentile(second_dist, 90)),
                "wickets_mean": float(np.mean(second_wkts)),
                "per_over_runs_mean": second_per_over_runs.mean(axis=0).tolist(),
                "per_over_wkts_mean": second_per_over_wkts.mean(axis=0).tolist(),
                "xi": s_xi.get("batters", []),
                "bowling_attack": s_atk,
            },
            "winner_probs": wp,
            "expected": expected,
            "n_sims": n_sims,
            "data_source": "player_aware_with_h2h",
        }

    def realistic_pre_match(self,
                            team_a_name: str,
                            team_a_stats: dict,
                            team_b_name: str,
                            team_b_stats: dict,
                            venue_phase_rpo: dict | None = None,
                            n_sims: int = 200,
                            team_a_bats_first: bool = True,
                            intel=None) -> dict:
        """
        Full match simulator using only player/team historical stats.
        Returns inning1, inning2, win probabilities, expected margin.
        """
        if team_a_bats_first:
            f_name, f_bat, f_bowl = team_a_name, team_a_stats, team_b_stats
            s_name, s_bat, s_bowl = team_b_name, team_b_stats, team_a_stats
        else:
            f_name, f_bat, f_bowl = team_b_name, team_b_stats, team_a_stats
            s_name, s_bat, s_bowl = team_a_name, team_a_stats, team_b_stats

        # Innings 1
        inn1 = self.realistic_innings(
            batting_team_stats=f_bat,
            bowling_team_stats=f_bowl,
            venue_phase_rpo=venue_phase_rpo,
            n_sims=n_sims,
            intel=intel,
        )
        first_dist = inn1["distribution"]

        # Innings 2 — for each first-innings sample, simulate a chase
        import numpy as np
        wins_first = 0
        wins_second = 0
        ties = 0
        second_dist = np.zeros(n_sims, dtype=int)
        margins_first = []
        margins_second = []
        # Aggregate per-over for chasing innings (for the chart)
        per_over_runs_2 = np.zeros((n_sims, 20))
        per_over_wkts_2 = np.zeros((n_sims, 20))

        for i in range(n_sims):
            target = int(first_dist[i]) + 1
            chase = self.realistic_innings(
                batting_team_stats=s_bat,
                bowling_team_stats=s_bowl,
                venue_phase_rpo=venue_phase_rpo,
                target=target,
                n_sims=1,   # single-shot chase per first-innings sample
                intel=intel,
            )
            chase_runs = int(chase["distribution"][0])
            chase_wkts = int(chase["wickets_mean"])
            second_dist[i] = chase_runs
            per_over_runs_2[i, :] = chase["per_over_runs_mean"]
            per_over_wkts_2[i, :] = chase["per_over_wkts_mean"]
            if chase_runs >= target:
                wins_second += 1
                margins_second.append(10 - chase_wkts)
            elif chase_runs == target - 1:
                ties += 1
            else:
                wins_first += 1
                margins_first.append(target - 1 - chase_runs)

        winner_probs = {
            f_name: wins_first / n_sims,
            s_name: wins_second / n_sims,
            "Tie":  ties / n_sims,
        }
        if winner_probs[f_name] >= winner_probs[s_name]:
            margin_runs = float(np.mean(margins_first)) if margins_first else 0
            expected = {"winner": f_name, "by": "runs",
                        "value": round(margin_runs, 1)}
        else:
            margin_w = float(np.mean(margins_second)) if margins_second else 0
            expected = {"winner": s_name, "by": "wickets",
                        "value": round(margin_w, 1)}

        return {
            "first_team": {
                "name": f_name, "role": "bat-first",
                "distribution": first_dist,
                "mean": float(np.mean(first_dist)),
                "median": float(np.median(first_dist)),
                "p10": float(np.percentile(first_dist, 10)),
                "p90": float(np.percentile(first_dist, 90)),
                "per_over_runs_mean": inn1["per_over_runs_mean"],
                "per_over_wkts_mean": inn1["per_over_wkts_mean"],
                "wickets_mean": inn1["wickets_mean"],
            },
            "second_team": {
                "name": s_name, "role": "chasing",
                "distribution": second_dist,
                "mean": float(np.mean(second_dist)),
                "median": float(np.median(second_dist)),
                "p10": float(np.percentile(second_dist, 10)),
                "p90": float(np.percentile(second_dist, 90)),
                "per_over_runs_mean": per_over_runs_2.mean(axis=0).tolist(),
                "per_over_wkts_mean": per_over_wkts_2.mean(axis=0).tolist(),
                "wickets_mean": float(np.mean([
                    sum(per_over_wkts_2[i, :]) for i in range(n_sims)
                ])),
            },
            "winner_probs": winner_probs,
            "expected": expected,
            "n_sims": n_sims,
        }

    def pre_match_full(self, team_a_name: str, team_b_name: str,
                       team_a_bat_rpo: float, team_a_bowl_eco: float,
                       team_b_bat_rpo: float, team_b_bowl_eco: float,
                       venue_phase_rpo: dict | None = None,
                       n_sims: int = 200,
                       team_a_bats_first: bool = True) -> dict:
        """
        Predict the FULL match (both innings) with a winner.

        Returns:
            {
              "team_a": {name, role: 'bat'|'chase', score_dist, mean, median, p10, p90},
              "team_b": {...},
              "winner_probs": {team_a_name: 0.62, team_b_name: 0.36, "Tie": 0.02},
              "expected_margin": {"winner": ..., "runs": 12, "wickets": 3},
            }
        """
        if team_a_bats_first:
            first_name, first_bat, first_bowl = team_a_name, team_a_bat_rpo, team_b_bowl_eco
            second_name, second_bat, second_bowl = team_b_name, team_b_bat_rpo, team_a_bowl_eco
        else:
            first_name, first_bat, first_bowl = team_b_name, team_b_bat_rpo, team_a_bowl_eco
            second_name, second_bat, second_bowl = team_a_name, team_a_bat_rpo, team_b_bowl_eco

        # Innings 1: simulate first innings totals
        inn1 = self.pre_match_simulate(
            batting_team_str_avg=first_bat,
            bowling_team_eco=first_bowl,
            venue_phase_rpo=venue_phase_rpo,
            n_sims=n_sims,
        )
        first_dist = inn1["distribution"]

        # Innings 2: for each first-innings total sample, simulate a chase
        # (capped at 120 balls / 10 wickets, ends if target reached)
        wins_first = 0
        wins_second = 0
        ties = 0
        second_dist = np.zeros(n_sims)
        margins_first = []   # list of run margins when team batting first wins
        margins_second = []  # list of wickets-in-hand when chasing team wins

        for i in range(n_sims):
            target = int(first_dist[i]) + 1
            # Quick chase sim — same engine as pre_match_simulate but stops at target
            state = MatchState(
                innings=2, over_num=0, ball_in_over=1,
                runs_scored=0, wickets_fallen=0, balls_bowled=0,
                target=target,
                venue_avg_rpb=(venue_phase_rpo or {}).get("powerplay", 8.4) / 6,
                striker=f"AvgBat_{second_bat:.0f}",
                non_striker=f"AvgBat_{second_bat:.0f}",
                bowler=f"AvgBowl_{second_bowl:.0f}",
            )
            phase_map = (venue_phase_rpo or
                         {"powerplay": 8.4, "middle": 7.9, "death": 10.6})
            for ball in range(120):
                if state.runs_scored >= target:
                    break
                if state.wickets_fallen >= 10:
                    break
                phase = ("powerplay" if state.over_num < 6
                         else "middle" if state.over_num < 15
                         else "death")
                state.venue_avg_rpb = phase_map.get(phase, 8.0) / 6
                ob = self.predict_next_ball_outcomes(state)
                outcomes = list(ob.keys())
                probs = np.array(list(ob.values()))
                probs = probs / probs.sum()
                outcome = np.random.choice(outcomes, p=probs)
                runs_map = {"dot": 0, "single": 1, "double": 2,
                            "boundary": 4, "six": 6, "wicket": 0, "extra": 1}
                runs = runs_map[outcome]
                state.runs_scored += runs
                state.recent_balls.append(runs)
                if outcome != "extra":
                    state.balls_bowled += 1
                    state.ball_in_over += 1
                if state.ball_in_over > 6:
                    state.ball_in_over = 1
                    state.striker, state.non_striker = state.non_striker, state.striker
                state.over_num = state.balls_bowled // 6
                if outcome == "wicket":
                    state.wickets_fallen += 1
                if runs in (1, 3):
                    state.striker, state.non_striker = state.non_striker, state.striker

            second_dist[i] = state.runs_scored
            if state.runs_scored >= target:
                wins_second += 1
                margins_second.append(10 - state.wickets_fallen)
            elif state.runs_scored == target - 1:
                ties += 1
            else:
                wins_first += 1
                margins_first.append(target - 1 - state.runs_scored)

        winner_probs = {
            first_name:  wins_first / n_sims,
            second_name: wins_second / n_sims,
            "Tie":       ties / n_sims,
        }
        # Decide expected outcome & margin
        if winner_probs[first_name] >= winner_probs[second_name]:
            margin_runs = float(np.mean(margins_first)) if margins_first else 0
            expected = {
                "winner": first_name,
                "by": "runs",
                "value": round(margin_runs, 1),
            }
        else:
            margin_wkts = float(np.mean(margins_second)) if margins_second else 0
            expected = {
                "winner": second_name,
                "by": "wickets",
                "value": round(margin_wkts, 1),
            }

        return {
            "first_team":  {
                "name": first_name, "role": "bat-first",
                "distribution": first_dist,
                "mean": float(np.mean(first_dist)),
                "median": float(np.median(first_dist)),
                "p10": float(np.percentile(first_dist, 10)),
                "p90": float(np.percentile(first_dist, 90)),
            },
            "second_team": {
                "name": second_name, "role": "chasing",
                "distribution": second_dist,
                "mean": float(np.mean(second_dist)),
                "median": float(np.median(second_dist)),
                "p10": float(np.percentile(second_dist, 10)),
                "p90": float(np.percentile(second_dist, 90)),
            },
            "winner_probs": winner_probs,
            "expected": expected,
            "n_sims": n_sims,
        }


# ----- Demo usage -----
if __name__ == "__main__":
    from data_loader import CricketDataLoader
    from feature_engine import FeatureEngine
    from model import BallOutcomeModel

    # Load/train model
    print("=== Loading data and training model ===")
    loader = CricketDataLoader(data_dir="data")
    raw = loader.load_all_matches()
    clean = loader.clean_data(raw)
    context = loader.compute_match_context(clean)

    engine = FeatureEngine()
    X, y = engine.build_features(context)

    ball_model = BallOutcomeModel(model_dir="models")
    ball_model.train(X, y)

    # Set up simulator
    sim = MonteCarloSimulator(ball_model, engine, n_simulations=2000)

    # --- Scenario: T20, 1st innings, 10th over, 75/2 ---
    print("\n" + "=" * 60)
    print("SIMULATION: 1st Innings, Over 10, Score 75/2")
    print("=" * 60)

    state = MatchState(
        innings=1,
        over_num=10,
        ball_in_over=1,
        runs_scored=75,
        wickets_fallen=2,
        balls_bowled=60,
        venue_avg_rpb=1.3,
        striker="India_Bat3",
        non_striker="India_Bat4",
        bowler="Australia_Bowl2",
        batters_remaining=[f"India_Bat{i}" for i in range(5, 12)],
        recent_balls=[1, 4, 0, 1, 2, 0, 1, 6, 0, 1, 1, 4]
    )

    # Simulate this over
    over_result = sim.simulate_over(state)
    print(f"\nOver 11 projection:")
    print(f"  Mean:   {over_result['mean']:.1f} runs")
    print(f"  Median: {over_result['median']:.1f} runs")
    print(f"  Std:    {over_result['std']:.1f}")
    print(f"  P(under 6.5): {over_result.get('prob_under_6.5', 'N/A')}")
    print(f"  P(over 8.5):  {over_result.get('prob_over_8.5', 'N/A')}")

    # Simulate innings total
    innings_result = sim.simulate_innings(state)
    print(f"\nInnings total projection (from 75/2 in 10 overs):")
    print(f"  Mean:   {innings_result['mean']:.0f}")
    print(f"  Median: {innings_result['median']:.0f}")
    print(f"  10th-90th percentile: {innings_result['percentiles'][10]:.0f} - "
          f"{innings_result['percentiles'][90]:.0f}")
    print(f"  Score ranges:")
    for range_label, prob in innings_result['prob_ranges'].items():
        bar = '█' * int(prob * 40)
        print(f"    {range_label}: {prob:.1%} {bar}")

    # Value detection
    print(f"\n--- Value Detection vs Bookmaker ---")
    value = sim.find_value_vs_bookmaker(
        state,
        bookmaker_line=7.5,
        bookmaker_odds_over=1.85,
        bookmaker_odds_under=1.95,
        market_type="over_runs"
    )
    print(f"  Bookmaker line: Over/Under {value['bookmaker_line']}")
    print(f"  Model mean: {value['model_mean']:.1f} runs")
    print(f"  Model P(over): {value['model_prob_over']:.1%} vs "
          f"Implied: {value['implied_prob_over']:.1%}")
    print(f"  Model P(under): {value['model_prob_under']:.1%} vs "
          f"Implied: {value['implied_prob_under']:.1%}")
    print(f"  Edge (over): {value['edge_over']:+.1%}")
    print(f"  Edge (under): {value['edge_under']:+.1%}")
    print(f"  >>> RECOMMENDATION: {value['recommendation']} "
          f"(confidence: {value['confidence']})")
    if value['quarter_kelly_stake'] > 0:
        print(f"  >>> Quarter-Kelly stake: {value['quarter_kelly_stake']:.1%} of bankroll")
