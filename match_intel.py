"""
match_intel.py
--------------
Pre-computed intelligence layer that the dashboard queries for context.

Crunches every IPL ball in `data/` once (takes ~10s) and caches the result
to `models/match_intel.pkl` so the dashboard loads instantly afterwards.

Provides:
  • Venue stats     — avg first-innings score, RPO by phase, par score
  • Batter careers  — strike rate, boundary %, dismissal rate, phase splits
  • Bowler careers  — economy, strike rate, wickets, phase splits
  • Head-to-head    — striker-vs-bowler historical matchup
"""

from __future__ import annotations

import os
import glob
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

CACHE_PATH = "models/match_intel.pkl"
CACHE_VERSION = 6  # bump to invalidate (XI filtered to current season + player recent form)


@dataclass
class MatchIntelligence:
    venue_stats: dict = field(default_factory=dict)
    batter_stats: dict = field(default_factory=dict)
    bowler_stats: dict = field(default_factory=dict)
    matchup_stats: dict = field(default_factory=dict)  # (striker, bowler) -> dict
    team_stats: dict = field(default_factory=dict)     # team_name -> stats
    h2h_stats: dict = field(default_factory=dict)      # (teamA, teamB) -> stats
    venue_team_stats: dict = field(default_factory=dict)  # (venue, team) -> stats
    venue_chase_stats: dict = field(default_factory=dict)  # venue -> chase stats
    team_recent: dict = field(default_factory=dict)    # team -> last-N batting scores
    # Real conditional ball-outcome distributions, derived from raw IPL data:
    # key = (phase, wkts_bucket, chase_bucket)
    # value = dict of {dot, single, double, triple, four, six, wicket, extra}
    ball_outcomes: dict = field(default_factory=dict)
    # Likely playing-11 per team derived from their last N matches:
    # team_name -> {
    #   "batters": [list of 11 names in typical batting order],
    #   "bowlers": [list of (name, typical_overs) up to 6 main bowlers],
    #   "based_on_matches": int,
    # }
    team_xi: dict = field(default_factory=dict)
    version: int = CACHE_VERSION

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def load_or_build(cls, data_dir: str = "data",
                      force: bool = False) -> "MatchIntelligence":
        """Load cached intel, or rebuild from CSVs if missing/outdated/forced."""
        if not force and os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as fh:
                    obj = pickle.load(fh)
                if getattr(obj, "version", 1) >= CACHE_VERSION:
                    return obj
                print(f"[intel] Cache version mismatch — rebuilding")
            except Exception:
                pass
        return cls.build(data_dir)

    @classmethod
    def build(cls, data_dir: str = "data") -> "MatchIntelligence":
        files = [
            f for f in glob.glob(os.path.join(data_dir, "*.csv"))
            if "_info" not in os.path.basename(f)
        ]
        print(f"[intel] Crunching {len(files)} match files…")

        frames = []
        for path in files:
            try:
                df = pd.read_csv(path, low_memory=False)
                if "ball" in df.columns and "runs_off_bat" in df.columns:
                    frames.append(df)
            except Exception:
                continue
        if not frames:
            print("[intel] No data — returning empty intel")
            return cls()

        df = pd.concat(frames, ignore_index=True)
        for col in ("runs_off_bat", "extras", "wides", "noballs"):
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        df["total_runs"] = df["runs_off_bat"] + df["extras"]
        df["is_legal"] = (df["wides"] == 0) & (df["noballs"] == 0)
        df["is_wicket"] = df.get("wicket_type", pd.Series(dtype=str)).fillna("").ne("")
        df["over_num"] = df["ball"].astype(str).str.split(".").str[0].astype(float).astype(int)
        df["phase"] = pd.cut(df["over_num"], bins=[-1, 5, 14, 20],
                             labels=["powerplay", "middle", "death"])

        intel = cls()

        # Venue stats
        for venue, vdf in df.groupby("venue", dropna=True):
            inn1 = vdf[vdf["innings"] == 1]
            if inn1.empty:
                continue
            totals = inn1.groupby("match_id")["total_runs"].sum()
            phase_rpo = {}
            for ph in ("powerplay", "middle", "death"):
                pdf = inn1[inn1["phase"] == ph]
                legal_balls = pdf["is_legal"].sum()
                if legal_balls > 0:
                    phase_rpo[ph] = float(pdf["total_runs"].sum() * 6 / legal_balls)
            intel.venue_stats[venue] = {
                "matches": int(totals.count()),
                "avg_first_innings": float(totals.mean()),
                "median_first_innings": float(totals.median()),
                "phase_rpo": phase_rpo,
                "par_score": float(totals.quantile(0.5)),
            }

        # Batter career
        for name, bdf in df.groupby("striker", dropna=True):
            if not name or pd.isna(name):
                continue
            balls = int(bdf["is_legal"].sum())
            if balls < 30:
                continue
            runs = int(bdf["runs_off_bat"].sum())
            outs = int(bdf["player_dismissed"].fillna("").eq(name).sum())
            phase_sr = {}
            for ph in ("powerplay", "middle", "death"):
                pdf = bdf[bdf["phase"] == ph]
                pballs = int(pdf["is_legal"].sum())
                if pballs >= 12:
                    phase_sr[ph] = float(pdf["runs_off_bat"].sum() * 100 / pballs)
            intel.batter_stats[name] = {
                "balls": balls,
                "runs": runs,
                "sr": float(runs * 100 / balls),
                "boundary_pct": float(((bdf["runs_off_bat"] == 4) |
                                        (bdf["runs_off_bat"] == 6)).sum() * 100 / balls),
                "dismissals": outs,
                "avg": float(runs / max(outs, 1)),
                "dot_pct": float((bdf["runs_off_bat"] == 0).sum() * 100 / balls),
                "phase_sr": phase_sr,
            }

        # Bowler career
        for name, bdf in df.groupby("bowler", dropna=True):
            if not name or pd.isna(name):
                continue
            balls = int(bdf["is_legal"].sum())
            if balls < 36:
                continue
            runs = int(bdf["total_runs"].sum())
            wkts = int(bdf["is_wicket"].sum())
            phase_eco = {}
            for ph in ("powerplay", "middle", "death"):
                pdf = bdf[bdf["phase"] == ph]
                pballs = int(pdf["is_legal"].sum())
                if pballs >= 12:
                    phase_eco[ph] = float(pdf["total_runs"].sum() * 6 / pballs)
            intel.bowler_stats[name] = {
                "balls": balls,
                "runs_conceded": runs,
                "economy": float(runs * 6 / balls),
                "wickets": wkts,
                "strike_rate": float(balls / max(wkts, 1)),
                "dot_pct": float((bdf["runs_off_bat"] == 0).sum() * 100 / balls),
                "phase_eco": phase_eco,
            }

        # ---- Striker × Bowler head-to-head (sparse, only ≥6 balls) ----
        h2h_groups = df.groupby(["striker", "bowler"], dropna=True)
        for (striker, bowler), grp in h2h_groups:
            if not striker or not bowler or pd.isna(striker) or pd.isna(bowler):
                continue
            balls = int(grp["is_legal"].sum())
            if balls < 6:
                continue
            runs = int(grp["runs_off_bat"].sum())
            wkts = int(grp["is_wicket"].sum())
            intel.matchup_stats[(striker, bowler)] = {
                "balls": balls,
                "runs": runs,
                "sr": float(runs * 100 / balls),
                "dismissals": wkts,
                "boundary_pct": float(((grp["runs_off_bat"] == 4) |
                                        (grp["runs_off_bat"] == 6)).sum() * 100 / balls),
                "dot_pct": float((grp["runs_off_bat"] == 0).sum() * 100 / balls),
            }

        # ---- Team-level batting & bowling averages ----
        for team, tdf in df.groupby("batting_team", dropna=True):
            if not team or pd.isna(team):
                continue
            balls = int(tdf["is_legal"].sum())
            if balls < 600:  # ~5 matches
                continue
            runs = int(tdf["total_runs"].sum())
            wkts = int(tdf["is_wicket"].sum())
            intel.team_stats.setdefault(team, {})["batting_rpo"] = float(runs * 6 / balls)
            intel.team_stats[team]["batting_wicket_rate"] = float(wkts / balls)
            intel.team_stats[team]["boundary_pct"] = float(
                ((tdf["runs_off_bat"] == 4) | (tdf["runs_off_bat"] == 6)).sum() * 100 / balls
            )
        for team, tdf in df.groupby("bowling_team", dropna=True):
            if not team or pd.isna(team):
                continue
            balls = int(tdf["is_legal"].sum())
            if balls < 600:
                continue
            runs = int(tdf["total_runs"].sum())
            wkts = int(tdf["is_wicket"].sum())
            intel.team_stats.setdefault(team, {})["bowling_eco"] = float(runs * 6 / balls)
            intel.team_stats[team]["bowling_wicket_rate"] = float(wkts / balls)

        # ---- Per-match outcomes from _info.csv files (toss/winner/chase) ----
        info_files = glob.glob(os.path.join(data_dir, "*_info.csv"))
        match_outcomes = []   # list of dicts per match
        for path in info_files:
            try:
                rec = {"venue": "", "teams": [], "toss_winner": "",
                       "toss_decision": "", "match_winner": "",
                       "innings_team": []}
                for line in open(path):
                    if not line.startswith("info,"):
                        continue
                    parts = line.strip().split(",", 2)
                    if len(parts) < 3:
                        continue
                    k, v = parts[1], parts[2].strip().strip('"')
                    if k == "venue":
                        rec["venue"] = v
                    elif k == "team":
                        rec["teams"].append(v)
                    elif k == "toss_winner":
                        rec["toss_winner"] = v
                    elif k == "toss_decision":
                        rec["toss_decision"] = v
                    elif k == "winner":
                        rec["match_winner"] = v
                if len(rec["teams"]) == 2 and rec["match_winner"]:
                    match_outcomes.append(rec)
            except Exception:
                continue
        print(f"[intel] Parsed {len(match_outcomes)} match outcomes from _info files")

        # ---- Toss & chase stats per team & venue ----
        # Per team: toss wins, match wins after winning toss, bat-first wins, chase wins
        for rec in match_outcomes:
            tw = rec["toss_winner"]
            mw = rec["match_winner"]
            decision = rec["toss_decision"]
            venue = rec["venue"]
            for team in rec["teams"]:
                ts = intel.team_stats.setdefault(team, {})
                ts.setdefault("matches_played", 0)
                ts.setdefault("matches_won", 0)
                ts.setdefault("toss_won", 0)
                ts.setdefault("bat_first_wins", 0)
                ts.setdefault("bat_first_played", 0)
                ts.setdefault("chase_wins", 0)
                ts.setdefault("chase_played", 0)
                ts["matches_played"] += 1
                if mw == team:
                    ts["matches_won"] += 1
                if tw == team:
                    ts["toss_won"] += 1
                # Did this team bat first?
                if tw == team and decision == "bat":
                    bat_first = True
                elif tw != team and decision == "field":
                    bat_first = True
                else:
                    bat_first = False
                if bat_first:
                    ts["bat_first_played"] += 1
                    if mw == team:
                        ts["bat_first_wins"] += 1
                else:
                    ts["chase_played"] += 1
                    if mw == team:
                        ts["chase_wins"] += 1

            # Venue chase stats
            if venue:
                vs = intel.venue_chase_stats.setdefault(venue, {
                    "matches": 0, "first_inn_wins": 0, "chase_wins": 0,
                    "toss_winner_wins": 0, "elected_bat": 0, "elected_field": 0,
                })
                vs["matches"] += 1
                # Determine bat-first team
                if decision == "bat":
                    bat_first_team = tw
                    vs["elected_bat"] += 1
                else:
                    bat_first_team = (rec["teams"][0] if rec["teams"][0] != tw
                                       else rec["teams"][1])
                    vs["elected_field"] += 1
                if mw == bat_first_team:
                    vs["first_inn_wins"] += 1
                else:
                    vs["chase_wins"] += 1
                if mw == tw:
                    vs["toss_winner_wins"] += 1

        # ---- H2H per team-pair ----
        for rec in match_outcomes:
            if len(rec["teams"]) < 2:
                continue
            a, b = sorted(rec["teams"])
            mw = rec["match_winner"]
            key = (a, b)
            h = intel.h2h_stats.setdefault(key, {
                "played": 0, f"{a}_wins": 0, f"{b}_wins": 0,
            })
            h["played"] += 1
            if mw == a:
                h[f"{a}_wins"] = h.get(f"{a}_wins", 0) + 1
            elif mw == b:
                h[f"{b}_wins"] = h.get(f"{b}_wins", 0) + 1

        # ---- Per-team last-10 batting totals (recent form) ----
        # Compute first-innings totals per match
        first_inn_totals = (df[df["innings"] == 1].groupby(["match_id", "batting_team"])
                              ["total_runs"].sum().reset_index())
        for team, tdf in first_inn_totals.groupby("batting_team"):
            scores = tdf["total_runs"].tolist()[-10:]
            intel.team_recent.setdefault(team, {})["recent_first_inn_scores"] = scores
            if scores:
                intel.team_recent[team]["recent_avg"] = float(sum(scores) / len(scores))
        # Same for second innings (chase totals)
        second_inn_totals = (df[df["innings"] == 2].groupby(["match_id", "batting_team"])
                              ["total_runs"].sum().reset_index())
        for team, tdf in second_inn_totals.groupby("batting_team"):
            scores = tdf["total_runs"].tolist()[-10:]
            if scores:
                intel.team_recent.setdefault(team, {})["recent_chase_scores"] = scores
                intel.team_recent[team]["recent_chase_avg"] = float(sum(scores) / len(scores))

        # ---- Real conditional ball-outcome distributions ----
        # Bucketize every legal delivery by:
        #   phase            (powerplay 0-5 / middle 6-14 / death 15-19)
        #   wickets_bucket   (0, 1, 2, 3, 4-5, 6+)
        #   chase_bucket     (none for inn1, easy/medium/hard for inn2)
        #
        # For each (phase, wkts, chase) tuple, count actual occurrences of
        # each outcome and compute the empirical probability.
        print("[intel] Computing real conditional ball-outcome distributions…")

        # We need cumulative wickets and current RRR per delivery.
        df_sorted = df.sort_values(["match_id", "innings", "over_num", "ball"]).copy()
        df_sorted["wkts_so_far"] = (
            df_sorted.groupby(["match_id", "innings"])["is_wicket"]
            .cumsum().astype(int) - df_sorted["is_wicket"].astype(int)
        )
        # First innings totals → target for innings 2
        first_inn_totals = (df_sorted[df_sorted["innings"] == 1]
                              .groupby("match_id")["total_runs"].sum())
        df_sorted = df_sorted.merge(
            first_inn_totals.rename("first_inn_total"),
            left_on="match_id", right_index=True, how="left",
        )
        df_sorted["target"] = df_sorted["first_inn_total"] + 1
        # cumulative runs in second innings
        df_sorted["runs_so_far"] = (
            df_sorted.groupby(["match_id", "innings"])["total_runs"]
            .cumsum().astype(int) - df_sorted["total_runs"].astype(int)
        )
        df_sorted["balls_so_far"] = (
            df_sorted.groupby(["match_id", "innings"])["is_legal"]
            .cumsum().astype(int) - df_sorted["is_legal"].astype(int)
        )

        def _wkts_bucket(w):
            if w == 0: return "0w"
            if w == 1: return "1w"
            if w == 2: return "2w"
            if w == 3: return "3w"
            if w in (4, 5): return "45w"
            return "6+w"

        def _chase_bucket(row):
            if row["innings"] != 2 or row["balls_so_far"] >= 120:
                return "none"
            balls_left = max(1, 120 - row["balls_so_far"])
            needed = max(0, row["target"] - row["runs_so_far"])
            rrr = needed * 6 / balls_left
            if needed <= 0:
                return "none"
            if rrr <= 8:    return "easy"
            if rrr <= 11:   return "medium"
            return "hard"

        df_sorted["wkts_bucket"] = df_sorted["wkts_so_far"].apply(_wkts_bucket)
        df_sorted["chase_bucket"] = df_sorted.apply(_chase_bucket, axis=1)

        def _outcome(row):
            if row["wides"] > 0 or row["noballs"] > 0:
                return "extra"
            if row["is_wicket"] == 1:
                return "wicket"
            r = int(row["runs_off_bat"])
            if r == 0: return "dot"
            if r == 1: return "single"
            if r == 2: return "double"
            if r == 3: return "triple"
            if r == 4: return "four"
            if r >= 6: return "six"
            return "dot"

        df_sorted["outcome"] = df_sorted.apply(_outcome, axis=1)

        # Group and count
        grouped = (df_sorted.groupby(["phase", "wkts_bucket", "chase_bucket", "outcome"])
                   .size().reset_index(name="count"))
        # Compute probabilities per (phase, wkts, chase) bucket
        bucket_totals = (grouped.groupby(["phase", "wkts_bucket", "chase_bucket"])
                         ["count"].sum().reset_index(name="total"))
        grouped = grouped.merge(bucket_totals,
                                 on=["phase", "wkts_bucket", "chase_bucket"])
        grouped["prob"] = grouped["count"] / grouped["total"]

        # Build the lookup table
        outcomes_all = ["dot", "single", "double", "triple",
                        "four", "six", "wicket", "extra"]
        for (phase, wkts, chase), bucket_df in grouped.groupby(
                ["phase", "wkts_bucket", "chase_bucket"]):
            dist = {o: 0.0 for o in outcomes_all}
            for _, r in bucket_df.iterrows():
                dist[r["outcome"]] = float(r["prob"])
            # Normalise (in case of 0-prob outcomes)
            s = sum(dist.values())
            if s > 0:
                dist = {k: v / s for k, v in dist.items()}
            intel.ball_outcomes[(str(phase), wkts, chase)] = {
                "dist": dist,
                "n": int(bucket_df["total"].iloc[0]),
            }
        print(f"[intel] Built {len(intel.ball_outcomes)} conditional ball-outcome buckets")

        # ---- Likely playing 11 per team — CURRENT SEASON ONLY ----
        # Squads change every IPL season (transfers, auctions, retirements).
        # Using all-time data gives stale XIs (e.g. CSK lineup from 2018 mixed
        # with 2026 lineup). So we filter to the most-recent season available.
        # Smart fallback: if a team has <3 matches in the current season,
        # blend in the previous season as well.
        print("[intel] Deriving likely XIs per team from CURRENT SEASON…")

        # Determine current season (most recent in data)
        seasons = sorted(set(str(s) for s in df["season"].dropna().unique()
                              if s and str(s) != "nan"), reverse=True)
        current_season = seasons[0] if seasons else None
        prev_season = seasons[1] if len(seasons) > 1 else None
        print(f"[intel] Current season: {current_season}  ·  prev: {prev_season}")

        # Map match_id → date
        if "start_date" in df.columns:
            match_dates = (df.groupby("match_id")["start_date"].first().to_dict())
        else:
            match_dates = {mid: "" for mid in df["match_id"].unique()}

        def _build_xi_for(team: str, season_filter) -> dict:
            """season_filter is a list of season strings to consider."""
            from collections import defaultdict
            team_innings = df[
                (df["batting_team"] == team) &
                (df["season"].astype(str).isin(season_filter))
            ]
            if team_innings.empty:
                return None
            sorted_ids = sorted(
                team_innings["match_id"].unique().tolist(),
                key=lambda m: str(match_dates.get(m, "")),
                reverse=True,
            )
            recent_innings = team_innings[team_innings["match_id"].isin(sorted_ids)]

            position_data = []
            for mid, mgrp in recent_innings.groupby("match_id"):
                wkts_so_far = (mgrp["is_wicket"].cumsum().astype(int)
                                - mgrp["is_wicket"].astype(int))
                mgrp = mgrp.assign(_wkts_at_arrival=wkts_so_far)
                first_app = (mgrp.groupby("striker", sort=False)["_wkts_at_arrival"]
                             .first().to_dict())
                for s, w in first_app.items():
                    if s and not pd.isna(s):
                        position_data.append((s, int(w)))
            if not position_data:
                return None

            pos_sum = defaultdict(int); pos_cnt = defaultdict(int)
            appearances = defaultdict(int)
            for s, p in position_data:
                pos_sum[s] += p; pos_cnt[s] += 1; appearances[s] += 1
            avg_pos = {s: pos_sum[s] / pos_cnt[s] for s in pos_sum}

            top_11 = sorted(appearances.items(), key=lambda x: -x[1])[:11]
            top_11_names = [n for n, _ in top_11]
            batters_in_order = sorted(top_11_names,
                                       key=lambda n: avg_pos.get(n, 99))

            team_bowl_innings = df[
                (df["bowling_team"] == team) &
                (df["season"].astype(str).isin(season_filter))
            ]
            recent_bowl_ids = sorted(
                team_bowl_innings["match_id"].unique().tolist(),
                key=lambda m: str(match_dates.get(m, "")),
                reverse=True,
            )
            recent_bowl = team_bowl_innings[
                team_bowl_innings["match_id"].isin(recent_bowl_ids)
            ]
            bowler_balls = (recent_bowl.groupby("bowler")["is_legal"]
                            .sum().sort_values(ascending=False))
            top_bowlers = []
            for name, balls in bowler_balls.head(6).items():
                if name and not pd.isna(name):
                    typical_overs = round(
                        balls / max(len(recent_bowl_ids), 1) / 6, 1)
                    top_bowlers.append((name, typical_overs))

            return {
                "batters": batters_in_order,
                "bowlers": top_bowlers,
                "based_on_matches": len(sorted_ids),
                "season_used": ",".join(season_filter),
            }

        for team in df["batting_team"].dropna().unique():
            if not team or pd.isna(team):
                continue
            # Try current season only
            xi = _build_xi_for(team, [current_season]) if current_season else None
            if not xi or xi["based_on_matches"] < 3:
                # Fallback: combine current + previous season
                if prev_season and current_season:
                    xi = _build_xi_for(team, [current_season, prev_season])
                elif prev_season:
                    xi = _build_xi_for(team, [prev_season])
            if xi and xi["batters"]:
                intel.team_xi[team] = xi

        # ---- Per-player RECENT FORM (last N legal balls in current+prev season) ----
        # Players' career stats include data from years ago. Recent form is more
        # predictive of current performance.
        print("[intel] Computing per-player recent form (current + prev season)…")
        recent_seasons = [s for s in (current_season, prev_season) if s]
        recent_df = df[df["season"].astype(str).isin(recent_seasons)] if recent_seasons else df.head(0)

        for name, bdf in recent_df.groupby("striker"):
            if not name or pd.isna(name):
                continue
            balls = int(bdf["is_legal"].sum())
            if balls < 30:
                continue
            runs = int(bdf["runs_off_bat"].sum())
            outs = int(bdf["player_dismissed"].fillna("").eq(name).sum())
            stats = intel.batter_stats.get(name, {})
            stats["recent_balls"] = balls
            stats["recent_runs"] = runs
            stats["recent_sr"] = float(runs * 100 / balls)
            stats["recent_avg"] = float(runs / max(outs, 1))
            stats["recent_boundary_pct"] = float(
                ((bdf["runs_off_bat"] == 4) | (bdf["runs_off_bat"] == 6)).sum() * 100 / balls
            )
            stats["recent_seasons"] = ",".join(recent_seasons)
            intel.batter_stats[name] = stats

        for name, bdf in recent_df.groupby("bowler"):
            if not name or pd.isna(name):
                continue
            balls = int(bdf["is_legal"].sum())
            if balls < 30:
                continue
            runs = int(bdf["total_runs"].sum())
            wkts = int(bdf["is_wicket"].sum())
            stats = intel.bowler_stats.get(name, {})
            stats["recent_balls"] = balls
            stats["recent_economy"] = float(runs * 6 / balls)
            stats["recent_wickets"] = wkts
            stats["recent_seasons"] = ",".join(recent_seasons)
            intel.bowler_stats[name] = stats

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(intel, fh)
        print(f"[intel] Built: {len(intel.venue_stats)} venues, "
              f"{len(intel.batter_stats)} batters, "
              f"{len(intel.bowler_stats)} bowlers, "
              f"{len(intel.matchup_stats)} matchups, "
              f"{len(intel.team_stats)} teams, "
              f"{len(intel.ball_outcomes)} ball-outcome buckets, "
              f"{len(intel.team_xi)} team XIs")
        return intel

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def venue(self, name: str) -> dict:
        if name in self.venue_stats:
            return self.venue_stats[name]
        # fuzzy: try matching by partial venue name
        for k in self.venue_stats:
            if name and (name in k or k in name):
                return self.venue_stats[k]
        return {"matches": 0, "avg_first_innings": 165.0,
                "phase_rpo": {"powerplay": 8.4, "middle": 7.9, "death": 10.6},
                "par_score": 165.0}

    def batter(self, name: str) -> Optional[dict]:
        return self.batter_stats.get(name)

    def bowler(self, name: str) -> Optional[dict]:
        return self.bowler_stats.get(name)

    def matchup(self, striker: str, bowler: str) -> Optional[dict]:
        """Career head-to-head between this striker and bowler."""
        return self.matchup_stats.get((striker, bowler))

    def team(self, team_name: str) -> dict:
        """Team batting + bowling averages from past IPL data."""
        return self.team_stats.get(team_name, {})

    def head_to_head(self, team_a: str, team_b: str) -> dict:
        """Past matches between two teams."""
        a, b = sorted([team_a, team_b])
        h = self.h2h_stats.get((a, b), {"played": 0})
        if h.get("played", 0) > 0:
            return {
                "played": h["played"],
                f"{team_a}_wins": h.get(f"{team_a}_wins", 0),
                f"{team_b}_wins": h.get(f"{team_b}_wins", 0),
                f"{team_a}_winrate": h.get(f"{team_a}_wins", 0) / h["played"],
                f"{team_b}_winrate": h.get(f"{team_b}_wins", 0) / h["played"],
            }
        return {"played": 0}

    def venue_chase(self, venue_name: str) -> dict:
        """Venue's bat-first vs chase outcomes."""
        if venue_name in self.venue_chase_stats:
            return self.venue_chase_stats[venue_name]
        # Fuzzy
        for k in self.venue_chase_stats:
            if venue_name and (venue_name in k or k in venue_name):
                return self.venue_chase_stats[k]
        return {"matches": 0, "first_inn_wins": 0, "chase_wins": 0,
                "toss_winner_wins": 0, "elected_bat": 0, "elected_field": 0}

    def team_recent_form(self, team_name: str) -> dict:
        return self.team_recent.get(team_name, {})

    def team_likely_xi(self, team_name: str) -> dict:
        """Likely playing-11 derived from team's last 10 IPL matches."""
        return self.team_xi.get(team_name, {"batters": [], "bowlers": [],
                                             "based_on_matches": 0})

    def ball_outcome_dist(self, phase: str, wkts: int,
                          chase_bucket: str = "none") -> Optional[dict]:
        """
        Real conditional probability of each ball outcome, drawn from
        the user's IPL data. Falls back to nearest bucket if exact missing.
        """
        wkts_b = ("0w" if wkts == 0 else
                  "1w" if wkts == 1 else
                  "2w" if wkts == 2 else
                  "3w" if wkts == 3 else
                  "45w" if wkts in (4, 5) else "6+w")
        # Direct lookup
        key = (phase, wkts_b, chase_bucket)
        if key in self.ball_outcomes:
            return self.ball_outcomes[key]
        # Fall back: same phase + wkts, no chase
        key = (phase, wkts_b, "none")
        if key in self.ball_outcomes:
            return self.ball_outcomes[key]
        # Fall back: same phase, 0 wkts, no chase
        key = (phase, "0w", "none")
        return self.ball_outcomes.get(key)


if __name__ == "__main__":
    intel = MatchIntelligence.build("data")
    print(f"Venues:  {len(intel.venue_stats)}")
    print(f"Batters: {len(intel.batter_stats)}")
    print(f"Bowlers: {len(intel.bowler_stats)}")
