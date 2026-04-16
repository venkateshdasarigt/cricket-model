"""
live_fetcher.py
---------------
Pluggable fetchers that produce a stream of `BallEvent`s for the predictor.

Why an abstraction?
-------------------
Cricket data sources keep changing. ESPN Cricinfo's internal API got
firewalled in 2024-2025. Cricbuzz killed its public endpoints around the
same time. To keep the prediction loop independent of the data source,
every fetcher implements the same `iter_balls()` generator interface.

Implementations included:
  - CricSheetReplayFetcher : replays a real IPL match ball-by-ball at a
    chosen speed. Works offline. No API keys needed. Best for testing
    the full live pipeline.
  - CricAPIFetcher         : polls cricketdata.org (free tier, 100 hits/day).
    Needs API key from https://cricketdata.org
  - CricbuzzRapidAPIFetcher: polls Cricbuzz via RapidAPI. Free tier ~100
    calls/day. Needs RapidAPI key from https://rapidapi.com/cricketapilive

To plug in a new source, subclass `LiveFetcher` and yield BallEvents.
"""

from __future__ import annotations

import os
import time
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from data_loader import CricketDataLoader


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (no python-dotenv dep)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


# ---------------------------------------------------------------------------
# Common event shape — every fetcher emits this
# ---------------------------------------------------------------------------

@dataclass
class BallEvent:
    """One delivery, normalized across all data sources."""
    match_id: str
    innings: int
    over: int            # 0-indexed
    ball_in_over: int    # 1-indexed (legal ball position)
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    runs_off_bat: int
    extras: int
    wides: int
    noballs: int
    is_wicket: bool
    wicket_type: str = ""
    player_dismissed: str = ""
    target: Optional[int] = None  # set on innings-2 deliveries

    @property
    def total_runs(self) -> int:
        return self.runs_off_bat + self.extras


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LiveFetcher(ABC):
    """All fetchers expose the same generator interface."""

    @abstractmethod
    def iter_balls(self) -> Iterator[BallEvent]:
        """Yield BallEvents in chronological order until the match ends."""
        ...


# ---------------------------------------------------------------------------
# 1) CricSheet replay — works offline, no API needed
# ---------------------------------------------------------------------------

class CricSheetReplayFetcher(LiveFetcher):
    """
    Replay a historical match (CricSheet CSV) ball-by-ball as if it were live.

    Useful for end-to-end testing of the live pipeline without paying
    for a live data feed.
    """

    def __init__(self, csv_path: str, ball_delay_s: float = 0.3):
        """
        Args:
            csv_path: path to a single-match CricSheet CSV (e.g. data/1370352.csv)
            ball_delay_s: seconds to sleep between balls. 0.0 = instant replay.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.csv_path = csv_path
        self.delay = ball_delay_s
        df = pd.read_csv(csv_path, low_memory=False)
        if "ball" not in df.columns:
            raise ValueError(f"{csv_path} is not a CricSheet ball-by-ball file")
        self.df = CricketDataLoader().clean_data(df)

    def iter_balls(self) -> Iterator[BallEvent]:
        # First innings total → target for innings 2
        first_inn_total = int(
            self.df.loc[self.df["innings"] == 1, "total_runs"].sum()
        )
        target = first_inn_total + 1

        for _, row in self.df.iterrows():
            yield BallEvent(
                match_id=str(row.get("match_id", os.path.basename(self.csv_path))),
                innings=int(row["innings"]),
                over=int(row["over_num"]),
                ball_in_over=int(row["ball_in_over"]),
                batting_team=str(row.get("batting_team", "")),
                bowling_team=str(row.get("bowling_team", "")),
                striker=str(row.get("striker", "")),
                non_striker=str(row.get("non_striker", "")),
                bowler=str(row.get("bowler", "")),
                runs_off_bat=int(row["runs_off_bat"]),
                extras=int(row["extras"]),
                wides=int(row["wides"]),
                noballs=int(row["noballs"]),
                is_wicket=bool(row["is_wicket"]),
                wicket_type=str(row.get("wicket_type", "") or ""),
                player_dismissed=str(row.get("player_dismissed", "") or ""),
                target=target if int(row["innings"]) == 2 else None,
            )
            if self.delay > 0:
                time.sleep(self.delay)


# ---------------------------------------------------------------------------
# 2) CricAPI v2 (cricketdata.org) — free tier 100 hits/day
# ---------------------------------------------------------------------------

class CricAPIFetcher(LiveFetcher):
    """
    Poll cricketdata.org for a live match.

    Get a free API key at https://cricketdata.org (100 hits/day).
    Set env var CRICAPI_KEY or pass `api_key` to the constructor.

    Note: free tier ball-by-ball coverage is patchy. Match-level state
    (score, wickets, current over) is reliable; full delivery commentary
    requires their paid plan. This fetcher emits a synthetic BallEvent
    each time the score changes between polls.
    """

    BASE = "https://api.cricapi.com/v1"

    def __init__(self, match_id: str, api_key: Optional[str] = None,
                 poll_seconds: int = 30):
        self.match_id = match_id
        self.api_key = api_key or os.environ.get("CRICAPI_KEY")
        if not self.api_key:
            raise RuntimeError(
                "CricAPI key required. Set CRICAPI_KEY env var or pass api_key. "
                "Get one free at https://cricketdata.org"
            )
        self.poll_seconds = poll_seconds

    def _get(self, path: str, params: dict) -> dict:
        params = {"apikey": self.api_key, **params}
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE}/{path}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "cricket-model/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def iter_balls(self) -> Iterator[BallEvent]:
        last_balls = -1
        last_runs = 0
        last_wkts = 0
        innings = 1
        target = None

        while True:
            try:
                data = self._get("match_info", {"id": self.match_id})
            except urllib.error.HTTPError as e:
                print(f"[CricAPI] HTTP {e.code}: {e.reason}")
                time.sleep(self.poll_seconds)
                continue
            except Exception as e:
                print(f"[CricAPI] {e}")
                time.sleep(self.poll_seconds)
                continue

            md = data.get("data", {})
            if md.get("matchEnded"):
                return

            score = md.get("score", [])
            if not score:
                time.sleep(self.poll_seconds)
                continue

            current = score[-1]
            inn = len(score)
            runs = current.get("r", 0)
            wkts = current.get("w", 0)
            overs = current.get("o", 0.0)
            balls_bowled = int(overs) * 6 + int(round((overs - int(overs)) * 10))

            # When innings flips, capture target
            if inn > innings:
                target = score[innings - 1].get("r", 0) + 1
                innings = inn

            # Emit one BallEvent per legal ball that occurred between polls
            new_balls = balls_bowled - last_balls
            if new_balls > 0:
                runs_added = runs - last_runs
                wkts_added = wkts - last_wkts
                # Distribute runs over new balls (best-effort without commentary)
                per_ball = runs_added // max(new_balls, 1)
                rem = runs_added - per_ball * new_balls
                for i in range(new_balls):
                    over_idx = (last_balls + i + 1 - 1) // 6
                    bio = ((last_balls + i + 1) - 1) % 6 + 1
                    yield BallEvent(
                        match_id=self.match_id,
                        innings=innings,
                        over=over_idx,
                        ball_in_over=bio,
                        batting_team=current.get("inning", "").replace(" Inning", ""),
                        bowling_team="Opponent",
                        striker="Live_Striker",
                        non_striker="Live_Non_Striker",
                        bowler="Live_Bowler",
                        runs_off_bat=per_ball + (1 if i < rem else 0),
                        extras=0, wides=0, noballs=0,
                        is_wicket=(i == new_balls - 1 and wkts_added > 0),
                        target=target,
                    )
                last_balls = balls_bowled
                last_runs = runs
                last_wkts = wkts

            time.sleep(self.poll_seconds)


# ---------------------------------------------------------------------------
# 3) Cricbuzz via RapidAPI — free tier 100 calls/day
# ---------------------------------------------------------------------------

class CricbuzzRapidAPIFetcher(LiveFetcher):
    """
    Poll the Cricbuzz Cricket API hosted on RapidAPI.

    Subscribe (free tier) at:
      https://rapidapi.com/cricketapilive/api/cricbuzz-cricket
    Set env var RAPIDAPI_KEY.

    Endpoint: /mcenter/v1/{matchId}/comm  (commentary including ball-by-ball)
    """

    HOST = "cricbuzz-cricket.p.rapidapi.com"

    def __init__(self, match_id: str, api_key: Optional[str] = None,
                 poll_seconds: int = 20):
        self.match_id = match_id
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise RuntimeError(
                "RapidAPI key required. Set RAPIDAPI_KEY env var. "
                "Subscribe free at https://rapidapi.com/cricketapilive/api/cricbuzz-cricket"
            )
        self.poll_seconds = poll_seconds

    def _commentary(self):
        url = f"https://{self.HOST}/mcenter/v1/{self.match_id}/comm"
        req = urllib.request.Request(url, headers={
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.HOST,
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def iter_balls(self) -> Iterator[BallEvent]:
        seen_ids = set()
        while True:
            try:
                data = self._commentary()
            except Exception as e:
                print(f"[Cricbuzz] {e}")
                time.sleep(self.poll_seconds)
                continue

            comms = data.get("commentaryList", []) or []
            # Cricbuzz returns newest-first; reverse to chronological
            for entry in reversed(comms):
                cid = entry.get("commentaryId") or entry.get("commTxt", "")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)

                # Only emit on actual deliveries (skip overs / breaks)
                event = entry.get("event", "")
                if event not in ("ball", "wicket", "boundary", ""):
                    continue

                yield BallEvent(
                    match_id=str(self.match_id),
                    innings=int(entry.get("inningsId", 1)),
                    over=int(entry.get("overNumber", 0)),
                    ball_in_over=int(entry.get("ballNbr", 1)),
                    batting_team=str(entry.get("battingTeam", "")),
                    bowling_team=str(entry.get("bowlingTeam", "")),
                    striker=str(entry.get("batsmanStriker", {}).get("batName", "")),
                    non_striker=str(entry.get("batsmanNonStriker", {}).get("batName", "")),
                    bowler=str(entry.get("bowlerStriker", {}).get("bowlName", "")),
                    runs_off_bat=int(entry.get("batsmanRuns", 0)),
                    extras=int(entry.get("extraRuns", 0)),
                    wides=int(entry.get("wides", 0)),
                    noballs=int(entry.get("noBalls", 0)),
                    is_wicket=(event == "wicket"),
                    wicket_type=str(entry.get("dismissalType", "")),
                    player_dismissed=str(entry.get("dismissedBatsman", "")),
                )

            time.sleep(self.poll_seconds)


# ---------------------------------------------------------------------------
# 4) Cricket Live Line Advance (cricket-live-line-advance.p.rapidapi.com)
# ---------------------------------------------------------------------------

class CricketLiveLineFetcher(LiveFetcher):
    """
    Live ball-by-ball from cricket-live-line-advance via RapidAPI.

    Free tier: 100 requests/day, 1000/hour.
    Endpoints used:
      • GET /matches/{matchId}/info                          — metadata
      • GET /matches/{matchId}/innings/{N}/commentary        — ball list

    Each commentary entry has: event ("ball" | "wicket" | "overend"),
    over, ball, batsman_id, bowler_id, run, four/six/wideball/noball,
    bat_run, plus a `players` dict at the top level for id->name lookup.
    """

    HOST = "cricket-live-line-advance.p.rapidapi.com"

    def __init__(self, match_id: int | str,
                 api_key: Optional[str] = None,
                 poll_seconds: int = 30):
        self.match_id = str(match_id)
        # Resolution order: explicit arg → keys.json active → env
        if api_key is None:
            try:
                from keys import active_key
                api_key = active_key()
            except Exception:
                api_key = None
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise RuntimeError(
                "RAPIDAPI_KEY missing. Add a key via the dashboard sidebar, "
                "or paste one in .env. "
                "Sign up free at "
                "https://rapidapi.com/apiservicesprovider/api/cricket-live-line-advance"
            )
        self.poll_seconds = poll_seconds
        self._target: Optional[int] = None  # set once innings 1 closes
        self._players: dict = {}            # id -> short name (cached per innings poll)
        self._teams: dict = {}              # team_id -> short_name

    # --- HTTP helper -------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"https://{self.HOST}{path}"
        req = urllib.request.Request(url, headers={
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.HOST,
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # --- Public helpers ---------------------------------------------

    def match_info(self) -> dict:
        """Match metadata: teams, venue, status, etc."""
        return self._get(f"/matches/{self.match_id}/info").get("response", {})

    def match_squads(self) -> dict:
        """Playing 11 + bench for both teams."""
        return self._get(f"/matches/{self.match_id}/squads").get("response", {})

    @classmethod
    def list_matches(cls, status: int = 3, api_key: Optional[str] = None,
                     limit: int = 30) -> list[dict]:
        """Status: 1=Scheduled  2=Completed  3=Live."""
        if api_key is None:
            try:
                from keys import active_key
                api_key = active_key()
            except Exception:
                pass
        api_key = api_key or os.environ.get("RAPIDAPI_KEY")
        if not api_key:
            raise RuntimeError(
                "No RapidAPI key set. Add one in the dashboard sidebar."
            )
        # NOTE: do NOT append &highlight_live_matches=1 — that flag overrides
        # the status filter and forces the response to only LIVE matches,
        # regardless of what `status` was requested. Confirmed on 2026-04-16.
        url = (f"https://{cls.HOST}/matches?status={status}"
               f"&per_paged={limit}&paged=1")
        req = urllib.request.Request(url, headers={
            "x-rapidapi-key": api_key, "x-rapidapi-host": cls.HOST,
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", {}).get("items", []) or []

    # --- Internal helpers --------------------------------------------

    def _ingest_lookup_tables(self, resp: dict) -> None:
        """Cache players + teams maps from a commentary response."""
        # players is a list of {pid, short_name, title, ...}
        players = resp.get("players") or []
        if isinstance(players, list):
            for info in players:
                pid = str(info.get("pid") or info.get("player_id") or info.get("id", ""))
                if pid and pid != "None":
                    self._players[pid] = (info.get("short_name")
                                          or info.get("title")
                                          or info.get("name") or pid)

        # teams is a dict: {"teama": {tid, abbr, title}, "teamb": {...}}
        teams = resp.get("teams") or {}
        if isinstance(teams, dict):
            for _slot, info in teams.items():
                if not isinstance(info, dict):
                    continue
                tid = str(info.get("tid") or info.get("team_id") or info.get("id", ""))
                if tid and tid != "None":
                    self._teams[tid] = (info.get("abbr")
                                        or info.get("short_name")
                                        or info.get("title") or tid)

    def _player_name(self, pid) -> str:
        if pid is None:
            return ""
        return self._players.get(str(pid), f"P{pid}")

    def _to_ball_event(self, c: dict, innings: int,
                       batting_team: str, bowling_team: str) -> Optional[BallEvent]:
        """Convert one commentary entry into a BallEvent (or None for non-ball events)."""
        ev = c.get("event")
        if ev not in ("ball", "wicket"):
            return None  # skip 'overend', 'overstart', etc.

        runs_bat = int(c.get("bat_run") or c.get("run") or 0)
        wides = int(c.get("wide_run") or 0)
        noballs = int(c.get("noball_run") or 0)
        byes = int(c.get("bye_run") or 0)
        legbyes = int(c.get("legbye_run") or 0)
        extras = wides + noballs + byes + legbyes

        return BallEvent(
            match_id=self.match_id,
            innings=innings,
            over=int(c.get("over", 0)),
            ball_in_over=int(c.get("ball", 1)),
            batting_team=batting_team,
            bowling_team=bowling_team,
            striker=self._player_name(c.get("batsman_id")),
            non_striker="",  # not provided per-ball; tracker carries previous
            bowler=self._player_name(c.get("bowler_id")),
            runs_off_bat=runs_bat,
            extras=extras,
            wides=wides,
            noballs=noballs,
            is_wicket=(ev == "wicket"),
            wicket_type=str(c.get("dismissal", "")),
            player_dismissed=self._player_name(c.get("batsman_id")) if ev == "wicket" else "",
            target=self._target if innings == 2 else None,
        )

    # --- Generator ---------------------------------------------------

    def iter_balls(self) -> Iterator[BallEvent]:
        """
        Yield BallEvents until the match ends.

        Logic per poll:
          1. Fetch innings-1 commentary; emit any new ball events
          2. If innings 1 has finished, capture target
          3. Fetch innings-2 commentary; emit any new ball events
          4. Sleep, repeat. Stop when match.status indicates Completed.
        """
        info = self.match_info()
        teama = (info.get("teama") or {}).get("short_name", "Team A")
        teamb = (info.get("teamb") or {}).get("short_name", "Team B")

        seen: set[str] = set()
        innings1_done = False
        match_done = False

        while not match_done:
            for inn in (1, 2):
                if inn == 2 and not innings1_done:
                    continue
                try:
                    payload = self._get(f"/matches/{self.match_id}/innings/{inn}/commentary")
                except urllib.error.HTTPError as e:
                    print(f"[CricketLiveLine] HTTP {e.code}: {e.reason} on innings {inn}")
                    time.sleep(self.poll_seconds)
                    continue
                except Exception as e:
                    print(f"[CricketLiveLine] {e}")
                    time.sleep(self.poll_seconds)
                    continue

                resp = payload.get("response") or {}
                if not resp:
                    continue

                self._ingest_lookup_tables(resp)
                inning_meta = resp.get("inning") or {}
                bat_team_id = str(inning_meta.get("batting_team_id", ""))
                fld_team_id = str(inning_meta.get("fielding_team_id", ""))
                batting_team = self._teams.get(bat_team_id, teama if inn == 1 else teamb)
                bowling_team = self._teams.get(fld_team_id, teamb if inn == 1 else teama)

                # Capture target after innings 1 ends
                if inn == 1 and inning_meta.get("status") in (2, "2"):
                    innings1_done = True
                    runs_str = inning_meta.get("scores", "0/0").split("/")[0]
                    try:
                        self._target = int(runs_str) + 1
                    except ValueError:
                        pass

                comm = resp.get("commentaries") or []
                # Order may be newest-first; ensure chronological by event_id
                comm_sorted = sorted(comm, key=lambda c: int(c.get("event_id") or 0))

                for c in comm_sorted:
                    eid = str(c.get("event_id", ""))
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    event = self._to_ball_event(c, inn, batting_team, bowling_team)
                    if event:
                        yield event

                # Match-complete check: status field at the top
                m = resp.get("match") or {}
                if m.get("status") in (2, "2"):  # 2 = Completed
                    if inn == 2:
                        match_done = True
                        break

            if not match_done:
                time.sleep(self.poll_seconds)
