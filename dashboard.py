"""
dashboard.py — Cricket AI Predictor (v2)
=========================================
Editorial Sports Almanac aesthetic. Four focused tabs:

  🔴 Now           — current score + AI next-ball outcomes + partnership
  📊 Phase Forecast — runs predicted for 1-6 / 7-10 / 11-15 / 16-20
  🎯 Pre-Match AI   — full match simulation before ball 0
  👥 Players        — squads, remaining batters, current matchup intel

Run:
    source venv/bin/activate
    streamlit run dashboard.py
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from feature_engine import FeatureEngine
from model import BallOutcomeModel
from simulator import MonteCarloSimulator, MatchState
from live_fetcher import (CricSheetReplayFetcher, BallEvent,
                          CricketLiveLineFetcher)
from live_predict import LiveStateTracker
from match_intel import MatchIntelligence


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cricket Almanac — AI Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# AESTHETIC: Editorial Sports Almanac
#   • Fraunces (display serif), Geist (body sans), JetBrains Mono (stats)
#   • Ink #0a0e1a, cricket-red #c1322a, paper-cream #f5efe2
# ---------------------------------------------------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Geist:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">

<style>
  :root {
    --ink:        #0a0e1a;
    --ink-soft:   #161b2d;
    --paper:      #f5efe2;
    --paper-dim:  #e8e0cc;
    --rule:       #2a3149;
    --red:        #c1322a;
    --red-soft:   #f4a8a2;
    --gold:       #c89834;
    --green:      #5a8b4a;
    --muted:      #8c93a8;
  }

  /* Reset Streamlit chrome */
  .stApp { background: var(--ink); color: var(--paper); }
  section[data-testid="stSidebar"] { background: var(--ink-soft); }
  header[data-testid="stHeader"] { background: transparent; }

  body, .stMarkdown, .stText, p, span, div, label, button {
    font-family: 'Geist', system-ui, sans-serif;
  }

  h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important;
               letter-spacing: -0.02em; color: var(--paper) !important; }

  /* The almanac scoreboard hero — torn-paper feel */
  .almanac-hero {
    background: var(--paper);
    color: var(--ink);
    padding: 28px 36px 26px 36px;
    margin-bottom: 24px;
    position: relative;
    box-shadow: 0 0 0 1px var(--rule), 12px 12px 0 var(--red);
    /* torn bottom edge */
    clip-path: polygon(
      0 0, 100% 0, 100% calc(100% - 8px),
      96% 100%, 92% calc(100% - 6px), 88% 100%, 84% calc(100% - 5px),
      80% 100%, 76% calc(100% - 7px), 72% 100%, 68% calc(100% - 4px),
      64% 100%, 60% calc(100% - 8px), 56% 100%, 52% calc(100% - 5px),
      48% 100%, 44% calc(100% - 7px), 40% 100%, 36% calc(100% - 4px),
      32% 100%, 28% calc(100% - 8px), 24% 100%, 20% calc(100% - 5px),
      16% 100%, 12% calc(100% - 7px), 8% 100%, 4% calc(100% - 4px),
      0 100%
    );
  }
  .almanac-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    text-transform: uppercase; letter-spacing: 3px; color: var(--red);
    font-weight: 800;
  }
  .almanac-teams {
    font-family: 'Fraunces', serif; font-size: 38px; font-weight: 800;
    line-height: 1.1; margin-top: 4px; letter-spacing: -0.03em;
  }
  .almanac-vs {
    font-family: 'Fraunces', serif; font-style: italic;
    color: var(--red); font-size: 28px; padding: 0 14px;
  }
  .almanac-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--ink-soft); margin-top: 8px; letter-spacing: 0.5px;
  }
  .almanac-ball-counter {
    font-family: 'JetBrains Mono', monospace; font-size: 24px;
    color: var(--ink); font-weight: 800; text-align: right;
  }
  .almanac-ball-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--muted);
    text-align: right;
  }

  /* SCOREBOARD CARD — main metric */
  .score-major {
    background: var(--ink-soft); border-left: 4px solid var(--red);
    padding: 22px 24px;
  }
  .score-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--muted);
  }
  .score-runs {
    font-family: 'Fraunces', serif; font-size: 64px; font-weight: 800;
    line-height: 1; color: var(--paper); margin: 4px 0;
    letter-spacing: -0.04em;
  }
  .score-overs {
    font-family: 'JetBrains Mono', monospace; font-size: 14px;
    color: var(--paper); opacity: 0.85;
  }
  .score-aux {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--muted); margin-top: 10px; padding-top: 10px;
    border-top: 1px dashed var(--rule);
  }

  /* AI prediction — feels different from raw score */
  .ai-card {
    background: var(--ink-soft);
    border: 1px solid var(--rule);
    padding: 18px 20px; height: 100%;
  }
  .ai-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--gold);
    font-weight: 800;
  }
  .ai-headline {
    font-family: 'Fraunces', serif; font-size: 36px; font-weight: 700;
    line-height: 1; color: var(--paper); margin: 6px 0 8px 0;
  }
  .ai-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--muted); letter-spacing: 0.5px;
  }
  .ai-big {
    font-family: 'Fraunces', serif; font-size: 56px; font-weight: 800;
    line-height: 1; letter-spacing: -0.03em;
  }
  .win-good { color: #7bcb6a !important; }
  .win-mid  { color: var(--gold) !important; }
  .win-bad  { color: var(--red-soft) !important; }

  /* Probability bars (next-ball outcomes) */
  .prob-row {
    display: flex; align-items: center; justify-content: space-between;
    margin: 6px 0;
  }
  .prob-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--paper); width: 80px; text-transform: uppercase;
    letter-spacing: 1px;
  }
  .prob-track {
    flex: 1; height: 8px; background: rgba(255,255,255,0.06);
    margin: 0 12px; position: relative;
  }
  .prob-fill {
    height: 100%; background: var(--paper);
  }
  .prob-fill.acc { background: var(--gold); }
  .prob-fill.bad { background: var(--red); }
  .prob-pct {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--paper); width: 56px; text-align: right; font-weight: 600;
  }

  /* Player cards */
  .player-card {
    background: var(--ink-soft); border: 1px solid var(--rule);
    padding: 16px 18px; height: 100%;
  }
  .player-role {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--gold);
    font-weight: 800;
  }
  .player-name {
    font-family: 'Fraunces', serif; font-size: 22px; font-weight: 700;
    color: var(--paper); margin: 4px 0 12px 0; line-height: 1.1;
  }
  .player-stat {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--paper); display: flex; justify-content: space-between;
    margin: 4px 0; opacity: 0.85;
  }
  .player-stat b { color: var(--paper); opacity: 1; }

  /* Pitch card — paper texture */
  .pitch-card {
    background: var(--paper); color: var(--ink);
    padding: 16px 20px; height: 100%;
    box-shadow: 6px 6px 0 var(--green);
  }
  .pitch-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--green);
    font-weight: 800;
  }
  .pitch-big {
    font-family: 'Fraunces', serif; font-size: 30px; font-weight: 800;
    color: var(--ink); margin: 4px 0;
  }
  .pitch-row {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--ink); display: flex; justify-content: space-between;
    margin: 3px 0;
  }

  /* Recent balls strip */
  .recent-ball {
    display: inline-block; min-width: 32px; height: 32px;
    line-height: 32px; text-align: center;
    margin: 0 4px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
  }
  .ball-dot { background: var(--ink-soft); color: var(--muted); border: 1px solid var(--rule); }
  .ball-1, .ball-2, .ball-3 { background: var(--ink-soft); color: var(--paper); border: 1px solid var(--rule); }
  .ball-4 { background: var(--gold); color: var(--ink); }
  .ball-6 { background: var(--red); color: var(--paper); }
  .ball-W { background: var(--paper); color: var(--red); border: 2px solid var(--red); }
  .ball-X { background: transparent; color: var(--muted); border: 1px dashed var(--muted); }

  /* Section labels — almanac headings */
  .almanac-section {
    font-family: 'Fraunces', serif; font-size: 18px; font-weight: 700;
    color: var(--paper); margin: 28px 0 12px 0;
    border-bottom: 2px solid var(--red); padding-bottom: 6px;
    letter-spacing: -0.01em;
  }

  /* Phase forecast cards */
  .phase-card {
    background: var(--ink-soft); border: 1px solid var(--rule);
    padding: 16px 18px; text-align: center;
    border-top: 4px solid var(--rule);
  }
  .phase-card.current { border-top: 4px solid var(--red); background: rgba(193,50,42,0.04); }
  .phase-card.completed { opacity: 0.45; }
  .phase-name {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: var(--gold);
    font-weight: 800;
  }
  .phase-runs {
    font-family: 'Fraunces', serif; font-size: 42px; font-weight: 800;
    color: var(--paper); margin: 6px 0; line-height: 1;
  }
  .phase-band {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--muted); letter-spacing: 0.5px;
  }

  /* Sidebar overrides */
  section[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 2px !important; color: var(--gold) !important;
    border-bottom: 1px solid var(--rule); padding-bottom: 6px;
  }

  /* Tabs */
  div[data-baseweb="tab-list"] {
    background: var(--ink-soft); border-bottom: 1px solid var(--rule);
    padding: 0 4px;
  }
  div[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; text-transform: uppercase;
    letter-spacing: 1.5px !important; padding: 14px 18px !important;
  }
  div[data-baseweb="tab"][aria-selected="true"] {
    color: var(--gold) !important; border-bottom: 2px solid var(--red) !important;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_engine():
    model = BallOutcomeModel(model_dir="models")
    model.load("ball_outcome_model.joblib")
    fe = FeatureEngine()
    sim = MonteCarloSimulator(model=model, feature_engine=fe, n_simulations=80)
    return sim


@st.cache_resource
def load_intel():
    return MatchIntelligence.load_or_build("data")


@st.cache_data
def list_replay_matches() -> list:
    rows = []
    files = [f for f in glob.glob("data/*.csv") if "_info" not in f]
    for csv in files:
        info = csv.replace(".csv", "_info.csv")
        teams, date, venue, toss = [], "", "", ""
        if os.path.exists(info):
            for line in open(info):
                if line.startswith("info,team,"):
                    teams.append(line.strip().split(",", 2)[2])
                elif line.startswith("info,date,"):
                    date = line.strip().split(",", 2)[2]
                elif line.startswith("info,venue,"):
                    venue = line.strip().split(",", 2)[2].strip('"')
                elif line.startswith("info,toss_winner,"):
                    toss = line.strip().split(",", 2)[2]
        if len(teams) >= 2:
            label = f"{date}  ·  {teams[0]} vs {teams[1]}"
            rows.append((csv, label, teams[0], teams[1], venue, toss, date))
    rows.sort(key=lambda r: r[6], reverse=True)
    return rows


@st.cache_data
def load_match_balls(csv_path: str) -> list[BallEvent]:
    fetcher = CricSheetReplayFetcher(csv_path, ball_delay_s=0.0)
    return list(fetcher.iter_balls())


# Tighter caching to stay friendly to the free tier (100 calls/day, 1000/hr)
@st.cache_data(ttl=300)  # 5 min — match listings change slowly
def list_live_api_matches(status: int):
    try:
        items = CricketLiveLineFetcher.list_matches(status=status, limit=30)
        rows = []
        for m in items:
            comp = (m.get("competition") or {}).get("abbr", "?")
            game = m.get("game_state_str") or m.get("status_str") or "?"
            rows.append((m.get("match_id"),
                         f"{m.get('short_title','?')}  ·  {comp}  ·  {game}", m))
        return rows
    except Exception as e:
        return [("__err__", str(e), {})]


@st.cache_data(ttl=60)  # bumped from 20s to 60s — was burning quota
def fetch_live_innings(match_id: int, innings: int) -> dict:
    fetcher = CricketLiveLineFetcher(match_id=match_id, poll_seconds=999)
    return fetcher._get(f"/matches/{match_id}/innings/{innings}/commentary")


# Lightweight in-process API call counter for live quota estimate
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "api_session_started" not in st.session_state:
    import time as _t
    st.session_state.api_session_started = _t.time()


@st.cache_data(ttl=1800)  # 30 min — squads barely change
def fetch_match_squads(match_id: int) -> dict:
    """Returns {'teama': {...}, 'teamb': {...}} or {} on error."""
    try:
        fetcher = CricketLiveLineFetcher(match_id=match_id, poll_seconds=999)
        return fetcher.match_squads()
    except Exception as e:
        return {"__error__": str(e)}


def is_rate_limited_err(err) -> bool:
    """Detect HTTP 429 from urllib's HTTPError or its string representation."""
    msg = str(err).lower()
    return ("429" in msg or "too many requests" in msg
            or "rate limit" in msg)


def _extract_squad_players(team_block) -> list[dict]:
    """Defensively pull a list of player dicts from arbitrary squad shape."""
    if not isinstance(team_block, dict):
        return []
    for key in ("playing_xi", "playing11", "squads", "players", "squad"):
        v = team_block.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for sub in ("playing", "starting", "xi", "playing_xi"):
                vv = v.get(sub)
                if isinstance(vv, list):
                    return vv
            # otherwise return values flattened
            out = []
            for slot, val in v.items():
                if isinstance(val, dict):
                    out.append(val)
            if out:
                return out
    return []


# ---------------------------------------------------------------------------
# Sidebar — source + match + controls
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 🔑 API Key manager — TOP of sidebar so it's always reachable, especially
# when the API is rate-limited and the rest of the page would st.stop()
# ---------------------------------------------------------------------------
import keys as _keymod

_kdata_outer = _keymod.load_keys()
_cloud_key = _keymod.streamlit_secret_key()
_active_source = _keymod.active_key_source()
_active_key_value = _keymod.active_key()
_active_tail = (_active_key_value[-6:] if _active_key_value else "—")

with st.sidebar.expander(
    f"🔑 RapidAPI keys  ·  using: {_active_source} (…{_active_tail})",
    expanded=False,
):
    _kdata = _keymod.load_keys()
    _klist = _kdata.get("keys") or []

    # Show priority + force-refresh button
    st.markdown(f"""
    <div style="background: rgba(200,152,52,0.08); border-left: 3px solid var(--gold);
                padding: 8px 10px; font-family:'JetBrains Mono',monospace;
                font-size:11px; color:var(--paper); margin-bottom:10px;">
      <b style="color:var(--gold);">PRIORITY (highest first):</b><br>
      1. Local key in this session (added below)<br>
      2. Streamlit Cloud secret<br>
      3. .env file<br>
      <br>
      <b>Currently active:</b>
      <span style="color:#7bcb6a;">{_active_source}</span><br>
      <b>Key tail:</b>
      <span style="color:var(--gold);">…{_active_tail}</span>
    </div>
    """, unsafe_allow_html=True)

    # Show cloud secret separately if it exists (so user knows it's there as fallback)
    if _cloud_key:
        st.caption(f"🌐 Streamlit secret detected: …{_cloud_key[-6:]} "
                   f"(used as fallback if no local key)")

    # Force-refresh button — clears all API caches without changing keys
    if st.button("🔄 Force refresh API caches",
                 width="stretch", key="btn_force_refresh"):
        st.cache_data.clear()
        st.success("All API caches cleared. Next match list call will be fresh.")
        st.rerun()

    if not _klist and not _cloud_key:
        st.markdown(
            "<div style='color:var(--muted); font-size:12px;'>"
            "No keys yet. Sign up free at "
            "<a href='https://rapidapi.com/apiservicesprovider/api/cricket-live-line-advance' "
            "style='color:var(--gold);' target='_blank'>"
            "rapidapi.com/.../cricket-live-line-advance</a> "
            "and paste the key below.</div>",
            unsafe_allow_html=True,
        )
    elif not _klist and _cloud_key:
        st.markdown(
            "<div style='color:var(--muted); font-size:12px;'>"
            "No additional local keys stored. The cloud secret above is "
            "being used. You can add more keys below to rotate between them."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace; "
            "font-size:11px; color:var(--muted); letter-spacing:1px;'>"
            f"{len(_klist)} KEY(S) STORED · ACTIVE = #{_kdata.get('active', 0)+1}</div>",
            unsafe_allow_html=True,
        )
        labels = [f"{i+1}. {k['label']}  ·  …{k['key'][-6:]}"
                  for i, k in enumerate(_klist)]
        chosen = st.radio("Active key", options=range(len(_klist)),
                          format_func=lambda i: labels[i],
                          index=_kdata.get("active", 0),
                          label_visibility="collapsed", key="key_radio")
        if chosen != _kdata.get("active", 0):
            _keymod.set_active(chosen)
            st.cache_data.clear()
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧪 Test active", width="stretch", key="btn_test_key"):
                ok, msg = _keymod.test_key(_klist[chosen]["key"])
                (st.success if ok else st.error)(msg)
        with col_b:
            if st.button("🗑 Delete", width="stretch", key="btn_del_key"):
                _keymod.remove_key(chosen)
                st.cache_data.clear()
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace; "
        "font-size:10px; color:var(--gold); letter-spacing:1px;'>"
        "➕ ADD ANOTHER KEY</div>",
        unsafe_allow_html=True,
    )
    new_label = st.text_input("Label (e.g. gmail account 2)",
                               placeholder="venk.backup@gmail.com",
                               key="new_key_label")
    new_key = st.text_input("Key (paste from RapidAPI playground)",
                             type="password",
                             placeholder="b6...027a",
                             key="new_key_value")

    add_col1, add_col2 = st.columns([1, 1])
    with add_col1:
        if st.button("🧪 Test only", width="stretch",
                     disabled=not new_key.strip(), key="btn_test_new"):
            ok, msg = _keymod.test_key(new_key.strip())
            st.session_state["new_key_test_result"] = (ok, msg)
    with add_col2:
        if st.button("➕ Add", type="primary", width="stretch",
                     disabled=not new_key.strip(), key="btn_add_key"):
            ok, msg = _keymod.test_key(new_key.strip())
            if ok:
                _keymod.add_key(new_label or "unlabeled", new_key.strip(),
                                make_active=True)
                st.cache_data.clear()
                st.success(f"Added & set active. {msg}")
                st.rerun()
            else:
                st.session_state["new_key_test_result"] = (False, msg)
                st.session_state["pending_add_key"] = (
                    new_label or "unlabeled", new_key.strip())

    # Show last test result + escape hatch
    test_result = st.session_state.get("new_key_test_result")
    if test_result:
        ok, msg = test_result
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
            pend = st.session_state.get("pending_add_key")
            if pend:
                st.markdown(
                    "<div style='font-size:11px; color:var(--muted); "
                    "margin: 6px 0;'>"
                    "Test failed but you can still save this key — "
                    "it'll work once you subscribe to cricket-live-line-advance "
                    "with that account.</div>",
                    unsafe_allow_html=True)
                if st.button("💾 Save anyway (skip test)",
                             width="stretch", key="btn_save_anyway"):
                    label, key_str = pend
                    _keymod.add_key(label, key_str, make_active=False)
                    st.session_state["pending_add_key"] = None
                    st.session_state["new_key_test_result"] = None
                    st.cache_data.clear()
                    st.success(
                        f"Saved as '{label}' (NOT set active — subscribe "
                        f"to cricket-live-line-advance with that account, "
                        f"then come back, set this key active, and re-test.")
                    st.rerun()


st.sidebar.markdown("### What do you want to do?")
workflow = st.sidebar.radio(
    "workflow",
    options=[
        "📺 Watch live match",
        "⏯ Replay past match",
        "🎯 Predict future match",
    ],
    label_visibility="collapsed",
)
mode_live    = workflow.startswith("📺")
mode_replay  = workflow.startswith("⏯")
mode_predict = workflow.startswith("🎯")
is_live = mode_live or mode_predict  # both use the API

st.sidebar.markdown("### Match")

csv_path = label = team1 = team2 = venue = toss = date = ""
api_match_id = None
match_format_str = ""

if mode_replay:
    matches = list_replay_matches()
    if not matches:
        st.sidebar.error("No CSVs in data/")
        st.stop()
    choice = st.sidebar.selectbox("Pick a match",
        options=range(len(matches)),
        format_func=lambda i: matches[i][1], index=0,
        label_visibility="collapsed")
    csv_path, label, team1, team2, venue, toss, date = matches[choice]
else:
    # Live or Predict — both hit the API but with different status filters
    if mode_live:
        default_status = ("Live (in progress)", 3)
        status_options = [
            ("Live (in progress)", 3),
            ("Recent (completed)", 2),
        ]
    else:  # mode_predict
        default_status = ("Scheduled (upcoming)", 1)
        status_options = [
            ("Scheduled (upcoming)", 1),
        ]

    status_pair = st.sidebar.selectbox(
        "Status",
        options=status_options,
        format_func=lambda x: x[0],
        index=0,
        label_visibility="collapsed",
    )
    api_rows = list_live_api_matches(status_pair[1])
    if not api_rows or api_rows[0][0] == "__err__":
        err_msg = api_rows[0][1] if api_rows else "No data"
        if is_rate_limited_err(err_msg):
            st.sidebar.markdown("""
            <div style="background: rgba(193,50,42,0.12); border: 1px solid var(--red);
                        padding: 10px 12px; font-family:'JetBrains Mono',monospace;
                        font-size:11px; color:var(--paper);">
              <b style="color:var(--red);">429 RATE LIMIT</b><br>
              <span style="color:var(--muted);">
                This key's free tier is exhausted. Three ways out:
                <ol style="margin:6px 0 0 16px; padding:0;">
                  <li>Open <b style="color:var(--gold);">🔑 RapidAPI keys</b>
                      above &amp; switch to a different key</li>
                  <li>Wait ~1 hour for per-hour cap to reset</li>
                  <li>Use <b style="color:var(--gold);">⏯ Replay past match</b>
                      mode — no API needed</li>
                </ol>
              </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.error(err_msg)
        st.stop()

    # ---- League / competition filter ----
    # Pull every distinct competition.abbr that came back. Case-insensitive
    # IPL detection covers IPL 2026, IPL 2025, "IPL", "Indian Premier League"
    leagues_present = []
    has_ipl = False
    for _, _, raw in api_rows:
        comp = (raw.get("competition") or {}).get("abbr", "?")
        if comp and comp not in leagues_present:
            leagues_present.append(comp)
        if "ipl" in comp.lower() or "indian premier" in comp.lower():
            has_ipl = True
    leagues_present.sort()

    # Build options
    league_opts = []
    if has_ipl:
        league_opts.append("🏏 IPL only")
    league_opts.append("🌍 All leagues")
    league_opts.extend(leagues_present)

    league_choice = st.sidebar.selectbox(
        "League filter",
        options=league_opts, index=0,
        label_visibility="collapsed",
    )

    # Show what leagues were actually returned by the API for transparency
    st.sidebar.caption(
        f"API returned: {', '.join(leagues_present[:6])}"
        + (f" +{len(leagues_present)-6} more" if len(leagues_present) > 6 else "")
    )

    # Apply filter (case-insensitive IPL match)
    if league_choice.startswith("🏏 IPL"):
        filtered = [r for r in api_rows
                    if "ipl" in (r[2].get("competition") or {}).get("abbr", "").lower()
                    or "indian premier" in (r[2].get("competition") or {}).get("abbr", "").lower()]
    elif league_choice.startswith("🌍 All"):
        filtered = api_rows
    else:
        filtered = [r for r in api_rows
                    if (r[2].get("competition") or {}).get("abbr") == league_choice]

    if not filtered:
        if league_choice.startswith("🏏 IPL"):
            st.sidebar.warning(
                f"No IPL matches in the '{status_pair[0]}' list right now. "
                f"IPL 2026 schedule may have a gap, or matches haven't been "
                f"announced yet. Try switching status (e.g. '📺 Watch live' "
                f"→ Recent (completed)) to see past IPL games."
            )
        filtered = api_rows  # fall back so the dropdown isn't empty

    api_choice = st.sidebar.selectbox(
        f"{len(filtered)} {league_choice.lstrip('🏏🌍 ').strip()} matches",
        options=range(len(filtered)),
        format_func=lambda i: filtered[i][1], index=0,
        label_visibility="collapsed",
    )
    api_match_id, label, raw = filtered[api_choice]
    team1 = (raw.get("teama") or {}).get("name", "Team A")
    team2 = (raw.get("teamb") or {}).get("name", "Team B")
    venue = (raw.get("venue") or {}).get("name", "")
    toss = raw.get("status_note") or ""
    date = (raw.get("date_start") or "").split(" ")[0]
    match_format_str = raw.get("format_str", "")

if not mode_predict:
    st.sidebar.markdown("### Controls")
    play = st.sidebar.toggle("Auto-play", value=True)
    if mode_replay:
        balls_per_tick = st.sidebar.slider("Balls per second", 1, 12, 4)
        poll_seconds = 60
    else:
        balls_per_tick = 1
        poll_seconds = st.sidebar.slider(
            "Poll interval (sec)", 30, 300, 90, step=15,
            help="Higher = friendlier to the free-tier API quota. "
                 "60s = ~120 calls/hr; 90s = ~80 calls/hr.",
        )
        st.sidebar.caption(
            f"≈ {int(3600 / poll_seconds * 2)} calls/hr at this interval "
            f"(2 calls per poll · innings 1+2)"
        )
    if st.sidebar.button("⟲ Reset (clear cache + state)", width="stretch"):
        for k in ("idx", "tracker", "history", "wp_history", "last_pred",
                  "current_match", "live_seen", "live_balls", "pending_pred"):
            st.session_state.pop(k, None)
        st.cache_data.clear()
        st.session_state.api_call_count = 0
        st.rerun()
else:
    # In Predict mode the match hasn't started — no live updates needed
    play = False
    balls_per_tick = 1
    poll_seconds = 60

# Quota tracker — only meaningful when API is in use
if is_live:
    import time as _t
    elapsed_min = max(1, (_t.time() - st.session_state.api_session_started) / 60)
    rate_per_hr = st.session_state.api_call_count / elapsed_min * 60
    st.sidebar.markdown("### API quota (this session)")
    st.sidebar.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                color:var(--paper); padding: 8px 0;">
      Calls so far: <b style="color:var(--gold);">{st.session_state.api_call_count}</b><br>
      Elapsed: <b>{elapsed_min:.0f} min</b><br>
      Rate: <b>{rate_per_hr:.0f}/hr</b>
      <span style="color:var(--muted);"> (free tier ≈ 1000/hr cap)</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### Inference")
sim_count = st.sidebar.slider("Monte-Carlo sims", 40, 400, 100, step=20)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

match_key = f"live:{api_match_id}" if is_live else f"replay:{csv_path}"
if st.session_state.get("current_match") != match_key:
    st.session_state.current_match = match_key
    st.session_state.idx = 0
    st.session_state.tracker = LiveStateTracker()
    st.session_state.history = []
    st.session_state.wp_history = []
    st.session_state.last_pred = None
    st.session_state.recent = []
    st.session_state.current_inn = 0
    st.session_state.live_seen = set()
    st.session_state.live_balls = []
    st.session_state.next_ball_outcomes = None
    st.session_state.phase_forecast = None


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

last_event: BallEvent | None = None

if not is_live:
    balls = load_match_balls(csv_path)
    total_balls = len(balls)
    if play and st.session_state.idx < total_balls:
        st_autorefresh(interval=max(int(1000 / balls_per_tick), 80),
                       key="replay_tick")
        end = min(st.session_state.idx + 1, total_balls)
    else:
        end = st.session_state.idx
    for i in range(st.session_state.idx, end):
        e = balls[i]
        if e.innings != st.session_state.current_inn:
            st.session_state.current_inn = e.innings
            st.session_state.recent = []
        st.session_state.tracker.apply(e)
        code = ("W" if e.is_wicket else
                "X" if (e.wides or e.noballs) else
                str(e.runs_off_bat))
        st.session_state.recent.append(code)
        if len(st.session_state.recent) > 18:
            st.session_state.recent.pop(0)
    st.session_state.idx = end
    last_event = balls[end - 1] if end > 0 else None
else:
    if play:
        st_autorefresh(interval=poll_seconds * 1000, key="live_tick")
    fetcher = CricketLiveLineFetcher(match_id=api_match_id, poll_seconds=poll_seconds)
    new_balls: list[BallEvent] = []
    rate_limited = False

    # Smart: only fetch innings 2 once innings 1 has reached >=119 balls
    # (saves ~50% of API calls during innings 1)
    fetch_innings_list = (1,)
    if (st.session_state.tracker.state and
            st.session_state.tracker.state.innings >= 2):
        fetch_innings_list = (1, 2)
    elif (st.session_state.tracker.state and
          st.session_state.tracker.state.balls_bowled >= 110):
        # Late innings 1 — innings 2 might start at any moment
        fetch_innings_list = (1, 2)

    for inn in fetch_innings_list:
        try:
            payload = fetch_live_innings(api_match_id, inn)
            st.session_state.api_call_count += 1
        except Exception as exc:
            if is_rate_limited_err(exc):
                rate_limited = True
            else:
                st.sidebar.warning(f"Innings {inn} fetch error: {str(exc)[:40]}")
            continue
        resp = payload.get("response") or {}
        if not resp:
            continue
        fetcher._ingest_lookup_tables(resp)
        inning_meta = resp.get("inning") or {}
        bat_team_id = str(inning_meta.get("batting_team_id", ""))
        fld_team_id = str(inning_meta.get("fielding_team_id", ""))
        bt = fetcher._teams.get(bat_team_id, team1 if inn == 1 else team2)
        ft = fetcher._teams.get(fld_team_id, team2 if inn == 1 else team1)
        if inn == 1 and inning_meta.get("status") in (2, "2"):
            try:
                fetcher._target = int(inning_meta.get("scores", "0/0").split("/")[0]) + 1
            except ValueError:
                pass
        comm_sorted = sorted(resp.get("commentaries") or [],
                             key=lambda c: int(c.get("event_id") or 0))
        for c in comm_sorted:
            eid = str(c.get("event_id", ""))
            if not eid or eid in st.session_state.live_seen:
                continue
            st.session_state.live_seen.add(eid)
            ev = fetcher._to_ball_event(c, inn, bt, ft)
            if ev:
                new_balls.append(ev)
    for e in new_balls:
        if e.innings != st.session_state.current_inn:
            st.session_state.current_inn = e.innings
            st.session_state.recent = []
        st.session_state.tracker.apply(e)
        code = ("W" if e.is_wicket else
                "X" if (e.wides or e.noballs) else
                str(e.runs_off_bat))
        st.session_state.recent.append(code)
        if len(st.session_state.recent) > 18:
            st.session_state.recent.pop(0)
        st.session_state.live_balls.append(e)
        st.session_state.idx += 1
    total_balls = max(st.session_state.idx, 1)
    end = st.session_state.idx
    last_event = (st.session_state.live_balls[-1]
                  if st.session_state.live_balls else None)

    if rate_limited:
        st.markdown("""
        <div style="background: rgba(193,50,42,0.10); border: 1px solid var(--red);
                    padding: 14px 18px; margin: 0 0 18px 0;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                      color:var(--red); letter-spacing:2px; font-weight:800;">
            ⚠️ API RATE LIMIT (429)
          </div>
          <div style="font-family:'Geist',sans-serif; color:var(--paper);
                      margin-top:6px; font-size:13px;">
            The free tier (100 calls/day · 1000/hour) is exceeded.
            Cached data is shown below. Things to try:
            <ul style="margin:6px 0 0 18px; color:var(--paper); font-size:12px;">
              <li>Wait <b>~1 hour</b> for the per-hour cap to reset</li>
              <li>Wait <b>until midnight UTC</b> for the per-day cap</li>
              <li>Increase the <b>Poll interval</b> in the sidebar (try 120s+)</li>
              <li>Toggle off <b>Auto-play</b> while you analyse</li>
              <li>Use <b>⏯ Replay past match</b> mode (no API needed)</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Predictions (only when something interesting happens)
# ---------------------------------------------------------------------------

state = st.session_state.tracker.state
sim = load_engine()
sim.n_sims = sim_count
intel = load_intel()

should_predict = (
    state is not None
    and last_event is not None
    and (last_event.is_wicket or last_event.ball_in_over == 6
         or st.session_state.last_pred is None)
)

if should_predict and state is not None and state.balls_bowled > 0:
    try:
        # ALL predictions now use REAL conditional distributions —
        # no more XGBoost model in the live prediction loop.
        over_pred = sim.realistic_next_over(state, intel=intel, n_sims=80)
        inn_pred = sim.realistic_projected_total(state, intel=intel)
        nb_outcomes = sim.predict_next_ball_outcomes(state, intel=intel)
        st.session_state.next_ball_outcomes = nb_outcomes
        st.session_state.last_pred = {
            "next_over_mean": float(over_pred["mean"]),
            "next_over_p_over_7_5": float(over_pred.get("prob_over_7.5", 0)),
            "projected_total": float(inn_pred["mean"]),
            "projected_p10": float(inn_pred["percentiles"][10]),
            "projected_p90": float(inn_pred["percentiles"][90]),
            "win_prob": float(inn_pred.get("win_prob", -1)),
        }

        # ---- Honest actual-vs-predicted chart logic ----
        # The prediction we just made (`over_pred`) is for the NEXT over.
        # If a previous over had a pending prediction attached, write the
        # comparison entry now: actual_in_just_completed_over vs that
        # pending prediction.
        completed_over = state.balls_bowled // 6   # 1-indexed completed over
        last_actual = sum(
            1 if c == "1" else 2 if c == "2" else 3 if c == "3" else
            4 if c == "4" else 6 if c == "6" else 0
            for c in st.session_state.recent[-6:] if c not in ("X",)
        )
        pending = st.session_state.get("pending_pred")
        if pending is not None and pending["over"] == completed_over and \
           pending["innings"] == state.innings:
            st.session_state.history.append({
                "innings": state.innings,
                "over": completed_over,
                "actual": last_actual,
                "predicted": float(pending["value"]),
            })
        # Now stash the just-made prediction for the NEXT over
        st.session_state.pending_pred = {
            "innings": state.innings,
            "over": completed_over + 1,
            "value": float(over_pred["mean"]),
        }

        if state.innings == 2 and "win_prob" in inn_pred:
            st.session_state.wp_history.append({
                "ball": state.balls_bowled,
                "win_prob": float(inn_pred["win_prob"]) * 100,
            })
    except Exception as exc:
        st.warning(f"Prediction error: {exc}")


# ---------------------------------------------------------------------------
# RENDER — Almanac hero header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="almanac-hero">
  <div style="display:flex; align-items:flex-start; justify-content:space-between;">
    <div>
      <div class="almanac-eyebrow">Cricket Almanac · AI Predictor · {date}</div>
      <div class="almanac-teams">
        {team1}<span class="almanac-vs">vs</span>{team2}
      </div>
      <div class="almanac-meta">
        📍 {venue or 'Venue TBD'}  ·  🪙 {toss or 'Toss pending'}
      </div>
    </div>
    <div>
      <div class="almanac-ball-label">Ball</div>
      <div class="almanac-ball-counter">{end:03d}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab_now, tab_phase, tab_pre, tab_players = st.tabs([
    "🔴 Now", "📊 Phase Forecast", "🎯 Pre-Match AI", "👥 Players & Squads"
])


# =============================================================================
# TAB 1 — NOW
# =============================================================================

with tab_now:
    if mode_predict:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:var(--muted);
                    font-family:'JetBrains Mono',monospace;">
          <div style="font-size:14px; letter-spacing:2px; color:var(--gold);">
            🎯 PREDICT FUTURE MATCH MODE
          </div>
          <div style="font-size:13px; margin-top:12px; max-width:500px; margin-left:auto; margin-right:auto;">
            This match hasn't started, so there's no live score to show.
            Switch to the <b style="color:var(--red);">🎯 Pre-Match AI</b>
            tab above to run the full-match prediction.
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif state is None:
        st.info("Press ▶ Auto-play in the sidebar to start.")
    else:
        batting_team = (team1 if last_event and team1.lower() in
                        (last_event.batting_team or "").lower()
                        else team2 if last_event else team1)

        # Row 1 — score + AI total + win prob
        c1, c2, c3 = st.columns([1.4, 1, 1])

        with c1:
            rrr_html = ""
            if state.innings == 2 and state.target:
                rrr_html = (
                    f"Target <b>{state.target}</b> · "
                    f"Need <b>{max(0, state.target - state.runs_scored)}</b> off "
                    f"<b>{120 - state.balls_bowled}</b> · "
                    f"RRR <b>{state.required_rr:.2f}</b>"
                )
            st.markdown(f"""
            <div class="score-major">
              <div class="score-eyebrow">Innings {state.innings} · {batting_team}</div>
              <div class="score-runs">{state.runs_scored}<span style="font-size:36px; opacity:0.7;">/{state.wickets_fallen}</span></div>
              <div class="score-overs">Overs {state.balls_bowled//6}.{state.balls_bowled%6}  ·  CRR {state.current_rr:.2f}</div>
              <div class="score-aux">{rrr_html}</div>
            </div>
            """, unsafe_allow_html=True)

        pred = st.session_state.last_pred or {}

        with c2:
            proj = pred.get("projected_total", 0)
            p10 = pred.get("projected_p10", 0)
            p90 = pred.get("projected_p90", 0)
            if proj:
                proj_html = f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">🤖 AI · Projected Total</div>
                  <div class="ai-headline">{proj:.0f}</div>
                  <div class="ai-sub">Bad day–Great day: {p10:.0f}–{p90:.0f} runs</div>
                </div>"""
            else:
                proj_html = """
                <div class="ai-card">
                  <div class="ai-eyebrow">🤖 AI · Projected Total</div>
                  <div class="ai-headline">—</div>
                  <div class="ai-sub">Awaiting first over</div>
                </div>"""
            st.markdown(proj_html, unsafe_allow_html=True)

        with c3:
            if state.innings == 2 and pred.get("win_prob", -1) >= 0:
                wp = pred["win_prob"] * 100
                cls = "win-good" if wp > 60 else "win-mid" if wp > 30 else "win-bad"
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">🏆 {batting_team} Win Probability</div>
                  <div class="ai-big {cls}">{wp:.0f}%</div>
                  <div class="ai-sub">Based on {sim_count} sims</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                vintel = intel.venue(venue)
                par = vintel.get("par_score", 165)
                diff = (proj - par) if proj else 0
                arrow = "↑" if diff >= 0 else "↓"
                cls = "win-good" if diff >= 10 else "win-bad" if diff <= -10 else "win-mid"
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">📊 vs Venue Par ({par:.0f})</div>
                  <div class="ai-big {cls}">{arrow} {abs(diff):.0f}</div>
                  <div class="ai-sub">Win prob unlocks at chase</div>
                </div>
                """, unsafe_allow_html=True)

        # Row 2 — next ball outcome bars + partnership + matchup
        st.markdown('<div class="almanac-section">Next Ball · Outcome Probabilities</div>',
                    unsafe_allow_html=True)

        ob = st.session_state.next_ball_outcomes or {}
        c1, c2, c3 = st.columns([1.4, 1, 1])

        with c1:
            if ob:
                rows = []
                for label_, key, css_cls in [
                    ("DOT", "dot", ""),
                    ("1 / 2", "single", ""),
                    ("BOUNDARY", "boundary", "acc"),
                    ("SIX", "six", "acc"),
                    ("WICKET", "wicket", "bad"),
                    ("EXTRA", "extra", ""),
                ]:
                    val = ob.get(key, 0) if key != "single" else (ob.get("single", 0) + ob.get("double", 0))
                    rows.append(
                        f'<div class="prob-row">'
                        f'  <span class="prob-label">{label_}</span>'
                        f'  <div class="prob-track"><div class="prob-fill {css_cls}" style="width:{val*100:.0f}%"></div></div>'
                        f'  <span class="prob-pct">{val*100:.1f}%</span>'
                        f'</div>'
                    )
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">AI · per-ball calibration</div>
                  {''.join(rows)}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="ai-card"><div class="ai-eyebrow">Awaiting state…</div></div>',
                            unsafe_allow_html=True)

        with c2:
            partner = st.session_state.tracker.partnership_summary()
            sr = (partner["runs"] * 100 / max(partner["balls"], 1))
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-eyebrow">Partnership</div>
              <div class="ai-headline">{partner['runs']}</div>
              <div class="ai-sub">{partner['balls']} balls · SR {sr:.1f} · RR {partner['rr']:.2f}</div>
              <div class="ai-sub" style="margin-top:8px; opacity:0.7;">
                {state.striker} & {state.non_striker}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            # Striker vs current bowler historical H2H
            h2h = intel.matchup(state.striker, state.bowler)
            next_over = pred.get("next_over_mean", 0)
            wkt_p = (ob.get("wicket", 0) * 100) if ob else 0
            if h2h:
                dis = h2h["dismissals"]
                sr = h2h["sr"]
                sr_color = ("win-good" if sr > 130 else
                            "win-bad" if sr < 100 else "win-mid")
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">⚔️ Career H2H · {state.striker.split()[-1] if state.striker else ''} vs {state.bowler.split()[-1] if state.bowler else ''}</div>
                  <div class="ai-big {sr_color}">{h2h['runs']}<span style="font-size:18px;color:var(--muted);"> ({h2h['balls']})</span></div>
                  <div class="ai-sub">SR <b>{sr:.0f}</b>  ·  out <b>{dis}</b> times  ·  4/6 % <b>{h2h['boundary_pct']:.0f}%</b></div>
                  <div class="ai-sub" style="margin-top:6px; opacity:0.7;">
                    Next over forecast: <b>{next_over:.1f}r</b> · wkt prob {wkt_p:.0f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-eyebrow">🤖 Next over forecast</div>
                  <div class="ai-headline">{next_over:.1f}</div>
                  <div class="ai-sub">runs predicted  ·  wkt prob {wkt_p:.0f}%</div>
                  <div class="ai-sub" style="margin-top:8px; opacity:0.7;">
                    No prior H2H data for this matchup
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Row 3 — recent balls strip
        st.markdown('<div class="almanac-section">Last 18 deliveries</div>',
                    unsafe_allow_html=True)
        strip = ""
        for c in st.session_state.recent:
            cls = ("ball-W" if c == "W" else
                   "ball-X" if c == "X" else
                   "ball-dot" if c == "0" else
                   "ball-4" if c == "4" else
                   "ball-6" if c == "6" else
                   f"ball-{c}")
            strip += f'<span class="recent-ball {cls}">{c}</span>'
        st.markdown(f'<div style="padding:8px 0 18px 0;">{strip or "<i style=\'color:var(--muted)\'>Awaiting first ball</i>"}</div>',
                    unsafe_allow_html=True)

        # Row 4 — momentum chart
        st.markdown('<div class="almanac-section">Momentum · Actual vs AI Predicted</div>',
                    unsafe_allow_html=True)
        if st.session_state.history:
            hdf = pd.DataFrame(st.session_state.history)
            hdf = hdf[hdf["innings"] == state.innings].copy()
            if not hdf.empty:
                # ---- Accuracy summary stat ----
                err = (hdf["actual"] - hdf["predicted"]).abs()
                avg_err = err.mean()
                n = len(hdf)
                st.markdown(f"""
                <div style="display:flex; gap:24px; margin: 6px 0 14px 0;
                            font-family:'JetBrains Mono',monospace; font-size:11px;
                            color:var(--muted); letter-spacing:1px;">
                  <span>📊 <b style="color:var(--gold);">GOLD</b> = ACTUAL RUNS · over already played</span>
                  <span>🤖 <b style="color:var(--red);">RED</b> = AI's PRE-OVER PREDICTION · what AI thought before it happened</span>
                  <span>🎯 ACCURACY · avg ±{avg_err:.1f} runs/over over {n} overs</span>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure()
                # Grouped bars: Actual gold + AI predicted red — same x position, shown side-by-side
                fig.add_trace(go.Bar(
                    x=hdf["over"], y=hdf["actual"],
                    name="Actual runs",
                    marker=dict(color="#c89834",
                                line=dict(color="#0a0e1a", width=1)),
                    text=hdf["actual"], textposition="outside",
                    textfont=dict(color="#c89834", size=10,
                                  family="JetBrains Mono"),
                    hovertemplate="<b>Over %{x}</b><br>ACTUAL: %{y} runs<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=hdf["over"], y=hdf["predicted"],
                    name="AI predicted (before this over)",
                    marker=dict(color="#c1322a", opacity=0.85,
                                line=dict(color="#0a0e1a", width=1)),
                    text=[f"{p:.0f}" for p in hdf["predicted"]],
                    textposition="outside",
                    textfont=dict(color="#c1322a", size=10,
                                  family="JetBrains Mono"),
                    hovertemplate="<b>Over %{x}</b><br>AI predicted: %{y:.1f} runs<extra></extra>",
                ))
                fig.add_hline(y=8, line_dash="dot",
                              line_color="#5a8b4a", line_width=1,
                              annotation_text="Par 8/over",
                              annotation_position="top right",
                              annotation_font=dict(color="#5a8b4a",
                                                   family="JetBrains Mono",
                                                   size=10))
                fig.update_layout(
                    template="plotly_dark", barmode="group",
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                    height=340, margin=dict(t=30, b=10, l=10, r=20),
                    xaxis=dict(
                        title=dict(text="OVER NUMBER",
                                   font=dict(family="JetBrains Mono", size=10,
                                             color="#8c93a8")),
                        tickfont=dict(family="JetBrains Mono", size=11,
                                      color="#f5efe2"),
                        dtick=1, showgrid=False, zeroline=False,
                    ),
                    yaxis=dict(
                        title=dict(text="RUNS IN OVER",
                                   font=dict(family="JetBrains Mono", size=10,
                                             color="#8c93a8")),
                        tickfont=dict(family="JetBrains Mono", size=11,
                                      color="#f5efe2"),
                        gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                    ),
                    legend=dict(orientation="h", y=-0.18, x=0.5,
                                xanchor="center",
                                font=dict(family="JetBrains Mono", size=11,
                                          color="#f5efe2"),
                                bgcolor="rgba(0,0,0,0)"),
                    bargap=0.25, bargroupgap=0.05,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.markdown('<div style="color:var(--muted); padding:20px 0;">'
                            'Predictions begin from over 2 (need one over of '
                            'history before AI can compare its pre-over '
                            'prediction to reality).</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--muted); padding:20px 0;">'
                        'Comparison chart appears after over 2.</div>',
                        unsafe_allow_html=True)


# =============================================================================
# TAB 2 — PHASE FORECAST
# =============================================================================

with tab_phase:
    if mode_predict:
        st.info("Phase forecast available in 📺 Watch live or ⏯ Replay modes. "
                "For an upcoming match, see the 🎯 Pre-Match AI tab.")
    elif state is None or state.balls_bowled == 0:
        st.info("Phase forecast appears once the match is in progress.")
    else:
        if (st.session_state.phase_forecast is None
                or last_event and last_event.ball_in_over == 6):
            try:
                st.session_state.phase_forecast = sim.predict_phase_segments(
                    state, intel=intel)
            except Exception as exc:
                st.warning(f"Phase forecast error: {exc}")

        forecasts = st.session_state.phase_forecast or []

        st.markdown('<div class="almanac-section">Phase-by-phase Run Forecast</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family: \'JetBrains Mono\', monospace; '
            'font-size:12px; color:var(--muted); margin-bottom:14px;">'
            'Each card forecasts runs the AI expects in that phase, with '
            'Range shows bad-day to great-day outcomes (80% of simulations fall inside). '
            'Red border = current phase.</div>',
            unsafe_allow_html=True)

        cols = st.columns(len(forecasts) if forecasts else 4)
        for i, ph in enumerate(forecasts):
            cls = ph["status"]  # current / completed / future
            with cols[i]:
                if ph["predicted_runs"] is None:
                    st.markdown(f"""
                    <div class="phase-card {cls}">
                      <div class="phase-name">{ph['label']}</div>
                      <div class="phase-runs" style="color:var(--muted);">—</div>
                      <div class="phase-band">phase complete</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="phase-card {cls}">
                      <div class="phase-name">{ph['label']}</div>
                      <div class="phase-runs">{ph['predicted_runs']:.0f}</div>
                      <div class="phase-band">Bad day–Great day: {ph['p10']:.0f}–{ph['p90']:.0f}</div>
                      <div class="phase-band" style="margin-top:6px; color:var(--red-soft);">
                        ~{ph['predicted_wickets']:.1f} wkts
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Sum line
        total_predicted = state.runs_scored + sum(
            ph["predicted_runs"] or 0 for ph in forecasts)
        st.markdown(f"""
        <div style="background: var(--ink-soft); border-left: 4px solid var(--gold);
                    padding: 16px 22px; margin-top: 24px;">
          <div class="ai-eyebrow">Sum · Current + All Future Phases</div>
          <div class="ai-big" style="margin-top:6px;">{total_predicted:.0f}
            <span style="font-size:18px; color:var(--muted); font-family:'JetBrains Mono',monospace;">
              projected innings total
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3 — PRE-MATCH AI
# =============================================================================

with tab_pre:
    n_buckets = len(intel.ball_outcomes) if hasattr(intel, "ball_outcomes") else 0
    st.markdown(f"""
    <div style="background: rgba(200,152,52,0.08); border-left: 3px solid var(--gold);
                padding: 14px 18px; margin-bottom: 18px;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                  color:var(--gold); letter-spacing:2px; font-weight:800;">
        🎯 PRE-MATCH PREDICTION · 100% DATA-DERIVED
      </div>
      <div style="font-family:'Geist',sans-serif; color:var(--paper);
                  margin-top:4px; font-size:13px; line-height:1.5;">
        Every number on this tab comes from real IPL data — no sliders, no
        guessing, no hand-tuned constants. The simulator samples from
        <b style="color:var(--gold);">{n_buckets} measured conditional
        probability buckets</b> (phase × wickets × chase pressure)
        derived from {len(intel.team_stats) if hasattr(intel,'team_stats') else 0} teams
        across <b style="color:var(--gold);">1,191 past IPL matches</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    vintel = intel.venue(venue)
    venue_phase = vintel.get("phase_rpo", {})
    venue_chase = intel.venue_chase(venue)

    a_intel = intel.team(team1) or {}
    b_intel = intel.team(team2) or {}
    a_recent = intel.team_recent_form(team1) or {}
    b_recent = intel.team_recent_form(team2) or {}
    h2h = intel.head_to_head(team1, team2)

    # ============================================================
    # ROW 1 — Quick context cards (read-only stats from history)
    # ============================================================
    st.markdown('<div class="almanac-section">Pre-Match Context (auto-derived from past)</div>',
                unsafe_allow_html=True)

    cc1, cc2, cc3, cc4 = st.columns(4)

    # Team A snapshot
    with cc1:
        a_avg = a_recent.get("recent_avg") or a_intel.get("batting_rpo", 0) * 20
        a_wins = a_intel.get("matches_won", 0)
        a_played = a_intel.get("matches_played", 0)
        a_winrate = (a_wins / a_played * 100) if a_played else 0
        st.markdown(f"""
        <div class="ai-card">
          <div class="ai-eyebrow">{team1[:25]}</div>
          <div class="ai-big">{a_avg:.0f}</div>
          <div class="ai-sub">Recent avg innings · {a_played} matches · {a_winrate:.0f}% win rate</div>
        </div>
        """, unsafe_allow_html=True)

    # Team B snapshot
    with cc2:
        b_avg = b_recent.get("recent_avg") or b_intel.get("batting_rpo", 0) * 20
        b_wins = b_intel.get("matches_won", 0)
        b_played = b_intel.get("matches_played", 0)
        b_winrate = (b_wins / b_played * 100) if b_played else 0
        st.markdown(f"""
        <div class="ai-card">
          <div class="ai-eyebrow">{team2[:25]}</div>
          <div class="ai-big">{b_avg:.0f}</div>
          <div class="ai-sub">Recent avg innings · {b_played} matches · {b_winrate:.0f}% win rate</div>
        </div>
        """, unsafe_allow_html=True)

    # H2H card
    with cc3:
        if h2h["played"] > 0:
            a_wr = h2h.get(f"{team1}_winrate", 0) * 100
            b_wr = h2h.get(f"{team2}_winrate", 0) * 100
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-eyebrow">⚔️ Head-to-head</div>
              <div class="ai-big">{h2h['played']}</div>
              <div class="ai-sub">past meetings  ·  {team1.split()[0]} {a_wr:.0f}%  vs  {team2.split()[0]} {b_wr:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-eyebrow">⚔️ Head-to-head</div>
              <div class="ai-big">—</div>
              <div class="ai-sub">No prior IPL meetings on record</div>
            </div>
            """, unsafe_allow_html=True)

    # Venue card
    with cc4:
        v_chase_pct = (venue_chase.get("chase_wins", 0) /
                       max(venue_chase.get("matches", 1), 1) * 100)
        st.markdown(f"""
        <div class="pitch-card">
          <div class="pitch-eyebrow">📍 Venue · {venue_chase.get('matches', 0)} games</div>
          <div class="pitch-big">{vintel.get('avg_first_innings', 165):.0f}</div>
          <div class="pitch-row"><span>Avg 1st inn</span></div>
          <div class="pitch-row"><span>Chase wins</span><b>{v_chase_pct:.0f}%</b></div>
          <div class="pitch-row"><span>Toss-winner wins</span><b>{venue_chase.get('toss_winner_wins',0)}/{venue_chase.get('matches',1)}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # ROW 2 — Toss & bat-first analysis
    # ============================================================
    st.markdown('<div class="almanac-section">Toss & Bat-first Analysis</div>',
                unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)

    with tc1:
        a_toss = a_intel.get("toss_won", 0)
        a_toss_pct = (a_toss / max(a_played, 1) * 100)
        a_bat_first_wr = (a_intel.get("bat_first_wins", 0) /
                           max(a_intel.get("bat_first_played", 1), 1) * 100)
        a_chase_wr = (a_intel.get("chase_wins", 0) /
                       max(a_intel.get("chase_played", 1), 1) * 100)
        st.markdown(f"""
        <div class="ai-card">
          <div class="ai-eyebrow">{team1}</div>
          <div class="player-stat" style="margin-top:10px;">
            <span>Toss-win rate</span><b>{a_toss_pct:.0f}%</b></div>
          <div class="player-stat">
            <span>When batting 1st (won {a_intel.get('bat_first_wins',0)}/{a_intel.get('bat_first_played',0)})</span>
            <b>{a_bat_first_wr:.0f}%</b></div>
          <div class="player-stat">
            <span>When chasing (won {a_intel.get('chase_wins',0)}/{a_intel.get('chase_played',0)})</span>
            <b>{a_chase_wr:.0f}%</b></div>
          <div class="player-stat">
            <span>Recent first-innings scores</span>
            <b>{', '.join(str(int(s)) for s in (a_recent.get('recent_first_inn_scores') or [])[-5:])}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with tc2:
        b_toss = b_intel.get("toss_won", 0)
        b_toss_pct = (b_toss / max(b_played, 1) * 100)
        b_bat_first_wr = (b_intel.get("bat_first_wins", 0) /
                           max(b_intel.get("bat_first_played", 1), 1) * 100)
        b_chase_wr = (b_intel.get("chase_wins", 0) /
                       max(b_intel.get("chase_played", 1), 1) * 100)
        st.markdown(f"""
        <div class="ai-card">
          <div class="ai-eyebrow">{team2}</div>
          <div class="player-stat" style="margin-top:10px;">
            <span>Toss-win rate</span><b>{b_toss_pct:.0f}%</b></div>
          <div class="player-stat">
            <span>When batting 1st (won {b_intel.get('bat_first_wins',0)}/{b_intel.get('bat_first_played',0)})</span>
            <b>{b_bat_first_wr:.0f}%</b></div>
          <div class="player-stat">
            <span>When chasing (won {b_intel.get('chase_wins',0)}/{b_intel.get('chase_played',0)})</span>
            <b>{b_chase_wr:.0f}%</b></div>
          <div class="player-stat">
            <span>Recent first-innings scores</span>
            <b>{', '.join(str(int(s)) for s in (b_recent.get('recent_first_inn_scores') or [])[-5:])}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # ROW 3 — RUN THE PREDICTION
    # ============================================================
    st.markdown('<div class="almanac-section">AI Prediction</div>',
                unsafe_allow_html=True)

    # Auto-decide who bats first based on historical preference at this venue
    venue_pref_chase = (venue_chase.get("chase_wins", 0) /
                         max(venue_chase.get("matches", 1), 1)) > 0.55
    smart_default = team2 if venue_pref_chase else team1

    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        st.markdown(
            f"<div style='color:var(--muted); font-size:12px; "
            f"font-family:\"JetBrains Mono\",monospace;'>"
            f"Default: <b style='color:var(--gold);'>{smart_default}</b> bats "
            f"first (this venue {'favors chasing' if venue_pref_chase else 'is bat-friendly'})"
            f"</div>",
            unsafe_allow_html=True)
        first_to_bat = st.radio(
            "Override: who bats first?",
            options=[smart_default,
                     team1 if smart_default != team1 else team2],
            horizontal=True, label_visibility="collapsed",
        )
    with rcol2:
        n_pm_sims = st.select_slider(
            "Simulations",
            options=[100, 200, 500, 1000], value=200,
        )

    run = st.button("🎯 Run Full-Match Prediction", type="primary",
                    width="stretch")

    if run:
        # Try player-aware simulator first (uses actual likely XIs)
        xi_a = intel.team_likely_xi(team1)
        xi_b = intel.team_likely_xi(team2)
        use_player_aware = bool(xi_a.get("batters") and xi_b.get("batters"))

        if use_player_aware:
            with st.spinner(f"Simulating {n_pm_sims} matches with PLAYER-AWARE engine "
                            f"(using actual likely XIs from past data)…"):
                fm = sim.player_aware_pre_match(
                    team_a_name=team1, team_a_xi=xi_a,
                    team_b_name=team2, team_b_xi=xi_b,
                    venue_phase_rpo=venue_phase if venue_phase else None,
                    n_sims=n_pm_sims,
                    team_a_bats_first=(first_to_bat == team1),
                    intel=intel,
                )
            engine_label = "🎯 PLAYER-AWARE · uses real likely XIs + every player's career stats + striker×bowler H2H"
        else:
            with st.spinner(f"Simulating {n_pm_sims} matches with team-level engine "
                            f"(player XIs not available)…"):
                fm = sim.realistic_pre_match(
                    team_a_name=team1, team_a_stats=a_intel or {},
                    team_b_name=team2, team_b_stats=b_intel or {},
                    venue_phase_rpo=venue_phase if venue_phase else None,
                    n_sims=n_pm_sims,
                    team_a_bats_first=(first_to_bat == team1),
                    intel=intel,
                )
            engine_label = "📊 TEAM-LEVEL · uses team averages (no player XI in cache)"

        st.markdown(f"""
        <div style="background: rgba(200,152,52,0.08); border-left: 3px solid var(--gold);
                    padding: 10px 14px; margin: 14px 0;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                      color:var(--gold); letter-spacing:1px;">
            {engine_label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Winner Headline ----
        wp = fm["winner_probs"]
        top_team, top_p = max(
            ((t, p) for t, p in wp.items() if t != "Tie"),
            key=lambda x: x[1],
        )
        exp = fm["expected"]
        margin_str = (f"by ~{exp['value']:.0f} runs"
                      if exp["by"] == "runs"
                      else f"by ~{exp['value']:.0f} wickets")

        st.markdown(f"""
        <div class="almanac-hero" style="margin-top: 18px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div class="almanac-eyebrow">AI prediction · {fm['n_sims']} sims · realistic engine</div>
              <div class="almanac-teams" style="font-size: 32px;">
                🏆 {exp['winner']}
              </div>
              <div class="almanac-meta">
                Expected to win {margin_str}
              </div>
            </div>
            <div>
              <div class="almanac-ball-label">WIN %</div>
              <div class="almanac-ball-counter" style="font-size: 38px; color: var(--red);">
                {top_p*100:.0f}%
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Win-prob breakdown ----
        st.markdown('<div class="almanac-section">Win Probability</div>',
                    unsafe_allow_html=True)
        wp_html = ""
        for team, prob in sorted(wp.items(), key=lambda x: -x[1]):
            if team == "Tie" and prob < 0.01:
                continue
            color_cls = ("acc" if team == top_team
                         else ("bad" if team == "Tie" else ""))
            wp_html += (
                f'<div class="prob-row">'
                f'  <span class="prob-label" style="width:170px;">{team[:22]}</span>'
                f'  <div class="prob-track"><div class="prob-fill {color_cls}" style="width:{prob*100:.0f}%"></div></div>'
                f'  <span class="prob-pct">{prob*100:.1f}%</span>'
                f'</div>'
            )
        st.markdown(f'<div class="ai-card">{wp_html}</div>',
                    unsafe_allow_html=True)

        # ---- Predicted Innings Totals ----
        st.markdown('<div class="almanac-section">Predicted Innings Totals</div>',
                    unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        ft = fm["first_team"]; sd = fm["second_team"]
        with sc1:
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-eyebrow">{ft['name']} · BAT FIRST</div>
              <div class="ai-big">{ft['median']:.0f}/{ft['wickets_mean']:.0f}</div>
              <div class="ai-sub">Bad day–Great day: {ft['p10']:.0f}–{ft['p90']:.0f} runs</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-eyebrow">{sd['name']} · CHASING</div>
              <div class="ai-big">{sd['median']:.0f}/{sd['wickets_mean']:.0f}</div>
              <div class="ai-sub">Bad day–Great day: {sd['p10']:.0f}–{sd['p90']:.0f} runs</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Predicted XIs (player-aware mode only) ----
        if use_player_aware:
            season_used = xi_a.get("season_used", "?")
            n_a = xi_a.get("based_on_matches", 0)
            n_b = xi_b.get("based_on_matches", 0)
            st.markdown(f'<div class="almanac-section">Likely Playing XIs '
                        f'(IPL {season_used} · '
                        f'{team1.split()[0]} {n_a} matches · '
                        f'{team2.split()[0]} {n_b} matches)</div>',
                        unsafe_allow_html=True)

            xi_cols = st.columns(2)
            for col, team_label, xi_obj in [
                (xi_cols[0], team1, xi_a),
                (xi_cols[1], team2, xi_b),
            ]:
                with col:
                    bat_rows = ""
                    for i, name in enumerate(xi_obj["batters"][:11], 1):
                        bs = intel.batter(name) or {}
                        # Prefer recent form, fall back to career
                        sr = bs.get("recent_sr") or bs.get("sr", 0)
                        career_sr = bs.get("sr", 0)
                        is_recent = "recent_sr" in bs
                        bp = bs.get("recent_boundary_pct") or bs.get("boundary_pct", 0)
                        sr_color = ("var(--red)" if sr > 160
                                    else "var(--gold)" if sr > 130
                                    else "var(--paper)")
                        sr_tag = "recent" if is_recent else "career"
                        bat_rows += (
                            f'<div class="player-stat" style="padding:5px 0; '
                            f'border-bottom:1px dashed var(--rule);">'
                            f'<span><b>{i}.</b> {name}</span>'
                            f'<b style="color:{sr_color};">SR {sr:.0f}'
                            f'<span style="font-size:9px; color:var(--muted); '
                            f'margin-left:4px;">{sr_tag}</span></b></div>'
                        )
                    bowl_rows = ""
                    for name, ovs in xi_obj.get("bowlers", []):
                        bs = intel.bowler(name) or {}
                        eco = bs.get("recent_economy") or bs.get("economy", 0)
                        is_recent = "recent_economy" in bs
                        eco_color = ("#7bcb6a" if eco < 8
                                     else "var(--gold)" if eco < 9
                                     else "var(--red-soft)")
                        eco_tag = "recent" if is_recent else "career"
                        bowl_rows += (
                            f'<div class="player-stat" style="padding:5px 0; '
                            f'border-bottom:1px dashed var(--rule);">'
                            f'<span>{name}  '
                            f'<span style="color:var(--muted); font-size:10px;">~{ovs}ov</span></span>'
                            f'<b style="color:{eco_color};">ECO {eco:.2f}'
                            f'<span style="font-size:9px; color:var(--muted); '
                            f'margin-left:4px;">{eco_tag}</span></b></div>'
                        )
                    st.markdown(f"""
                    <div class="player-card">
                      <div class="player-role">⚡ {team_label} · BATTING ORDER</div>
                      {bat_rows}
                    </div>
                    <div class="player-card" style="margin-top:12px;">
                      <div class="player-role">🎯 {team_label} · BOWLING ATTACK</div>
                      {bowl_rows}
                    </div>
                    """, unsafe_allow_html=True)

        # ---- Over-by-over expected runs + wickets table ----
        st.markdown('<div class="almanac-section">Expected Over-by-over (both innings)</div>',
                    unsafe_allow_html=True)
        oo_fig = go.Figure()
        ft_runs = ft["per_over_runs_mean"][:20]
        sd_runs = sd["per_over_runs_mean"][:20]
        ft_wkts = ft["per_over_wkts_mean"][:20]
        sd_wkts = sd["per_over_wkts_mean"][:20]
        overs = list(range(1, 21))
        oo_fig.add_trace(go.Bar(
            x=overs, y=ft_runs, name=f"{ft['name']} runs",
            marker=dict(color="#c89834", opacity=0.85,
                        line=dict(color="#0a0e1a", width=1)),
        ))
        oo_fig.add_trace(go.Bar(
            x=overs, y=sd_runs, name=f"{sd['name']} runs",
            marker=dict(color="#c1322a", opacity=0.85,
                        line=dict(color="#0a0e1a", width=1)),
        ))
        # Wicket markers
        oo_fig.add_trace(go.Scatter(
            x=overs, y=[15 + w*5 for w in ft_wkts],
            name=f"{ft['name']} cumul wickets (×5+15 for visibility)",
            mode="markers", marker=dict(symbol="x", color="#fff", size=7),
            yaxis="y2",
        ))
        oo_fig.update_layout(
            template="plotly_dark", barmode="group",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            height=340, margin=dict(t=20, b=10, l=10, r=20),
            xaxis=dict(
                title=dict(text="OVER", font=dict(family="JetBrains Mono",
                                                  size=10, color="#8c93a8")),
                tickfont=dict(family="JetBrains Mono", size=11,
                              color="#f5efe2"),
                dtick=1, showgrid=False, zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="EXPECTED RUNS",
                           font=dict(family="JetBrains Mono", size=10,
                                     color="#8c93a8")),
                tickfont=dict(family="JetBrains Mono", size=11,
                              color="#f5efe2"),
                gridcolor="rgba(255,255,255,0.04)", zeroline=False,
            ),
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        showticklabels=False),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                        font=dict(family="JetBrains Mono", size=11,
                                  color="#f5efe2")),
            bargap=0.25, bargroupgap=0.05,
        )
        st.plotly_chart(oo_fig, width="stretch")

        # ---- Score distribution (overlaid histograms) ----
        st.markdown('<div class="almanac-section">Score Distributions</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ft["distribution"], nbinsx=24,
            name=f"{ft['name']} (bat first)",
            marker=dict(color="#c89834", opacity=0.75,
                        line=dict(color="#0a0e1a", width=1)),
        ))
        fig.add_trace(go.Histogram(
            x=sd["distribution"], nbinsx=24,
            name=f"{sd['name']} (chasing)",
            marker=dict(color="#c1322a", opacity=0.75,
                        line=dict(color="#0a0e1a", width=1)),
        ))
        fig.add_vline(x=ft['median'], line_dash="dash",
                      line_color="#c89834", line_width=2)
        fig.add_vline(x=sd['median'], line_dash="dash",
                      line_color="#c1322a", line_width=2)
        fig.update_layout(
            template="plotly_dark", barmode="overlay",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            height=320, margin=dict(t=20, b=10, l=10, r=20),
            xaxis=dict(
                title=dict(text="INNINGS TOTAL",
                           font=dict(family="JetBrains Mono", size=10,
                                     color="#8c93a8")),
                tickfont=dict(family="JetBrains Mono", size=11,
                              color="#f5efe2"),
                showgrid=False, zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="SIMULATIONS",
                           font=dict(family="JetBrains Mono", size=10,
                                     color="#8c93a8")),
                tickfont=dict(family="JetBrains Mono", size=11,
                              color="#f5efe2"),
                gridcolor="rgba(255,255,255,0.04)", zeroline=False,
            ),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                        font=dict(family="JetBrains Mono", size=11,
                                  color="#f5efe2")),
        )
        st.plotly_chart(fig, width="stretch")


# =============================================================================
# TAB 4 — PLAYERS & SQUADS
# =============================================================================

with tab_players:
    if mode_predict:
        st.info("In Predict-future-match mode, the playing 11 may not be "
                "announced yet. The 🎯 Pre-Match AI tab uses each team's "
                "season-average stats from past IPL data instead.")
    elif state is None:
        st.info("Player intel appears once the match is in progress.")
    else:
        # Current matchup
        st.markdown('<div class="almanac-section">Current Matchup</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        striker_career = intel.batter(state.striker) or {}
        striker_live = st.session_state.tracker.current_batter_card(state.striker)
        with c1:
            st.markdown(f"""
            <div class="player-card" style="border-left: 4px solid var(--red);">
              <div class="player-role">⚡ On Strike</div>
              <div class="player-name">{state.striker}</div>
              <div class="player-stat"><span>This innings</span>
                <b>{striker_live['runs']} ({striker_live['balls']}) · SR {striker_live['sr']:.0f}</b></div>
              <div class="player-stat"><span>Career SR</span>
                <b>{striker_career.get('sr', 0):.1f}</b></div>
              <div class="player-stat"><span>Career boundary %</span>
                <b>{striker_career.get('boundary_pct', 0):.1f}%</b></div>
              <div class="player-stat"><span>Career avg</span>
                <b>{striker_career.get('avg', 0):.1f}</b></div>
              <div class="player-stat"><span>Balls faced (career)</span>
                <b>{striker_career.get('balls', 0):,}</b></div>
            </div>
            """, unsafe_allow_html=True)

        nstriker_career = intel.batter(state.non_striker) or {}
        nstriker_live = st.session_state.tracker.current_batter_card(state.non_striker)
        with c2:
            st.markdown(f"""
            <div class="player-card">
              <div class="player-role">Non-striker</div>
              <div class="player-name">{state.non_striker}</div>
              <div class="player-stat"><span>This innings</span>
                <b>{nstriker_live['runs']} ({nstriker_live['balls']}) · SR {nstriker_live['sr']:.0f}</b></div>
              <div class="player-stat"><span>Career SR</span>
                <b>{nstriker_career.get('sr', 0):.1f}</b></div>
              <div class="player-stat"><span>Career boundary %</span>
                <b>{nstriker_career.get('boundary_pct', 0):.1f}%</b></div>
              <div class="player-stat"><span>Career avg</span>
                <b>{nstriker_career.get('avg', 0):.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        bowler_career = intel.bowler(state.bowler) or {}
        bowler_live = st.session_state.tracker.current_bowler_card(state.bowler)
        phase_eco = bowler_career.get("phase_eco", {}) if bowler_career else {}
        phase_now = ("powerplay" if state.balls_bowled // 6 < 6
                     else "middle" if state.balls_bowled // 6 < 15
                     else "death")
        with c3:
            st.markdown(f"""
            <div class="player-card" style="border-left: 4px solid var(--gold);">
              <div class="player-role">🎯 Bowling</div>
              <div class="player-name">{state.bowler}</div>
              <div class="player-stat"><span>This innings</span>
                <b>{bowler_live['overs']} · {bowler_live['runs']} runs · {bowler_live['wickets']} wkts · ECO {bowler_live['economy']:.2f}</b></div>
              <div class="player-stat"><span>Career economy</span>
                <b>{bowler_career.get('economy', 0):.2f}</b></div>
              <div class="player-stat"><span>Career economy in {phase_now}</span>
                <b>{phase_eco.get(phase_now, bowler_career.get('economy', 0)):.2f}</b></div>
              <div class="player-stat"><span>Career wickets</span>
                <b>{bowler_career.get('wickets', 0)}</b></div>
              <div class="player-stat"><span>Career dot %</span>
                <b>{bowler_career.get('dot_pct', 0):.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Yet to bat
        st.markdown('<div class="almanac-section">Yet to Bat / Already Out</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        already_out = st.session_state.tracker.dismissed
        with c1:
            st.markdown('<div class="ai-eyebrow" style="margin-bottom:10px;">'
                        f'Dismissed ({len(already_out)})</div>',
                        unsafe_allow_html=True)
            if already_out:
                rows = ""
                for name in already_out:
                    bcard = st.session_state.tracker.current_batter_card(name)
                    rows += (
                        f'<div class="player-stat" style="padding:6px 0; '
                        f'border-bottom:1px dashed var(--rule);">'
                        f'<span>{name}</span>'
                        f'<b>{bcard["runs"]} ({bcard["balls"]})</b></div>'
                    )
                st.markdown(f'<div class="player-card">{rows}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:var(--muted);">No wickets yet.</div>',
                            unsafe_allow_html=True)

        with c2:
            # Current pair
            st.markdown('<div class="ai-eyebrow" style="margin-bottom:10px;">'
                        'At the crease</div>', unsafe_allow_html=True)
            pair_html = ""
            for name in (state.striker, state.non_striker):
                bcard = st.session_state.tracker.current_batter_card(name)
                pair_html += (
                    f'<div class="player-stat" style="padding:6px 0; '
                    f'border-bottom:1px dashed var(--rule);">'
                    f'<span>{name}</span>'
                    f'<b>{bcard["runs"]}* ({bcard["balls"]})</b></div>'
                )
            st.markdown(f'<div class="player-card">{pair_html}</div>',
                        unsafe_allow_html=True)

        # Squads panel (live mode only — needs API)
        if is_live and api_match_id:
            st.markdown('<div class="almanac-section">Playing 11 — both teams</div>',
                        unsafe_allow_html=True)
            squads = fetch_match_squads(api_match_id)
            if "__error__" in squads:
                st.markdown(
                    f'<div style="color:var(--muted); padding:10px 0;">'
                    f'Squad fetch failed: {squads["__error__"][:80]} '
                    f'(usually rate limit; cached squads return on next session)</div>',
                    unsafe_allow_html=True)
            else:
                sq_cols = st.columns(2)
                for col, slot, default_team in [
                    (sq_cols[0], "teama", team1),
                    (sq_cols[1], "teamb", team2),
                ]:
                    team_block = squads.get(slot, {})
                    team_name = (team_block.get("name") or
                                 team_block.get("title") or default_team)
                    players_list = _extract_squad_players(team_block)
                    with col:
                        rows = ""
                        for p in players_list[:11]:
                            pname = (p.get("short_name") or p.get("name")
                                     or p.get("title") or "?")
                            role = (p.get("playing_role") or p.get("role")
                                    or p.get("role_str") or "")
                            bat = intel.batter(pname) or {}
                            bowl = intel.bowler(pname) or {}
                            stat_str = ""
                            if bat.get("sr"):
                                stat_str += f" SR {bat['sr']:.0f}"
                            if bowl.get("economy"):
                                stat_str += f" ECO {bowl['economy']:.1f}"
                            rows += (
                                f'<div class="player-stat" style="padding:6px 0;'
                                f' border-bottom:1px dashed var(--rule);">'
                                f'<span>{pname}'
                                f'<span style="color:var(--muted); font-size:10px;'
                                f' margin-left:8px;">{role}</span></span>'
                                f'<b style="color:var(--gold);">{stat_str}</b></div>'
                            )
                        if not rows:
                            rows = (f'<div style="color:var(--muted);">'
                                    f'No squad data yet for {team_name}</div>')
                        st.markdown(f"""
                        <div class="player-card">
                          <div class="player-role">{team_name}</div>
                          {rows}
                        </div>
                        """, unsafe_allow_html=True)


# Footer
st.markdown(
    '<div style="text-align:center; color:var(--muted); font-size:10px; '
    'font-family:\'JetBrains Mono\',monospace; letter-spacing:2px; '
    'margin-top:40px; padding-top:20px; border-top:1px solid var(--rule);">'
    '★ CRICKET ALMANAC · POWERED BY XGBOOST ON 1,191 IPL MATCHES ★'
    '</div>', unsafe_allow_html=True)
