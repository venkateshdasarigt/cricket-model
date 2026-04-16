"""
keys.py
-------
Tiny key-manager for RapidAPI keys so you can rotate between accounts
without editing files. Stores everything in `keys.json` (gitignored).

File format
-----------
{
  "active": 0,
  "keys": [
    {"label": "venk@gmail.com",   "key": "b6...027a"},
    {"label": "venk2@gmail.com",  "key": "abc...xyz"}
  ]
}

First-run migration: if `keys.json` doesn't exist but `.env` has a
RAPIDAPI_KEY, it auto-creates `keys.json` with that key as entry 0.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

KEYS_PATH = "keys.json"
ENV_PATH = ".env"
HOST = "cricket-live-line-advance.p.rapidapi.com"


def _read_env_key() -> Optional[str]:
    if not os.path.exists(ENV_PATH):
        return None
    for line in open(ENV_PATH):
        line = line.strip()
        if line.startswith("RAPIDAPI_KEY=") and "=" in line:
            return line.split("=", 1)[1].strip()
    return None


def _migrate_from_env() -> dict:
    """First-run: pull RAPIDAPI_KEY from .env into a fresh keys.json."""
    env_key = _read_env_key()
    payload = {
        "active": 0,
        "keys": (
            [{"label": "primary (from .env)", "key": env_key}]
            if env_key else []
        ),
    }
    save_keys(payload)
    return payload


def load_keys() -> dict:
    """Load keys.json, migrating from .env on first run."""
    if not os.path.exists(KEYS_PATH):
        return _migrate_from_env()
    try:
        with open(KEYS_PATH) as fh:
            data = json.load(fh)
        if "keys" not in data:
            data["keys"] = []
        if "active" not in data:
            data["active"] = 0
        return data
    except Exception:
        return _migrate_from_env()


def save_keys(data: dict) -> None:
    with open(KEYS_PATH, "w") as fh:
        json.dump(data, fh, indent=2)


def streamlit_secret_key() -> Optional[str]:
    """Read RAPIDAPI_KEY from Streamlit secrets (Cloud deploy). None if absent."""
    try:
        import streamlit as st
    except Exception:
        return None
    if not hasattr(st, "secrets"):
        return None
    # Try multiple access patterns — Streamlit secrets are picky
    for accessor in (
        lambda: st.secrets["RAPIDAPI_KEY"],   # dict-style
        lambda: st.secrets.RAPIDAPI_KEY,       # attr-style
        lambda: st.secrets.get("RAPIDAPI_KEY"),  # .get
    ):
        try:
            v = accessor()
            if v:
                return str(v).strip()
        except Exception:
            continue
    return None


def active_key() -> Optional[str]:
    """
    Return the currently-active API key.

    Resolution order:
      1. Streamlit Cloud secret  (RAPIDAPI_KEY in Secrets dashboard)
      2. Active key from keys.json   (local interactive use)
      3. RAPIDAPI_KEY env var       (legacy .env file)
      4. None
    """
    k = streamlit_secret_key()
    if k:
        return k

    data = load_keys()
    keys = data.get("keys") or []
    if keys:
        idx = max(0, min(data.get("active", 0), len(keys) - 1))
        return keys[idx]["key"]

    return os.environ.get("RAPIDAPI_KEY")


def add_key(label: str, key: str, make_active: bool = True) -> dict:
    data = load_keys()
    data["keys"].append({"label": label.strip() or "unlabeled", "key": key.strip()})
    if make_active:
        data["active"] = len(data["keys"]) - 1
    save_keys(data)
    return data


def remove_key(index: int) -> dict:
    data = load_keys()
    if 0 <= index < len(data["keys"]):
        data["keys"].pop(index)
        if data["active"] >= len(data["keys"]):
            data["active"] = max(0, len(data["keys"]) - 1)
        save_keys(data)
    return data


def set_active(index: int) -> dict:
    data = load_keys()
    if 0 <= index < len(data["keys"]):
        data["active"] = index
        save_keys(data)
    return data


def test_key(key: str, timeout: int = 20) -> tuple[bool, str]:
    """
    Make one tiny API call to verify the key works.
    Returns (ok, message).

    Distinguishes:
      • OK                — call succeeded
      • Invalid (401)     — bad key entirely
      • Not subscribed    — valid account key but not subbed to this API
      • Rate limited (429)— valid + subbed but quota hit
      • Timeout           — usually means not subscribed (RapidAPI hangs)
                            or network is slow
    """
    url = f"https://{HOST}/matches?status=2&per_paged=1&paged=1"
    req = urllib.request.Request(url, headers={
        "x-rapidapi-key": key, "x-rapidapi-host": HOST,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "ok":
            n = len(data.get("response", {}).get("items", []))
            return True, f"✓ Key works · returned {n} matches"
        msg = str(data.get("message") or data)[:160]
        return False, f"× API replied but status != ok: {msg}"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, ("× 401 Unauthorized — the key itself is invalid. "
                           "Did you copy the FULL key including the trailing chars?")
        if e.code == 403:
            try:
                body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                body = ""
            return False, ("× 403 NOT SUBSCRIBED — your account isn't "
                           "subscribed to cricket-live-line-advance yet. "
                           "Go to https://rapidapi.com/apiservicesprovider/api/"
                           "cricket-live-line-advance/pricing and click "
                           "'Start Free Plan'. Then try again. "
                           f"({body[:80]})")
        if e.code == 429:
            return False, ("× 429 RATE LIMITED — key is valid + subscribed "
                           "but the daily/hourly cap is hit on this account.")
        return False, f"× HTTP {e.code}: {e.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        msg = str(e)
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return False, ("× Request timed out (>{0}s). Most common cause: "
                           "the account that owns this key hasn't subscribed "
                           "to cricket-live-line-advance yet. "
                           "Subscribe (free) at https://rapidapi.com/"
                           "apiservicesprovider/api/cricket-live-line-advance/"
                           "pricing → Start Free Plan.".format(timeout))
        return False, f"× Network error: {msg}"
    except Exception as e:
        return False, f"× {e}"
