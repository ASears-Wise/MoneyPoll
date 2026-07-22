"""
Google Civic Information API helpers.

IMPORTANT (2025+): Google turned down the Representatives API
(representativeInfoByAddress / representativeInfoByDivision) in April 2025.
Use static legislator data for current members.

Still available for election / voter information (VIP-supported elections):
  - elections.electionQuery
  - elections.voterInfoQuery

Get a free API key: Google Cloud Console → enable "Google Civic Information API".
Set GOOGLE_CIVIC_API_KEY in .env or the environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from config import CACHE_DIR, CIVIC_BASE, CIVIC_CACHE_TTL_HOURS, ROOT

load_dotenv(ROOT / ".env")

# Conservative client-side pacing
_MIN_INTERVAL_SEC = 0.15
_last_call = 0.0


def get_api_key() -> str | None:
    """Resolve Civic key from env, then Streamlit secrets (Cloud)."""
    key = os.getenv("GOOGLE_CIVIC_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        for name in ("GOOGLE_CIVIC_API_KEY", "google_civic_api_key"):
            if name in st.secrets:
                val = str(st.secrets[name]).strip()
                if val:
                    return val
    except Exception:
        pass
    return None


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_DIR / f"civic_{h}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    if age_h > CIVIC_CACHE_TTL_HOURS:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_call = time.time()


def _get(
    endpoint: str,
    params: dict[str, Any],
    api_key: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    GET helper with cache + rate limit.

    Raises RuntimeError if no API key; requests.HTTPError on API failure.
    """
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError(
            "GOOGLE_CIVIC_API_KEY not set. Add it to .env (see .env.example)."
        )

    params = {**params, "key": key}
    cache_key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "key")
    cpath = _cache_path(cache_key)
    if use_cache:
        cached = _read_cache(cpath)
        if cached is not None:
            return cached

    _throttle()
    url = f"{CIVIC_BASE}/{endpoint.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if use_cache:
        _write_cache(cpath, data)
    return data


def get_elections(api_key: str | None = None) -> list[dict[str, Any]]:
    """
    List elections known to the Civic API (VIP).

    Example:
        elections = get_elections()
        for e in elections:
            print(e["id"], e["name"], e.get("electionDay"))
    """
    data = _get("elections", {}, api_key=api_key)
    return data.get("elections", [])


def get_voter_info(
    address: str,
    election_id: str | int,
    api_key: str | None = None,
    official_only: bool = False,
) -> dict[str, Any]:
    """
    Voter information for an address + election, including contests/candidates
    when VIP has data for that election.

    Example:
        info = get_voter_info("1600 Pennsylvania Ave NW, Washington DC", "2000")
        for c in info.get("contests", []):
            print(c.get("office"), c.get("candidates"))
    """
    params: dict[str, Any] = {
        "address": address,
        "electionId": str(election_id),
    }
    if official_only:
        params["officialOnly"] = "true"
    return _get("voterinfo", params, api_key=api_key)


def search_divisions(
    query: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Search Open Civic Data divisions by query string.

    Example:
        search_divisions("cd:12")
    """
    return _get("divisions", {"query": query}, api_key=api_key)


def extract_house_contests(voter_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter voterInfo contests that look like U.S. House races."""
    out = []
    for c in voter_info.get("contests", []) or []:
        office = (c.get("office") or "").lower()
        level = " ".join(c.get("level") or []).lower() if isinstance(c.get("level"), list) else str(c.get("level") or "").lower()
        if "representative" in office or "house" in office or "congressional" in office:
            out.append(c)
        elif "country" in level and "representative" in office:
            out.append(c)
    return out


def safe_civic_call(fn, *args, **kwargs) -> tuple[Any | None, str | None]:
    """
    Call a civic function; return (result, error_message).
    Never raises — suitable for Streamlit UI.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 — surface any API/network issue in UI
        return None, str(e)


# ---------------------------------------------------------------------------
# NOTE: Representatives API (removed April 2025)
#
# Previously:
#   GET /representatives
#   GET /representatives/{ocdId}
#
# Do NOT call these endpoints. Use unitedstates/congress-legislators or
# similar static sources via scripts/build_master_dataset.py.
# ---------------------------------------------------------------------------
