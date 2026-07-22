"""
OpenStates API v3 client — state legislators & jurisdictions.

Docs: https://docs.openstates.org/api-v3/
Get a free key: https://openstates.org/accounts/profile/
Set OPENSTATES_API_KEY in .env or Streamlit secrets.
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

from config import OPENSTATES_BASE, OPENSTATES_CACHE_TTL_HOURS, ROOT, STATE_CACHE_DIR

load_dotenv(ROOT / ".env")

_MIN_INTERVAL = 0.2
_last = 0.0


def get_api_key() -> str | None:
    key = os.getenv("OPENSTATES_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        for name in ("OPENSTATES_API_KEY", "openstates_api_key"):
            if name in st.secrets:
                val = str(st.secrets[name]).strip()
                if val:
                    return val
    except Exception:
        pass
    return None


def _cache_path(key: str) -> Path:
    STATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:28]
    return STATE_CACHE_DIR / f"os_{h}.json"


def _get(path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict[str, Any]:
    global _last
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("OPENSTATES_API_KEY not set (.env or Streamlit secrets).")

    params = dict(params or {})
    cache_key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    cpath = _cache_path(cache_key)
    if use_cache and cpath.exists():
        age_h = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_h <= OPENSTATES_CACHE_TTL_HOURS:
            try:
                return json.loads(cpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    elapsed = time.time() - _last
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last = time.time()

    headers = {"X-API-KEY": api_key}
    url = f"{OPENSTATES_BASE.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    if resp.status_code == 429:
        time.sleep(2.0)
        resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if use_cache:
        cpath.write_text(json.dumps(data), encoding="utf-8")
    return data


def list_people(
    jurisdiction: str,
    *,
    org_classification: str | None = None,
    per_page: int = 50,
    max_pages: int = 40,
) -> list[dict[str, Any]]:
    """
    jurisdiction e.g. 'az' or 'ocd-jurisdiction/country:us/state:az/government'
    org_classification: 'lower' | 'upper' | None
    """
    # Normalize to openstates jurisdiction id form used by API
    j = jurisdiction.strip().lower()
    if not j.startswith("ocd-jurisdiction"):
        j = f"ocd-jurisdiction/country:us/state:{j}/government"

    out: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        params: dict[str, Any] = {
            "jurisdiction": j,
            "per_page": per_page,
            "page": page,
        }
        if org_classification:
            params["org_classification"] = org_classification
        data = _get("people", params)
        results = data.get("results") or []
        out.extend(results)
        pagination = data.get("pagination") or {}
        # OpenStates may use max_page or links
        if not results:
            break
        max_page = pagination.get("max_page") or pagination.get("total_pages")
        if max_page is not None and page >= int(max_page):
            break
        if len(results) < per_page:
            break
        page += 1
    return out


def party_to_code(party: str | None) -> str:
    if not party:
        return ""
    p = str(party).lower()
    if "republican" in p:
        return "R"
    if "democrat" in p:
        return "D"
    return "O"


def people_to_overlay_rows(people: list[dict[str, Any]], chamber: str) -> list[dict[str, Any]]:
    """Map OpenStates people records to district overlay fields."""
    rows = []
    for p in people:
        current = p.get("current_role") or {}
        # roles list fallback
        roles = p.get("roles") or []
        role = current if current else (roles[0] if roles else {})
        org = (role.get("org_classification") or role.get("type") or "").lower()
        if chamber == "lower" and org and org not in ("lower", "legislature"):
            # NE unicameral often "legislature"
            if chamber == "lower" and org not in ("lower", "legislature", ""):
                continue
        if chamber == "upper" and org and org != "upper":
            continue

        dist = role.get("district")
        if dist is None:
            continue
        try:
            num = int(str(dist).split("-")[-1].strip())
        except ValueError:
            # keep string digits
            digits = "".join(c for c in str(dist) if c.isdigit())
            if not digits:
                continue
            num = int(digits)

        # state from jurisdiction
        juris = (p.get("jurisdiction") or {}).get("id") or role.get("jurisdiction") or ""
        st = ""
        if "state:" in str(juris):
            st = str(juris).split("state:")[1].split("/")[0].upper()
        if not st and p.get("jurisdiction"):
            st = str(p["jurisdiction"].get("name", ""))[:2].upper()

        party = ""
        parties = p.get("party") or []
        if isinstance(parties, list) and parties:
            party = party_to_code(parties[0].get("name") if isinstance(parties[0], dict) else parties[0])
        elif isinstance(parties, str):
            party = party_to_code(parties)

        name = p.get("name") or f"{p.get('given_name', '')} {p.get('family_name', '')}".strip()
        tag = "L" if chamber == "lower" else "U"
        did = f"{st}-{tag}-{num:03d}" if st else None
        if not did:
            continue
        rows.append(
            {
                "district_id": did,
                "state": st,
                "rep_name": name,
                "rep_party": party,
                "party_control": party if party in ("R", "D") else None,
                "openstates_id": p.get("id"),
            }
        )
    return rows


def safe_openstates_call(fn, *args, **kwargs) -> tuple[Any | None, str | None]:
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)
