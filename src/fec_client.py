"""
OpenFEC API client — House candidate totals & independent expenditures.

Docs: https://api.open.fec.gov/developers/
Base: https://api.open.fec.gov/v1/

Set FEC_API_KEY in .env (or DEMO_KEY for very low rate limits).
Never commit real keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests
from dotenv import load_dotenv

from config import CACHE_DIR, ROOT

load_dotenv(ROOT / ".env")

FEC_BASE = "https://api.open.fec.gov/v1"
_MIN_INTERVAL = 0.12  # polite client-side pacing
_last_call = 0.0

# Map FEC party codes → R/D/O
_PARTY_SIDE = {
    "REP": "R",
    "DEM": "D",
    "LIB": "O",
    "GRE": "O",
    "IND": "O",
    "NNE": "O",
    "UNK": "O",
}


def get_api_key() -> str | None:
    """Resolve FEC key from env, then Streamlit secrets (Cloud)."""
    key = os.getenv("FEC_API_KEY", "").strip()
    if key:
        return key
    # Streamlit Community Cloud: Settings → Secrets
    try:
        import streamlit as st

        for name in ("FEC_API_KEY", "fec_api_key", "OPENFEC_API_KEY"):
            if name in st.secrets:
                val = str(st.secrets[name]).strip()
                if val:
                    return val
    except Exception:
        pass
    return None


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:28]
    return CACHE_DIR / f"fec_{h}.json"


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call = time.time()


def _get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    use_cache: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict[str, Any]:
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError("FEC_API_KEY not set. Add it to .env (see .env.example).")

    params = dict(params or {})
    params["api_key"] = key
    cache_key = path + "?" + "&".join(
        f"{k}={v}" for k, v in sorted(params.items()) if k != "api_key"
    )
    cpath = _cache_path(cache_key)
    if use_cache and cpath.exists():
        age_h = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_h <= cache_ttl_hours:
            try:
                return json.loads(cpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    _throttle()
    url = f"{FEC_BASE}/{path.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=90)
    if resp.status_code == 429:
        time.sleep(2.0)
        _throttle()
        resp = requests.get(url, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if use_cache:
        cpath.write_text(json.dumps(data), encoding="utf-8")
    return data


def paginate(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    per_page: int = 100,
    max_pages: int | None = None,
    use_cache: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield all result rows from a paginated OpenFEC endpoint."""
    page = 1
    while True:
        p = dict(params or {})
        p["per_page"] = per_page
        p["page"] = page
        data = _get(path, p, api_key=api_key, use_cache=use_cache)
        results = data.get("results") or []
        for row in results:
            yield row
        pagination = data.get("pagination") or {}
        pages = int(pagination.get("pages") or 1)
        if page >= pages:
            break
        page += 1
        if max_pages is not None and page > max_pages:
            break


def district_id_from_row(row: dict[str, Any]) -> str | None:
    """Build XX-## / XX-00 from FEC state + district fields."""
    state = (row.get("state") or row.get("candidate_office_state") or "").strip().upper()
    if not state or len(state) != 2:
        return None
    dist = row.get("district")
    if dist is None:
        dist = row.get("district_number")
    if dist is None:
        dist = row.get("candidate_office_district")
    if dist is None or dist == "" or str(dist).upper() in ("00", "0", "99", "NONE"):
        # at-large / unknown
        if state in {"AK", "DE", "ND", "SD", "VT", "WY", "MT"} and str(dist) in ("00", "0", "1", "01", "1.0"):
            # MT now has 2 seats post-2020; only true at-large:
            if state in {"AK", "DE", "ND", "SD", "VT", "WY"}:
                return f"{state}-00"
        if state in {"AK", "DE", "ND", "SD", "VT", "WY"}:
            return f"{state}-00"
        return None
    try:
        n = int(float(str(dist)))
    except (TypeError, ValueError):
        return None
    if n == 0:
        return f"{state}-00"
    # at-large sometimes coded as 1
    if state in {"AK", "DE", "ND", "SD", "VT", "WY"}:
        return f"{state}-00"
    return f"{state}-{n:02d}"


def party_side(party: str | None) -> str:
    if not party:
        return "O"
    p = str(party).upper().strip()
    if p in _PARTY_SIDE:
        return _PARTY_SIDE[p]
    if p.startswith("R"):
        return "R"
    if p.startswith("D"):
        return "D"
    return "O"


def fetch_house_candidate_totals(
    cycle: int,
    *,
    api_key: str | None = None,
    election_full: bool = True,
    active_only: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """
    All House candidates/totals for a two-year cycle (2022, 2024, 2026, …).

    Endpoint: GET /candidates/totals/
    """
    params: dict[str, Any] = {
        "office": "H",
        "cycle": cycle,
        "election_full": "true" if election_full else "false",
        "sort": "name",
    }
    if active_only:
        params["is_active_candidate"] = "true"

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(
        paginate("candidates/totals/", params, api_key=api_key, per_page=100), 1
    ):
        rows.append(row)
        if progress and i % 500 == 0:
            print(f"  … {cycle}: {i} candidate rows")
    if progress:
        print(f"  cycle {cycle}: {len(rows)} candidate total rows")
    return rows


def fetch_ie_by_district_cycle(
    state: str,
    district: int | str,
    cycle: int,
    *,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Independent expenditure aggregates by candidate for one House district.

    Endpoint: GET /schedules/schedule_e/by_candidate/
    Requires office=house + state + district.
    """
    # FEC wants district as 00 / 01 style for at-large vs numbered
    if isinstance(district, str):
        d = district
    else:
        d = f"{int(district):02d}"
    params = {
        "cycle": cycle,
        "office": "house",
        "state": state.upper(),
        "district": d,
    }
    try:
        return list(paginate("schedules/schedule_e/by_candidate/", params, api_key=api_key))
    except requests.HTTPError:
        # some at-large coded as 00 vs 01
        if d in ("00", "01"):
            alt = "01" if d == "00" else "00"
            params["district"] = alt
            try:
                return list(
                    paginate("schedules/schedule_e/by_candidate/", params, api_key=api_key)
                )
            except requests.HTTPError:
                return []
        return []


def fetch_ie_by_candidate_ids(
    candidate_ids: list[str],
    cycle: int,
    *,
    api_key: str | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Pull IE totals for specific candidate_ids (batched one-by-one; cached)."""
    all_rows: list[dict[str, Any]] = []
    for i, cid in enumerate(candidate_ids, 1):
        if not cid:
            continue
        params = {"cycle": cycle, "candidate_id": cid}
        try:
            rows = list(
                paginate("schedules/schedule_e/by_candidate/", params, api_key=api_key)
            )
            all_rows.extend(rows)
        except requests.HTTPError as e:
            if progress:
                print(f"  IE candidate {cid}: {e}")
        if progress and i % 50 == 0:
            print(f"  … IE candidates {i}/{len(candidate_ids)}")
    return all_rows


def aggregate_district_finance(
    candidate_rows: list[dict[str, Any]],
    cycle: int,
) -> pd.DataFrame:
    """
    Collapse candidate totals → one row per district_id with R/D raised & spent.
    """
    raised: dict[str, dict[str, float]] = defaultdict(lambda: {"R": 0.0, "D": 0.0, "O": 0.0})
    spent: dict[str, dict[str, float]] = defaultdict(lambda: {"R": 0.0, "D": 0.0, "O": 0.0})
    n_cand: dict[str, dict[str, int]] = defaultdict(lambda: {"R": 0, "D": 0, "O": 0})
    # incumbents / top raisers for candidate lists
    cand_lists: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in candidate_rows:
        did = district_id_from_row(row)
        if not did:
            continue
        side = party_side(row.get("party"))
        rec = float(row.get("receipts") or 0)
        dis = float(row.get("disbursements") or 0)
        raised[did][side] += rec
        spent[did][side] += dis
        n_cand[did][side] += 1
        cand_lists[did].append(
            {
                "name": row.get("name"),
                "party": row.get("party"),
                "side": side,
                "receipts": rec,
                "disbursements": dis,
                "candidate_id": row.get("candidate_id"),
                "incumbent_challenge": row.get("incumbent_challenge"),
                "incumbent_challenge_full": row.get("incumbent_challenge_full"),
                "cash_on_hand": float(row.get("cash_on_hand_end_period") or 0)
                if row.get("cash_on_hand_end_period") is not None
                else None,
            }
        )

    records = []
    for did in sorted(set(raised) | set(spent)):
        records.append(
            {
                "district_id": did,
                "cycle": cycle,
                f"fec_raised_r_{cycle}": raised[did]["R"],
                f"fec_spent_r_{cycle}": spent[did]["R"],
                f"fec_raised_d_{cycle}": raised[did]["D"],
                f"fec_spent_d_{cycle}": spent[did]["D"],
                f"fec_raised_o_{cycle}": raised[did]["O"],
                f"fec_n_cand_r_{cycle}": n_cand[did]["R"],
                f"fec_n_cand_d_{cycle}": n_cand[did]["D"],
                f"fec_candidates_{cycle}": json.dumps(
                    sorted(cand_lists[did], key=lambda x: -x["receipts"])[:12]
                ),
            }
        )
    return pd.DataFrame(records)


def aggregate_outside_spending(
    ie_rows: list[dict[str, Any]],
    cycle: int,
) -> pd.DataFrame:
    """
    Sum independent expenditures (support+oppose) per district from by_candidate IE rows.
    OpenFEC by_candidate rows typically include candidate_id, total, support_oppose, etc.
    """
    totals: dict[str, float] = defaultdict(float)
    for row in ie_rows:
        did = row.get("_district_id") or district_id_from_row(row)
        if not did:
            cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            if cand:
                did = district_id_from_row(
                    {
                        "state": cand.get("state") or row.get("candidate_office_state"),
                        "district": cand.get("district") or row.get("candidate_office_district"),
                    }
                )
        if not did:
            continue
        amt = row.get("total")
        if amt is None:
            amt = row.get("total_ytd") or row.get("expenditure_amount") or 0
        totals[did] += abs(float(amt or 0))

    return pd.DataFrame(
        [
            {"district_id": did, f"fec_outside_{cycle}": tot}
            for did, tot in sorted(totals.items())
        ]
    )


def fetch_house_ie_national(
    cycle: int,
    *,
    api_key: str | None = None,
    district_ids: list[str] | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Pull House IE by_candidate for each district (office+state+district required by API).

    district_ids like ['AZ-01', 'WY-00']. If None, uses full 435-seat apportionment
    from a lightweight state seat map.
    """
    if district_ids is None:
        # Post-2020 House apportionment (435)
        seats = {
            "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
            "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
            "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
            "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
            "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
            "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
            "WI": 8, "WY": 1,
        }
        district_ids = []
        for st, n in seats.items():
            if n == 1:
                district_ids.append(f"{st}-00")
            else:
                for d in range(1, n + 1):
                    district_ids.append(f"{st}-{d:02d}")

    all_rows: list[dict[str, Any]] = []
    for i, did in enumerate(district_ids, 1):
        try:
            st, num = did.split("-")
            rows = fetch_ie_by_district_cycle(st, num, cycle, api_key=api_key)
            # stamp district_id for aggregation
            for r in rows:
                r.setdefault("state", st)
                r.setdefault("district", num)
                r["_district_id"] = did
            all_rows.extend(rows)
            if progress and (i % 25 == 0 or rows):
                print(f"  IE {cycle} {did}: {len(rows)} rows ({i}/{len(district_ids)})")
        except Exception as e:  # noqa: BLE001
            if progress:
                print(f"  IE {cycle} {did}: failed {e}")
    return all_rows


def build_cycle_frames(
    cycles: list[int] | None = None,
    *,
    api_key: str | None = None,
    fetch_outside: bool = True,
    outside_cycle: int = 2026,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Fetch and aggregate multiple cycles into wide district-level frames.

    Returns (finance_wide, outside_df_or_None).
    finance_wide has one row per district_id with fec_*_{cycle} columns.
    """
    cycles = cycles or [2026, 2024, 2022]
    frames = []
    for cy in cycles:
        if progress:
            print(f"Fetching House candidate totals for {cy}…")
        rows = fetch_house_candidate_totals(cy, api_key=api_key, progress=progress)
        frames.append(aggregate_district_finance(rows, cy))

    wide = frames[0]
    for fr in frames[1:]:
        wide = wide.merge(fr, on="district_id", how="outer")

    # drop helper cycle cols
    drop_cols = [c for c in wide.columns if c == "cycle" or c.startswith("cycle_")]
    wide = wide.drop(columns=drop_cols, errors="ignore")

    outside = None
    if fetch_outside:
        if progress:
            print(f"Fetching independent expenditures (House) for {outside_cycle}…")
        ie_rows = fetch_house_ie_national(outside_cycle, api_key=api_key, progress=progress)
        outside = aggregate_outside_spending(ie_rows, outside_cycle)

    return wide, outside


def merge_fec_into_master(
    master: pd.DataFrame,
    finance: pd.DataFrame,
    outside: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Overlay FEC aggregates onto master; recompute hist_cost_to_compete when possible."""
    out = master.copy()
    # Drop existing FEC numeric columns we'll replace (keep mock if district missing)
    fec_cols = [c for c in finance.columns if c != "district_id"]
    if outside is not None:
        fec_cols = list(set(fec_cols) | set(c for c in outside.columns if c != "district_id"))

    # Merge finance
    fin = finance.copy()
    # drop cycle leftover columns from multi-merge
    for c in list(fin.columns):
        if c.startswith("cycle"):
            fin = fin.drop(columns=[c])

    out = out.merge(fin, on="district_id", how="left", suffixes=("", "_fecnew"))

    # Prefer new FEC values where present
    for c in fin.columns:
        if c == "district_id":
            continue
        new_c = f"{c}_fecnew" if f"{c}_fecnew" in out.columns else None
        if new_c:
            out[c] = out[new_c].combine_first(out[c])
            out = out.drop(columns=[new_c])
        elif c in out.columns and f"{c}_x" in out.columns:
            pass
        # columns only from merge with _x/_y
        if f"{c}_y" in out.columns:
            out[c] = out[f"{c}_y"].combine_first(out.get(f"{c}_x", out.get(c)))
            out = out.drop(columns=[x for x in (f"{c}_x", f"{c}_y") if x in out.columns])

    # Clean duplicate suffix columns for standard fields
    for cycle in (2022, 2024, 2026):
        for base in (
            f"fec_raised_r_{cycle}",
            f"fec_spent_r_{cycle}",
            f"fec_raised_d_{cycle}",
            f"fec_spent_d_{cycle}",
        ):
            if f"{base}_y" in out.columns:
                out[base] = out[f"{base}_y"].combine_first(out.get(f"{base}_x", pd.Series([None] * len(out))))
                out = out.drop(columns=[c for c in (f"{base}_x", f"{base}_y") if c in out.columns])
            if f"{base}_fecnew" in out.columns:
                out[base] = out[f"{base}_fecnew"].combine_first(out.get(base))
                out = out.drop(columns=[f"{base}_fecnew"])

    if outside is not None and len(outside):
        out = out.drop(columns=[c for c in out.columns if c.startswith("fec_outside_")], errors="ignore")
        out = out.merge(outside, on="district_id", how="left")
        if "fec_outside_2024" not in out.columns and "fec_outside_2024_y" in out.columns:
            out["fec_outside_2024"] = out["fec_outside_2024_y"]

    # hist_cost_to_compete from latest complete cycle with data (prefer current 2026)
    def _hist(row: pd.Series) -> float:
        for cy in (2026, 2024, 2022):
            r = float(row.get(f"fec_raised_r_{cy}") or 0)
            d = float(row.get(f"fec_raised_d_{cy}") or 0)
            if r + d > 0:
                return (r + d) * 0.35
        return float(row.get("hist_cost_to_compete") or 500_000)

    out["hist_cost_to_compete"] = out.apply(_hist, axis=1)

    # Enrich 2026 candidates JSON when available
    if "fec_candidates_2026" in out.columns:
        def _cand_json(row: pd.Series) -> str:
            raw = row.get("fec_candidates_2026")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return row.get("civic_candidates_json") or "{}"
            try:
                cands = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                return row.get("civic_candidates_json") or "{}"
            payload = {
                "office": f"U.S. House {row['district_id']}",
                "source": "openfec",
                "cycle": 2026,
                "candidates": [
                    {
                        "name": c.get("name"),
                        "party": c.get("party"),
                        "candidate_id": c.get("candidate_id"),
                        "receipts": c.get("receipts"),
                        "incumbent_challenge": c.get("incumbent_challenge_full"),
                    }
                    for c in (cands or [])
                ],
            }
            return json.dumps(payload)

        out["civic_candidates_json"] = out.apply(_cand_json, axis=1)

    flags = out.get("data_source_flags", pd.Series(["base"] * len(out))).astype(str)
    out["data_source_flags"] = flags.apply(
        lambda s: s if "openfec" in s else (s + "|openfec").strip("|")
    )
    out["notes"] = out.get("notes", "").astype(str)
    out["notes"] = out["notes"].apply(
        lambda n: (n + "; FEC via OpenFEC API").replace(";;", ";") if "OpenFEC" not in n else n
    )
    return out


def safe_fec_call(fn, *args, **kwargs) -> tuple[Any | None, str | None]:
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)
