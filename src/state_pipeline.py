"""
OpenStates overlay pipeline for state legislative masters.

Used by CLI (build_state_masters.py) and Streamlit cloud-side refresh.
"""
from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

from config import STATE_DATA_DIR, STATE_LOWER_PARQUET, STATE_UPPER_PARQUET
from src.openstates_client import get_api_key, list_people, people_to_overlay_rows
from src.state_data import load_state_master

Chamber = Literal["lower", "upper"]


def apply_overlay(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay is None or overlay.empty:
        return base
    out = base.copy()
    o = overlay.drop_duplicates("district_id").set_index("district_id")
    for col in ("rep_name", "rep_party", "party_control"):
        if col not in o.columns:
            continue
        mapped = out["district_id"].map(o[col])
        # Prefer live non-null overlay
        out[col] = mapped.where(mapped.notna() & (mapped.astype(str) != ""), out[col])
    # Recompute is_open_seat lightly when we got a real name
    if "rep_name" in out.columns:
        out["is_open_seat"] = out["rep_name"].astype(str).str.upper().eq("OPEN SEAT")
        out["incumbent_running"] = ~out["is_open_seat"]
    flags = out.get("data_source_flags", pd.Series(["base"] * len(out))).astype(str)
    out["data_source_flags"] = flags.apply(
        lambda s: s if "openstates" in s else f"{s}|openstates".strip("|")
    )
    out["notes"] = out.get("notes", "").astype(str).apply(
        lambda n: n
        if "OpenStates" in n
        else (n + "; members via OpenStates API").replace(";;", ";")
    )
    return out


def fetch_openstates_for_states(
    states: Sequence[str],
    chamber: Chamber,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, st in enumerate(states, 1):
        st = str(st).upper()
        if progress:
            print(f"  OpenStates {st} {chamber} ({i}/{len(states)})…")
        try:
            # Prefer chamber classification; fall back to all people in jurisdiction
            people = list_people(st.lower(), org_classification=chamber)
            if not people:
                people = list_people(st.lower(), org_classification=None)
            part = people_to_overlay_rows(people, chamber)
            if progress:
                print(f"    mapped {len(part)} seats from {len(people)} people")
            rows.extend(part)
        except Exception as e:  # noqa: BLE001
            if progress:
                print(f"    failed: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_and_overlay_openstates(
    chamber: Chamber,
    *,
    states: Sequence[str] | None = None,
    progress: bool = False,
    persist: bool = True,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Load base master, overlay OpenStates members for selected (or all) states.

    Returns (df, label, stats).
    """
    if not get_api_key():
        raise RuntimeError(
            "OPENSTATES_API_KEY not set. Add it to Streamlit Secrets or .env "
            '(TOML: OPENSTATES_API_KEY = "…").'
        )

    base, base_label = load_state_master(chamber)
    state_list = list(states) if states else sorted(base["state"].dropna().unique().tolist())
    overlay = fetch_openstates_for_states(state_list, chamber, progress=progress)
    merged = apply_overlay(base, overlay)

    n_mapped = 0
    if not overlay.empty and "district_id" in overlay.columns:
        n_mapped = int(overlay["district_id"].nunique())
    n_live_names = 0
    if "data_source_flags" in merged.columns:
        # seats where name no longer looks mock-generic is hard; count overlay hits
        if not overlay.empty:
            hit = set(overlay["district_id"].astype(str))
            n_live_names = int(merged["district_id"].isin(hit).sum())

    if persist:
        try:
            STATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = STATE_LOWER_PARQUET if chamber == "lower" else STATE_UPPER_PARQUET
            merged.to_parquet(path, index=False)
        except OSError:
            pass

    stats = {
        "chamber": chamber,
        "states_requested": len(state_list),
        "overlay_rows": len(overlay),
        "seats_matched": n_live_names,
        "unique_districts_in_overlay": n_mapped,
        "total_seats": len(merged),
    }
    label = f"OpenStates overlay · {chamber} · {n_live_names}/{len(merged)} seats matched"
    return merged, label, stats


def master_needs_openstates(df: pd.DataFrame, label: str) -> bool:
    """True if we should auto-pull OpenStates (mock-only, no openstates flag)."""
    if "data_source_flags" in df.columns:
        if df["data_source_flags"].astype(str).str.contains("openstates", case=False, na=False).any():
            return False
    return "mock" in label.lower() or (
        "data_source_flags" in df.columns
        and df["data_source_flags"].astype(str).str.contains("mock", case=False, na=False).any()
    )
