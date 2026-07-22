#!/usr/bin/env python3
"""
Build state lower/upper masters from mock + optional OpenStates overlay.

Usage:
    python scripts/build_state_masters.py
    python scripts/build_state_masters.py --from-mock
    python scripts/build_state_masters.py --openstates --states AZ GA MI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import STATE_DATA_DIR, STATE_LOWER_PARQUET, STATE_UPPER_PARQUET  # noqa: E402
from src.openstates_client import (  # noqa: E402
    get_api_key,
    list_people,
    people_to_overlay_rows,
)
from src.state_data import load_state_master  # noqa: E402


def apply_overlay(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return base
    out = base.copy()
    o = overlay.drop_duplicates("district_id").set_index("district_id")
    for col in ("rep_name", "rep_party", "party_control"):
        if col not in o.columns:
            continue
        mapped = out["district_id"].map(o[col])
        out[col] = mapped.combine_first(out[col])
    out["data_source_flags"] = out["data_source_flags"].astype(str).apply(
        lambda s: s if "openstates" in s else f"{s}|openstates".strip("|")
    )
    return out


def fetch_openstates_for_states(states: list[str], chamber: str) -> pd.DataFrame:
    rows = []
    for st in states:
        print(f"  OpenStates {st} {chamber}…")
        try:
            people = list_people(st.lower(), org_classification=chamber if chamber != "lower" else None)
            # For lower, also try without filter then filter client-side if empty
            if not people and chamber == "lower":
                people = list_people(st.lower())
            part = people_to_overlay_rows(people, chamber)
            print(f"    {len(part)} mapped seats")
            rows.extend(part)
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-mock", action="store_true", help="Regenerate mock then copy to live")
    ap.add_argument("--openstates", action="store_true", help="Overlay OpenStates people")
    ap.add_argument("--states", nargs="*", default=None, help="State codes for OpenStates (default: all)")
    args = ap.parse_args()

    if args.from_mock:
        from scripts.generate_state_mock_data import main as gen

        gen()

    lower, _ = load_state_master("lower")
    upper, _ = load_state_master("upper")

    if args.openstates:
        if not get_api_key():
            raise SystemExit("OPENSTATES_API_KEY required for --openstates")
        states = args.states or sorted(lower["state"].unique().tolist())
        print("Fetching OpenStates lower…")
        lo = fetch_openstates_for_states(states, "lower")
        print("Fetching OpenStates upper…")
        uo = fetch_openstates_for_states(states, "upper")
        lower = apply_overlay(lower, lo)
        upper = apply_overlay(upper, uo)

    STATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    lower.to_parquet(STATE_LOWER_PARQUET, index=False)
    upper.to_parquet(STATE_UPPER_PARQUET, index=False)
    print(f"Wrote {len(lower)} lower → {STATE_LOWER_PARQUET}")
    print(f"Wrote {len(upper)} upper → {STATE_UPPER_PARQUET}")


if __name__ == "__main__":
    main()
