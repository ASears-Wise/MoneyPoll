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
from src.openstates_client import get_api_key  # noqa: E402
from src.state_data import load_state_master  # noqa: E402
from src.state_pipeline import fetch_and_overlay_openstates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-mock", action="store_true", help="Regenerate mock then copy to live")
    ap.add_argument("--openstates", action="store_true", help="Overlay OpenStates people")
    ap.add_argument("--states", nargs="*", default=None, help="State codes for OpenStates (default: all)")
    args = ap.parse_args()

    if args.from_mock:
        from generate_state_mock_data import main as gen

        gen()

    if args.openstates:
        if not get_api_key():
            raise SystemExit("OPENSTATES_API_KEY required for --openstates")
        print("Fetching OpenStates lower…")
        lower, lab_l, st_l = fetch_and_overlay_openstates(
            "lower", states=args.states, progress=True, persist=True
        )
        print(lab_l, st_l)
        print("Fetching OpenStates upper…")
        upper, lab_u, st_u = fetch_and_overlay_openstates(
            "upper", states=args.states, progress=True, persist=True
        )
        print(lab_u, st_u)
        return

    lower, _ = load_state_master("lower")
    upper, _ = load_state_master("upper")
    STATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    lower.to_parquet(STATE_LOWER_PARQUET, index=False)
    upper.to_parquet(STATE_UPPER_PARQUET, index=False)
    print(f"Wrote {len(lower)} lower → {STATE_LOWER_PARQUET}")
    print(f"Wrote {len(upper)} upper → {STATE_UPPER_PARQUET}")


if __name__ == "__main__":
    main()
