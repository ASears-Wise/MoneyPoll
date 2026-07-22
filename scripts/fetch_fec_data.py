#!/usr/bin/env python3
"""
Fetch House campaign finance from the OpenFEC API and merge into master.parquet.

Usage:
    python scripts/fetch_fec_data.py
    python scripts/fetch_fec_data.py --cycles 2022 2024 2026
    python scripts/fetch_fec_data.py --no-outside

Requires FEC_API_KEY in .env or environment.
Docs: https://api.open.fec.gov/developers/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fec_client import get_api_key  # noqa: E402
from src.fec_pipeline import fetch_and_merge_fec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch OpenFEC House finance → master.parquet")
    ap.add_argument("--cycles", nargs="+", type=int, default=[2026, 2024, 2022])
    ap.add_argument("--no-outside", action="store_true", help="Skip IE (faster)")
    ap.add_argument("--outside-cycle", type=int, default=2026)
    args = ap.parse_args()

    if not get_api_key():
        print("ERROR: FEC_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    print(f"Cycles: {args.cycles}")
    print(f"Outside/IE: {not args.no_outside} (cycle {args.outside_cycle})")

    df, label, stats = fetch_and_merge_fec(
        cycles=args.cycles,
        include_outside=not args.no_outside,
        outside_cycle=args.outside_cycle,
        progress=True,
        persist=True,
    )
    print(f"\n{label}: {len(df)} rows")
    for cy in args.cycles:
        k = f"districts_with_receipts_{cy}"
        r = f"raised_{cy}"
        if k in stats:
            print(f"  {cy}: {stats[k]} districts; total raised ≈ ${stats[r]/1e9:.2f}B")
    if stats.get("include_outside"):
        print(
            f"  outside 2024: {stats.get('districts_with_outside')} districts; "
            f"${stats.get('outside_2024', 0)/1e6:.0f}M"
        )
    try:
        from src.ui_common import write_refresh_meta
        from datetime import datetime, timezone

        write_refresh_meta(
            {
                "federal_refreshed_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "federal_source": "openfec",
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"(meta skip: {e})")
    print("Done.")


if __name__ == "__main__":
    main()
