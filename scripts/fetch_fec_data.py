#!/usr/bin/env python3
"""
Fetch House campaign finance from the OpenFEC API and merge into master.parquet.

Usage:
    python scripts/fetch_fec_data.py
    python scripts/fetch_fec_data.py --cycles 2022 2024 2026
    python scripts/fetch_fec_data.py --no-outside          # skip IE (faster)
    python scripts/fetch_fec_data.py --outside-only 2024

Requires FEC_API_KEY in .env (https://api.data.gov/signup/ — used by OpenFEC).
Docs: https://api.open.fec.gov/developers/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MASTER_PARQUET, MOCK_MASTER_PARQUET, RAW_DIR  # noqa: E402
from src.data_loader import load_master  # noqa: E402
from src.fec_client import build_cycle_frames, get_api_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch OpenFEC House finance → master.parquet")
    ap.add_argument(
        "--cycles",
        nargs="+",
        type=int,
        default=[2022, 2024, 2026],
        help="Election cycles (two-year periods)",
    )
    ap.add_argument(
        "--no-outside",
        action="store_true",
        help="Skip independent expenditure pull (much faster)",
    )
    ap.add_argument(
        "--outside-cycle",
        type=int,
        default=2024,
        help="Cycle for outside/IE spending aggregate",
    )
    ap.add_argument(
        "--save-raw",
        action="store_true",
        help="Also write intermediate parquet under data/raw/",
    )
    args = ap.parse_args()

    if not get_api_key():
        print("ERROR: FEC_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    print(f"Cycles: {args.cycles}")
    print(f"Outside/IE: {not args.no_outside} (cycle {args.outside_cycle})")

    finance, outside = build_cycle_frames(
        cycles=args.cycles,
        fetch_outside=not args.no_outside,
        outside_cycle=args.outside_cycle,
        progress=True,
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        finance.to_parquet(RAW_DIR / "fec_district_finance.parquet", index=False)
        print(f"Wrote {RAW_DIR / 'fec_district_finance.parquet'} ({len(finance)} districts)")
        if outside is not None:
            outside.to_parquet(RAW_DIR / "fec_outside.parquet", index=False)
            print(f"Wrote {RAW_DIR / 'fec_outside.parquet'}")

    # Always save finance snapshot for re-merge without re-fetch
    finance.to_parquet(RAW_DIR / "fec_district_finance.parquet", index=False)
    if outside is not None and len(outside):
        outside.to_parquet(RAW_DIR / "fec_outside.parquet", index=False)

    if MASTER_PARQUET.exists() or MOCK_MASTER_PARQUET.exists():
        master, src = load_master()
        print(f"Loaded master from {src} ({len(master)} rows)")
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_mock_data import generate

        master = generate()
        print("Generated mock master base")

    # Simpler merge: drop target FEC columns then join
    drop_prefixes = (
        "fec_raised_",
        "fec_spent_",
        "fec_outside_",
        "fec_n_cand_",
        "fec_candidates_",
        "fec_raised_o_",
    )
    keep = [c for c in master.columns if not any(c.startswith(p) for p in drop_prefixes)]
    base = master[keep].copy()

    merged = base.merge(finance, on="district_id", how="left")
    # drop accidental cycle cols
    merged = merged.drop(columns=[c for c in merged.columns if c == "cycle" or c.startswith("cycle_")], errors="ignore")

    if outside is not None and len(outside):
        oc = [c for c in outside.columns if c != "district_id"]
        merged = merged.drop(columns=[c for c in oc if c in merged.columns], errors="ignore")
        merged = merged.merge(outside, on="district_id", how="left")

    # hist cost + flags via shared helper (re-apply on already-merged)
    # Reconstruct using merge_fec_into_master for candidate JSON / hist
    # Re-attach dropped mock fec if needed — we already dropped; fillna 0 for missing
    for cy in args.cycles:
        for col in (
            f"fec_raised_r_{cy}",
            f"fec_spent_r_{cy}",
            f"fec_raised_d_{cy}",
            f"fec_spent_d_{cy}",
        ):
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    if "fec_outside_2024" not in merged.columns:
        merged["fec_outside_2024"] = 0.0
    else:
        merged["fec_outside_2024"] = pd.to_numeric(merged["fec_outside_2024"], errors="coerce").fillna(0.0)

    def _hist(row: pd.Series) -> float:
        for cy in (2024, 2022, 2026):
            r = float(row.get(f"fec_raised_r_{cy}") or 0)
            d = float(row.get(f"fec_raised_d_{cy}") or 0)
            if r + d > 0:
                return (r + d) * 0.35
        return float(row.get("hist_cost_to_compete") or 500_000)

    merged["hist_cost_to_compete"] = merged.apply(_hist, axis=1)

    # 2026 candidates → civic_candidates_json
    import json

    if "fec_candidates_2026" in merged.columns:
        def _cj(row: pd.Series) -> str:
            raw = row.get("fec_candidates_2026")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return row.get("civic_candidates_json") if isinstance(row.get("civic_candidates_json"), str) else "{}"
            try:
                cands = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                return "{}"
            return json.dumps(
                {
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
            )

        merged["civic_candidates_json"] = merged.apply(_cj, axis=1)

    flags = merged.get("data_source_flags", pd.Series(["base"] * len(merged))).astype(str)
    merged["data_source_flags"] = flags.apply(
        lambda s: s if "openfec" in s else f"{s}|openfec".strip("|")
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MASTER_PARQUET, index=False)
    print(f"\nWrote {len(merged)} rows → {MASTER_PARQUET}")

    # Coverage stats
    for cy in args.cycles:
        col = f"fec_raised_r_{cy}"
        if col in merged.columns:
            n = int((merged[col].fillna(0) + merged[f"fec_raised_d_{cy}"].fillna(0) > 0).sum())
            tot = float(merged[col].fillna(0).sum() + merged[f"fec_raised_d_{cy}"].fillna(0).sum())
            print(f"  {cy}: {n} districts with R/D receipts; total raised ≈ ${tot/1e9:.2f}B")

    print("Done. Restart Streamlit or clear cache to load new data.")


if __name__ == "__main__":
    main()
