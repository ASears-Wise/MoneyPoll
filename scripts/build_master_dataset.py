#!/usr/bin/env python3
"""
Build / refresh master.parquet from mock + optional real sources.

Pipeline steps (all optional except mock base):
  1. Start from mock or existing master
  2. Overlay legislator roster JSON if provided
  3. Overlay PVI CSV if provided
  4. Optional Google Civic sample (expensive — default off)
  5. Write data/master.parquet

Usage:
    python scripts/build_master_dataset.py
    python scripts/build_master_dataset.py --pvi data/raw/pvi.csv
    python scripts/build_master_dataset.py --legislators data/raw/legislators-current.json

TODO:
  - Census ACS API pull by congressional district
  - unitedstates/congress-legislators auto-download

FEC: prefer `python scripts/fetch_fec_data.py` (OpenFEC API).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MASTER_PARQUET, MOCK_MASTER_PARQUET, RAW_DIR  # noqa: E402


def load_base() -> pd.DataFrame:
    if MASTER_PARQUET.exists():
        return pd.read_parquet(MASTER_PARQUET)
    if MOCK_MASTER_PARQUET.exists():
        return pd.read_parquet(MOCK_MASTER_PARQUET)
    # Generate mock
    from scripts.generate_mock_data import generate  # type: ignore

    # direct import path
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_mock_data import generate as gen  # noqa: E402

    return gen()


def merge_pvi(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Expect CSV with columns: district_id, pvi
    Optional: cook_rating
    """
    pvi = pd.read_csv(path)
    if "district_id" not in pvi.columns or "pvi" not in pvi.columns:
        raise SystemExit("PVI CSV needs district_id, pvi columns")
    out = df.drop(columns=[c for c in ("pvi", "cook_rating") if c in df.columns], errors="ignore")
    cols = ["district_id", "pvi"] + (["cook_rating"] if "cook_rating" in pvi.columns else [])
    out = out.merge(pvi[cols], on="district_id", how="left")
    # Keep old pvi where missing
    if out["pvi"].isna().any() and "pvi" in df.columns:
        out["pvi"] = out["pvi"].fillna(df.set_index("district_id").reindex(out["district_id"])["pvi"].values)
    out["data_source_flags"] = out.get("data_source_flags", "base") + "|pvi_csv"
    return out


def merge_legislators(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Minimal JSON list:
    [{"district_id": "CA-12", "rep_name": "...", "rep_party": "D", "first_elected": 2012}, ...]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    leg = pd.DataFrame(data)
    if "district_id" not in leg.columns:
        raise SystemExit("legislators JSON needs district_id")
    out = df.copy()
    for col in ("rep_name", "rep_party", "first_elected", "party_control"):
        if col in leg.columns:
            m = leg.set_index("district_id")[col]
            out[col] = out["district_id"].map(m).fillna(out[col] if col in out.columns else None)
    out["data_source_flags"] = out["data_source_flags"].astype(str) + "|legislators"
    return out


def optional_civic_probe(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Probe Civic elections list; attach first election id to notes (not full 435)."""
    try:
        from src.civic_client import get_api_key, get_elections
    except ImportError:
        return df
    if not get_api_key():
        print("No GOOGLE_CIVIC_API_KEY — skipping Civic probe")
        return df
    try:
        elections = get_elections()
        print(f"Civic elections available: {len(elections)}")
        if elections:
            eid = elections[0].get("id")
            df = df.copy()
            df["civic_election_id"] = df["civic_election_id"].fillna(eid)
            df["data_source_flags"] = df["data_source_flags"].astype(str) + "|civic_probe"
    except Exception as e:  # noqa: BLE001
        print(f"Civic probe failed: {e}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build House Moneyball master dataset")
    ap.add_argument("--pvi", type=Path, help="CSV with district_id,pvi")
    ap.add_argument("--legislators", type=Path, help="JSON legislator overlay")
    ap.add_argument("--civic-probe", action="store_true", help="Hit Civic elections endpoint")
    ap.add_argument("--from-mock", action="store_true", help="Force regenerate from mock generator")
    ap.add_argument(
        "--fec",
        action="store_true",
        help="Fetch OpenFEC House totals (calls scripts/fetch_fec_data logic; needs FEC_API_KEY)",
    )
    ap.add_argument(
        "--fec-from-cache",
        action="store_true",
        help="Merge data/raw/fec_district_finance.parquet if present (no API)",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_mock:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_mock_data import generate

        df = generate()
    else:
        # Prefer generate_mock_data module path
        try:
            df = load_base()
        except Exception:
            sys.path.insert(0, str(ROOT / "scripts"))
            from generate_mock_data import generate

            df = generate()

    if args.pvi:
        df = merge_pvi(df, args.pvi)
        print(f"Merged PVI from {args.pvi}")
    if args.legislators:
        df = merge_legislators(df, args.legislators)
        print(f"Merged legislators from {args.legislators}")
    if args.civic_probe:
        df = optional_civic_probe(df)

    if args.fec:
        from src.fec_client import build_cycle_frames, get_api_key

        if not get_api_key():
            raise SystemExit("FEC_API_KEY required for --fec")
        finance, outside = build_cycle_frames(progress=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        finance.to_parquet(RAW_DIR / "fec_district_finance.parquet", index=False)
        if outside is not None:
            outside.to_parquet(RAW_DIR / "fec_outside.parquet", index=False)
        drop_prefixes = (
            "fec_raised_",
            "fec_spent_",
            "fec_outside_",
            "fec_n_cand_",
            "fec_candidates_",
            "fec_raised_o_",
        )
        keep = [c for c in df.columns if not any(c.startswith(p) for p in drop_prefixes)]
        df = df[keep].merge(finance, on="district_id", how="left")
        if outside is not None and len(outside):
            df = df.merge(outside, on="district_id", how="left")
        df["data_source_flags"] = df["data_source_flags"].astype(str) + "|openfec"
        print("Merged live OpenFEC aggregates")

    if args.fec_from_cache:
        fin_path = RAW_DIR / "fec_district_finance.parquet"
        if not fin_path.exists():
            raise SystemExit(f"Missing {fin_path}; run scripts/fetch_fec_data.py first")
        finance = pd.read_parquet(fin_path)
        drop_prefixes = ("fec_raised_", "fec_spent_", "fec_outside_", "fec_n_cand_", "fec_candidates_", "fec_raised_o_")
        keep = [c for c in df.columns if not any(c.startswith(p) for p in drop_prefixes)]
        df = df[keep].merge(finance, on="district_id", how="left")
        out_path = RAW_DIR / "fec_outside.parquet"
        if out_path.exists():
            df = df.merge(pd.read_parquet(out_path), on="district_id", how="left")
        df["data_source_flags"] = df["data_source_flags"].astype(str) + "|openfec_cache"
        print("Merged cached OpenFEC aggregates")

    df.to_parquet(MASTER_PARQUET, index=False)
    print(f"Wrote {len(df)} rows → {MASTER_PARQUET}")
    print(df["party_control"].value_counts().to_string())


if __name__ == "__main__":
    main()
