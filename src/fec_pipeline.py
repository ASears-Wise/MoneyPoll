"""
Merge OpenFEC aggregates into the master district table.

Used by scripts/fetch_fec_data.py and the Streamlit cloud-side refresh path.
"""
from __future__ import annotations

import json
from typing import Sequence

import pandas as pd

from config import DATA_DIR, MASTER_PARQUET, MOCK_MASTER_PARQUET, RAW_DIR
from src.data_loader import load_master
from src.fec_client import build_cycle_frames, get_api_key


_DROP_PREFIXES = (
    "fec_raised_",
    "fec_spent_",
    "fec_outside_",
    "fec_n_cand_",
    "fec_candidates_",
    "fec_raised_o_",
)


def _base_master() -> pd.DataFrame:
    if MASTER_PARQUET.exists() or MOCK_MASTER_PARQUET.exists():
        df, _ = load_master()
        return df
    # Generate mock skeleton
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from generate_mock_data import generate

    return generate()


def merge_finance_into_master(
    master: pd.DataFrame,
    finance: pd.DataFrame,
    outside: pd.DataFrame | None,
    cycles: Sequence[int],
) -> pd.DataFrame:
    """Drop prior FEC columns and left-join new aggregates; recompute hist cost."""
    keep = [c for c in master.columns if not any(c.startswith(p) for p in _DROP_PREFIXES)]
    merged = master[keep].merge(finance, on="district_id", how="left")
    merged = merged.drop(
        columns=[c for c in merged.columns if c == "cycle" or str(c).startswith("cycle_")],
        errors="ignore",
    )

    if outside is not None and len(outside):
        oc = [c for c in outside.columns if c != "district_id"]
        merged = merged.drop(columns=[c for c in oc if c in merged.columns], errors="ignore")
        merged = merged.merge(outside, on="district_id", how="left")

    for cy in cycles:
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
        merged["fec_outside_2024"] = pd.to_numeric(
            merged["fec_outside_2024"], errors="coerce"
        ).fillna(0.0)

    def _hist(row: pd.Series) -> float:
        for cy in (2024, 2022, 2026):
            r = float(row.get(f"fec_raised_r_{cy}") or 0)
            d = float(row.get(f"fec_raised_d_{cy}") or 0)
            if r + d > 0:
                return (r + d) * 0.35
        return float(row.get("hist_cost_to_compete") or 500_000)

    merged["hist_cost_to_compete"] = merged.apply(_hist, axis=1)

    if "fec_candidates_2026" in merged.columns:

        def _cj(row: pd.Series) -> str:
            raw = row.get("fec_candidates_2026")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                existing = row.get("civic_candidates_json")
                return existing if isinstance(existing, str) else "{}"
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
    return merged


def fetch_and_merge_fec(
    *,
    cycles: Sequence[int] | None = None,
    include_outside: bool = False,
    outside_cycle: int = 2024,
    progress: bool = False,
    persist: bool = True,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Pull OpenFEC House totals (and optional IE), merge into base master.

    Returns (df, source_label, stats).
    """
    if not get_api_key():
        raise RuntimeError("FEC_API_KEY not set (env or Streamlit secrets).")

    cycles = list(cycles or [2022, 2024, 2026])
    finance, outside = build_cycle_frames(
        cycles=cycles,
        fetch_outside=include_outside,
        outside_cycle=outside_cycle,
        progress=progress,
    )

    if persist:
        try:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            finance.to_parquet(RAW_DIR / "fec_district_finance.parquet", index=False)
            if outside is not None and len(outside):
                outside.to_parquet(RAW_DIR / "fec_outside.parquet", index=False)
        except OSError:
            pass

    master = _base_master()
    # If master already has openfec, still re-merge on top of non-FEC columns
    merged = merge_finance_into_master(master, finance, outside, cycles)

    if persist:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(MASTER_PARQUET, index=False)
        except OSError:
            pass

    stats: dict = {"cycles": cycles, "n_districts": len(merged), "include_outside": include_outside}
    for cy in cycles:
        col = f"fec_raised_r_{cy}"
        if col in merged.columns:
            n = int(
                (
                    merged[col].fillna(0) + merged[f"fec_raised_d_{cy}"].fillna(0) > 0
                ).sum()
            )
            tot = float(
                merged[col].fillna(0).sum() + merged[f"fec_raised_d_{cy}"].fillna(0).sum()
            )
            stats[f"raised_{cy}"] = tot
            stats[f"districts_with_receipts_{cy}"] = n
    if "fec_outside_2024" in merged.columns:
        stats["outside_2024"] = float(merged["fec_outside_2024"].fillna(0).sum())
        stats["districts_with_outside"] = int((merged["fec_outside_2024"].fillna(0) > 0).sum())

    label = "OpenFEC live merge" + (" + IE" if include_outside else " (candidate totals)")
    return merged, label, stats
