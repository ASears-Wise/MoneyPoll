"""Load master district table and GeoJSON; mock fallback."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import GEOJSON_PATH, MASTER_PARQUET, MOCK_MASTER_PARQUET, ROOT


def ensure_mock_data() -> Path:
    """Generate mock master if neither master nor mock exists."""
    if MASTER_PARQUET.exists():
        return MASTER_PARQUET
    if MOCK_MASTER_PARQUET.exists():
        return MOCK_MASTER_PARQUET
    script = ROOT / "scripts" / "generate_mock_data.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(ROOT))
    if MASTER_PARQUET.exists():
        return MASTER_PARQUET
    return MOCK_MASTER_PARQUET


def load_master(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """
    Load master district dataframe.

    Returns (df, source_label).
    """
    if path is not None:
        p = path
        label = str(p)
    elif MASTER_PARQUET.exists():
        p = MASTER_PARQUET
        label = "data/master.parquet"
    elif MOCK_MASTER_PARQUET.exists():
        p = MOCK_MASTER_PARQUET
        label = "data/mock/master.parquet (mock)"
    else:
        p = ensure_mock_data()
        label = f"{p.relative_to(ROOT)} (auto-generated mock)"

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)

    # Normalize dtypes
    for col in ("incumbent_running", "is_open_seat"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    if "pvi" in df.columns:
        df["pvi"] = pd.to_numeric(df["pvi"], errors="coerce")

    # Detect mock
    if "data_source_flags" in df.columns and df["data_source_flags"].astype(str).str.contains("mock").any():
        if "mock" not in label:
            label = label + " [contains mock flags]"

    return df, label


def load_geojson(path: Path | None = None) -> dict[str, Any] | None:
    """Load congressional district GeoJSON if present and non-empty."""
    p = path or GEOJSON_PATH
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if len(data.get("features") or []) < 50:
        return None
    return data


_STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
_AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}


def normalize_geo_district_id(props: dict[str, Any]) -> str | None:
    """
    Best-effort extract district_id (e.g. CA-12) from common GeoJSON property names.
    """
    for k in ("district_id", "DISTRICT_ID", "district_id_alt"):
        if k in props and props[k]:
            v = str(props[k]).upper()
            if "-" in v and len(v) <= 8:
                return v

    # Jeffrey B. Lewis / UCLA CDMaps: statename + district
    sn = props.get("statename") or props.get("STATENAME")
    if sn is not None and props.get("district") is not None:
        abbr = _STATE_NAME_TO_ABBR.get(str(sn).title())
        if abbr:
            try:
                n = int(float(props["district"]))
                if abbr in _AT_LARGE:
                    return f"{abbr}-00"
                return f"{abbr}-{n:02d}"
            except (TypeError, ValueError):
                pass

    state = props.get("state") or props.get("STATE") or props.get("STUSPS") or props.get("STATE_ABBR")
    cd = (
        props.get("district")
        or props.get("CD")
        or props.get("CD119FP")
        or props.get("CD118FP")
        or props.get("CD116FP")
        or props.get("DISTRICT")
        or props.get("District")
    )
    if state and cd is not None:
        st = str(state).upper()
        if len(st) > 2:
            st = _STATE_NAME_TO_ABBR.get(str(state).title(), st)
        try:
            n = int(float(str(cd)))
            if st in _AT_LARGE:
                return f"{st}-00"
            return f"{st}-{n:02d}"
        except ValueError:
            return f"{st}-{str(cd)}"

    return None


def list_states(df: pd.DataFrame) -> list[str]:
    return sorted(df["state"].dropna().unique().tolist())
