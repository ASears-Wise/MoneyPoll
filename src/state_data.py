"""Load state legislative masters (lower/upper) with mock fallback."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config import (
    ROOT,
    STATE_LOWER_PARQUET,
    STATE_MOCK_LOWER,
    STATE_MOCK_UPPER,
    STATE_RAW_DIR,
    STATE_UPPER_PARQUET,
)

Chamber = Literal["lower", "upper"]


def _paths(chamber: Chamber) -> tuple[Path, Path]:
    if chamber == "lower":
        return STATE_LOWER_PARQUET, STATE_MOCK_LOWER
    return STATE_UPPER_PARQUET, STATE_MOCK_UPPER


def ensure_state_mock() -> None:
    live_l, mock_l = _paths("lower")
    live_u, mock_u = _paths("upper")
    if mock_l.exists() and mock_u.exists():
        return
    script = ROOT / "scripts" / "generate_state_mock_data.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(ROOT))


def load_state_master(chamber: Chamber) -> tuple[pd.DataFrame, str]:
    ensure_state_mock()
    live, mock = _paths(chamber)
    if live.exists():
        p, label = live, f"data/state/{chamber}_master.parquet"
    elif mock.exists():
        p, label = mock, f"data/state/mock/{chamber}_master.parquet (mock)"
    else:
        ensure_state_mock()
        p, label = mock, f"data/state/mock/{chamber}_master.parquet (auto)"

    df = pd.read_parquet(p) if p.suffix != ".csv" else pd.read_csv(p)
    for col in ("incumbent_running", "is_open_seat"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    if "pvi" in df.columns:
        df["pvi"] = pd.to_numeric(df["pvi"], errors="coerce")

    if "data_source_flags" in df.columns:
        if df["data_source_flags"].astype(str).str.contains("mock").any() and "mock" not in label:
            label += " [contains mock flags]"
    return df, label


def load_state_geojson(chamber: Chamber, state: str | None = None) -> dict[str, Any] | None:
    """
    Optional GeoJSON under data/state/raw/.
    Prefer state-specific file: {ST}_{lower|upper}.geojson
    or national sldl.geojson / sldu.geojson
    """
    STATE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    if state:
        st = state.upper()
        candidates.append(STATE_RAW_DIR / f"{st}_{chamber}.geojson")
    tag = "sldl" if chamber == "lower" else "sldu"
    candidates.append(STATE_RAW_DIR / f"{tag}.geojson")
    candidates.append(STATE_RAW_DIR / f"{chamber}.geojson")

    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        feats = data.get("features") or []
        if len(feats) < 1:
            continue
        if state:
            st = state.upper()
            filtered = []
            for f in feats:
                props = f.get("properties") or {}
                pst = (
                    props.get("state")
                    or props.get("STUSPS")
                    or props.get("STATE")
                    or ""
                )
                did = str(props.get("district_id") or "")
                if str(pst).upper() == st or did.startswith(f"{st}-"):
                    filtered.append(f)
            if filtered:
                return {"type": "FeatureCollection", "features": filtered}
            # if file was already state-specific
            if f"_{chamber}" in p.name or p.name.startswith(st):
                return data
        else:
            return data
    return None


def list_states(df: pd.DataFrame) -> list[str]:
    return sorted(df["state"].dropna().unique().tolist())


def parse_state_candidates(row: pd.Series) -> list[dict[str, Any]]:
    raw = row.get("candidates_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return list(data.get("candidates") or [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []
