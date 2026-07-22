"""Shared UI helpers: presets, cached RIVS, data status, slim columns."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import DATA_DIR, RIVS_DEFAULTS, ROOT, STATE_DATA_DIR
from src.rivs import compute_rivs
from src.state_rivs import compute_state_rivs

META_PATH = DATA_DIR / "refresh_meta.json"
STATE_META_PATH = STATE_DATA_DIR / "refresh_meta.json"

# Default visible table columns (fast scan)
FEDERAL_SLIM_COLS = [
    "rivs_rank",
    "district_id",
    "state",
    "party_control",
    "mode",
    "rivs",
    "incremental_cost",
    "rep_name",
]
FEDERAL_EXTRA_COLS = [
    "base_mode",
    "pvi",
    "cook_rating",
    "baseline_win_prob_r",
    "expected_prob_gain",
    "cost_per_gain",
    "seat_priority",
    "is_open_seat",
    "hist_cost_to_compete",
    "abandon_reason",
    "median_income",
    "pop_total",
]
STATE_SLIM_COLS = [
    "rivs_rank",
    "district_id",
    "seat_label",
    "state",
    "party_control",
    "mode",
    "rivs",
    "incremental_cost",
    "rep_name",
]
STATE_EXTRA_COLS = [
    "pvi",
    "baseline_win_prob_r",
    "expected_prob_gain",
    "seats_to_majority",
    "hist_cost_to_compete",
    "abandon_reason",
]

PRESETS: dict[str, dict[str, Any]] = {
    "Balanced (default)": {
        "modes": ["attack", "defend", "abandon", "safe"],
        "w_attack": 1.2,
        "w_defend": 1.0,
        "cost_sensitivity": 1.0,
        "risk_tolerance": 0.52,
        "abandon_enabled": True,
        "description": "Standard attack/defend balance with abandon on.",
    },
    "Flips only": {
        "modes": ["attack"],
        "w_attack": 1.5,
        "w_defend": 0.5,
        "cost_sensitivity": 1.0,
        "risk_tolerance": 0.52,
        "abandon_enabled": True,
        "description": "Show only flip targets; weight attack higher.",
    },
    "Protect majority": {
        "modes": ["defend", "abandon"],
        "w_attack": 0.6,
        "w_defend": 1.5,
        "cost_sensitivity": 1.0,
        "risk_tolerance": 0.55,
        "abandon_enabled": True,
        "description": "Focus on holds; slightly higher win-prob threshold.",
    },
    "Hide abandon": {
        "modes": ["attack", "defend", "safe"],
        "w_attack": 1.2,
        "w_defend": 1.0,
        "cost_sensitivity": 1.0,
        "risk_tolerance": 0.52,
        "abandon_enabled": False,
        "description": "No abandon flagging; show competitive + safe.",
    },
    "Top ROI (strict cost)": {
        "modes": ["attack", "defend"],
        "w_attack": 1.3,
        "w_defend": 1.1,
        "cost_sensitivity": 1.8,
        "risk_tolerance": 0.52,
        "abandon_enabled": True,
        "description": "Penalize expensive races; attack/defend only.",
    },
}

# Swing-ish pack for state default filter (optional quick filter)
PRIORITY_STATES = ["AZ", "GA", "MI", "NC", "NV", "PA", "WI", "OH", "FL", "TX"]


def data_status_badge(label: str, df: pd.DataFrame | None = None) -> str:
    """Human-readable data provenance strip."""
    parts = [f"Source: `{label}`"]
    if df is not None and "data_source_flags" in df.columns:
        flags = "|".join(sorted({str(x) for x in df["data_source_flags"].dropna().unique()}))
        if "openfec" in flags:
            parts.append("FEC: live/snapshot")
        elif "mock" in flags:
            parts.append("FEC: synthetic")
        if "openstates" in flags:
            parts.append("Members: OpenStates")
        elif "mock" in flags:
            parts.append("Lean/members: synthetic")
    meta = read_refresh_meta()
    if meta.get("federal_refreshed_at"):
        parts.append(f"Federal refresh: {meta['federal_refreshed_at']}")
    if meta.get("state_refreshed_at"):
        parts.append(f"State refresh: {meta['state_refreshed_at']}")
    return " · ".join(parts)


def read_refresh_meta() -> dict[str, Any]:
    import json

    out: dict[str, Any] = {}
    for path in (META_PATH, STATE_META_PATH):
        if path.exists():
            try:
                out.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    return out


def write_refresh_meta(updates: dict[str, Any], *, state: bool = False) -> None:
    import json

    path = STATE_META_PATH if state else META_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cur = {}
    cur.update(updates)
    cur["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path.write_text(json.dumps(cur, indent=2), encoding="utf-8")


@st.cache_data(show_spinner=False)
def cached_federal_rivs(
    df: pd.DataFrame,
    cost_sensitivity: float,
    long_term_multiplier: float,
    risk_tolerance: float,
    w_attack: float,
    w_defend: float,
    incumbent_bonus: float,
    open_seat_volatility: float,
    target_seats: int,
    abandon_enabled: bool,
    abandon_cost_percentile: float,
    abandon_min_cost_m: float,
    abandon_alt_races: int,
    abandon_alt_gain_ratio: float,
) -> pd.DataFrame:
    return compute_rivs(
        df,
        cost_sensitivity=cost_sensitivity,
        long_term_multiplier=long_term_multiplier,
        risk_tolerance=risk_tolerance,
        w_attack=w_attack,
        w_defend=w_defend,
        incumbent_bonus=incumbent_bonus,
        open_seat_volatility=open_seat_volatility,
        target_seats=target_seats,
        abandon_enabled=abandon_enabled,
        abandon_cost_percentile=abandon_cost_percentile,
        abandon_min_cost_m=abandon_min_cost_m,
        abandon_alt_races=abandon_alt_races,
        abandon_alt_gain_ratio=abandon_alt_gain_ratio,
    )


@st.cache_data(show_spinner=False)
def cached_state_rivs(
    df: pd.DataFrame,
    cost_sensitivity: float,
    long_term_multiplier: float,
    risk_tolerance: float,
    w_attack: float,
    w_defend: float,
    incumbent_bonus: float,
    open_seat_volatility: float,
    abandon_enabled: bool,
    abandon_cost_percentile: float,
    abandon_min_cost_m: float,
    abandon_alt_races: int,
    abandon_alt_gain_ratio: float,
) -> pd.DataFrame:
    return compute_state_rivs(
        df,
        cost_sensitivity=cost_sensitivity,
        long_term_multiplier=long_term_multiplier,
        risk_tolerance=risk_tolerance,
        w_attack=w_attack,
        w_defend=w_defend,
        incumbent_bonus=incumbent_bonus,
        open_seat_volatility=open_seat_volatility,
        abandon_enabled=abandon_enabled,
        abandon_cost_percentile=abandon_cost_percentile,
        abandon_min_cost_m=abandon_min_cost_m,
        abandon_alt_races=abandon_alt_races,
        abandon_alt_gain_ratio=abandon_alt_gain_ratio,
    )


def apply_preset_to_session(prefix: str, preset_name: str) -> None:
    p = PRESETS.get(preset_name) or PRESETS["Balanced (default)"]
    st.session_state[f"{prefix}_preset_modes"] = list(p["modes"])
    st.session_state[f"{prefix}_w_attack"] = p["w_attack"]
    st.session_state[f"{prefix}_w_defend"] = p["w_defend"]
    st.session_state[f"{prefix}_cost_sensitivity"] = p["cost_sensitivity"]
    st.session_state[f"{prefix}_risk"] = p["risk_tolerance"]
    st.session_state[f"{prefix}_abandon_enabled"] = p["abandon_enabled"]


def rivs_kwargs_from_form(values: dict[str, Any]) -> dict[str, Any]:
    d = RIVS_DEFAULTS
    return {
        "cost_sensitivity": values.get("cost_sensitivity", d["cost_sensitivity"]),
        "long_term_multiplier": values.get("long_term_multiplier", d["long_term_multiplier"]),
        "risk_tolerance": values.get("risk_tolerance", d["risk_tolerance"]),
        "w_attack": values.get("w_attack", d["w_attack"]),
        "w_defend": values.get("w_defend", d["w_defend"]),
        "incumbent_bonus": values.get("incumbent_bonus", d["incumbent_bonus"]),
        "open_seat_volatility": values.get("open_seat_volatility", d["open_seat_volatility"]),
        "target_seats": values.get("target_seats", 230),
        "abandon_enabled": values.get("abandon_enabled", d["abandon_enabled"]),
        "abandon_cost_percentile": values.get(
            "abandon_cost_percentile", d["abandon_cost_percentile"]
        ),
        "abandon_min_cost_m": values.get("abandon_min_cost_m", d["abandon_min_cost_m"]),
        "abandon_alt_races": values.get("abandon_alt_races", d["abandon_alt_races"]),
        "abandon_alt_gain_ratio": values.get(
            "abandon_alt_gain_ratio", d["abandon_alt_gain_ratio"]
        ),
    }
