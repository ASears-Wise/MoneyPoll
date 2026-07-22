"""
Display number formatting for tables, metrics, and charts.

Rules:
- Integers and whole-number floats: 1,234,567 (no decimals)
- Money: $1,234,567 (no cents unless fractional dollars needed)
- Percents / small fractions: keep needed decimals only
- Probabilities 0–1: show as percent with 1 decimal if needed
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

# Columns treated as USD (substring match, case-insensitive)
_MONEY_EXACT = {
    "incremental_cost",
    "hist_cost_to_compete",
    "cost_to_be_competitive",
    "cost_per_gain",
    "median_income",
    "spend",
    "fec_outside_2024",
    "fec_outside_2026",
}
_MONEY_PREFIXES = (
    "fec_raised_",
    "fec_spent_",
    "fec_outside_",
    "state_raised_",
    "state_spent_",
)

# Columns that are probabilities in 0–1 range → percent display
_PCT_COLS = {
    "baseline_win_prob_r",
    "expected_prob_gain",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_asian",
    "pct_ba_plus",
    "pct_urban",
    "frac",
}

# Keep limited decimals (scores / factors)
_DECIMAL_COLS = {
    "rivs": 2,
    "pvi": 1,
    "seat_priority": 2,
    "long_term_factor": 2,
    "top_rivs": 2,
    "expected_r_seats": 1,
    "total_prob_gain": 2,
}

# Integer-ish counts / ranks / years
_INT_COLS = {
    "rivs_rank",
    "pop_total",
    "vap",
    "district_num",
    "tenure_years",
    "first_elected",
    "chamber_total_seats",
    "chamber_majority_threshold",
    "chamber_target_seats",
    "state_r_held",
    "seats_to_majority",
    "r_held",
    "chamber_size",
    "majority",
    "seats_short",
    "n_attack",
    "n_defend",
    "n_abandon",
    "receipts",
}


def is_money_col(name: str) -> bool:
    n = name.lower()
    if n in _MONEY_EXACT or name in _MONEY_EXACT:
        return True
    return any(n.startswith(p) or name.startswith(p) for p in _MONEY_PREFIXES)


def fmt_int(x: Any) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return str(x)


def fmt_money(x: Any, *, whole_dollars: bool = True) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if whole_dollars or abs(v - round(v)) < 0.005:
        return f"${int(round(v)):,}"
    return f"${v:,.2f}"


def fmt_pct(x: Any, *, decimals: int = 1, already_percent: bool = False) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not already_percent and abs(v) <= 1.5:
        v = v * 100.0
    if decimals <= 0 or abs(v - round(v)) < 10 ** (-decimals) / 2:
        return f"{int(round(v))}%"
    return f"{v:.{decimals}f}%"


def fmt_num(x: Any, *, decimals: int | None = None) -> str:
    """General number: commas; decimals only if needed or forced."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if decimals is not None:
        if decimals <= 0:
            return f"{int(round(v)):,}"
        s = f"{v:,.{decimals}f}"
        # strip trailing zeros after decimal
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    # up to 2 decimals, strip trailing zeros
    s = f"{v:,.2f}".rstrip("0").rstrip(".")
    return s


def fmt_pvi(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    # one decimal only if needed
    if abs(v - round(v)) < 0.05:
        return f"{int(round(v)):+d}"
    return f"{v:+.1f}"


def format_cell(col: str, val: Any) -> Any:
    """Format a single cell for display tables."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (bool, np.bool_)):
        return "Yes" if val else "No"
    if isinstance(val, str):
        return val

    if is_money_col(col):
        return fmt_money(val)
    if col in _PCT_COLS or col.startswith("pct_"):
        # expected_prob_gain is small probability points, not always %
        if col == "expected_prob_gain":
            return fmt_num(val, decimals=3 if abs(float(val)) < 0.01 else 2)
        return fmt_pct(val)
    if col == "pvi":
        return fmt_pvi(val)
    if col in _DECIMAL_COLS:
        return fmt_num(val, decimals=_DECIMAL_COLS[col])
    if col in _INT_COLS or col.endswith("_rank") or col.endswith("_seats"):
        return fmt_int(val)
    if isinstance(val, (int, np.integer)):
        return fmt_int(val)
    if isinstance(val, (float, np.floating)):
        return fmt_num(val)
    return val


def format_display_frame(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a copy with human-readable number/$ formatting for Streamlit tables."""
    if df is None or df.empty:
        return df
    cols = [c for c in (columns or df.columns) if c in df.columns]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = [format_cell(c, v) for v in df[c].tolist()]
    return out


# Column help text for table headers (shown via caption / docs)
COLUMN_HELP: dict[str, str] = {
    "rivs_rank": "Rank by RIVS (1 = highest investment value)",
    "district_id": "District identifier",
    "seat_label": "Display label for the seat",
    "state": "Two-letter state code",
    "party_control": "Current party holding the seat (R/D)",
    "mode": "attack / defend / abandon / safe",
    "base_mode": "Mode before abandon overlay",
    "rivs": "Republican Investment Value Score — higher is better ROI",
    "pvi": "Lean score (R positive). Synthetic or PVI-like, not official Cook unless supplied",
    "cook_rating": "Competitiveness label derived from lean",
    "baseline_win_prob_r": "Modeled Republican win probability before new spending",
    "expected_prob_gain": "Expected gain in win probability from funding to threshold",
    "incremental_cost": "Modeled $ to fund expected win-probability gain (RIVS cost)",
    "cost_to_be_competitive": "Cost to be competitive — FEC/historical spend proxy for a viable race",
    "cost_per_gain": "Incremental cost divided by probability gain ($ per ΔP)",
    "seat_priority": "Strategic weight (marginality × path to majority × mode)",
    "rep_name": "Sitting member or OPEN SEAT",
    "is_open_seat": "Whether the seat is open (no incumbent running)",
    "hist_cost_to_compete": "Same as cost to be competitive (raw column name)",
    "fec_raised_r_2026": "Republican raised (2026 cycle)",
    "fec_raised_d_2026": "Democratic raised (2026 cycle)",
    "state_raised_r_2026": "Republican raised proxy (2026)",
    "state_raised_d_2026": "Democratic raised proxy (2026)",
    "abandon_reason": "Why this seat was flagged abandon",
    "median_income": "Median household income (ACS-style)",
    "pop_total": "Total population",
    "vap": "Voting-age population",
    "seats_to_majority": "Seats short of chamber majority for R in this state",
    "chamber_majority_threshold": "Seats needed for majority in this chamber",
    "r_held": "Republican-held seats in chamber",
    "chamber_size": "Total seats in the chamber",
    "majority": "Majority threshold",
    "seats_short": "How many seats R needs to reach majority",
    "r_controls": "Whether R currently holds majority",
    "expected_r_seats": "Sum of baseline R win probabilities in chamber",
    "n_attack": "Count of attack-mode seats",
    "n_defend": "Count of defend-mode seats",
    "n_abandon": "Count of abandon-mode seats",
    "top_rivs": "Highest RIVS in that state chamber",
    "spend": "Budget-sim dollars allocated",
    "gain": "Probability gain from budget allocation",
    "frac": "Fraction of full incremental cost funded",
}
