"""
Chamber-level RIVS: apply federal RIVS within each (state, chamber) majority context.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import RIVS_DEFAULTS
from src.rivs import budget_simulation, compute_rivs


def compute_state_rivs(
    df: pd.DataFrame,
    **rivs_kwargs: Any,
) -> pd.DataFrame:
    """
    Score seats for path to **that state's chamber majority**.

    Uses each row's chamber_majority_threshold when present; otherwise
    floor(n/2)+1 for seats in the same state.
    """
    if df.empty:
        return df.copy()

    pieces = []
    for state, g in df.groupby("state", sort=False):
        g = g.copy()
        if "chamber_majority_threshold" in g.columns and g["chamber_majority_threshold"].notna().any():
            target = int(g["chamber_majority_threshold"].dropna().iloc[0])
        else:
            n = len(g)
            target = n // 2 + 1
        # Lower absolute abandon floors for state races (cheaper than federal)
        kwargs = {
            "abandon_min_cost_m": rivs_kwargs.get(
                "abandon_min_cost_m",
                min(float(RIVS_DEFAULTS["abandon_min_cost_m"]), 1.5),
            ),
            **{k: v for k, v in rivs_kwargs.items() if k != "abandon_min_cost_m"},
            "target_seats": target,
        }
        # State finance columns → hist already set; map state_* to fec_* names if rivs expects outside
        if "fec_outside_2024" not in g.columns:
            g["fec_outside_2024"] = 0.0
        scored = compute_rivs(g, **kwargs)
        scored["chamber_target_seats"] = target
        scored["state_r_held"] = int((g["party_control"] == "R").sum())
        scored["seats_to_majority"] = max(0, target - scored["state_r_held"].iloc[0])
        pieces.append(scored)

    out = pd.concat(pieces, ignore_index=True)
    # Global rank for national prioritization across selected states
    out["rivs_rank"] = out["rivs"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("rivs", ascending=False).reset_index(drop=True)
    return out


def chamber_majority_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    """One row per state: R held, target, seats short, mean RIVS of attack seats."""
    if ranked.empty:
        return pd.DataFrame()
    rows = []
    for state, g in ranked.groupby("state"):
        target = int(g["chamber_target_seats"].iloc[0]) if "chamber_target_seats" in g.columns else len(g) // 2 + 1
        r_held = int((g["party_control"] == "R").sum())
        exp = float(g["baseline_win_prob_r"].sum())
        rows.append(
            {
                "state": state,
                "r_held": r_held,
                "chamber_size": len(g),
                "majority": target,
                "seats_short": max(0, target - r_held),
                "r_controls": r_held >= target,
                "expected_r_seats": exp,
                "n_attack": int((g["mode"] == "attack").sum()),
                "n_defend": int((g["mode"] == "defend").sum()),
                "n_abandon": int((g["mode"] == "abandon").sum()),
                "top_rivs": float(g["rivs"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["seats_short", "top_rivs"], ascending=[True, False])


def state_budget_simulation(
    ranked: pd.DataFrame,
    budget_usd: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Greedy RIVS budget over filtered state seats; report chambers helped."""
    sim = budget_simulation(ranked, budget_usd, **kwargs)
    # Chambers that receive any funded district
    funded_states = {f["district_id"].split("-")[0] for f in sim.get("funded", [])}
    sim["states_funded"] = sorted(funded_states)
    sim["n_states_funded"] = len(funded_states)
    return sim
