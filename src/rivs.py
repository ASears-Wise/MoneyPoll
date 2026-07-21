"""
Republican Investment Value Score (RIVS)

RIVS = (Expected_Probability_Gain × Seat_Priority) / Incremental_Cost_to_Reach_Threshold

All inputs are tunable; see config.RIVS_DEFAULTS.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_TARGET_SEATS, RIVS_DEFAULTS, TOTAL_HOUSE_SEATS


def _logistic_from_pvi(pvi: np.ndarray, scale: float = 8.0) -> np.ndarray:
    """Map Cook-style signed PVI (R positive) to R win probability prior."""
    return 1.0 / (1.0 + np.exp(-pvi / scale))


def baseline_win_prob_r(
    df: pd.DataFrame,
    incumbent_bonus: float = RIVS_DEFAULTS["incumbent_bonus"],
    open_seat_volatility: float = RIVS_DEFAULTS["open_seat_volatility"],
) -> pd.Series:
    """
    Baseline R win probability from PVI + incumbent / open-seat factors.

    - R incumbent running: +incumbent_bonus
    - D incumbent running: -incumbent_bonus
    - Open seat: pull toward 0.5 by open_seat_volatility amount
    """
    pvi = df["pvi"].astype(float).to_numpy()
    base = _logistic_from_pvi(pvi)

    party = df["party_control"].astype(str).str.upper()
    incumbent_running = df["incumbent_running"].astype(bool)
    is_open = df["is_open_seat"].astype(bool)

    adj = np.zeros(len(df), dtype=float)
    r_inc = (party == "R") & incumbent_running & ~is_open
    d_inc = (party == "D") & incumbent_running & ~is_open
    adj[r_inc.to_numpy()] += incumbent_bonus
    adj[d_inc.to_numpy()] -= incumbent_bonus

    # Open seats: blend toward 0.5
    open_mask = is_open.to_numpy()
    base_open = base.copy()
    base_open[open_mask] = (1.0 - open_seat_volatility) * base[open_mask] + open_seat_volatility * 0.5
    base = np.where(open_mask, base_open, base)
    base = np.clip(base + adj, 0.02, 0.98)
    return pd.Series(base, index=df.index, name="baseline_win_prob_r")


def expected_probability_gain(
    p0: np.ndarray,
    p_star: float,
    mode: np.ndarray,
) -> np.ndarray:
    """
    Probability gain from spending toward threshold.

    Attack (D-held / flip): gain = max(0, P* - P0)
    Defend (R-held): gain from fortifying toward max(P*, P0 + buffer)
    Safe deep seats: near-zero gain
    """
    gain = np.zeros_like(p0, dtype=float)
    for i, m in enumerate(mode):
        m = str(m).lower()
        if m == "attack":
            gain[i] = max(0.0, p_star - p0[i])
        elif m == "defend":
            # Value of pushing further above waterline / back to P*
            target = max(p_star, min(0.72, p0[i] + 0.08))
            gain[i] = max(0.0, target - p0[i])
            # Also value if currently underwater
            if p0[i] < p_star:
                gain[i] = max(gain[i], p_star - p0[i])
        else:  # safe
            gain[i] = max(0.0, 0.02 * (1.0 - abs(p0[i] - 0.5)))
    return gain


def seat_priority(
    df: pd.DataFrame,
    p0: np.ndarray,
    gain: np.ndarray,
    w_attack: float,
    w_defend: float,
    target_seats: int = DEFAULT_TARGET_SEATS,
) -> np.ndarray:
    """
    Higher priority for marginal seats and path to target majority.
    Mode weights scale attack vs defend.
    """
    mode = df["mode"].astype(str).str.lower().to_numpy()
    n = len(df)

    # Competitiveness: seats near 50% matter more
    marginal = 1.0 - 2.0 * np.abs(p0 - 0.5)
    marginal = np.clip(marginal, 0.05, 1.0)

    # Path-to-majority: estimate seats R already "likely holds"
    likely_r = float(np.sum(p0 >= 0.5))
    seats_needed = max(0, target_seats - likely_r)
    urgency = 1.0 + min(1.5, seats_needed / 30.0)

    mode_w = np.ones(n)
    mode_w[mode == "attack"] = w_attack
    mode_w[mode == "defend"] = w_defend
    mode_w[mode == "safe"] = 0.15

    # Prefer districts with meaningful gain opportunities
    gain_factor = 0.3 + 0.7 * (gain / (gain.max() + 1e-9))

    return marginal * urgency * mode_w * gain_factor


def incremental_cost(
    df: pd.DataFrame,
    gain: np.ndarray,
    cost_sensitivity: float,
) -> np.ndarray:
    """
    $ cost to realize probability gain, from historical FEC competitive spend proxy.
    """
    hist = df["hist_cost_to_compete"].astype(float).to_numpy()
    # More gain / harder environment → higher cost
    pvi = np.abs(df["pvi"].astype(float).to_numpy())
    hardness = 1.0 + pvi / 20.0
    # Floor hist so low-activity districts (real FEC safe seats) don't dominate RIVS
    hist = np.maximum(hist, 250_000.0)
    cost = hist * hardness * cost_sensitivity * (0.4 + 1.6 * gain)
    # Floor to avoid explode / divide-by-zero
    return np.maximum(cost, 250_000.0)


def long_term_bonus(
    df: pd.DataFrame,
    long_term_multiplier: float,
) -> np.ndarray:
    """Bonus factor for open seats / building infrastructure."""
    is_open = df["is_open_seat"].astype(bool).to_numpy()
    bonus = np.ones(len(df))
    bonus[is_open] *= long_term_multiplier
    # Slight bonus for younger tenure opposition (fragile holds)
    if "tenure_years" in df.columns:
        tenure = df["tenure_years"].fillna(10).to_numpy()
        party = df["party_control"].astype(str).str.upper().to_numpy()
        fragile_d = (party == "D") & (tenure < 4)
        bonus[fragile_d] *= 1.0 + 0.15 * (long_term_multiplier - 1.0)
    return bonus


def derive_mode(df: pd.DataFrame) -> pd.Series:
    """Classify attack / defend / safe from control + PVI."""
    modes = []
    for _, row in df.iterrows():
        party = str(row["party_control"]).upper()
        pvi = float(row["pvi"])
        if party == "D":
            modes.append("attack" if pvi > -12 else "safe")
        elif party == "R":
            modes.append("defend" if pvi < 12 else "safe")
        else:
            modes.append("attack")
    return pd.Series(modes, index=df.index, name="mode")


def compute_rivs(
    df: pd.DataFrame,
    *,
    cost_sensitivity: float = RIVS_DEFAULTS["cost_sensitivity"],
    long_term_multiplier: float = RIVS_DEFAULTS["long_term_multiplier"],
    risk_tolerance: float = RIVS_DEFAULTS["risk_tolerance"],
    w_attack: float = RIVS_DEFAULTS["w_attack"],
    w_defend: float = RIVS_DEFAULTS["w_defend"],
    incumbent_bonus: float = RIVS_DEFAULTS["incumbent_bonus"],
    open_seat_volatility: float = RIVS_DEFAULTS["open_seat_volatility"],
    target_seats: int = DEFAULT_TARGET_SEATS,
) -> pd.DataFrame:
    """
    Return a copy of df with RIVS columns attached and ranked.
    """
    out = df.copy()
    if "mode" not in out.columns or out["mode"].isna().any():
        out["mode"] = derive_mode(out)

    p0 = baseline_win_prob_r(
        out,
        incumbent_bonus=incumbent_bonus,
        open_seat_volatility=open_seat_volatility,
    ).to_numpy()
    out["baseline_win_prob_r"] = p0

    mode = out["mode"].to_numpy()
    gain = expected_probability_gain(p0, risk_tolerance, mode)
    out["expected_prob_gain"] = gain

    cost = incremental_cost(out, gain, cost_sensitivity)
    out["incremental_cost"] = cost

    priority = seat_priority(
        out, p0, gain, w_attack, w_defend, target_seats=target_seats
    )
    out["seat_priority"] = priority

    lt = long_term_bonus(out, long_term_multiplier)
    out["long_term_factor"] = lt

    # Core formula
    rivs = (gain * priority * lt) / cost
    # Scale to a readable score range (~0–100 with mock FEC magnitudes)
    scale = 1e8
    out["rivs_raw"] = rivs
    out["rivs"] = rivs * scale

    out["rivs_rank"] = out["rivs"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("rivs", ascending=False).reset_index(drop=True)
    return out


def budget_simulation(
    ranked: pd.DataFrame,
    budget_usd: float,
) -> dict[str, Any]:
    """
    Greedy allocation to highest RIVS districts until budget exhausted.
    Expected seats ≈ baseline R seats + sum of gains for funded districts.
    """
    remaining = float(budget_usd)
    funded = []
    total_gain = 0.0
    spent = 0.0

    for _, row in ranked.iterrows():
        cost = float(row["incremental_cost"])
        if cost <= 0 or remaining < cost * 0.25:
            continue
        # Partial funding: scale gain by fraction if partial last seat
        if remaining >= cost:
            frac = 1.0
            pay = cost
        else:
            frac = remaining / cost
            pay = remaining
        remaining -= pay
        spent += pay
        g = float(row["expected_prob_gain"]) * frac
        total_gain += g
        funded.append(
            {
                "district_id": row["district_id"],
                "spend": pay,
                "frac": frac,
                "gain": g,
                "rivs": float(row["rivs"]),
                "mode": row["mode"],
            }
        )
        if remaining <= 1.0:
            break

    p0 = ranked["baseline_win_prob_r"].to_numpy()
    baseline_seats = float(np.sum(p0))
    expected_seats = baseline_seats + total_gain
    # Hard cap at 435
    expected_seats = min(float(TOTAL_HOUSE_SEATS), expected_seats)

    return {
        "budget_usd": budget_usd,
        "spent_usd": spent,
        "n_districts_funded": len(funded),
        "total_prob_gain": total_gain,
        "baseline_expected_r_seats": baseline_seats,
        "expected_r_seats_after": expected_seats,
        "funded": funded,
    }


def format_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"
