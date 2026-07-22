"""
Republican Investment Value Score (RIVS)

RIVS = (Expected_Probability_Gain × Seat_Priority) / Incremental_Cost_to_Reach_Threshold

Modes:
  attack  — D-held / flip opportunities
  defend  — R-held seats that need fortification
  safe    — deep seats, little investment edge
  abandon — competitive but so capital-intensive that $ is better spread
            across multiple other races (opportunity-cost flag)

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

    Attack: max(0, P* - P0)
    Defend: fortify toward buffer / P*
    Abandon: still compute structural gain (for diagnostics) but priority later zeros
    Safe: near-zero residual
    """
    gain = np.zeros_like(p0, dtype=float)
    for i, m in enumerate(mode):
        m = str(m).lower()
        if m in ("attack", "abandon"):
            # abandon keeps attack/defend-like gain for cost diagnostics;
            # final mode may override from provisional attack/defend
            gain[i] = max(0.0, p_star - p0[i])
        elif m == "defend":
            target = max(p_star, min(0.72, p0[i] + 0.08))
            gain[i] = max(0.0, target - p0[i])
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
    """Higher priority for marginal seats; abandon gets near-zero weight."""
    mode = df["mode"].astype(str).str.lower().to_numpy()
    n = len(df)

    marginal = 1.0 - 2.0 * np.abs(p0 - 0.5)
    marginal = np.clip(marginal, 0.05, 1.0)

    likely_r = float(np.sum(p0 >= 0.5))
    seats_needed = max(0, target_seats - likely_r)
    urgency = 1.0 + min(1.5, seats_needed / 30.0)

    mode_w = np.ones(n)
    mode_w[mode == "attack"] = w_attack
    mode_w[mode == "defend"] = w_defend
    mode_w[mode == "safe"] = 0.15
    mode_w[mode == "abandon"] = 0.02  # deprioritize hard — do not pour in more $

    gain_factor = 0.3 + 0.7 * (gain / (gain.max() + 1e-9))
    return marginal * urgency * mode_w * gain_factor


def incremental_cost(
    df: pd.DataFrame,
    gain: np.ndarray,
    cost_sensitivity: float,
) -> np.ndarray:
    """$ cost to realize probability gain, from historical FEC competitive spend proxy."""
    hist = df["hist_cost_to_compete"].astype(float).to_numpy()
    pvi = np.abs(df["pvi"].astype(float).to_numpy())
    hardness = 1.0 + pvi / 20.0
    # Include outside spend when available — money arms races inflate true cost
    outside = np.zeros(len(df), dtype=float)
    if "fec_outside_2024" in df.columns:
        outside = pd.to_numeric(df["fec_outside_2024"], errors="coerce").fillna(0).to_numpy()
    hist = hist + 0.25 * outside
    hist = np.maximum(hist, 250_000.0)
    cost = hist * hardness * cost_sensitivity * (0.4 + 1.6 * gain)
    return np.maximum(cost, 250_000.0)


def long_term_bonus(
    df: pd.DataFrame,
    long_term_multiplier: float,
) -> np.ndarray:
    """Bonus factor for open seats / building infrastructure."""
    is_open = df["is_open_seat"].astype(bool).to_numpy()
    bonus = np.ones(len(df))
    bonus[is_open] *= long_term_multiplier
    if "tenure_years" in df.columns:
        tenure = df["tenure_years"].fillna(10).to_numpy()
        party = df["party_control"].astype(str).str.upper().to_numpy()
        fragile_d = (party == "D") & (tenure < 4)
        bonus[fragile_d] *= 1.0 + 0.15 * (long_term_multiplier - 1.0)
    return bonus


def derive_base_mode(df: pd.DataFrame) -> pd.Series:
    """First-pass classify attack / defend / safe from control + PVI (no abandon yet)."""
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
    return pd.Series(modes, index=df.index, name="base_mode")


# Back-compat alias used by older callers
def derive_mode(df: pd.DataFrame) -> pd.Series:
    return derive_base_mode(df).rename("mode")


def apply_abandon_flags(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    cost_percentile: float = RIVS_DEFAULTS["abandon_cost_percentile"],
    min_cost_m: float = RIVS_DEFAULTS["abandon_min_cost_m"],
    alt_races: int = RIVS_DEFAULTS["abandon_alt_races"],
    alt_gain_ratio: float = RIVS_DEFAULTS["abandon_alt_gain_ratio"],
) -> pd.DataFrame:
    """
    Re-label attack/defend seats as **abandon** when concentration of spend is untenable.

    A competitive seat is abandoned when either:
      A) Cost-per-probability-point is above the competitive-seat percentile threshold
         AND absolute incremental cost ≥ min_cost_m, OR
      B) Opportunity cost: this seat's cost ≥ alt_races × median competitive cost
         AND alt_races × median competitive gain > this seat's gain × alt_gain_ratio

    Rationale: dollars locked into one mega-race could fund several cheaper
    attack/defend opportunities with more total expected seat gain.
    """
    out = df.copy()
    out["abandon_reason"] = ""
    out["cost_per_gain"] = out["incremental_cost"] / np.maximum(
        out["expected_prob_gain"].astype(float).to_numpy(), 1e-4
    )

    if not enabled or len(out) == 0:
        return out

    base = out["base_mode"].astype(str).str.lower()
    competitive = base.isin(["attack", "defend"])
    if not competitive.any():
        return out

    min_cost = float(min_cost_m) * 1_000_000.0
    cpg = out["cost_per_gain"].to_numpy()
    cost = out["incremental_cost"].astype(float).to_numpy()
    gain = out["expected_prob_gain"].astype(float).to_numpy()

    comp_idx = competitive.to_numpy()
    cpg_comp = cpg[comp_idx]
    cost_comp = cost[comp_idx]
    gain_comp = gain[comp_idx]

    if len(cpg_comp) == 0:
        return out

    cpg_cut = float(np.quantile(cpg_comp, cost_percentile))
    med_cost = float(np.median(cost_comp))
    med_gain = float(np.median(gain_comp)) if np.any(gain_comp > 0) else 0.01

    reasons: list[str] = [""] * len(out)
    new_mode = out["base_mode"].astype(str).str.lower().tolist()

    for i in range(len(out)):
        if not comp_idx[i]:
            continue
        if cost[i] < min_cost:
            continue

        inefficient = cpg[i] >= cpg_cut and cost[i] >= min_cost
        # Opportunity: one mega-check ≈ N median races, and N races buy more gain
        n = max(2, int(alt_races))
        opportunity = (
            cost[i] >= n * med_cost
            and (n * med_gain) > (gain[i] * float(alt_gain_ratio))
        )

        if inefficient or opportunity:
            new_mode[i] = "abandon"
            bits = []
            if inefficient:
                bits.append(
                    f"cost/gain in top {(1 - cost_percentile) * 100:.0f}% of competitive "
                    f"(${cpg[i]/1e6:.1f}M per ΔP vs cut ${cpg_cut/1e6:.1f}M)"
                )
            if opportunity:
                bits.append(
                    f"opportunity cost: seat cost {format_money(cost[i])} ≥ {n}× median "
                    f"competitive ({format_money(med_cost)}); "
                    f"{n} median races ≈ {n * med_gain:.3f} ΔP vs this seat {gain[i]:.3f} ΔP"
                )
            reasons[i] = "; ".join(bits)

    out["mode"] = new_mode
    out["abandon_reason"] = reasons
    out["is_abandon"] = out["mode"] == "abandon"
    return out


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
    abandon_enabled: bool = RIVS_DEFAULTS["abandon_enabled"],
    abandon_cost_percentile: float = RIVS_DEFAULTS["abandon_cost_percentile"],
    abandon_min_cost_m: float = RIVS_DEFAULTS["abandon_min_cost_m"],
    abandon_alt_races: int = RIVS_DEFAULTS["abandon_alt_races"],
    abandon_alt_gain_ratio: float = RIVS_DEFAULTS["abandon_alt_gain_ratio"],
) -> pd.DataFrame:
    """
    Return a copy of df with RIVS columns attached and ranked.

    Pipeline:
      1. base_mode (attack/defend/safe)
      2. P0, gain, cost under base modes
      3. abandon overlay on expensive competitive seats
      4. recompute priority (abandon deprioritized) + final RIVS
    """
    out = df.copy()
    out["base_mode"] = derive_base_mode(out)
    # Provisional mode for gain math = base (abandon applied after cost known)
    out["mode"] = out["base_mode"]

    p0 = baseline_win_prob_r(
        out,
        incumbent_bonus=incumbent_bonus,
        open_seat_volatility=open_seat_volatility,
    ).to_numpy()
    out["baseline_win_prob_r"] = p0

    # Gain uses base attack/defend semantics
    mode_for_gain = out["base_mode"].to_numpy()
    gain = expected_probability_gain(p0, risk_tolerance, mode_for_gain)
    # For base defend seats, use defend gain formula
    for i, m in enumerate(mode_for_gain):
        if str(m).lower() == "defend":
            target = max(risk_tolerance, min(0.72, p0[i] + 0.08))
            g = max(0.0, target - p0[i])
            if p0[i] < risk_tolerance:
                g = max(g, risk_tolerance - p0[i])
            gain[i] = g
        elif str(m).lower() == "attack":
            gain[i] = max(0.0, risk_tolerance - p0[i])
        else:
            gain[i] = max(0.0, 0.02 * (1.0 - abs(p0[i] - 0.5)))
    out["expected_prob_gain"] = gain

    cost = incremental_cost(out, gain, cost_sensitivity)
    out["incremental_cost"] = cost

    # Abandon overlay
    out = apply_abandon_flags(
        out,
        enabled=abandon_enabled,
        cost_percentile=abandon_cost_percentile,
        min_cost_m=abandon_min_cost_m,
        alt_races=int(abandon_alt_races),
        alt_gain_ratio=abandon_alt_gain_ratio,
    )

    priority = seat_priority(
        out, p0, out["expected_prob_gain"].to_numpy(), w_attack, w_defend, target_seats=target_seats
    )
    out["seat_priority"] = priority

    lt = long_term_bonus(out, long_term_multiplier)
    out["long_term_factor"] = lt

    rivs = (out["expected_prob_gain"].to_numpy() * priority * lt) / np.maximum(
        out["incremental_cost"].to_numpy(), 1.0
    )
    # Abandon: force score to bottom of investable range (still ranked for visibility)
    abandon_mask = out["mode"].astype(str).str.lower().eq("abandon").to_numpy()
    rivs = rivs.copy()
    rivs[abandon_mask] *= 0.05

    scale = 1e8
    out["rivs_raw"] = rivs
    out["rivs"] = rivs * scale

    out["rivs_rank"] = out["rivs"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("rivs", ascending=False).reset_index(drop=True)
    return out


def budget_simulation(
    ranked: pd.DataFrame,
    budget_usd: float,
    *,
    skip_abandon: bool = True,
    skip_safe: bool = True,
) -> dict[str, Any]:
    """
    Greedy allocation to highest RIVS districts until budget exhausted.
    By default skips abandon (and safe) so capital is not sunk into untenable races.
    """
    remaining = float(budget_usd)
    funded = []
    total_gain = 0.0
    spent = 0.0
    skipped_abandon = 0

    for _, row in ranked.iterrows():
        mode = str(row.get("mode", "")).lower()
        if skip_abandon and mode == "abandon":
            skipped_abandon += 1
            continue
        if skip_safe and mode == "safe":
            continue
        cost = float(row["incremental_cost"])
        if cost <= 0 or remaining < cost * 0.25:
            continue
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
    expected_seats = min(float(TOTAL_HOUSE_SEATS), baseline_seats + total_gain)

    return {
        "budget_usd": budget_usd,
        "spent_usd": spent,
        "n_districts_funded": len(funded),
        "total_prob_gain": total_gain,
        "baseline_expected_r_seats": baseline_seats,
        "expected_r_seats_after": expected_seats,
        "funded": funded,
        "skipped_abandon": skipped_abandon,
    }


def format_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"
