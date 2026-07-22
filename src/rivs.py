"""
Republican Investment Value Score (RIVS) — tighter strategy edition

RIVS = (Expected_Probability_Gain × Seat_Priority × Strategy_Fit) / Incremental_Cost

Clear separation of benefits:
  attack  — **weak Democrats** (D-held, near-even or R-leaning; open/fragile D)
  defend  — **strong Republican holds worth locking** (R-held, still R-favored but
            not so safe that $ is wasted; plus underwater R rescues)
  safe    — deep seats; near-zero RIVS
  abandon — outrageously expensive competitive seats; crushed RIVS

All inputs are tunable; see config.RIVS_DEFAULTS.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_TARGET_SEATS, RIVS_DEFAULTS, TOTAL_HOUSE_SEATS


def _logistic_from_pvi(pvi: np.ndarray, scale: float = 7.0) -> np.ndarray:
    """Map Cook-style signed PVI (R positive) to R win probability prior."""
    return 1.0 / (1.0 + np.exp(-pvi / scale))


def baseline_win_prob_r(
    df: pd.DataFrame,
    incumbent_bonus: float = RIVS_DEFAULTS["incumbent_bonus"],
    open_seat_volatility: float = RIVS_DEFAULTS["open_seat_volatility"],
) -> pd.Series:
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


def derive_base_mode(df: pd.DataFrame) -> pd.Series:
    """
    attack: D-held and not deep blue (PVI > -7)  → weak/competitive Democrats
    defend: R-held and not deep red (PVI < 8)     → holds worth fortifying
    safe:   everything else
    """
    modes = []
    for _, row in df.iterrows():
        party = str(row["party_control"]).upper()
        pvi = float(row["pvi"])
        if party == "D":
            modes.append("attack" if pvi > -7 else "safe")
        elif party == "R":
            modes.append("defend" if pvi < 8 else "safe")
        else:
            modes.append("attack")
    return pd.Series(modes, index=df.index, name="base_mode")


def derive_mode(df: pd.DataFrame) -> pd.Series:
    return derive_base_mode(df).rename("mode")


def _weak_democrat_score(df: pd.DataFrame) -> np.ndarray:
    """
    0–1: how *weak* the Democratic hold is (higher = better attack target).

    Weak D signals:
      - PVI closer to even or R-leaning (pvi higher / less negative)
      - open seat
      - short D tenure
    """
    pvi = df["pvi"].astype(float).to_numpy()
    party = df["party_control"].astype(str).str.upper().to_numpy()
    is_open = df["is_open_seat"].astype(bool).to_numpy() if "is_open_seat" in df.columns else np.zeros(len(df), dtype=bool)
    tenure = (
        df["tenure_years"].fillna(12).to_numpy()
        if "tenure_years" in df.columns
        else np.full(len(df), 12.0)
    )

    # Map D-held pvi in (-7, +5) → weakness 0..1 (pvi=-7 weakish, pvi=0 very weak D)
    # For attack candidates only; others 0
    weakness = np.zeros(len(df))
    d_mask = party == "D"
    # Linear: pvi=-7 → 0.15, pvi=0 → 0.85, pvi=+3 → 1.0
    w = (pvi - (-7.0)) / 10.0
    w = np.clip(w, 0.0, 1.0)
    weakness = np.where(d_mask, 0.15 + 0.85 * w, 0.0)
    weakness = np.where(d_mask & is_open, np.minimum(1.0, weakness + 0.2), weakness)
    weakness = np.where(d_mask & (tenure < 4), np.minimum(1.0, weakness + 0.12), weakness)
    # Deep blue never weak
    weakness = np.where(d_mask & (pvi <= -10), 0.0, weakness)
    return weakness


def _strong_republican_defend_score(df: pd.DataFrame) -> np.ndarray:
    """
    0–1: value of *defending* an R seat to keep Republicans strong.

    Highest for R-held seats that are still R-favored but not auto-safe
    (roughly PVI 0 to +7): invest to lock in strength.
    Also elevate underwater R (pvi < 0) as rescue defends.
    Near-zero for deep solid R (PVI ≥ 10) — already strong, $ has little edge.
    """
    pvi = df["pvi"].astype(float).to_numpy()
    party = df["party_control"].astype(str).str.upper().to_numpy()
    is_open = df["is_open_seat"].astype(bool).to_numpy() if "is_open_seat" in df.columns else np.zeros(len(df), dtype=bool)

    score = np.zeros(len(df))
    r_mask = party == "R"
    # Sweet spot: R+0 to R+7 (competitive-but-strong R holds)
    in_band = r_mask & (pvi >= -2) & (pvi < 8)
    # Peak around R+2 to R+4
    peak = 1.0 - np.abs(pvi - 3.0) / 8.0
    peak = np.clip(peak, 0.2, 1.0)
    score = np.where(in_band, peak, score)
    # Rescue: R-held but D-leaning
    rescue = r_mask & (pvi < 0)
    score = np.where(rescue, np.maximum(score, 0.75 + np.clip(-pvi, 0, 5) * 0.05), score)
    # Open R seat → high defend value
    score = np.where(r_mask & is_open & (pvi < 10), np.minimum(1.0, score + 0.2), score)
    # Solid R: crush
    score = np.where(r_mask & (pvi >= 10), 0.05, score)
    score = np.where(~r_mask, 0.0, score)
    return np.clip(score, 0.0, 1.0)


def expected_probability_gain(
    p0: np.ndarray,
    p_star: float,
    mode: np.ndarray,
    *,
    weak_d: np.ndarray | None = None,
    strong_r: np.ndarray | None = None,
) -> np.ndarray:
    """
    Probability gain — amplified for weak-D attacks and strong-R defends.
    """
    gain = np.zeros_like(p0, dtype=float)
    weak_d = weak_d if weak_d is not None else np.zeros_like(p0)
    strong_r = strong_r if strong_r is not None else np.zeros_like(p0)

    for i, m in enumerate(mode):
        m = str(m).lower()
        if m in ("attack", "abandon"):
            raw = max(0.0, p_star - p0[i])
            # Weak D: more gain credit (easier path to flip)
            gain[i] = raw * (0.65 + 0.55 * weak_d[i])
        elif m == "defend":
            # Lock in R strength: buffer above waterline
            target = max(p_star + 0.05, min(0.68, p0[i] + 0.10))
            raw = max(0.0, target - p0[i])
            if p0[i] < p_star:
                raw = max(raw, p_star - p0[i])
            # Strong R holds (R-favored competitive) get more defend credit
            gain[i] = raw * (0.70 + 0.50 * strong_r[i])
        else:  # safe
            gain[i] = max(0.0, 0.005 * (1.0 - abs(p0[i] - 0.5)))
    return gain


def seat_priority(
    df: pd.DataFrame,
    p0: np.ndarray,
    gain: np.ndarray,
    w_attack: float,
    w_defend: float,
    target_seats: int = DEFAULT_TARGET_SEATS,
    *,
    weak_d: np.ndarray | None = None,
    strong_r: np.ndarray | None = None,
) -> np.ndarray:
    """
    Priority tightly couples mode × strategy fit:
      attack priority ∝ weak Democrat score
      defend priority ∝ strong Republican hold score
      abandon / safe → near zero
    """
    mode = df["mode"].astype(str).str.lower().to_numpy()
    n = len(df)
    weak_d = weak_d if weak_d is not None else _weak_democrat_score(df)
    strong_r = strong_r if strong_r is not None else _strong_republican_defend_score(df)

    # Competitiveness (still matters, but strategy fit dominates)
    marginal = 1.0 - 2.0 * np.abs(p0 - 0.5)
    marginal = np.clip(marginal, 0.08, 1.0)

    likely_r = float(np.sum(p0 >= 0.5))
    seats_needed = max(0, target_seats - likely_r)
    urgency = 1.0 + min(1.2, seats_needed / 25.0)

    # Strategy fit vector
    fit = np.zeros(n)
    fit = np.where(mode == "attack", 0.25 + 0.75 * weak_d, fit)
    fit = np.where(mode == "defend", 0.25 + 0.75 * strong_r, fit)
    fit = np.where(mode == "safe", 0.04, fit)
    fit = np.where(mode == "abandon", 0.02, fit)

    mode_w = np.ones(n)
    mode_w[mode == "attack"] = w_attack
    mode_w[mode == "defend"] = w_defend
    mode_w[mode == "safe"] = 0.08
    mode_w[mode == "abandon"] = 0.02

    gain_factor = 0.25 + 0.75 * (gain / (gain.max() + 1e-9))
    return marginal * urgency * mode_w * fit * gain_factor


def incremental_cost(
    df: pd.DataFrame,
    gain: np.ndarray,
    cost_sensitivity: float,
) -> np.ndarray:
    """$ cost to realize probability gain — expensive seats stay expensive."""
    hist = df["hist_cost_to_compete"].astype(float).to_numpy()
    pvi = np.abs(df["pvi"].astype(float).to_numpy())
    hardness = 1.0 + (pvi / 14.0) ** 1.15
    outside = np.zeros(len(df), dtype=float)
    if "fec_outside_2024" in df.columns:
        outside = pd.to_numeric(df["fec_outside_2024"], errors="coerce").fillna(0).to_numpy()
    if "fec_outside_2026" in df.columns:
        o6 = pd.to_numeric(df["fec_outside_2026"], errors="coerce").fillna(0).to_numpy()
        outside = np.maximum(outside, o6)
    hist = hist + 0.30 * outside
    hist = np.maximum(hist, 300_000.0)
    cost = hist * hardness * cost_sensitivity * (0.45 + 1.4 * np.maximum(gain, 0.02))
    return np.maximum(cost, 300_000.0)


def long_term_bonus(
    df: pd.DataFrame,
    long_term_multiplier: float,
) -> np.ndarray:
    """Bonus for open seats and fragile D incumbents (weak Dems)."""
    is_open = df["is_open_seat"].astype(bool).to_numpy()
    bonus = np.ones(len(df))
    bonus[is_open] *= long_term_multiplier
    if "tenure_years" in df.columns:
        tenure = df["tenure_years"].fillna(10).to_numpy()
        party = df["party_control"].astype(str).str.upper().to_numpy()
        fragile_d = (party == "D") & (tenure < 4)
        bonus[fragile_d] *= 1.0 + 0.25 * (long_term_multiplier - 1.0)
    return bonus


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
    Mark outrageously expensive competitive seats as **abandon**.

    Triggers (any):
      A) Cost/gain in top tail of competitive seats AND cost ≥ min_cost_m
      B) Opportunity cost: cost ≥ N × median competitive AND N median races buy more gain
      C) Absolute megaprice: cost ≥ $8M and expected gain < 0.12 (clear money pit)
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

        inefficient = cost[i] >= min_cost and cpg[i] >= cpg_cut
        n = max(2, int(alt_races))
        opportunity = (
            cost[i] >= n * med_cost
            and (n * med_gain) > (gain[i] * float(alt_gain_ratio))
        )
        megapit = cost[i] >= 8_000_000 and gain[i] < 0.12

        if inefficient or opportunity or megapit:
            new_mode[i] = "abandon"
            bits = []
            if megapit:
                bits.append(
                    f"outrageous price: {format_money(cost[i])} for only {gain[i]:.3f} ΔP"
                )
            if inefficient:
                bits.append(
                    f"cost/gain in top {(1 - cost_percentile) * 100:.0f}% of competitive "
                    f"({format_money(cpg[i])} per ΔP)"
                )
            if opportunity:
                bits.append(
                    f"opportunity cost: seat {format_money(cost[i])} ≥ {n}× median "
                    f"({format_money(med_cost)}); {n} cheaper races buy more gain"
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
    Pipeline:
      1. base_mode + weak-D / strong-R scores
      2. P0, gain, cost
      3. abandon outrageously expensive seats
      4. priority (attack weak D, defend strong R) + RIVS
    """
    out = df.copy()
    out["base_mode"] = derive_base_mode(out)
    out["mode"] = out["base_mode"]

    out["weak_democrat_score"] = _weak_democrat_score(out)
    out["strong_republican_score"] = _strong_republican_defend_score(out)
    weak_d = out["weak_democrat_score"].to_numpy()
    strong_r = out["strong_republican_score"].to_numpy()

    p0 = baseline_win_prob_r(
        out,
        incumbent_bonus=incumbent_bonus,
        open_seat_volatility=open_seat_volatility,
    ).to_numpy()
    out["baseline_win_prob_r"] = p0

    gain = expected_probability_gain(
        p0,
        risk_tolerance,
        out["base_mode"].to_numpy(),
        weak_d=weak_d,
        strong_r=strong_r,
    )
    out["expected_prob_gain"] = gain

    cost = incremental_cost(out, gain, cost_sensitivity)
    out["incremental_cost"] = cost

    out = apply_abandon_flags(
        out,
        enabled=abandon_enabled,
        cost_percentile=abandon_cost_percentile,
        min_cost_m=abandon_min_cost_m,
        alt_races=int(abandon_alt_races),
        alt_gain_ratio=abandon_alt_gain_ratio,
    )

    priority = seat_priority(
        out,
        p0,
        out["expected_prob_gain"].to_numpy(),
        w_attack,
        w_defend,
        target_seats=target_seats,
        weak_d=weak_d,
        strong_r=strong_r,
    )
    out["seat_priority"] = priority

    lt = long_term_bonus(out, long_term_multiplier)
    out["long_term_factor"] = lt

    rivs = (
        out["expected_prob_gain"].to_numpy() * priority * lt
    ) / np.maximum(out["incremental_cost"].to_numpy(), 1.0)

    # Crush abandon + safe so rankings scream strategy
    mode = out["mode"].astype(str).str.lower().to_numpy()
    rivs = rivs.copy()
    rivs[mode == "abandon"] *= 0.02
    rivs[mode == "safe"] *= 0.08
    # Extra boost so weak-D attacks and strong-R defends separate clearly
    rivs[mode == "attack"] *= 1.0 + 0.35 * weak_d[mode == "attack"] if np.any(mode == "attack") else 1.0
    # vector-safe boosts:
    attack_m = mode == "attack"
    defend_m = mode == "defend"
    rivs[attack_m] = rivs[attack_m] * (1.0 + 0.40 * weak_d[attack_m])
    rivs[defend_m] = rivs[defend_m] * (1.0 + 0.40 * strong_r[defend_m])

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
    """Greedy top-RIVS fill; skips abandon/safe by default."""
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
            frac, pay = 1.0, cost
        else:
            frac, pay = remaining / cost, remaining
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
