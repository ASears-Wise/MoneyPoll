#!/usr/bin/env python3
"""
Generate deterministic mock masters for all 50-state lower/upper chambers.

Usage:
    python scripts/generate_state_mock_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    STATE_CHAMBER_SEATS,
    STATE_DATA_DIR,
    STATE_LEAN_PROXY,
    STATE_MOCK_DIR,
    STATE_MOCK_LOWER,
    STATE_MOCK_UPPER,
)

FIRST = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
]
LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
]


def majority_threshold(n: int) -> int:
    return n // 2 + 1 if n > 0 else 0


def district_id(state: str, chamber: str, num: int) -> str:
    tag = "L" if chamber == "lower" else "U"
    return f"{state}-{tag}-{num:03d}"


def ocd_id(state: str, chamber: str, num: int) -> str:
    kind = "sldl" if chamber == "lower" else "sldu"
    return f"ocd-division/country:us/state:{state.lower()}/{kind}:{num}"


def generate_chamber(chamber: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (0 if chamber == "lower" else 99))
    rows = []
    name_i = 0

    for state, seats in sorted(STATE_CHAMBER_SEATS.items()):
        n = int(seats.get(chamber, 0) or 0)
        if n <= 0:
            continue
        lean0 = float(STATE_LEAN_PROXY.get(state, 0.0))
        maj = majority_threshold(n)
        for d in range(1, n + 1):
            # Within-state spread; a few competitive seats
            within = float(rng.normal(0, 10))
            if d <= max(1, n // 8):
                within = float(rng.uniform(-6, 6)) - lean0 * 0.25
            pvi = float(np.clip(lean0 * 0.55 + within, -40, 40))
            pvi = round(pvi, 1)

            if pvi > 2:
                party = "R"
            elif pvi < -2:
                party = "D"
            else:
                party = "R" if rng.random() > 0.5 else "D"
            if abs(pvi) < 5 and rng.random() < 0.3:
                party = "D" if party == "R" else "R"

            is_open = bool(rng.random() < 0.07)
            incumbent_running = not is_open and bool(rng.random() < 0.9)
            first_elected = int(rng.integers(2008, 2025)) if not is_open else None
            tenure = (2026 - first_elected) if first_elected else 0.0
            rep_name = "OPEN SEAT"
            if not is_open:
                rep_name = f"{FIRST[name_i % len(FIRST)]} {LAST[name_i % len(LAST)]}"
                name_i += 1

            compete = max(0.12, 1.0 - abs(pvi) / 28.0)
            base = 80_000 + compete * float(rng.uniform(150_000, 1_200_000))
            if party == "R":
                r_m, d_m = 1.1, 0.9
            else:
                r_m, d_m = 0.9, 1.1
            raised_r = base * r_m * float(rng.uniform(0.7, 1.2))
            raised_d = base * d_m * float(rng.uniform(0.7, 1.2))
            hist = (raised_r + raised_d) * 0.4 * (0.5 + compete)

            if party == "D":
                mode = "attack" if pvi > -14 else "safe"
            else:
                mode = "defend" if pvi < 14 else "safe"

            did = district_id(state, chamber, d)
            cands = {
                "office": f"{state} {'House' if chamber == 'lower' else 'Senate'} District {d}",
                "source": "mock",
                "candidates": (
                    [{"name": rep_name, "party": party}] if not is_open else []
                )
                + (
                    [
                        {
                            "name": f"{FIRST[rng.integers(0, len(FIRST))]} {LAST[rng.integers(0, len(LAST))]}",
                            "party": "R" if party == "D" else "D",
                        }
                    ]
                    if mode != "safe" or rng.random() < 0.25
                    else []
                ),
            }

            rows.append(
                {
                    "district_id": did,
                    "state": state,
                    "chamber": chamber,
                    "district_num": d,
                    "ocd_id": ocd_id(state, chamber, d),
                    "seat_label": f"{state} {'HD' if chamber == 'lower' else 'SD'}-{d}",
                    "party_control": party,
                    "rep_name": rep_name,
                    "rep_party": party if not is_open else "",
                    "first_elected": first_elected if first_elected else pd.NA,
                    "tenure_years": float(tenure),
                    "pvi": pvi,
                    "cook_rating": (
                        "Toss-up"
                        if abs(pvi) < 2
                        else f"{'Lean' if abs(pvi) < 7 else 'Likely'} {'R' if pvi > 0 else 'D'}"
                    ),
                    "incumbent_running": incumbent_running,
                    "is_open_seat": is_open,
                    "mode": mode,
                    "chamber_total_seats": n,
                    "chamber_majority_threshold": maj,
                    "state_raised_r_2026": round(raised_r, 0),
                    "state_spent_r_2026": round(raised_r * 0.9, 0),
                    "state_raised_d_2026": round(raised_d, 0),
                    "state_spent_d_2026": round(raised_d * 0.9, 0),
                    "state_raised_r_2024": round(raised_r * 0.9, 0),
                    "state_spent_r_2024": round(raised_r * 0.85, 0),
                    "state_raised_d_2024": round(raised_d * 0.9, 0),
                    "state_spent_d_2024": round(raised_d * 0.85, 0),
                    "state_raised_r_2022": round(raised_r * 0.85, 0),
                    "state_spent_r_2022": round(raised_r * 0.8, 0),
                    "state_raised_d_2022": round(raised_d * 0.85, 0),
                    "state_spent_d_2022": round(raised_d * 0.8, 0),
                    "hist_cost_to_compete": round(hist, 0),
                    "candidates_json": json.dumps(cands),
                    "notes": "Synthetic state-leg row — replace via OpenStates / FTM pipeline",
                    "data_source_flags": "mock",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    STATE_MOCK_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    lower = generate_chamber("lower", seed=11)
    upper = generate_chamber("upper", seed=22)
    lower.to_parquet(STATE_MOCK_LOWER, index=False)
    upper.to_parquet(STATE_MOCK_UPPER, index=False)
    # Default live paths = mock until build script overwrites
    lower.to_parquet(STATE_DATA_DIR / "lower_master.parquet", index=False)
    upper.to_parquet(STATE_DATA_DIR / "upper_master.parquet", index=False)
    lower.to_csv(STATE_MOCK_DIR / "lower_master.csv", index=False)
    upper.to_csv(STATE_MOCK_DIR / "upper_master.csv", index=False)
    print(f"Lower: {len(lower)} seats → {STATE_MOCK_LOWER}")
    print(f"Upper: {len(upper)} seats → {STATE_MOCK_UPPER}")
    print(lower.groupby("party_control").size().to_string())
    print("---")
    print(upper.groupby("party_control").size().to_string())


if __name__ == "__main__":
    main()
