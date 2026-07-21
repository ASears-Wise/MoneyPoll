#!/usr/bin/env python3
"""
Generate a deterministic 435-district mock master dataset.

Usage:
    python scripts/generate_mock_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root or scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MOCK_DIR, MOCK_MASTER_PARQUET, DATA_DIR  # noqa: E402

# Post-2020 House apportionment (seats per state)
APPORTIONMENT: dict[str, int] = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
    "WI": 8, "WY": 1,
}

# Rough state lean bias (positive = R-leaning)
STATE_LEAN: dict[str, float] = {
    "AL": 14, "AK": 6, "AZ": 2, "AR": 15, "CA": -12, "CO": -3, "CT": -7, "DE": -7,
    "FL": 3, "GA": 1, "HI": -14, "ID": 18, "IL": -7, "IN": 10, "IA": 4, "KS": 10,
    "KY": 15, "LA": 11, "ME": -1, "MD": -12, "MA": -14, "MI": 0, "MN": -1, "MS": 10,
    "MO": 10, "MT": 10, "NE": 12, "NV": 0, "NH": -1, "NJ": -5, "NM": -3, "NY": -10,
    "NC": 1, "ND": 18, "OH": 5, "OK": 18, "OR": -5, "PA": 0, "RI": -10, "SC": 8,
    "SD": 16, "TN": 14, "TX": 4, "UT": 12, "VT": -15, "VA": -2, "WA": -6, "WV": 20,
    "WI": 0, "WY": 25,
}

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
    "Anthony", "Betty", "Mark", "Margaret", "Donald", "Sandra", "Steven", "Ashley",
    "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna", "Kenneth", "Michelle",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


def district_id(state: str, num: int, n_seats: int) -> str:
    if n_seats == 1:
        return f"{state}-00"
    return f"{state}-{num:02d}"


def cook_rating_from_pvi(pvi: float) -> str:
    a = abs(pvi)
    side = "R" if pvi >= 0 else "D"
    if a < 1.5:
        return "Toss-up"
    if a < 3.5:
        return f"Tilt {side}"
    if a < 6:
        return f"Lean {side}"
    if a < 10:
        return f"Likely {side}"
    return f"Solid {side}"


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    name_i = 0

    for state, n_seats in sorted(APPORTIONMENT.items()):
        lean = STATE_LEAN.get(state, 0.0)
        for d in range(1, n_seats + 1):
            # Within-state variation
            within = float(rng.normal(0, 8))
            # Make a few competitive seats per large state
            if n_seats >= 5 and d <= max(1, n_seats // 6):
                within = float(rng.uniform(-4, 4)) - lean * 0.3
            pvi = float(np.clip(lean + within, -35, 35))
            pvi = round(pvi, 1)

            if pvi > 1.5:
                party = "R"
            elif pvi < -1.5:
                party = "D"
            else:
                party = "R" if rng.random() > 0.5 else "D"

            # Flip some close seats for realism
            if abs(pvi) < 4 and rng.random() < 0.35:
                party = "D" if party == "R" else "R"

            is_open = bool(rng.random() < 0.08)
            incumbent_running = not is_open and bool(rng.random() < 0.88)
            first_elected = int(rng.integers(2004, 2025)) if not is_open else None
            tenure = (2026 - first_elected) if first_elected else 0.0

            rep_name = "OPEN SEAT"
            if not is_open:
                rep_name = f"{FIRST_NAMES[name_i % len(FIRST_NAMES)]} {LAST_NAMES[name_i % len(LAST_NAMES)]}"
                name_i += 1

            # FEC magnitudes: competitive races spend more
            compete = max(0.15, 1.0 - abs(pvi) / 25.0)
            base_spend = 1.2e6 + compete * float(rng.uniform(2e6, 12e6))

            if party == "R":
                r_mult, d_mult = 1.1, 0.85
            else:
                r_mult, d_mult = 0.85, 1.1

            fec_r_24 = base_spend * r_mult * float(rng.uniform(0.8, 1.2))
            fec_d_24 = base_spend * d_mult * float(rng.uniform(0.8, 1.2))
            fec_r_22 = fec_r_24 * float(rng.uniform(0.6, 1.1))
            fec_d_22 = fec_d_24 * float(rng.uniform(0.6, 1.1))

            hist_cost = (fec_r_24 + fec_d_24) * 0.35 * (0.5 + compete)

            # Demographics
            pop = int(rng.integers(690_000, 820_000))
            vap = int(pop * float(rng.uniform(0.72, 0.82)))
            # Race shares (simplified; not real ACS)
            if state in ("CA", "TX", "NM", "AZ", "NV", "FL"):
                hisp = float(rng.uniform(0.18, 0.55))
            else:
                hisp = float(rng.uniform(0.03, 0.22))
            black = float(rng.uniform(0.02, 0.45 if state in ("MS", "GA", "AL", "LA", "MD", "SC") else 0.18))
            asian = float(rng.uniform(0.01, 0.25 if state in ("CA", "HI", "NY", "WA") else 0.08))
            white = max(0.05, 1.0 - hisp - black - asian - float(rng.uniform(0.02, 0.08)))
            s = white + black + hisp + asian
            white, black, hisp, asian = white / s, black / s, hisp / s, asian / s

            median_income = float(rng.normal(72000, 18000))
            median_income = float(np.clip(median_income, 38000, 160000))
            pct_ba = float(np.clip(rng.normal(0.32, 0.12), 0.12, 0.72))
            pct_urban = float(np.clip(rng.beta(2, 1.2), 0.15, 0.99))

            # Mode
            if party == "D":
                mode = "attack" if pvi > -12 else "safe"
            else:
                mode = "defend" if pvi < 12 else "safe"

            did = district_id(state, d if n_seats > 1 else 0, n_seats)
            # Census-like GEOID: SS + CD (2 digits, 00 at-large)
            state_fips = {
                "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
                "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
                "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
                "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
                "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
                "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
                "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
                "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
                "WI": "55", "WY": "56",
            }
            cd = "00" if n_seats == 1 else f"{d:02d}"
            geoid = state_fips[state] + cd

            challengers = []
            if mode != "safe" or rng.random() < 0.3:
                challengers.append(
                    {
                        "name": f"{FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]} {LAST_NAMES[rng.integers(0, len(LAST_NAMES))]}",
                        "party": "R" if party == "D" else "D",
                    }
                )

            civic_candidates = {
                "office": f"U.S. Representative, District {cd}",
                "candidates": (
                    [{"name": rep_name, "party": party}] if not is_open else []
                )
                + challengers,
                "source": "mock",
            }

            rows.append(
                {
                    "district_id": did,
                    "state": state,
                    "district_num": 0 if n_seats == 1 else d,
                    "geoid": geoid,
                    "party_control": party,
                    "rep_name": rep_name,
                    "rep_party": party if not is_open else "",
                    "first_elected": first_elected if first_elected else pd.NA,
                    "tenure_years": float(tenure),
                    "pvi": pvi,
                    "cook_rating": cook_rating_from_pvi(pvi),
                    "baseline_win_prob_r": np.nan,  # filled by RIVS
                    "incumbent_running": incumbent_running,
                    "is_open_seat": is_open,
                    "mode": mode,
                    "fec_raised_r_2024": round(fec_r_24, 0),
                    "fec_spent_r_2024": round(fec_r_24 * float(rng.uniform(0.75, 0.98)), 0),
                    "fec_raised_d_2024": round(fec_d_24, 0),
                    "fec_spent_d_2024": round(fec_d_24 * float(rng.uniform(0.75, 0.98)), 0),
                    "fec_outside_2024": round(base_spend * compete * float(rng.uniform(0.2, 1.5)), 0),
                    "fec_raised_r_2022": round(fec_r_22, 0),
                    "fec_spent_r_2022": round(fec_r_22 * float(rng.uniform(0.75, 0.98)), 0),
                    "fec_raised_d_2022": round(fec_d_22, 0),
                    "fec_spent_d_2022": round(fec_d_22 * float(rng.uniform(0.75, 0.98)), 0),
                    "fec_raised_r_2026": round(fec_r_24 * float(rng.uniform(0.15, 0.55)), 0),
                    "fec_spent_r_2026": round(fec_r_24 * float(rng.uniform(0.05, 0.35)), 0),
                    "fec_raised_d_2026": round(fec_d_24 * float(rng.uniform(0.15, 0.55)), 0),
                    "fec_spent_d_2026": round(fec_d_24 * float(rng.uniform(0.05, 0.35)), 0),
                    "hist_cost_to_compete": round(hist_cost, 0),
                    "incremental_cost": np.nan,
                    "pop_total": pop,
                    "vap": vap,
                    "median_income": round(median_income, 0),
                    "pct_white": round(white, 4),
                    "pct_black": round(black, 4),
                    "pct_hispanic": round(hisp, 4),
                    "pct_asian": round(asian, 4),
                    "pct_ba_plus": round(pct_ba, 4),
                    "pct_urban": round(pct_urban, 4),
                    "civic_election_id": pd.NA,
                    "civic_candidates_json": json.dumps(civic_candidates),
                    "notes": "Synthetic mock row — replace via scripts/build_master_dataset.py",
                    "data_source_flags": "mock",
                }
            )

    df = pd.DataFrame(rows)
    assert len(df) == 435, f"Expected 435 districts, got {len(df)}"
    return df


def main() -> None:
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_parquet(MOCK_MASTER_PARQUET, index=False)
    # Also write as default master if missing
    master = DATA_DIR / "master.parquet"
    df.to_parquet(master, index=False)
    # CSV for inspection
    csv_path = MOCK_DIR / "master.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} districts → {MOCK_MASTER_PARQUET}")
    print(f"Also wrote → {master}")
    print(f"CSV → {csv_path}")
    print(df.groupby("party_control").size().to_string())


if __name__ == "__main__":
    main()
