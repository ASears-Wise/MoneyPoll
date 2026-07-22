#!/usr/bin/env python3
"""
Recalibrate federal House master: realistic leans/control + cost-to-compete.

Root cause of bad rankings (e.g. TX-07 as a cheap toss-up): synthetic mock
PVI was merged with real FEC finance, so random leans + low FEC minority spend
produced absurd RIVS.

This script:
  1. Assigns approximate Cook-style PVI and party control from a calibrated model
     (2024 House map shape + known safe seats; not a licensed Cook product).
  2. Aligns party_control with lean (no R seat with D+ lean, etc.).
  3. Sets hist_cost_to_compete so deep seats are expensive to flip (not $200K).
  4. Marks mode attack/defend/safe consistently.
  5. Writes data/master.parquet + mock snapshot.

Usage:
    python scripts/recalibrate_house_master.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MASTER_PARQUET, MOCK_MASTER_PARQUET  # noqa: E402
from scripts.generate_mock_data import APPORTIONMENT, cook_rating_from_pvi  # noqa: E402

# ---------------------------------------------------------------------------
# Known / high-confidence leans (signed R+). Override model for accuracy.
# Sources: public 2024 results + commonly cited PVI-like ranges; TX-07 user note.
# ---------------------------------------------------------------------------
KNOWN_PVI: dict[str, float] = {
    # Texas — post-2020 / 2024-era seats (approximate; TX-07 set per user D+24)
    "TX-01": 26, "TX-02": 13, "TX-03": 12, "TX-04": 18, "TX-05": 14,
    "TX-06": 12, "TX-07": -24, "TX-08": 22, "TX-09": -28, "TX-10": 11,
    "TX-11": 28, "TX-12": 14, "TX-13": 30, "TX-14": 16, "TX-15": 3,
    "TX-16": -18, "TX-17": 12, "TX-18": -26, "TX-19": 28, "TX-20": -16,
    "TX-21": 10, "TX-22": 8, "TX-23": 4, "TX-24": 8, "TX-25": 14,
    "TX-26": 14, "TX-27": 14, "TX-28": -2, "TX-29": -22, "TX-30": -28,
    "TX-31": 10, "TX-32": -8, "TX-33": -26, "TX-34": -6, "TX-35": -20,
    "TX-36": 16, "TX-37": -26, "TX-38": 12,
    # Deep blue metros
    "CA-12": -36, "CA-11": -34, "CA-10": -20, "NY-13": -36, "NY-15": -34,
    "IL-07": -34, "MA-07": -32, "WA-07": -32, "MD-04": -32, "GA-05": -30,
    "PA-03": -36, "MI-13": -28, "MN-05": -28,
    # Classic battlegrounds / toss-ups (approx)
    "AZ-01": 2, "AZ-06": 2, "CA-13": 0, "CA-22": 2, "CA-27": -1,
    "CA-41": 2, "CA-45": 1, "CO-08": 0, "IA-01": 2, "IA-03": 1,
    "ME-02": 2, "MI-07": 1, "MI-08": 0, "MI-10": 2, "NC-01": 1,
    "NE-02": 1, "NH-01": -1, "NJ-07": 1, "NM-02": 1, "NV-01": -2,
    "NV-03": 0, "NV-04": -2, "NY-03": 2, "NY-04": 2, "NY-17": 1,
    "NY-19": 1, "NY-22": 1, "OH-09": 2, "OH-13": 1, "OR-05": 1,
    "PA-07": 0, "PA-08": 1, "PA-10": 2, "PA-17": -1, "VA-02": 1,
    "VA-07": 0, "WA-03": 2, "WI-01": 2, "WI-03": 1, "AK-00": 6,
    "MT-01": 6, "MT-02": 14,
}

# 2024-ish D-held seats (119th) — incomplete list of solid + known D holds
KNOWN_D_HELD = {
    "TX-07", "TX-09", "TX-16", "TX-18", "TX-20", "TX-28", "TX-29", "TX-30",
    "TX-32", "TX-33", "TX-34", "TX-35", "TX-37",
    "CA-02", "CA-04", "CA-06", "CA-07", "CA-08", "CA-09", "CA-10", "CA-11",
    "CA-12", "CA-14", "CA-15", "CA-16", "CA-17", "CA-18", "CA-19", "CA-21",
    "CA-24", "CA-25", "CA-26", "CA-28", "CA-29", "CA-30", "CA-31", "CA-32",
    "CA-33", "CA-34", "CA-35", "CA-36", "CA-37", "CA-38", "CA-39", "CA-42",
    "CA-43", "CA-44", "CA-46", "CA-47", "CA-49", "CA-51", "CA-52",
    "NY-05", "NY-06", "NY-07", "NY-08", "NY-09", "NY-10", "NY-12", "NY-13",
    "NY-14", "NY-15", "NY-16", "NY-18", "NY-20", "NY-25", "NY-26",
    "IL-01", "IL-02", "IL-03", "IL-04", "IL-05", "IL-06", "IL-07", "IL-08",
    "IL-09", "IL-10", "IL-11", "IL-13", "IL-14", "IL-17",
    "MA-01", "MA-02", "MA-03", "MA-04", "MA-05", "MA-06", "MA-07", "MA-08", "MA-09",
    "MD-02", "MD-03", "MD-04", "MD-05", "MD-06", "MD-07", "MD-08",
    "WA-01", "WA-02", "WA-06", "WA-07", "WA-08", "WA-09", "WA-10",
    "OR-01", "OR-03", "OR-04", "OR-06",
    "CO-01", "CO-02", "CO-06", "CO-07",
    "VA-03", "VA-04", "VA-08", "VA-10", "VA-11",
    "GA-02", "GA-04", "GA-05", "GA-06", "GA-07", "GA-13",
    "NC-02", "NC-04", "NC-12", "NC-14",
    "PA-02", "PA-03", "PA-04", "PA-05", "PA-06", "PA-12", "PA-17",
    "MI-06", "MI-11", "MI-12", "MI-13",
    "MN-02", "MN-03", "MN-04", "MN-05",
    "WI-02", "WI-04",
    "AZ-03", "AZ-04", "AZ-07",
    "NV-01", "NV-04",
    "NM-01", "NM-03",
    "NJ-01", "NJ-03", "NJ-05", "NJ-06", "NJ-08", "NJ-09", "NJ-10", "NJ-11", "NJ-12",
    "CT-01", "CT-02", "CT-03", "CT-04", "CT-05",
    "RI-01", "RI-02", "DE-00", "VT-00", "HI-01", "HI-02",
    "ME-01", "NH-02",
    "FL-09", "FL-10", "FL-14", "FL-20", "FL-22", "FL-23", "FL-24", "FL-25",
    "OH-01", "OH-03", "OH-09", "OH-11", "OH-13",
    "IN-01", "KY-03", "LA-02", "MO-01", "MO-05", "TN-09",
    "AL-07", "MS-02", "SC-06",
    "CA-13", "CA-22", "CA-27", "CA-45",  # competitive / toss-up leans handled via PVI
}

# Approximate 2024 Trump state margin (R positive points) for residual seats
STATE_PRES_MARGIN_R: dict[str, float] = {
    "AL": 30, "AK": 13, "AZ": 5, "AR": 30, "CA": -20, "CO": -11, "CT": -18, "DE": -15,
    "FL": 13, "GA": 2, "HI": -23, "ID": 36, "IL": -11, "IN": 19, "IA": 13, "KS": 16,
    "KY": 30, "LA": 22, "ME": -7, "MD": -28, "MA": -25, "MI": 1, "MN": -4, "MS": 23,
    "MO": 18, "MT": 20, "NE": 20, "NV": 3, "NH": -3, "NJ": -6, "NM": -6, "NY": -12,
    "NC": 3, "ND": 36, "OH": 11, "OK": 34, "OR": -14, "PA": 2, "RI": -14, "SC": 18,
    "SD": 29, "TN": 29, "TX": 14, "UT": 22, "VT": -31, "VA": -6, "WA": -18, "WV": 42,
    "WI": 1, "WY": 46,
}


def district_id(state: str, num: int, n: int) -> str:
    return f"{state}-00" if n == 1 else f"{state}-{num:02d}"


def assign_leans() -> pd.DataFrame:
    """Build district_id → pvi, party_control for all 435."""
    rng = np.random.default_rng(2026)
    rows = []
    for state, n in sorted(APPORTIONMENT.items()):
        base = STATE_PRES_MARGIN_R.get(state, 0.0) * 0.55  # compress to CD-like PVI scale
        for d in range(1, n + 1):
            did = district_id(state, d if n > 1 else 0, n)

            # --- Party first (whitelist D holds; default R otherwise) ---
            if did in KNOWN_D_HELD:
                party = "D"
            elif did in KNOWN_PVI:
                party = "D" if KNOWN_PVI[did] < 0 else "R"
            else:
                # Not a curated D seat → Republican hold (plus blue-state residuals below)
                party = "R"

            # Blue states still need many D seats beyond the explicit list
            # If state is deep blue and seat not listed, assign leftover Ds by district rank
            # (handled in second pass)

            # --- PVI ---
            if did in KNOWN_PVI:
                pvi = float(KNOWN_PVI[did])
            elif party == "D":
                pvi = float(np.clip(min(base - 11.0, -8.0) + float(rng.normal(0, 2.0)), -40, -3.5))
                pvi = round(pvi, 1)
            else:
                # R hold: lean at least R+3 unless battleground known via KNOWN_PVI
                t = (d - 0.5) / max(n, 1)
                noise = float(rng.normal(0, 2.0))
                pvi = float(np.clip(max(base, 0) + 4 + 8 * t + noise, 3.0, 38.0))
                pvi = round(pvi, 1)

            # Consistency
            if pvi <= -2.5:
                party = "D"
            if pvi >= 2.5:
                party = "R"

            rows.append({"district_id": did, "pvi": float(pvi), "party_control": party})

    leans = pd.DataFrame(rows)

    # Second pass: fill blue-state D seat counts to approximate 2024 (~215 D)
    # For CA, NY, MA, etc., ensure enough D seats
    BLUE_STATE_D_TARGETS = {
        "CA": 40, "NY": 19, "IL": 14, "PA": 7, "OH": 5, "MI": 6, "NJ": 9,
        "MA": 9, "WA": 8, "VA": 6, "MD": 7, "MN": 4, "WI": 2, "OR": 4,
        "CT": 5, "CO": 5, "GA": 5, "NC": 4, "AZ": 3, "NV": 2, "NM": 2,
        "FL": 8, "TX": 13, "IN": 2, "KY": 1, "LA": 1, "MO": 2, "TN": 1,
        "AL": 1, "MS": 1, "SC": 1, "RI": 2, "HI": 2, "DE": 1, "VT": 1,
        "ME": 1, "NH": 1,
    }
    rng2 = np.random.default_rng(7)
    for state, target_d in BLUE_STATE_D_TARGETS.items():
        idx = leans.index[leans["district_id"].str.startswith(state + "-")]
        sub = leans.loc[idx]
        cur_d = (sub["party_control"] == "D").sum()
        if cur_d >= target_d:
            continue
        # Convert lowest-PVI R seats (most D-friendly) to D until target
        r_seats = sub[sub["party_control"] == "R"].sort_values("pvi")
        need = int(target_d - cur_d)
        for i, (_, row) in enumerate(r_seats.iterrows()):
            if i >= need:
                break
            di = row.name
            leans.at[di, "party_control"] = "D"
            new_pvi = float(np.clip(min(row["pvi"], -3.5) - abs(float(rng2.normal(2, 1.5))), -30, -2))
            # Competitive if near even conversion
            if abs(row["pvi"]) < 8:
                new_pvi = float(np.clip(-abs(row["pvi"]) - 0.5, -6, -1))
            leans.at[di, "pvi"] = round(new_pvi, 1)

    # Re-assert known overrides
    for did, pvi in KNOWN_PVI.items():
        m = leans["district_id"] == did
        leans.loc[m, "pvi"] = pvi
        leans.loc[m, "party_control"] = "D" if pvi < 0 else "R"
    for did in KNOWN_D_HELD:
        m = leans["district_id"] == did
        leans.loc[m, "party_control"] = "D"
        if did not in KNOWN_PVI:
            leans.loc[m, "pvi"] = leans.loc[m, "pvi"].apply(lambda x: min(float(x), -4.0))

    return leans.reset_index(drop=True)


def cost_to_be_competitive(pvi: float, party: str, fec_r: float, fec_d: float) -> float:
    """
    Dollars to run a *competitive* race in this district.

    Deep seats cost far more to contest for the minority party; majority-party
    maintenance is cheaper. Uses FEC when it signals a real arms race.
    """
    a = abs(float(pvi))
    fec_total = float(fec_r or 0) + float(fec_d or 0)
    # Structural floor from lean (quadratic): D+24 cannot be "a few hundred thousand"
    if a < 1.5:
        structural = 3_500_000
    elif a < 3.5:
        structural = 2_800_000
    elif a < 6:
        structural = 2_200_000
    elif a < 10:
        structural = 1_600_000 + (a - 6) * 200_000
    elif a < 15:
        structural = 2_500_000 + (a - 10) * 400_000
    else:
        # Solid seats: expensive to flip
        structural = 4_000_000 + (a - 15) ** 2 * 80_000

    # FEC competitive spend signal (when both sides spent)
    fec_signal = 0.0
    if fec_r > 100_000 and fec_d > 100_000:
        fec_signal = 0.45 * (fec_r + fec_d)
    elif fec_total > 0:
        fec_signal = 0.25 * fec_total

    cost = max(structural, fec_signal, 250_000)
    # Cap absurd outliers
    cost = min(cost, 25_000_000)
    return float(round(cost, 0))


def derive_mode(pvi: float, party: str) -> str:
    if party == "D":
        if pvi > -8:
            return "attack"
        return "safe"
    if party == "R":
        if pvi < 8:
            return "defend"
        return "safe"
    return "attack"


def apply_to_master(master: pd.DataFrame, leans: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    m = leans.set_index("district_id")
    out["pvi"] = out["district_id"].map(m["pvi"])
    out["party_control"] = out["district_id"].map(m["party_control"])
    out["rep_party"] = out["party_control"]
    out["cook_rating"] = out["pvi"].map(cook_rating_from_pvi)

    # Names: keep if openfec/openstates; else placeholder
    def _rep_name(row):
        if str(row.get("data_source_flags", "")).find("openstates") >= 0:
            return row.get("rep_name")
        # Keep non-fake names if present from previous; else generic
        name = str(row.get("rep_name") or "")
        if name and name != "OPEN SEAT" and "mock" not in str(row.get("data_source_flags", "")):
            # still mock names from generator — replace with party placeholder for clarity
            pass
        return f"{'Democratic' if row['party_control']=='D' else 'Republican'} incumbent" if not row.get("is_open_seat") else "OPEN SEAT"

    # Softer: only fix OPEN and clearly synthetic patterns
    out["mode"] = [
        derive_mode(float(p), str(party))
        for p, party in zip(out["pvi"], out["party_control"])
    ]

    fec_r = out["fec_raised_r_2024"] if "fec_raised_r_2024" in out.columns else 0
    fec_d = out["fec_raised_d_2024"] if "fec_raised_d_2024" in out.columns else 0
    # Prefer 2026 when present
    if "fec_raised_r_2026" in out.columns:
        fec_r = out["fec_raised_r_2026"].fillna(0) + out.get("fec_raised_r_2024", 0).fillna(0) * 0
        # blend: use max of cycles for signal
        fec_r = pd.concat(
            [
                out["fec_raised_r_2026"].fillna(0),
                out["fec_raised_r_2024"].fillna(0) if "fec_raised_r_2024" in out.columns else 0,
            ],
            axis=1,
        ).max(axis=1)
        fec_d = pd.concat(
            [
                out["fec_raised_d_2026"].fillna(0),
                out["fec_raised_d_2024"].fillna(0) if "fec_raised_d_2024" in out.columns else 0,
            ],
            axis=1,
        ).max(axis=1)

    costs = [
        cost_to_be_competitive(float(p), str(party), float(fr), float(fd))
        for p, party, fr, fd in zip(out["pvi"], out["party_control"], fec_r, fec_d)
    ]
    out["hist_cost_to_compete"] = costs
    out["cost_to_be_competitive"] = costs

    flags = out.get("data_source_flags", pd.Series(["mock"] * len(out))).astype(str)
    out["data_source_flags"] = flags.apply(
        lambda s: s
        if "leans_v2" in s
        else (s.replace("mock", "mock+leans_v2") if "mock" in s else s + "|leans_v2")
    )
    out["notes"] = (
        "Leans/control recalibrated (approx. 2024-map PVI-like; not licensed Cook). "
        "FEC finance retained where present. Cost to compete uses lean hardness + FEC."
    )
    return out


def main() -> None:
    if MASTER_PARQUET.exists():
        master = pd.read_parquet(MASTER_PARQUET)
    elif MOCK_MASTER_PARQUET.exists():
        master = pd.read_parquet(MOCK_MASTER_PARQUET)
    else:
        from scripts.generate_mock_data import generate

        master = generate()

    leans = assign_leans()
    assert len(leans) == 435, len(leans)
    out = apply_to_master(master, leans)

    # Sanity checks
    tx7 = out[out["district_id"] == "TX-07"].iloc[0]
    assert tx7["party_control"] == "D", tx7["party_control"]
    assert tx7["pvi"] <= -20, tx7["pvi"]
    assert tx7["hist_cost_to_compete"] >= 2_000_000, tx7["hist_cost_to_compete"]
    assert tx7["mode"] == "safe", tx7["mode"]

    mismatch = out[
        ((out["party_control"] == "R") & (out["pvi"] < -3))
        | ((out["party_control"] == "D") & (out["pvi"] > 3))
    ]
    assert len(mismatch) == 0, mismatch[["district_id", "party_control", "pvi"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MASTER_PARQUET, index=False)
    out.to_parquet(MOCK_MASTER_PARQUET, index=False)
    ref = DATA_DIR / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    leans.to_csv(ref / "house_leans_approx.csv", index=False)
    out[
        [
            "district_id",
            "state",
            "party_control",
            "pvi",
            "cook_rating",
            "mode",
            "hist_cost_to_compete",
        ]
    ].to_csv(ref / "house_master_lean_summary.csv", index=False)

    print(f"Wrote {MASTER_PARQUET} ({len(out)} rows)")
    print(
        "TX-07:",
        out[out.district_id == "TX-07"][
            ["party_control", "pvi", "mode", "hist_cost_to_compete"]
        ].to_string(index=False),
    )
    print("party counts:\n", out["party_control"].value_counts().to_string())
    print("mode counts:\n", out["mode"].value_counts().to_string())
    from src.rivs import compute_rivs

    r = compute_rivs(out)
    print("Top 10 RIVS after fix:")
    print(
        r.nsmallest(10, "rivs_rank")[
            ["district_id", "party_control", "pvi", "mode", "rivs", "hist_cost_to_compete"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
