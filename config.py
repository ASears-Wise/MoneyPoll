"""House Moneyball configuration: paths, defaults, RIVS bounds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MOCK_DIR = DATA_DIR / "mock"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"

MASTER_PARQUET = DATA_DIR / "master.parquet"
MOCK_MASTER_PARQUET = MOCK_DIR / "master.parquet"
GEOJSON_PATH = RAW_DIR / "cds.geojson"

# Target majority for Republican House strategy narrative
DEFAULT_TARGET_SEATS = 230
TOTAL_HOUSE_SEATS = 435

# RIVS slider defaults & bounds
RIVS_DEFAULTS = {
    "cost_sensitivity": 1.0,  # scales incremental cost
    "long_term_multiplier": 1.15,
    "risk_tolerance": 0.52,  # target win probability P*
    "w_attack": 1.2,
    "w_defend": 1.0,
    "incumbent_bonus": 0.06,  # additive to baseline R win prob if R incumbent
    "open_seat_volatility": 0.04,
    "budget_millions": 50.0,
}

RIVS_BOUNDS = {
    "cost_sensitivity": (0.25, 3.0),
    "long_term_multiplier": (0.8, 2.0),
    "risk_tolerance": (0.48, 0.65),
    "w_attack": (0.0, 2.5),
    "w_defend": (0.0, 2.5),
    "incumbent_bonus": (0.0, 0.15),
    "open_seat_volatility": (0.0, 0.12),
    "budget_millions": (1.0, 500.0),
}

# Civic API
CIVIC_BASE = "https://www.googleapis.com/civicinfo/v2"
CIVIC_CACHE_TTL_HOURS = 24

# OpenFEC API — https://api.open.fec.gov/developers/
FEC_BASE = "https://api.open.fec.gov/v1"
FEC_CACHE_TTL_HOURS = 24
FEC_DEFAULT_CYCLES = (2022, 2024, 2026)

APP_TITLE = "House Moneyball"
APP_SUBTITLE = (
    "Republican Investment Value Score (RIVS) — find high-ROI House districts "
    "toward a comfortable 230-seat majority."
)
