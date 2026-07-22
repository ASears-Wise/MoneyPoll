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

# State legislative data
STATE_DATA_DIR = DATA_DIR / "state"
STATE_MOCK_DIR = STATE_DATA_DIR / "mock"
STATE_RAW_DIR = STATE_DATA_DIR / "raw"
STATE_CACHE_DIR = STATE_DATA_DIR / "cache"
STATE_LOWER_PARQUET = STATE_DATA_DIR / "lower_master.parquet"
STATE_UPPER_PARQUET = STATE_DATA_DIR / "upper_master.parquet"
STATE_MOCK_LOWER = STATE_MOCK_DIR / "lower_master.parquet"
STATE_MOCK_UPPER = STATE_MOCK_DIR / "upper_master.parquet"

# OpenStates API
OPENSTATES_BASE = "https://v3.openstates.org"
OPENSTATES_CACHE_TTL_HOURS = 24

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
    # Abandon: high-cost races where $ is better spread across multiple alternatives
    "abandon_enabled": True,
    "abandon_cost_percentile": 0.82,  # flag competitive seats above this $/gain percentile
    "abandon_min_cost_m": 6.0,  # $M floor — won't abandon cheap races even if inefficient
    "abandon_alt_races": 3,  # if this seat's cost ≥ N × median competitive cost…
    "abandon_alt_gain_ratio": 1.15,  # …and N median seats yield more gain → abandon
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
    "abandon_cost_percentile": (0.60, 0.98),
    "abandon_min_cost_m": (1.0, 25.0),
    "abandon_alt_races": (2, 8),
    "abandon_alt_gain_ratio": (1.0, 2.5),
}

# Civic API
CIVIC_BASE = "https://www.googleapis.com/civicinfo/v2"
CIVIC_CACHE_TTL_HOURS = 24

# OpenFEC API — https://api.open.fec.gov/developers/
FEC_BASE = "https://api.open.fec.gov/v1"
FEC_CACHE_TTL_HOURS = 24
# Current election cycle first (display + hist-cost preference)
DEFAULT_FEC_CYCLE = 2026
FEC_DEFAULT_CYCLES = (2026, 2024, 2022)

APP_TITLE = "House Moneyball"
APP_SUBTITLE = (
    "Republican Investment Value Score (RIVS) — federal House and state legislative "
    "chambers: attack, defend, or abandon high-cost races on the path to majority."
)

# Approximate chamber sizes (post-2020 maps). NE unicameral → lower only.
STATE_CHAMBER_SEATS: dict[str, dict[str, int]] = {
    "AL": {"lower": 105, "upper": 35},
    "AK": {"lower": 40, "upper": 20},
    "AZ": {"lower": 60, "upper": 30},
    "AR": {"lower": 100, "upper": 35},
    "CA": {"lower": 80, "upper": 40},
    "CO": {"lower": 65, "upper": 35},
    "CT": {"lower": 151, "upper": 36},
    "DE": {"lower": 41, "upper": 21},
    "FL": {"lower": 120, "upper": 40},
    "GA": {"lower": 180, "upper": 56},
    "HI": {"lower": 51, "upper": 25},
    "ID": {"lower": 70, "upper": 35},
    "IL": {"lower": 118, "upper": 59},
    "IN": {"lower": 100, "upper": 50},
    "IA": {"lower": 100, "upper": 50},
    "KS": {"lower": 125, "upper": 40},
    "KY": {"lower": 100, "upper": 38},
    "LA": {"lower": 105, "upper": 39},
    "ME": {"lower": 151, "upper": 35},
    "MD": {"lower": 141, "upper": 47},
    "MA": {"lower": 160, "upper": 40},
    "MI": {"lower": 110, "upper": 38},
    "MN": {"lower": 134, "upper": 67},
    "MS": {"lower": 122, "upper": 52},
    "MO": {"lower": 163, "upper": 34},
    "MT": {"lower": 100, "upper": 50},
    "NE": {"lower": 49, "upper": 0},  # unicameral
    "NV": {"lower": 42, "upper": 21},
    "NH": {"lower": 400, "upper": 24},
    "NJ": {"lower": 80, "upper": 40},
    "NM": {"lower": 70, "upper": 42},
    "NY": {"lower": 150, "upper": 63},
    "NC": {"lower": 120, "upper": 50},
    "ND": {"lower": 94, "upper": 47},
    "OH": {"lower": 99, "upper": 33},
    "OK": {"lower": 101, "upper": 48},
    "OR": {"lower": 60, "upper": 30},
    "PA": {"lower": 203, "upper": 50},
    "RI": {"lower": 75, "upper": 38},
    "SC": {"lower": 124, "upper": 46},
    "SD": {"lower": 70, "upper": 35},
    "TN": {"lower": 99, "upper": 33},
    "TX": {"lower": 150, "upper": 31},
    "UT": {"lower": 75, "upper": 29},
    "VT": {"lower": 150, "upper": 30},
    "VA": {"lower": 100, "upper": 40},
    "WA": {"lower": 98, "upper": 49},
    "WV": {"lower": 100, "upper": 34},
    "WI": {"lower": 99, "upper": 33},
    "WY": {"lower": 62, "upper": 31},
}

# Rough state presidential lean proxy (R positive) for synthetic district PVI
STATE_LEAN_PROXY: dict[str, float] = {
    "AL": 25, "AK": 10, "AZ": 3, "AR": 27, "CA": -22, "CO": -8, "CT": -14, "DE": -15,
    "FL": 3, "GA": 2, "HI": -30, "ID": 30, "IL": -14, "IN": 16, "IA": 8, "KS": 15,
    "KY": 26, "LA": 19, "ME": -2, "MD": -26, "MA": -30, "MI": 1, "MN": -2, "MS": 17,
    "MO": 15, "MT": 16, "NE": 20, "NV": 0, "NH": -1, "NJ": -12, "NM": -8, "NY": -18,
    "NC": 2, "ND": 33, "OH": 8, "OK": 32, "OR": -12, "PA": 1, "RI": -20, "SC": 14,
    "SD": 28, "TN": 23, "TX": 6, "UT": 20, "VT": -32, "VA": -5, "WA": -14, "WV": 39,
    "WI": 1, "WY": 43,
}
