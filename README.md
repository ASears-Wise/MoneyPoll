# House Moneyball ⚾

Interactive **Streamlit** dashboard that ranks all **435 U.S. House districts** by a tunable **Republican Investment Value Score (RIVS)** — highlighting high-ROI *attack* (flips) and *defend* (holds) opportunities, plus **abandon** races where concentrating spend is untenable versus spreading capital across multiple cheaper seats — toward a comfortable **230-seat** Republican majority.

Runs **locally on macOS** with **mock data out of the box** (no API keys required). Optional Google Civic Information API integration for election/contest data.

---

## Features

1. **Interactive map** — Folium choropleth by RIVS (or state-centroid fallback without GeoJSON)
2. **District detail** — demographics, representative, 2026 candidates, FEC charts by cycle
3. **Sortable table** + CSV export
4. **RIVS algorithm** with sidebar sliders + **budget simulation**
5. **Filters** — state, party, mode, RIVS threshold, FEC cycle

---

## RIVS formula

\[
\mathrm{RIVS} = \frac{\mathrm{Expected\_Probability\_Gain} \times \mathrm{Seat\_Priority} \times \mathrm{Long\_Term}}{\mathrm{Incremental\_Cost}}
\]

| Piece | Meaning |
|-------|---------|
| Baseline \(P_0\) | Logistic of PVI + incumbent / open-seat adjustments |
| Expected gain | Distance toward target win probability \(P^*\) (risk tolerance) |
| Seat priority | Marginal seats × path to target majority × attack/defend weights |
| Incremental cost | Historical competitive FEC proxy (+ outside) × difficulty × cost sensitivity |
| Long-term | Bonus for open seats / infrastructure |
| **Abandon** | Competitive but capital-intensive: high $/ΔP **or** opportunity cost vs N median races — redeploy |

**Budget sim:** greedily fund top RIVS districts until \$X is spent (skips abandon by default); report expected R seats ≈ \(\sum P_0 + \sum \Delta P\).

---

## macOS setup

### 1. Python 3.11+

```bash
brew install python@3.11
# or: pyenv install 3.11.9 && pyenv local 3.11.9
```

### 2. Clone / enter project

```bash
cd "/Users/alec/Coding Projects/moneypoll"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Mock data (auto on first app load, or explicit)

```bash
python scripts/generate_mock_data.py
```

### 4. API keys (`.env`)

```bash
cp .env.example .env
# edit .env
```

| Variable | Purpose |
|----------|---------|
| `FEC_API_KEY` | [OpenFEC](https://api.open.fec.gov/developers/) / [api.data.gov](https://api.data.gov/signup/) key for candidate receipts, disbursements, and IE |
| `GOOGLE_CIVIC_API_KEY` | Optional — elections / voterInfo only |

**Streamlit Community Cloud:** put the same keys under **App settings → Secrets** (TOML). The app reads `st.secrets` as well as env vars:

```toml
FEC_API_KEY = "your-key"
# GOOGLE_CIVIC_API_KEY = "optional"
```

**OpenFEC pull (historical + ongoing House cycles):**

```bash
# 2022, 2024, 2026 House candidate totals + 2024 outside spending
python scripts/fetch_fec_data.py

# Faster: skip independent expenditures
python scripts/fetch_fec_data.py --no-outside

# Custom cycles
python scripts/fetch_fec_data.py --cycles 2024 2026 --no-outside
```

This writes `data/raw/fec_district_finance.parquet` and updates `data/master.parquet` (`fec_raised_*`, `fec_spent_*`, `hist_cost_to_compete`, 2026 candidate lists). Responses are cached under `data/cache/`.

**Important:** Google **turned down the Representatives API in April 2025**. Members come from the master table (mock or roster overlay). Civic is only for **elections / voterInfo** (`src/civic_client.py`).

### 5. Optional: district boundaries (full choropleth)

```bash
python scripts/download_geojson.py
# saves data/raw/cds.geojson when the public URL succeeds
```

Manual fallback: download CD GeoJSON from [UCLA CDMaps](https://cdmaps.polisci.ucla.edu/) or Census TIGER cartographic boundaries and save as `data/raw/cds.geojson`. Join keys: `district_id` / `STATE`+`CD*` / `GEOID`.

### 6. Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## Project layout

```
moneypoll/
├── app.py                 # Streamlit UI
├── config.py
├── requirements.txt
├── src/
│   ├── rivs.py            # RIVS + budget simulation
│   ├── data_loader.py
│   ├── map_builder.py
│   ├── charts.py
│   └── civic_client.py    # Google Civic helpers + cache
├── scripts/
│   ├── generate_mock_data.py
│   ├── download_geojson.py
│   └── build_master_dataset.py
└── data/
    ├── mock/
    ├── raw/               # GeoJSON, FEC zips, overlays
    └── cache/             # Civic responses
```

---

## Master dataframe schema

One row per district (`district_id` like `PA-08`, at-large `AK-00`).

| Column | Description |
|--------|-------------|
| `district_id`, `state`, `district_num`, `geoid` | Identifiers |
| `party_control`, `rep_name`, `rep_party`, `first_elected`, `tenure_years` | Member |
| `pvi`, `cook_rating` | Lean (signed PVI: R positive) |
| `incumbent_running`, `is_open_seat`, `mode` | Race context (`attack`/`defend`/`abandon`/`safe`) |
| `fec_*_{2022,2024,2026}` | Raised/spent by side; `fec_outside_2024` |
| `hist_cost_to_compete` | Competitive spend proxy for RIVS cost |
| `pop_total`, `vap`, `median_income`, `pct_*`, `pct_ba_plus`, `pct_urban` | Demographics |
| `civic_election_id`, `civic_candidates_json` | Civic / mock contest payload |
| `data_source_flags` | Provenance |

RIVS columns (`rivs`, `rivs_rank`, `baseline_win_prob_r`, …) are **computed at runtime** from sidebar parameters.

### Overlay real data

```bash
# CSV: district_id,pvi[,cook_rating]
python scripts/build_master_dataset.py --pvi data/raw/pvi.csv

# JSON list of {district_id, rep_name, rep_party, first_elected, party_control}
python scripts/build_master_dataset.py --legislators data/raw/legislators.json

# Probe Civic elections list (requires API key)
python scripts/build_master_dataset.py --civic-probe
```

---

## Google Civic — code examples

```python
from src.civic_client import get_elections, get_voter_info, extract_house_contests

elections = get_elections()  # needs GOOGLE_CIVIC_API_KEY
info = get_voter_info("100 Main St, Phoenix AZ", election_id=elections[0]["id"])
house = extract_house_contests(info)
```

Responses are cached under `data/cache/` (TTL in `config.py`). Rate limiting is client-side.

---

## Limitations

- Default dataset is **synthetic** — for product/demo use until you wire real feeds.
- RIVS is a **heuristic**, not a causal persuasion model.
- PVI/ratings are **not** official Cook products unless you supply licensed data.
- Civic midterm House contest coverage depends on VIP publication.
- OpenFEC coverage depends on filings; early-cycle 2026 totals can be sparse.

---

## License / disclaimer

For research and strategy exploration. Not affiliated with any campaign, party committee, Cook Political Report, or MLB. Campaign finance figures (when real) should be verified against [FEC.gov](https://www.fec.gov/).
