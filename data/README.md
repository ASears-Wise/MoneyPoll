# Data layout

| Path | Purpose |
|------|---------|
| `mock/master.parquet` | Synthetic 435-district master (always regenerable) |
| `master.parquet` | Active master used by the app (mock or built) |
| `raw/cds.geojson` | Congressional district boundaries |
| `cache/` | Google Civic API response cache |

## Master schema (one row per district)

See plan / README. Key fields:

- `district_id` — e.g. `TX-07`, at-large `WY-00`
- `party_control`, `rep_name`, `pvi`, `mode` (`attack`/`defend`/`safe`)
- FEC columns: `fec_raised_r_2024`, `fec_spent_d_2022`, …
- ACS-style demographics: `pop_total`, `vap`, `pct_*`, `median_income`
- `civic_candidates_json` — JSON blob of contest candidates
- `data_source_flags` — provenance (`mock|pvi_csv|…`)

## OpenFEC refresh

```bash
# Candidate receipts/disbursements (2022/2024/2026) + 2024 IE outside
python scripts/fetch_fec_data.py

# Faster re-run without IE
python scripts/fetch_fec_data.py --no-outside
```

Requires `FEC_API_KEY` in `.env`. Intermediate files: `raw/fec_district_finance.parquet`, `raw/fec_outside.parquet`.

## Refresh TODOs

1. Replace mock PVI with licensed/public lean ratings  
2. Census ACS 5-year by CD  
3. Legislators from unitedstates/congress-legislators  
4. Civic voterInfo when 2026 VIP elections are published  

