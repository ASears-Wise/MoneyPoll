"""
House Moneyball — Streamlit dashboard
Republican Investment Value Score (RIVS) for U.S. House districts.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_TARGET_SEATS,
    RIVS_BOUNDS,
    RIVS_DEFAULTS,
    TOTAL_HOUSE_SEATS,
)
from src.charts import demographics_pie, fec_bars, parse_candidates, top_rivs_bar  # noqa: E402
from src.civic_client import get_api_key as get_civic_key  # noqa: E402
from src.civic_client import get_elections, safe_civic_call  # noqa: E402
from src.fec_client import get_api_key as get_fec_key  # noqa: E402
from src.data_loader import list_states, load_geojson, load_master  # noqa: E402
from src.map_builder import build_district_map, build_simple_state_centroids_map  # noqa: E402
from src.rivs import budget_simulation, compute_rivs, format_money  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading district master data…")
def _cached_master() -> tuple[pd.DataFrame, str]:
    return load_master()


@st.cache_data(show_spinner=False)
def _cached_geojson() -> dict | None:
    return load_geojson()


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.title("⚾ Controls")
    st.sidebar.caption("Filters & RIVS algorithm knobs")

    states = list_states(df)
    sel_states = st.sidebar.multiselect("State(s)", states, default=[])
    party = st.sidebar.multiselect(
        "Party control",
        ["R", "D", "VACANT"],
        default=["R", "D", "VACANT"],
    )
    modes = st.sidebar.multiselect(
        "Mode",
        ["attack", "defend", "abandon", "safe"],
        default=["attack", "defend", "abandon", "safe"],
        help=(
            "attack = flip targets · defend = protect holds · "
            "abandon = competitive but capital is better spent across many races · "
            "safe = deep seats"
        ),
    )
    rivs_min = st.sidebar.slider("Min RIVS (after compute)", 0.0, 50.0, 0.0, 0.5)
    competitive_only = st.sidebar.checkbox(
        "Map: competitive only (attack/defend)",
        False,
        help="Hides safe and abandon on the map layer filter",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("RIVS parameters")
    with st.sidebar.expander("What is RIVS?", expanded=False):
        st.markdown(
            r"""
**RIVS** = \(\frac{\text{Expected Prob Gain} \times \text{Seat Priority}}{\text{Incremental Cost}}\)

- **Expected Prob Gain**: how much R win probability you can buy toward the risk threshold  
- **Seat Priority**: marginal seats + path to majority + attack/defend weights  
- **Incremental Cost**: FEC-based cost to compete (+ outside spend), scaled by difficulty  
- **Long-term factor**: open seats / infrastructure bonus  
- **Abandon**: competitive seats where $ is better split across multiple cheaper races  
            """
        )

    b = RIVS_BOUNDS
    d = RIVS_DEFAULTS
    cost_sensitivity = st.sidebar.slider(
        "Cost sensitivity",
        b["cost_sensitivity"][0],
        b["cost_sensitivity"][1],
        d["cost_sensitivity"],
        0.05,
        help="Higher = treat dollars as more expensive → lower RIVS for costly districts",
    )
    long_term = st.sidebar.slider(
        "Long-term multiplier",
        b["long_term_multiplier"][0],
        b["long_term_multiplier"][1],
        d["long_term_multiplier"],
        0.05,
        help="Boost open seats / long-term infrastructure value",
    )
    risk = st.sidebar.slider(
        "Risk tolerance (target P*)",
        b["risk_tolerance"][0],
        b["risk_tolerance"][1],
        d["risk_tolerance"],
        0.01,
        help="Target R win probability threshold for 'enough' investment",
    )
    w_attack = st.sidebar.slider(
        "Attack weight (flips)",
        b["w_attack"][0],
        b["w_attack"][1],
        d["w_attack"],
        0.05,
    )
    w_defend = st.sidebar.slider(
        "Defend weight (holds)",
        b["w_defend"][0],
        b["w_defend"][1],
        d["w_defend"],
        0.05,
    )
    inc_bonus = st.sidebar.slider(
        "Incumbent bonus",
        b["incumbent_bonus"][0],
        b["incumbent_bonus"][1],
        d["incumbent_bonus"],
        0.01,
    )
    open_vol = st.sidebar.slider(
        "Open-seat volatility",
        b["open_seat_volatility"][0],
        b["open_seat_volatility"][1],
        d["open_seat_volatility"],
        0.01,
    )

    st.sidebar.markdown("##### Abandon (don't double-down)")
    abandon_enabled = st.sidebar.checkbox(
        "Enable abandon mode",
        value=bool(d["abandon_enabled"]),
        help="Flag races where concentrating spend is untenable vs spreading $",
    )
    abandon_pct = st.sidebar.slider(
        "Abandon cost/gain percentile",
        b["abandon_cost_percentile"][0],
        b["abandon_cost_percentile"][1],
        d["abandon_cost_percentile"],
        0.01,
        help="Competitive seats above this $/ΔP percentile become abandon (if also ≥ min cost)",
        disabled=not abandon_enabled,
    )
    abandon_min_m = st.sidebar.slider(
        "Abandon min cost ($M)",
        b["abandon_min_cost_m"][0],
        b["abandon_min_cost_m"][1],
        d["abandon_min_cost_m"],
        0.5,
        help="Never abandon races cheaper than this absolute floor",
        disabled=not abandon_enabled,
    )
    abandon_alts = st.sidebar.slider(
        "Opportunity: alt races (N)",
        b["abandon_alt_races"][0],
        b["abandon_alt_races"][1],
        int(d["abandon_alt_races"]),
        1,
        help="If this seat costs ≥ N × median competitive race, compare gains",
        disabled=not abandon_enabled,
    )
    abandon_gain_ratio = st.sidebar.slider(
        "Opportunity: gain ratio",
        b["abandon_alt_gain_ratio"][0],
        b["abandon_alt_gain_ratio"][1],
        d["abandon_alt_gain_ratio"],
        0.05,
        help="Abandon if N median races yield more than this seat × ratio",
        disabled=not abandon_enabled,
    )

    target_seats = st.sidebar.number_input(
        "Target R seats",
        min_value=218,
        max_value=TOTAL_HOUSE_SEATS,
        value=DEFAULT_TARGET_SEATS,
        step=1,
    )
    budget_m = st.sidebar.slider(
        "Budget simulation ($M)",
        b["budget_millions"][0],
        b["budget_millions"][1],
        d["budget_millions"],
        1.0,
    )
    skip_abandon_budget = st.sidebar.checkbox(
        "Budget sim: skip abandon seats",
        value=True,
        help="Do not allocate simulated dollars into abandon races",
    )
    fec_cycle = st.sidebar.selectbox("FEC cycle (detail charts)", ["2026", "2024", "2022"], index=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data APIs")
    fec_ok = get_fec_key() is not None
    civic_ok = get_civic_key() is not None
    st.sidebar.write("OpenFEC key:", "✅ set" if fec_ok else "⚪ not set")
    st.sidebar.write("Google Civic key:", "✅ set" if civic_ok else "⚪ not set (optional)")
    st.sidebar.caption(
        "Refresh finance: `python scripts/fetch_fec_data.py` "
        "([OpenFEC docs](https://api.open.fec.gov/developers/))"
    )
    if st.sidebar.button("List Civic elections (live)"):
        if not civic_ok:
            st.sidebar.warning("Set GOOGLE_CIVIC_API_KEY in .env")
        else:
            result, err = safe_civic_call(get_elections)
            if err:
                st.sidebar.error(err)
            else:
                st.sidebar.success(f"{len(result or [])} elections returned")
                st.sidebar.json((result or [])[:5])

    return {
        "sel_states": sel_states,
        "party": party,
        "modes": modes,
        "rivs_min": rivs_min,
        "competitive_only": competitive_only,
        "cost_sensitivity": cost_sensitivity,
        "long_term": long_term,
        "risk": risk,
        "w_attack": w_attack,
        "w_defend": w_defend,
        "inc_bonus": inc_bonus,
        "open_vol": open_vol,
        "target_seats": int(target_seats),
        "budget_m": budget_m,
        "fec_cycle": fec_cycle,
        "abandon_enabled": abandon_enabled,
        "abandon_cost_percentile": abandon_pct,
        "abandon_min_cost_m": abandon_min_m,
        "abandon_alt_races": abandon_alts,
        "abandon_alt_gain_ratio": abandon_gain_ratio,
        "skip_abandon_budget": skip_abandon_budget,
    }


def apply_filters(ranked: pd.DataFrame, ctl: dict) -> pd.DataFrame:
    out = ranked.copy()
    if ctl["sel_states"]:
        out = out[out["state"].isin(ctl["sel_states"])]
    if ctl["party"]:
        out = out[out["party_control"].isin(ctl["party"])]
    if ctl["modes"]:
        out = out[out["mode"].isin(ctl["modes"])]
    out = out[out["rivs"] >= ctl["rivs_min"]]
    return out.reset_index(drop=True)


def render_detail(row: pd.Series, fec_cycle: str) -> None:
    st.subheader(f"{row['district_id']} — detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RIVS", f"{float(row['rivs']):.2f}", help=f"Rank #{int(row['rivs_rank'])}")
    c2.metric("PVI (R+)", f"{float(row['pvi']):+.1f}")
    c3.metric("R win P₀", f"{float(row['baseline_win_prob_r']):.1%}")
    mode = str(row["mode"]).lower()
    c4.metric("Mode", mode.title())

    if mode == "abandon":
        st.warning(
            "**Abandon** — concentrating more money here is likely untenable. "
            "Redeploy toward multiple higher-RIVS attack/defend seats instead."
        )
        reason = row.get("abandon_reason") or ""
        if reason:
            st.caption(f"Why: {reason}")
        base = row.get("base_mode")
        if base and str(base) != mode:
            st.caption(f"Underlying strategic type before abandon: **{base}**")

    st.markdown(
        f"**Representative:** {row.get('rep_name', '—')} "
        f"({row.get('rep_party', '—')}) · "
        f"**Rating:** {row.get('cook_rating', '—')} · "
        f"**Open seat:** {'Yes' if row.get('is_open_seat') else 'No'}"
    )
    if pd.notna(row.get("first_elected")):
        st.caption(
            f"First elected: {int(row['first_elected'])} · "
            f"Tenure ≈ {float(row.get('tenure_years') or 0):.0f} yrs · "
            f"Incumbent running: {bool(row.get('incumbent_running'))}"
        )

    st.markdown("##### Demographics (ACS-style)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Population", f"{int(row['pop_total']):,}")
    d2.metric("VAP", f"{int(row['vap']):,}")
    d3.metric("Median income", f"${float(row['median_income']):,.0f}")
    d4.metric("BA+", f"{float(row['pct_ba_plus']):.1%}")
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(demographics_pie(row), use_container_width=True)
    with e2:
        st.write(
            {
                "pct_urban": f"{float(row['pct_urban']):.1%}",
                "pct_white": f"{float(row['pct_white']):.1%}",
                "pct_black": f"{float(row['pct_black']):.1%}",
                "pct_hispanic": f"{float(row['pct_hispanic']):.1%}",
                "pct_asian": f"{float(row['pct_asian']):.1%}",
            }
        )

    st.markdown("##### 2026 race / candidates")
    cands = parse_candidates(row)
    if cands:
        st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)
    else:
        st.info("No candidate payload in master row (mock or Civic cache empty).")

    st.markdown("##### RIVS breakdown")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Prob gain", f"{float(row['expected_prob_gain']):.3f}")
    b2.metric("Seat priority", f"{float(row['seat_priority']):.3f}")
    b3.metric("Incremental cost", format_money(float(row["incremental_cost"])))
    b4.metric("Long-term factor", f"{float(row['long_term_factor']):.2f}")

    st.markdown(f"##### FEC finance — {fec_cycle}")
    st.plotly_chart(fec_bars(row, cycle=fec_cycle), use_container_width=True)


def main() -> None:
    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")

    raw, source_label = _cached_master()
    geo = _cached_geojson()
    ctl = sidebar_controls(raw)

    ranked = compute_rivs(
        raw,
        cost_sensitivity=ctl["cost_sensitivity"],
        long_term_multiplier=ctl["long_term"],
        risk_tolerance=ctl["risk"],
        w_attack=ctl["w_attack"],
        w_defend=ctl["w_defend"],
        incumbent_bonus=ctl["inc_bonus"],
        open_seat_volatility=ctl["open_vol"],
        target_seats=ctl["target_seats"],
        abandon_enabled=ctl["abandon_enabled"],
        abandon_cost_percentile=ctl["abandon_cost_percentile"],
        abandon_min_cost_m=ctl["abandon_min_cost_m"],
        abandon_alt_races=ctl["abandon_alt_races"],
        abandon_alt_gain_ratio=ctl["abandon_alt_gain_ratio"],
    )
    filtered = apply_filters(ranked, ctl)
    sim = budget_simulation(
        ranked,
        ctl["budget_m"] * 1_000_000,
        skip_abandon=ctl["skip_abandon_budget"],
    )

    # KPIs
    r_held = int((ranked["party_control"] == "R").sum())
    n_abandon = int((ranked["mode"] == "abandon").sum())
    n_attack = int((ranked["mode"] == "attack").sum())
    n_defend = int((ranked["mode"] == "defend").sum())
    baseline_seats = sim["baseline_expected_r_seats"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("R-held (control)", r_held)
    k2.metric("Expected R seats (P₀ sum)", f"{baseline_seats:.1f}")
    k3.metric(
        f"After ${ctl['budget_m']:.0f}M top-RIVS",
        f"{sim['expected_r_seats_after']:.1f}",
        delta=f"+{sim['total_prob_gain']:.2f} seats",
    )
    k4.metric("Attack / Defend", f"{n_attack} / {n_defend}")
    k5.metric("Abandon", n_abandon, help="High-cost races — redeploy $ elsewhere")
    k6.metric("Districts shown", len(filtered))

    st.caption(
        f"Data: `{source_label}` · "
        f"Budget sim funds **{sim['n_districts_funded']}** districts "
        f"({format_money(sim['spent_usd'])} spent"
        + (
            f", skipped {sim.get('skipped_abandon', 0)} abandon"
            if ctl["skip_abandon_budget"]
            else ""
        )
        + "). "
        f"GeoJSON: {'loaded' if geo else 'missing — using state centroids fallback'}."
    )

    if "selected_district" not in st.session_state:
        st.session_state.selected_district = (
            str(filtered.iloc[0]["district_id"]) if len(filtered) else None
        )

    tab_map, tab_table, tab_method = st.tabs(["🗺️ Map & detail", "📋 Table", "📖 Methodology"])

    with tab_map:
        left, right = st.columns([1.35, 1.0])
        with left:
            st.markdown("##### RIVS choropleth")
            search = st.text_input("Jump to district (e.g. AZ-01)", "")
            if search.strip():
                q = search.strip().upper()
                if q in set(ranked["district_id"]):
                    st.session_state.selected_district = q

            pick = st.selectbox(
                "Select district",
                filtered["district_id"].tolist() if len(filtered) else ranked["district_id"].tolist(),
                index=0,
            )
            if pick:
                st.session_state.selected_district = pick

            try:
                from streamlit_folium import st_folium

                if geo is not None:
                    fmap = build_district_map(
                        filtered if len(filtered) else ranked,
                        geo,
                        selected=st.session_state.selected_district,
                        show_competitive_only=ctl["competitive_only"],
                    )
                else:
                    fmap = build_simple_state_centroids_map(
                        filtered if len(filtered) else ranked
                    )
                    st.info(
                        "District GeoJSON not found at `data/raw/cds.geojson`. "
                        "Showing state-level RIVS markers. "
                        "Run `python scripts/download_geojson.py` for full choropleth."
                    )
                st_folium(fmap, width=None, height=520, returned_objects=[])
            except ImportError:
                st.warning("streamlit-folium not installed; map disabled.")

            st.plotly_chart(
                top_rivs_bar(filtered if len(filtered) else ranked, n=15),
                use_container_width=True,
            )

        with right:
            sel = st.session_state.selected_district
            match = ranked[ranked["district_id"] == sel]
            if len(match):
                render_detail(match.iloc[0], ctl["fec_cycle"])
            else:
                st.warning("Select a district.")

        st.markdown("##### Budget simulation — funded districts")
        if sim["funded"]:
            fund_df = pd.DataFrame(sim["funded"])
            fund_df["spend"] = fund_df["spend"].map(format_money)
            fund_df["gain"] = fund_df["gain"].map(lambda x: f"{x:.3f}")
            st.dataframe(fund_df, use_container_width=True, hide_index=True)
        else:
            st.write("No districts funded at this budget / filter combination.")

    with tab_table:
        show_cols = [
            "rivs_rank",
            "district_id",
            "state",
            "party_control",
            "mode",
            "base_mode",
            "rivs",
            "pvi",
            "cook_rating",
            "baseline_win_prob_r",
            "expected_prob_gain",
            "incremental_cost",
            "cost_per_gain",
            "seat_priority",
            "rep_name",
            "is_open_seat",
            "hist_cost_to_compete",
            "abandon_reason",
            "median_income",
            "pop_total",
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        view = filtered[show_cols].copy()
        st.dataframe(
            view.style.format(
                {
                    "rivs": "{:.2f}",
                    "pvi": "{:+.1f}",
                    "baseline_win_prob_r": "{:.1%}",
                    "expected_prob_gain": "{:.3f}",
                    "incremental_cost": "${:,.0f}",
                    "seat_priority": "{:.3f}",
                    "hist_cost_to_compete": "${:,.0f}",
                    "median_income": "${:,.0f}",
                    "pop_total": "{:,.0f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            height=560,
        )
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv,
            file_name="house_moneyball_filtered.csv",
            mime="text/csv",
        )

    with tab_method:
        st.markdown(
            """
### Republican Investment Value Score (RIVS)

$$
\\mathrm{RIVS} = \\frac{\\mathrm{Expected\\_Probability\\_Gain} \\times \\mathrm{Seat\\_Priority} \\times \\mathrm{Long\\_Term}}{\\mathrm{Incremental\\_Cost}}
$$

| Component | Construction |
|-----------|----------------|
| Baseline win prob \(P_0\) | Logistic of PVI + incumbent bonus/penalty + open-seat blend |
| Expected gain | Attack: \(\\max(0, P^*-P_0)\); Defend: fortify toward buffer / \(P^*\) |
| Seat priority | Marginality × path-to-target majority × attack/defend weights (abandon ≈ 0) |
| Incremental cost | FEC competitive proxy + 25% of 2024 outside spend × difficulty |
| Long-term | Multiplier for open seats / fragile opposition |
| **Abandon** | Competitive seats with extreme $/ΔP **or** opportunity cost vs N median races — redeploy, do not double-down |

**Budget simulation:** Greedy fill from highest RIVS until budget exhausted (skips abandon/safe by default); expected seats ≈ \(\\sum P_0 + \\sum \\Delta P_{\\mathrm{funded}}\).

### Data sources

| Layer | Source |
|-------|--------|
| Members / party | Static roster (or mock). **Google Civic Representatives API was turned down April 2025.** |
| Elections / contests | Google Civic `electionQuery` / `voterInfoQuery` when VIP has data; cached under `data/cache/` |
| Boundaries | UCLA CDMaps / Jeffrey B. Lewis GeoJSON → `data/raw/cds.geojson` |
| Demographics | Census ACS 5-year by CD (or mock) |
| FEC | OpenFEC API (`scripts/fetch_fec_data.py`) → per-district raised/spent + IE |
| Lean | PVI-like scores (mock or your CSV) — not a licensed Cook product unless you supply one |

### Limitations

1. **Mock-first defaults** are synthetic and for UI/algorithm demo — not actionable campaign intelligence.  
2. **PVI / ratings** here are approximations unless you replace them with licensed or official ratings.  
3. **RIVS is a heuristic**, not a causal model of persuasion or turnout; FEC spend ≠ vote share.  
4. **Civic coverage** for 2026 House contests is incomplete until VIP publishes election feeds.  
5. **GeoJSON** must be downloaded separately for true 435-district choropleths.  
6. **Target 230** is a strategic narrative knob, not a forecast.

### Update TODOs

- [ ] Drop real PVI / Cook or equivalent CSV into the build script  
- [x] OpenFEC API → district FEC aggregates (`src/fec_client.py`)  

- [ ] Pull ACS via Census API (`CENSUS_API_KEY`) by congressional district  
- [ ] Optional: sample Civic `voterInfoQuery` per district once 2026 VIP elections appear  
- [ ] Refresh legislator YAML from `unitedstates/congress-legislators`
            """
        )

    st.markdown("---")
    st.caption(
        "House Moneyball · local Streamlit tool · not affiliated with MLB Moneyball, "
        "Cook Political Report, or any campaign committee."
    )


if __name__ == "__main__":
    main()
