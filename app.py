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
from src.fec_pipeline import fetch_and_merge_fec  # noqa: E402
from src.data_loader import list_states, load_geojson, load_master  # noqa: E402
from src.map_builder import build_district_map, build_simple_state_centroids_map  # noqa: E402
from src.formatters import (  # noqa: E402
    COLUMN_HELP,
    format_display_frame,
    fmt_int,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_pvi,
)
from src.rivs import budget_simulation, compute_rivs, format_money  # noqa: E402
from src.ui_state import render_state_chamber_tab  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cache OpenFEC pull for 12h so Streamlit Cloud does not re-hit the API every rerun.
# include_outside=True is much slower (per-district IE); default off for cloud.
@st.cache_data(ttl=12 * 3600, show_spinner="Fetching OpenFEC House finance (may take several minutes)…")
def _cached_fec_master(include_outside: bool, cache_bust: int) -> tuple[pd.DataFrame, str, dict]:
    """cache_bust increments to force a manual refresh."""
    return fetch_and_merge_fec(
        cycles=[2022, 2024, 2026],
        include_outside=include_outside,
        progress=False,
        persist=True,
    )


@st.cache_data(show_spinner="Loading district master data…")
def _cached_disk_master() -> tuple[pd.DataFrame, str]:
    return load_master()


@st.cache_data(show_spinner=False)
def _cached_geojson() -> dict | None:
    return load_geojson()


def _master_is_mock_only(df: pd.DataFrame, label: str) -> bool:
    flags = (
        df["data_source_flags"].astype(str)
        if "data_source_flags" in df.columns
        else pd.Series([""] * len(df))
    )
    has_openfec = flags.str.contains("openfec", case=False, na=False).any()
    if has_openfec:
        return False
    return "mock" in label.lower() or flags.str.contains("mock", case=False, na=False).any()


def resolve_master(
    *,
    force_refresh: bool = False,
    include_outside: bool = False,
    auto_fetch_if_mock: bool = True,
) -> tuple[pd.DataFrame, str, dict | None]:
    """
    Prefer OpenFEC-backed master when FEC_API_KEY is available (env or Streamlit secrets).

    On Streamlit Cloud: auto-pull once (cached 12h) if disk data is still mock-only.
    """
    fec_ok = get_fec_key() is not None
    stats = None

    if "fec_cache_bust" not in st.session_state:
        st.session_state.fec_cache_bust = 0
    if force_refresh:
        st.session_state.fec_cache_bust += 1
        _cached_fec_master.clear()
        _cached_disk_master.clear()

    if fec_ok and force_refresh:
        df, label, stats = _cached_fec_master(include_outside, st.session_state.fec_cache_bust)
        return df, label, stats

    # Disk / mock first (fast path)
    try:
        disk_df, disk_label = _cached_disk_master()
    except Exception:
        disk_df, disk_label = load_master()

    if fec_ok and auto_fetch_if_mock and _master_is_mock_only(disk_df, disk_label):
        df, label, stats = _cached_fec_master(include_outside, st.session_state.fec_cache_bust)
        return df, label, stats

    # Key set and master already has openfec from prior persist
    if fec_ok and not _master_is_mock_only(disk_df, disk_label):
        return disk_df, disk_label, None

    return disk_df, disk_label, None


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.title("⚾ Controls")
    st.sidebar.caption("Hover (?) icons for brief explainers")

    states = list_states(df)
    sel_states = st.sidebar.multiselect(
        "State(s)",
        states,
        default=[],
        help="Limit the map and tables to selected states. Leave empty for all 50 + DC districts in the master.",
    )
    party = st.sidebar.multiselect(
        "Party control",
        ["R", "D", "VACANT"],
        default=["R", "D", "VACANT"],
        help="Filter by party currently holding the seat (R = Republican, D = Democrat).",
    )
    modes = st.sidebar.multiselect(
        "Mode",
        ["attack", "defend", "abandon", "safe"],
        default=["attack", "defend", "abandon", "safe"],
        help=(
            "attack = flip targets · defend = protect holds · "
            "abandon = capital better spent across many races · "
            "safe = deep seats with little ROI"
        ),
    )
    rivs_min = st.sidebar.slider(
        "Min RIVS (after compute)",
        0.0,
        50.0,
        0.0,
        0.5,
        help="Hide districts whose Republican Investment Value Score is below this floor.",
    )
    competitive_only = st.sidebar.checkbox(
        "Map: competitive only (attack/defend)",
        False,
        help="On the map, show only attack/defend seats (hide safe and abandon polygons).",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("RIVS parameters")
    st.sidebar.caption("Tune the moneyball score formula live (hover controls for detail).")
    with st.sidebar.expander("What is RIVS?", expanded=False):
        st.markdown(
            r"""
**RIVS** = \(\frac{\text{Expected Prob Gain} \times \text{Seat Priority}}{\text{Incremental Cost}}\)

- **Expected Prob Gain**: win probability you can buy toward the risk threshold  
- **Seat Priority**: marginal seats × path to majority × attack/defend weights  
- **Incremental Cost**: FEC-based cost to compete (+ outside spend)  
- **Long-term factor**: open seats / infrastructure bonus  
- **Abandon**: competitive seats where $ is better split across cheaper races  
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
        help="Scales modeled spend. Higher = dollars are scarcer → costly districts score lower.",
    )
    long_term = st.sidebar.slider(
        "Long-term multiplier",
        b["long_term_multiplier"][0],
        b["long_term_multiplier"][1],
        d["long_term_multiplier"],
        0.05,
        help="Boosts open seats and long-term infrastructure value in RIVS.",
    )
    risk = st.sidebar.slider(
        "Risk tolerance (target P*)",
        b["risk_tolerance"][0],
        b["risk_tolerance"][1],
        d["risk_tolerance"],
        0.01,
        help="Target Republican win probability that counts as 'enough' investment (e.g. 0.52).",
    )
    w_attack = st.sidebar.slider(
        "Attack weight (flips)",
        b["w_attack"][0],
        b["w_attack"][1],
        d["w_attack"],
        0.05,
        help="Priority multiplier for flipping opposition-held seats.",
    )
    w_defend = st.sidebar.slider(
        "Defend weight (holds)",
        b["w_defend"][0],
        b["w_defend"][1],
        d["w_defend"],
        0.05,
        help="Priority multiplier for protecting vulnerable Republican seats.",
    )
    inc_bonus = st.sidebar.slider(
        "Incumbent bonus",
        b["incumbent_bonus"][0],
        b["incumbent_bonus"][1],
        d["incumbent_bonus"],
        0.01,
        help="How much an incumbent shifts baseline win probability (R up / D down).",
    )
    open_vol = st.sidebar.slider(
        "Open-seat volatility",
        b["open_seat_volatility"][0],
        b["open_seat_volatility"][1],
        d["open_seat_volatility"],
        0.01,
        help="Pulls open-seat win odds toward a coin flip (more uncertainty).",
    )

    st.sidebar.markdown("##### Abandon (don't double-down)")
    abandon_enabled = st.sidebar.checkbox(
        "Enable abandon mode",
        value=bool(d["abandon_enabled"]),
        help="Flag races where concentrating spend is untenable vs spreading $ across other seats.",
    )
    abandon_pct = st.sidebar.slider(
        "Abandon cost/gain percentile",
        b["abandon_cost_percentile"][0],
        b["abandon_cost_percentile"][1],
        d["abandon_cost_percentile"],
        0.01,
        help="Among competitive seats, those above this $/gain percentile can be marked abandon.",
        disabled=not abandon_enabled,
    )
    abandon_min_m = st.sidebar.slider(
        "Abandon min cost ($M)",
        b["abandon_min_cost_m"][0],
        b["abandon_min_cost_m"][1],
        d["abandon_min_cost_m"],
        0.5,
        help="Never abandon a race cheaper than this absolute dollar floor.",
        disabled=not abandon_enabled,
    )
    abandon_alts = st.sidebar.slider(
        "Opportunity: alt races (N)",
        b["abandon_alt_races"][0],
        b["abandon_alt_races"][1],
        int(d["abandon_alt_races"]),
        1,
        help="If this seat costs ≥ N × median competitive race, compare total gains.",
        disabled=not abandon_enabled,
    )
    abandon_gain_ratio = st.sidebar.slider(
        "Opportunity: gain ratio",
        b["abandon_alt_gain_ratio"][0],
        b["abandon_alt_gain_ratio"][1],
        d["abandon_alt_gain_ratio"],
        0.05,
        help="Abandon if N median races yield more gain than this seat × ratio.",
        disabled=not abandon_enabled,
    )

    target_seats = st.sidebar.number_input(
        "Target R seats",
        min_value=218,
        max_value=TOTAL_HOUSE_SEATS,
        value=DEFAULT_TARGET_SEATS,
        step=1,
        help="Strategic House majority target (default 230 = comfortable majority).",
    )
    budget_m = st.sidebar.slider(
        "Budget simulation ($M)",
        b["budget_millions"][0],
        b["budget_millions"][1],
        d["budget_millions"],
        1.0,
        help="Simulated dollars to allocate greedily down the RIVS ranking.",
    )
    skip_abandon_budget = st.sidebar.checkbox(
        "Budget sim: skip abandon seats",
        value=True,
        help="Do not pour simulated dollars into abandon-flagged races.",
    )
    fec_cycle = st.sidebar.selectbox(
        "FEC cycle (detail charts)",
        ["2026", "2024", "2022"],
        index=1,
        help="Which election cycle’s raised/spent totals to chart in district detail.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data APIs")
    st.sidebar.caption("Live data keys and refresh controls.")
    fec_ok = get_fec_key() is not None
    civic_ok = get_civic_key() is not None
    st.sidebar.write("OpenFEC key:", "✅ set" if fec_ok else "⚪ not set")
    st.sidebar.write("Google Civic key:", "✅ set" if civic_ok else "⚪ not set (optional)")

    st.sidebar.markdown("##### OpenFEC cloud refresh")
    include_outside = st.sidebar.checkbox(
        "Include 2024 outside spending (IE)",
        value=False,
        help="Also pull independent expenditures by district (slow: all 435). Default = candidate totals only.",
        disabled=not fec_ok,
    )
    force_fec = st.sidebar.button(
        "Refresh FEC data now",
        type="primary",
        disabled=not fec_ok,
        help="Force a new OpenFEC pull and rebuild the master table (otherwise cached ~12h).",
    )
    if not fec_ok:
        st.sidebar.caption(
            "Add `FEC_API_KEY` under Streamlit **Settings → Secrets**, then reboot the app."
        )
    else:
        st.sidebar.caption(
            "With a key set, mock-only data auto-refreshes once from OpenFEC (12h cache). "
            "Use the button to force a new pull."
        )

    if st.sidebar.button(
        "List Civic elections (live)",
        help="Call Google Civic electionQuery to list VIP-supported elections (needs Civic key).",
    ):
        if not civic_ok:
            st.sidebar.warning("Set GOOGLE_CIVIC_API_KEY in secrets or .env")
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
        "force_fec": force_fec,
        "include_outside": include_outside,
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
    c1.metric(
        "RIVS",
        fmt_num(row["rivs"], decimals=2),
        help=f"Republican Investment Value Score. Rank #{fmt_int(row['rivs_rank'])} (higher = better ROI).",
    )
    c2.metric(
        "PVI (R+)",
        fmt_pvi(row["pvi"]),
        help="Lean score with R positive. Synthetic/PVI-like unless you supply official ratings.",
    )
    c3.metric(
        "R win P₀",
        fmt_pct(row["baseline_win_prob_r"]),
        help="Modeled Republican win probability before new spending.",
    )
    mode = str(row["mode"]).lower()
    c4.metric(
        "Mode",
        mode.title(),
        help="attack=flip · defend=hold · abandon=redeploy $ · safe=deep seat",
    )

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
            f"First elected: {fmt_int(row['first_elected'])} · "
            f"Tenure ≈ {fmt_int(row.get('tenure_years') or 0)} yrs · "
            f"Incumbent running: {bool(row.get('incumbent_running'))}"
        )

    st.markdown("##### Demographics (ACS-style)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Population", fmt_int(row["pop_total"]), help="Total population estimate for the district.")
    d2.metric("VAP", fmt_int(row["vap"]), help="Voting-age population.")
    d3.metric("Median income", fmt_money(row["median_income"]), help="Median household income.")
    d4.metric("BA+", fmt_pct(row["pct_ba_plus"]), help="Share of adults with bachelor’s degree or higher.")
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(demographics_pie(row), use_container_width=True)
    with e2:
        st.write(
            {
                "pct_urban": fmt_pct(row["pct_urban"]),
                "pct_white": fmt_pct(row["pct_white"]),
                "pct_black": fmt_pct(row["pct_black"]),
                "pct_hispanic": fmt_pct(row["pct_hispanic"]),
                "pct_asian": fmt_pct(row["pct_asian"]),
            }
        )

    st.markdown("##### 2026 race / candidates")
    cands = parse_candidates(row)
    if cands:
        cdf = pd.DataFrame(cands)
        if "receipts" in cdf.columns:
            cdf["receipts"] = cdf["receipts"].map(fmt_money)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.info("No candidate payload in master row (mock or Civic cache empty).")

    st.markdown("##### RIVS breakdown")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "Prob gain",
        fmt_num(row["expected_prob_gain"], decimals=3),
        help="Expected increase in R win probability if fully funded to threshold.",
    )
    b2.metric(
        "Seat priority",
        fmt_num(row["seat_priority"], decimals=2),
        help="Strategic weight: marginality × path to majority × mode weights.",
    )
    b3.metric(
        "Incremental cost",
        fmt_money(row["incremental_cost"]),
        help="Modeled dollars needed to realize the probability gain.",
    )
    b4.metric(
        "Long-term factor",
        fmt_num(row["long_term_factor"], decimals=2),
        help="Multiplier for open seats / infrastructure value.",
    )

    st.markdown(f"##### FEC finance — {fec_cycle}")
    st.caption("Raised and spent by party from OpenFEC (or mock if not refreshed).")
    st.plotly_chart(fec_bars(row, cycle=fec_cycle), use_container_width=True)


def render_federal_house_tab() -> None:
    """US House — existing moneyball view."""
    geo = _cached_geojson()
    # Fast bootstrap for sidebar filters (states list, etc.)
    try:
        bootstrap, bootstrap_label = _cached_disk_master()
    except Exception:
        bootstrap, bootstrap_label = load_master()

    st.sidebar.markdown("### 🇺🇸 US House controls")
    ctl = sidebar_controls(bootstrap)

    fec_stats = None
    source_label = bootstrap_label
    raw = bootstrap
    try:
        raw, source_label, fec_stats = resolve_master(
            force_refresh=bool(ctl.get("force_fec")),
            include_outside=bool(ctl.get("include_outside")),
            auto_fetch_if_mock=True,
        )
    except Exception as e:
        st.error(f"OpenFEC refresh failed — using disk/mock data. ({e})")
        raw, source_label = bootstrap, bootstrap_label

    if fec_stats:
        st.success(
            f"OpenFEC data loaded · "
            f"2024 raised ≈ ${fec_stats.get('raised_2024', 0)/1e9:.2f}B · "
            f"districts with 2024 receipts: {fec_stats.get('districts_with_receipts_2024', '—')}"
            + (
                f" · outside 2024 ≈ ${fec_stats.get('outside_2024', 0)/1e6:.0f}M"
                if fec_stats.get("include_outside")
                else ""
            )
        )

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
    k1.metric(
        "R-held (control)",
        fmt_int(r_held),
        help="House seats currently coded Republican in the master table.",
    )
    k2.metric(
        "Expected R seats (P₀ sum)",
        fmt_num(baseline_seats, decimals=1),
        help="Sum of baseline R win probabilities across all districts (soft seat count).",
    )
    k3.metric(
        f"After {fmt_money(ctl['budget_m'] * 1_000_000)} top-RIVS",
        fmt_num(sim["expected_r_seats_after"], decimals=1),
        delta=f"+{fmt_num(sim['total_prob_gain'], decimals=2)} seats",
        help="Expected R seats after greedily funding top RIVS districts up to the budget.",
    )
    k4.metric(
        "Attack / Defend",
        f"{fmt_int(n_attack)} / {fmt_int(n_defend)}",
        help="Counts of flip targets vs vulnerable holds under current RIVS modes.",
    )
    k5.metric(
        "Abandon",
        fmt_int(n_abandon),
        help="High-cost races where money is better redeployed across multiple seats.",
    )
    k6.metric(
        "Districts shown",
        fmt_int(len(filtered)),
        help="Districts remaining after sidebar filters.",
    )

    st.caption(
        f"Data: `{source_label}` · "
        f"Budget sim funds **{fmt_int(sim['n_districts_funded'])}** districts "
        f"({fmt_money(sim['spent_usd'])} spent"
        + (
            f", skipped {fmt_int(sim.get('skipped_abandon', 0))} abandon"
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
        st.caption("Greedy fill from highest RIVS until the budget is spent (skips abandon/safe by default).")
        if sim["funded"]:
            fund_df = pd.DataFrame(sim["funded"])
            st.dataframe(
                format_display_frame(fund_df),
                use_container_width=True,
                hide_index=True,
            )
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
        st.caption(
            " | ".join(f"**{c}**: {COLUMN_HELP[c]}" for c in show_cols if c in COLUMN_HELP)[:900]
            + ("…" if len(show_cols) > 8 else "")
        )
        st.dataframe(
            format_display_frame(filtered, show_cols),
            use_container_width=True,
            height=560,
        )
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv,
            file_name="house_moneyball_filtered.csv",
            mime="text/csv",
            help="Raw numeric CSV (unformatted) for Excel/Sheets.",
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
        "US House view · OpenFEC finance when configured · "
        "not affiliated with MLB Moneyball or Cook Political Report."
    )


def main() -> None:
    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")

    tab_fed, tab_lower, tab_upper = st.tabs(
        ["🇺🇸 US House", "🏛️ State House", "🏛️ State Senate"]
    )
    with tab_fed:
        render_federal_house_tab()
    with tab_lower:
        # State-tab RIVS knobs live in-tab; keep federal sidebar only on House
        render_state_chamber_tab("lower")
    with tab_upper:
        render_state_chamber_tab("upper")

    st.markdown("---")
    st.caption(
        "House Moneyball · federal + state legislative RIVS · "
        "not affiliated with MLB Moneyball, Cook Political Report, or any campaign committee."
    )


if __name__ == "__main__":
    main()
