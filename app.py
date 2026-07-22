"""
House Moneyball — Streamlit dashboard

Snapshot-first: opens instantly from committed parquet.
Live FEC / OpenStates only on manual Refresh.
RIVS knobs live in forms (Apply scores) so sliders do not thrash.

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
from src.data_loader import list_states, load_geojson, load_master  # noqa: E402
from src.fec_client import get_api_key as get_fec_key  # noqa: E402
from src.fec_pipeline import fetch_and_merge_fec  # noqa: E402
from src.formatters import (  # noqa: E402
    COLUMN_HELP,
    format_display_frame,
    fmt_int,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_pvi,
)
from src.home import render_home  # noqa: E402
from src.map_builder import build_district_map, build_simple_state_centroids_map  # noqa: E402
from src.rivs import budget_simulation  # noqa: E402
from src.state_data import load_state_master  # noqa: E402
from src.ui_common import (  # noqa: E402
    FEDERAL_EXTRA_COLS,
    FEDERAL_SLIM_COLS,
    PRESETS,
    cached_federal_rivs,
    data_status_badge,
    write_refresh_meta,
)
from src.ui_state import render_state_chamber_tab  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=12 * 3600, show_spinner="Fetching OpenFEC (manual refresh)…")
def _cached_fec_master(include_outside: bool, cache_bust: int):
    return fetch_and_merge_fec(
        cycles=[2022, 2024, 2026],
        include_outside=include_outside,
        progress=False,
        persist=True,
    )


@st.cache_data(show_spinner=False)
def _cached_disk_master():
    return load_master()


@st.cache_data(show_spinner=False)
def _cached_geojson():
    return load_geojson()


@st.cache_data(show_spinner=False)
def _cached_state(chamber: str):
    return load_state_master(chamber)  # type: ignore[arg-type]


def resolve_master(
    *,
    force_refresh: bool = False,
    include_outside: bool = False,
) -> tuple[pd.DataFrame, str, dict | None]:
    """Snapshot-first. Live OpenFEC only when force_refresh=True."""
    stats = None
    if "fec_cache_bust" not in st.session_state:
        st.session_state.fec_cache_bust = 0

    if force_refresh and get_fec_key():
        st.session_state.fec_cache_bust += 1
        _cached_fec_master.clear()
        _cached_disk_master.clear()
        df, label, stats = _cached_fec_master(
            include_outside, st.session_state.fec_cache_bust
        )
        try:
            write_refresh_meta(
                {
                    "federal_refreshed_at": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .strftime("%Y-%m-%d %H:%M UTC"),
                    "federal_source": "openfec_live",
                }
            )
        except Exception:
            pass
        return df, label, stats

    try:
        return (*_cached_disk_master(), None)
    except Exception:
        return (*load_master(), None)


def _default_scoring() -> dict:
    d = RIVS_DEFAULTS
    return {
        "modes": ["attack", "defend", "abandon", "safe"],
        "cost_sensitivity": d["cost_sensitivity"],
        "long_term": d["long_term_multiplier"],
        "risk": d["risk_tolerance"],
        "w_attack": d["w_attack"],
        "w_defend": d["w_defend"],
        "inc_bonus": d["incumbent_bonus"],
        "open_vol": d["open_seat_volatility"],
        "target_seats": DEFAULT_TARGET_SEATS,
        "abandon_enabled": d["abandon_enabled"],
        "abandon_cost_percentile": d["abandon_cost_percentile"],
        "abandon_min_cost_m": d["abandon_min_cost_m"],
        "abandon_alt_races": d["abandon_alt_races"],
        "abandon_alt_gain_ratio": d["abandon_alt_gain_ratio"],
        "budget_m": d["budget_millions"],
        "skip_abandon_budget": True,
        "sel_states": [],
        "party": ["R", "D", "VACANT"],
        "rivs_min": 0.0,
        "show_map": False,
        "competitive_only": False,
        "fec_cycle": "2024",
        "show_extra_cols": False,
        "top_n_table": 100,
    }


def federal_sidebar(df: pd.DataFrame) -> dict:
    """Filters + scoring form (Apply) + manual data refresh."""
    st.sidebar.markdown("### 🇺🇸 US House")
    st.sidebar.caption("Snapshot loads instantly. Scoring applies only when you click **Apply**.")

    # Presets (outside form so one click can reset session scoring draft)
    preset = st.sidebar.selectbox(
        "Preset",
        list(PRESETS.keys()),
        help="Opinionated starting points. Click Apply scores after choosing.",
    )
    if st.sidebar.button("Load preset values", help="Copy preset into the scoring form fields."):
        p = PRESETS[preset]
        st.session_state["fed_modes"] = list(p["modes"])
        st.session_state["fed_wa"] = p["w_attack"]
        st.session_state["fed_wd"] = p["w_defend"]
        st.session_state["fed_cost"] = p["cost_sensitivity"]
        st.session_state["fed_risk"] = p["risk_tolerance"]
        st.session_state["fed_ab_en"] = p["abandon_enabled"]
        st.sidebar.success(PRESETS[preset]["description"])

    states = list_states(df)
    with st.sidebar.form("fed_scoring_form"):
        st.markdown("##### Filters")
        sel_states = st.multiselect(
            "State(s)",
            states,
            default=st.session_state.get("fed_states", []),
            help="Empty = all districts.",
        )
        party = st.multiselect(
            "Party control",
            ["R", "D", "VACANT"],
            default=st.session_state.get("fed_party", ["R", "D", "VACANT"]),
            help="Filter by party holding the seat.",
        )
        modes = st.multiselect(
            "Mode",
            ["attack", "defend", "abandon", "safe"],
            default=st.session_state.get("fed_modes", ["attack", "defend", "abandon", "safe"]),
            help="attack=flip · defend=hold · abandon=redeploy · safe=deep",
        )
        rivs_min = st.slider(
            "Min RIVS",
            0.0,
            50.0,
            float(st.session_state.get("fed_rivs_min", 0.0)),
            0.5,
            help="Hide seats below this score after compute.",
        )
        top_n = st.slider(
            "Table top N by rank",
            25,
            435,
            int(st.session_state.get("fed_top_n", 100)),
            25,
            help="Only show the top N ranked seats in the table (faster).",
        )
        show_extra = st.checkbox(
            "Show extra table columns",
            value=bool(st.session_state.get("fed_extra", False)),
            help="Hide advanced columns by default for a cleaner table.",
        )
        show_map = st.checkbox(
            "Show map",
            value=bool(st.session_state.get("fed_map", False)),
            help="Maps are heavy — off by default. Enable when you need geography.",
        )
        competitive_only = st.checkbox(
            "Map: attack/defend only",
            value=bool(st.session_state.get("fed_comp", False)),
            help="When map is on, hide safe/abandon polygons.",
        )

        st.markdown("##### Scoring")
        b, d = RIVS_BOUNDS, RIVS_DEFAULTS
        cost_sensitivity = st.slider(
            "Cost sensitivity",
            b["cost_sensitivity"][0],
            b["cost_sensitivity"][1],
            float(st.session_state.get("fed_cost", d["cost_sensitivity"])),
            0.05,
            help="Higher = penalize expensive seats more.",
        )
        risk = st.slider(
            "Risk tolerance (P*)",
            b["risk_tolerance"][0],
            b["risk_tolerance"][1],
            float(st.session_state.get("fed_risk", d["risk_tolerance"])),
            0.01,
            help="Target win probability for 'enough' investment.",
        )
        w_attack = st.slider(
            "Attack weight",
            b["w_attack"][0],
            b["w_attack"][1],
            float(st.session_state.get("fed_wa", d["w_attack"])),
            0.05,
            help="Priority for flips.",
        )
        w_defend = st.slider(
            "Defend weight",
            b["w_defend"][0],
            b["w_defend"][1],
            float(st.session_state.get("fed_wd", d["w_defend"])),
            0.05,
            help="Priority for holds.",
        )
        budget_m = st.slider(
            "Budget sim ($M)",
            b["budget_millions"][0],
            b["budget_millions"][1],
            float(st.session_state.get("fed_bud", d["budget_millions"])),
            1.0,
            help="Dollars for greedy top-RIVS allocation.",
        )
        target_seats = st.number_input(
            "Target R seats",
            218,
            TOTAL_HOUSE_SEATS,
            int(st.session_state.get("fed_target", DEFAULT_TARGET_SEATS)),
            1,
            help="Strategic majority goal (default 230).",
        )
        skip_abandon = st.checkbox(
            "Budget: skip abandon",
            value=bool(st.session_state.get("fed_skip_ab", True)),
            help="Do not fund abandon seats in the budget simulation.",
        )

        with st.expander("Advanced scoring", expanded=False):
            long_term = st.slider(
                "Long-term multiplier",
                b["long_term_multiplier"][0],
                b["long_term_multiplier"][1],
                d["long_term_multiplier"],
                0.05,
                help="Boost open seats / infrastructure.",
            )
            inc_bonus = st.slider(
                "Incumbent bonus",
                b["incumbent_bonus"][0],
                b["incumbent_bonus"][1],
                d["incumbent_bonus"],
                0.01,
                help="Incumbent shift to baseline win prob.",
            )
            open_vol = st.slider(
                "Open-seat volatility",
                b["open_seat_volatility"][0],
                b["open_seat_volatility"][1],
                d["open_seat_volatility"],
                0.01,
                help="Pull open seats toward a coin flip.",
            )
            abandon_enabled = st.checkbox(
                "Enable abandon",
                value=bool(st.session_state.get("fed_ab_en", d["abandon_enabled"])),
                help="Flag races where $ is better spread across many seats.",
            )
            abandon_pct = st.slider(
                "Abandon cost/gain percentile",
                b["abandon_cost_percentile"][0],
                b["abandon_cost_percentile"][1],
                d["abandon_cost_percentile"],
                0.01,
                disabled=not abandon_enabled,
            )
            abandon_min_m = st.slider(
                "Abandon min cost ($M)",
                b["abandon_min_cost_m"][0],
                b["abandon_min_cost_m"][1],
                d["abandon_min_cost_m"],
                0.5,
                disabled=not abandon_enabled,
            )
            abandon_alts = st.slider(
                "Opportunity alt races N",
                b["abandon_alt_races"][0],
                b["abandon_alt_races"][1],
                int(d["abandon_alt_races"]),
                1,
                disabled=not abandon_enabled,
            )
            abandon_gain_ratio = st.slider(
                "Opportunity gain ratio",
                b["abandon_alt_gain_ratio"][0],
                b["abandon_alt_gain_ratio"][1],
                d["abandon_alt_gain_ratio"],
                0.05,
                disabled=not abandon_enabled,
            )

        fec_cycle = st.selectbox(
            "FEC cycle (detail)",
            ["2026", "2024", "2022"],
            index=1,
            help="Cycle shown on district finance charts.",
        )
        applied = st.form_submit_button("Apply scores", type="primary", use_container_width=True)

    # Persist form defaults into session for next open
    if applied:
        st.session_state["fed_states"] = sel_states
        st.session_state["fed_party"] = party
        st.session_state["fed_modes"] = modes
        st.session_state["fed_rivs_min"] = rivs_min
        st.session_state["fed_top_n"] = top_n
        st.session_state["fed_extra"] = show_extra
        st.session_state["fed_map"] = show_map
        st.session_state["fed_comp"] = competitive_only
        st.session_state["fed_cost"] = cost_sensitivity
        st.session_state["fed_risk"] = risk
        st.session_state["fed_wa"] = w_attack
        st.session_state["fed_wd"] = w_defend
        st.session_state["fed_bud"] = budget_m
        st.session_state["fed_target"] = target_seats
        st.session_state["fed_skip_ab"] = skip_abandon
        st.session_state["fed_ab_en"] = abandon_enabled
        st.session_state["fed_scoring"] = {
            "modes": modes,
            "sel_states": sel_states,
            "party": party,
            "rivs_min": rivs_min,
            "top_n_table": top_n,
            "show_extra_cols": show_extra,
            "show_map": show_map,
            "competitive_only": competitive_only,
            "cost_sensitivity": cost_sensitivity,
            "long_term": long_term,
            "risk": risk,
            "w_attack": w_attack,
            "w_defend": w_defend,
            "inc_bonus": inc_bonus,
            "open_vol": open_vol,
            "target_seats": int(target_seats),
            "abandon_enabled": abandon_enabled,
            "abandon_cost_percentile": abandon_pct,
            "abandon_min_cost_m": abandon_min_m,
            "abandon_alt_races": int(abandon_alts),
            "abandon_alt_gain_ratio": abandon_gain_ratio,
            "budget_m": budget_m,
            "skip_abandon_budget": skip_abandon,
            "fec_cycle": fec_cycle,
        }

    # Manual data refresh (outside form)
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Data (manual only)")
    fec_ok = get_fec_key() is not None
    st.sidebar.write("OpenFEC key:", "✅ set" if fec_ok else "⚪ not set")
    include_outside = st.sidebar.checkbox(
        "Include 2024 IE outside",
        False,
        help="Slow full-district IE pull. Off by default.",
        disabled=not fec_ok,
    )
    force_fec = st.sidebar.button(
        "Refresh FEC now",
        type="secondary",
        disabled=not fec_ok,
        help="Live OpenFEC pull — may take several minutes. Snapshot is used until then.",
    )
    if st.sidebar.button("List Civic elections", help="Optional Google Civic check."):
        if not get_civic_key():
            st.sidebar.warning("No GOOGLE_CIVIC_API_KEY")
        else:
            result, err = safe_civic_call(get_elections)
            st.sidebar.error(err) if err else st.sidebar.json((result or [])[:3])

    # Active scoring = last applied or defaults
    scoring = st.session_state.get("fed_scoring") or _default_scoring()
    scoring["force_fec"] = force_fec
    scoring["include_outside"] = include_outside
    return scoring


def apply_filters(ranked: pd.DataFrame, ctl: dict) -> pd.DataFrame:
    out = ranked.copy()
    if ctl.get("sel_states"):
        out = out[out["state"].isin(ctl["sel_states"])]
    if ctl.get("party"):
        out = out[out["party_control"].isin(ctl["party"])]
    if ctl.get("modes"):
        out = out[out["mode"].isin(ctl["modes"])]
    out = out[out["rivs"] >= float(ctl.get("rivs_min") or 0)]
    out = out.sort_values("rivs_rank").reset_index(drop=True)
    top_n = int(ctl.get("top_n_table") or 100)
    if len(out) > top_n:
        out = out.head(top_n).reset_index(drop=True)
    return out


def render_detail(row: pd.Series, fec_cycle: str) -> None:
    st.subheader(f"{row['district_id']} — detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RIVS", fmt_num(row["rivs"], decimals=2), help=f"Rank #{fmt_int(row['rivs_rank'])}")
    c2.metric("PVI (R+)", fmt_pvi(row["pvi"]), help="Lean score, R positive.")
    c3.metric("R win P₀", fmt_pct(row["baseline_win_prob_r"]), help="Baseline R win probability.")
    mode = str(row["mode"]).lower()
    c4.metric("Mode", mode.title(), help="attack / defend / abandon / safe")
    if mode == "abandon":
        st.warning("**Abandon** — redeploy $ to higher-ROI seats.")
        if row.get("abandon_reason"):
            st.caption(str(row["abandon_reason"]))
    st.markdown(
        f"**Rep:** {row.get('rep_name', '—')} ({row.get('rep_party', '—')}) · "
        f"**Rating:** {row.get('cook_rating', '—')}"
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Population", fmt_int(row.get("pop_total")), help="District population.")
    d2.metric("VAP", fmt_int(row.get("vap")), help="Voting-age population.")
    d3.metric("Median income", fmt_money(row.get("median_income")), help="ACS-style income.")
    d4.metric("BA+", fmt_pct(row.get("pct_ba_plus")), help="Bachelor’s or higher.")
    st.plotly_chart(demographics_pie(row), use_container_width=True)
    cands = parse_candidates(row)
    if cands:
        cdf = pd.DataFrame(cands)
        if "receipts" in cdf.columns:
            cdf["receipts"] = cdf["receipts"].map(fmt_money)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Prob gain", fmt_num(row["expected_prob_gain"], decimals=3))
    b2.metric("Seat priority", fmt_num(row["seat_priority"], decimals=2))
    b3.metric("Incremental cost", fmt_money(row["incremental_cost"]))
    b4.metric("Long-term factor", fmt_num(row["long_term_factor"], decimals=2))
    st.plotly_chart(fec_bars(row, cycle=fec_cycle), use_container_width=True)


def render_federal_house_tab() -> None:
    """US House tab — snapshot-first, form scoring, optional map."""
    bootstrap, bootstrap_label = _cached_disk_master()
    ctl = federal_sidebar(bootstrap)

    fec_stats = None
    try:
        raw, source_label, fec_stats = resolve_master(
            force_refresh=bool(ctl.get("force_fec")),
            include_outside=bool(ctl.get("include_outside")),
        )
    except Exception as e:
        st.error(f"OpenFEC refresh failed — using snapshot. ({e})")
        raw, source_label = bootstrap, bootstrap_label

    if fec_stats:
        st.success(
            f"OpenFEC refreshed · districts with 2024 receipts: "
            f"{fmt_int(fec_stats.get('districts_with_receipts_2024', 0))}"
        )

    st.caption(data_status_badge(source_label, raw))

    ranked = cached_federal_rivs(
        raw,
        float(ctl["cost_sensitivity"]),
        float(ctl["long_term"]),
        float(ctl["risk"]),
        float(ctl["w_attack"]),
        float(ctl["w_defend"]),
        float(ctl["inc_bonus"]),
        float(ctl["open_vol"]),
        int(ctl["target_seats"]),
        bool(ctl["abandon_enabled"]),
        float(ctl["abandon_cost_percentile"]),
        float(ctl["abandon_min_cost_m"]),
        int(ctl["abandon_alt_races"]),
        float(ctl["abandon_alt_gain_ratio"]),
    )
    filtered = apply_filters(ranked, ctl)
    sim = budget_simulation(
        ranked,
        float(ctl["budget_m"]) * 1_000_000,
        skip_abandon=bool(ctl.get("skip_abandon_budget", True)),
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("R-held", fmt_int((ranked["party_control"] == "R").sum()), help="Republican seats.")
    k2.metric(
        "Expected R (P₀)",
        fmt_num(sim["baseline_expected_r_seats"], decimals=1),
        help="Sum of baseline win probs.",
    )
    k3.metric(
        f"After {fmt_money(float(ctl['budget_m']) * 1e6)}",
        fmt_num(sim["expected_r_seats_after"], decimals=1),
        delta=f"+{fmt_num(sim['total_prob_gain'], decimals=2)}",
        help="Post budget-sim expected seats.",
    )
    k4.metric(
        "Attack / Defend",
        f"{fmt_int((ranked['mode']=='attack').sum())} / {fmt_int((ranked['mode']=='defend').sum())}",
        help="Mode counts under current scoring.",
    )
    k5.metric("Abandon", fmt_int((ranked["mode"] == "abandon").sum()), help="Redeploy candidates.")
    k6.metric("Table rows", fmt_int(len(filtered)), help="Filters + top N.")

    tab_plan, tab_map, tab_budget, tab_method = st.tabs(
        ["📋 Plan & table", "🗺️ Map", "💰 Budget", "📖 Methodology"]
    )

    with tab_plan:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.plotly_chart(
                top_rivs_bar(filtered if len(filtered) else ranked, n=15),
                use_container_width=True,
            )
            cols = list(FEDERAL_SLIM_COLS)
            if ctl.get("show_extra_cols"):
                cols += [c for c in FEDERAL_EXTRA_COLS if c not in cols]
            cols = [c for c in cols if c in filtered.columns]
            helps = " · ".join(f"{c}: {COLUMN_HELP[c]}" for c in cols if c in COLUMN_HELP)
            st.caption(helps[:500] + ("…" if len(helps) > 500 else ""))
            st.dataframe(
                format_display_frame(filtered, cols),
                use_container_width=True,
                height=420,
            )
            st.download_button(
                "Download filtered CSV (raw)",
                filtered.to_csv(index=False).encode("utf-8"),
                "house_moneyball_filtered.csv",
                "text/csv",
            )
        with right:
            ids = (
                filtered["district_id"].tolist()
                if len(filtered)
                else ranked["district_id"].tolist()
            )
            pick = st.selectbox("District detail", ids, help="Detail panel for one seat.")
            match = ranked[ranked["district_id"] == pick]
            if len(match):
                render_detail(match.iloc[0], str(ctl.get("fec_cycle", "2024")))

    with tab_map:
        if not ctl.get("show_map"):
            st.info(
                "Map is **off** for speed. In the sidebar form enable **Show map**, then **Apply scores**."
            )
        else:
            geo = _cached_geojson()
            try:
                from streamlit_folium import st_folium

                view = filtered if len(filtered) else ranked
                if geo is not None:
                    fmap = build_district_map(
                        view,
                        geo,
                        show_competitive_only=bool(ctl.get("competitive_only")),
                    )
                else:
                    fmap = build_simple_state_centroids_map(view)
                    st.caption("No `data/raw/cds.geojson` — centroids only.")
                st_folium(fmap, width=None, height=520, returned_objects=[])
            except ImportError:
                st.warning("streamlit-folium not installed")

    with tab_budget:
        st.caption("Greedy allocation by RIVS until budget exhausted.")
        if sim["funded"]:
            st.dataframe(
                format_display_frame(pd.DataFrame(sim["funded"])),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No districts funded.")

    with tab_method:
        st.markdown(
            """
### Performance design
| Feature | Behavior |
|---------|----------|
| Snapshot | Committed parquet opens immediately |
| Live FEC | Sidebar **Refresh FEC now** only |
| Scoring | **Apply scores** form (no recompute on every slider drag) |
| Table | Top N + slim columns |
| Map | Opt-in |
| Nightly data | GitHub Action `.github/workflows/refresh-data.yml` |
            """
        )


def main() -> None:
    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")

    # Single active view so the left toolbar is not a stack of federal + state controls
    view = st.sidebar.radio(
        "View",
        ["🏠 Home", "🇺🇸 US House", "🏛️ State House", "🏛️ State Senate"],
        help=(
            "Switch views here. The left toolbar changes for each view "
            "(US House vs a specific state chamber)."
        ),
    )
    st.sidebar.markdown("---")

    if view == "🏠 Home":
        try:
            fed_df, fed_lab = _cached_disk_master()
        except Exception:
            fed_df, fed_lab = load_master()
        try:
            low_df, low_lab = _cached_state("lower")
        except Exception:
            low_df, low_lab = load_state_master("lower")
        try:
            up_df, up_lab = _cached_state("upper")
        except Exception:
            up_df, up_lab = load_state_master("upper")
        render_home(fed_df, fed_lab, low_df, low_lab, up_df, up_lab)
    elif view == "🇺🇸 US House":
        render_federal_house_tab()
    elif view == "🏛️ State House":
        render_state_chamber_tab("lower")
    else:
        render_state_chamber_tab("upper")

    st.caption(
        "House Moneyball · snapshot-first · pick a state for chamber views · "
        "not affiliated with MLB Moneyball or Cook Political Report."
    )


if __name__ == "__main__":
    main()
