"""Streamlit UI for state House / Senate chamber tabs."""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import RIVS_BOUNDS, RIVS_DEFAULTS
from src.charts import top_rivs_bar
from src.map_builder import build_district_map, build_simple_state_centroids_map
from src.openstates_client import get_api_key as get_openstates_key
from src.rivs import format_money
from src.state_data import list_states, load_state_geojson, load_state_master, parse_state_candidates
from src.state_pipeline import fetch_and_overlay_openstates, master_needs_openstates
from src.state_rivs import chamber_majority_summary, compute_state_rivs, state_budget_simulation

Chamber = Literal["lower", "upper"]


@st.cache_data(
    ttl=12 * 3600,
    show_spinner="Fetching OpenStates legislators (all states — may take several minutes)…",
)
def _cached_openstates_master(chamber: str, cache_bust: int) -> tuple[pd.DataFrame, str, dict]:
    return fetch_and_overlay_openstates(
        chamber,  # type: ignore[arg-type]
        states=None,
        progress=False,
        persist=True,
    )


def resolve_state_master(
    chamber: Chamber,
    *,
    force_refresh: bool = False,
    auto_fetch: bool = True,
) -> tuple[pd.DataFrame, str, dict | None]:
    """Disk/mock master, with OpenStates overlay when API key is set."""
    bust_key = f"os_cache_bust_{chamber}"
    if bust_key not in st.session_state:
        st.session_state[bust_key] = 0
    if force_refresh:
        st.session_state[bust_key] += 1
        _cached_openstates_master.clear()

    key_ok = get_openstates_key() is not None
    raw, label = load_state_master(chamber)
    stats = None

    if key_ok and (force_refresh or (auto_fetch and master_needs_openstates(raw, label))):
        try:
            raw, label, stats = _cached_openstates_master(chamber, st.session_state[bust_key])
        except Exception as e:  # noqa: BLE001
            st.warning(f"OpenStates refresh failed — using `{label}`. ({e})")
    return raw, label, stats


def _state_finance_bars(row: pd.Series, cycle: str = "2024") -> go.Figure:
    c = str(cycle)
    labels = ["R Raised", "R Spent", "D Raised", "D Spent"]
    keys = [
        f"state_raised_r_{c}",
        f"state_spent_r_{c}",
        f"state_raised_d_{c}",
        f"state_spent_d_{c}",
    ]
    vals = [float(row[k]) if k in row.index and pd.notna(row[k]) else 0.0 for k in keys]
    colors = ["#e81b23", "#a01218", "#00aef3", "#006b9a"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[f"${v/1e3:.0f}K" for v in vals],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"State campaign finance (proxy) — {c}",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def _rivs_knobs(prefix: str) -> dict[str, Any]:
    """Sidebar knobs namespaced by chamber tab."""
    b, d = RIVS_BOUNDS, RIVS_DEFAULTS
    with st.sidebar.expander(f"RIVS ({prefix})", expanded=False):
        cost_sensitivity = st.slider(
            f"Cost sensitivity ({prefix})",
            b["cost_sensitivity"][0],
            b["cost_sensitivity"][1],
            d["cost_sensitivity"],
            0.05,
            key=f"{prefix}_cost",
        )
        risk = st.slider(
            f"Risk tolerance ({prefix})",
            b["risk_tolerance"][0],
            b["risk_tolerance"][1],
            d["risk_tolerance"],
            0.01,
            key=f"{prefix}_risk",
        )
        w_attack = st.slider(
            f"Attack weight ({prefix})",
            b["w_attack"][0],
            b["w_attack"][1],
            d["w_attack"],
            0.05,
            key=f"{prefix}_wa",
        )
        w_defend = st.slider(
            f"Defend weight ({prefix})",
            b["w_defend"][0],
            b["w_defend"][1],
            d["w_defend"],
            0.05,
            key=f"{prefix}_wd",
        )
        abandon_enabled = st.checkbox(
            f"Enable abandon ({prefix})",
            value=True,
            key=f"{prefix}_ab_en",
        )
        abandon_min = st.slider(
            f"Abandon min cost $M ({prefix})",
            0.2,
            5.0,
            1.0,
            0.1,
            key=f"{prefix}_ab_min",
            disabled=not abandon_enabled,
        )
        budget_m = st.slider(
            f"Budget sim $M ({prefix})",
            0.5,
            50.0,
            5.0,
            0.5,
            key=f"{prefix}_bud",
        )
    return {
        "cost_sensitivity": cost_sensitivity,
        "risk_tolerance": risk,
        "w_attack": w_attack,
        "w_defend": w_defend,
        "abandon_enabled": abandon_enabled,
        "abandon_min_cost_m": abandon_min,
        "long_term_multiplier": d["long_term_multiplier"],
        "incumbent_bonus": d["incumbent_bonus"],
        "open_seat_volatility": d["open_seat_volatility"],
        "abandon_cost_percentile": d["abandon_cost_percentile"],
        "abandon_alt_races": d["abandon_alt_races"],
        "abandon_alt_gain_ratio": d["abandon_alt_gain_ratio"],
        "budget_m": budget_m,
    }


def render_state_chamber_tab(chamber: Chamber) -> None:
    title = "State House (lower chamber)" if chamber == "lower" else "State Senate (upper chamber)"
    st.subheader(title)
    st.caption(
        "RIVS targets **chamber majority in each state** (not federal 230). "
        "Lean/finance default to synthetic proxies until OpenStates / FTM refresh."
    )

    if chamber == "upper":
        st.info("Nebraska is unicameral — no upper-chamber seats in this tab.")

    prefix = "L" if chamber == "lower" else "U"
    os_key = get_openstates_key() is not None

    with st.sidebar.expander(f"OpenStates ({prefix})", expanded=os_key):
        st.write("API key:", "✅ set" if os_key else "⚪ not set")
        st.caption(
            'Streamlit Secrets TOML: `OPENSTATES_API_KEY = "…"` '
            "(exact name). Auto-overlays members when data is still mock-only."
        )
        force_os = st.button(
            f"Refresh OpenStates ({prefix})",
            type="primary",
            disabled=not os_key,
            key=f"{prefix}_os_refresh",
            help="Re-pull legislators for all states in this chamber (cached 12h).",
        )
        if not os_key:
            st.caption("Add key under App settings → Secrets, then reboot the app.")

    try:
        raw, source_label, os_stats = resolve_state_master(
            chamber,
            force_refresh=bool(force_os) if os_key else False,
            auto_fetch=True,
        )
    except Exception as e:
        st.error(f"Could not load state master: {e}")
        st.info("Run: `python scripts/generate_state_mock_data.py`")
        return

    if os_stats:
        st.success(
            f"OpenStates · {os_stats.get('seats_matched', 0)}/"
            f"{os_stats.get('total_seats', 0)} seats matched · "
            f"{os_stats.get('states_requested', 0)} states"
        )

    if raw.empty:
        st.warning("No seats in this chamber master.")
        return

    states = list_states(raw)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_states = st.multiselect(
            "State filter",
            states,
            default=[],
            key=f"{prefix}_states",
            help="Empty = all states. Select 1–few for faster maps.",
        )
    with c2:
        modes = st.multiselect(
            "Mode",
            ["attack", "defend", "abandon", "safe"],
            default=["attack", "defend", "abandon", "safe"],
            key=f"{prefix}_modes",
        )
    with c3:
        map_state = st.selectbox(
            "Map focus state",
            ["(centroids)"] + states,
            key=f"{prefix}_map_st",
            help="Pick a state to load district polygons when GeoJSON is available.",
        )

    knobs = _rivs_knobs(prefix)

    st.caption(
        f"Data: `{source_label}` · OpenStates key: "
        f"{'✅ set' if os_key else '⚪ not set (mock members)'}"
    )

    ranked = compute_state_rivs(
        raw,
        cost_sensitivity=knobs["cost_sensitivity"],
        long_term_multiplier=knobs["long_term_multiplier"],
        risk_tolerance=knobs["risk_tolerance"],
        w_attack=knobs["w_attack"],
        w_defend=knobs["w_defend"],
        incumbent_bonus=knobs["incumbent_bonus"],
        open_seat_volatility=knobs["open_seat_volatility"],
        abandon_enabled=knobs["abandon_enabled"],
        abandon_cost_percentile=knobs["abandon_cost_percentile"],
        abandon_min_cost_m=knobs["abandon_min_cost_m"],
        abandon_alt_races=knobs["abandon_alt_races"],
        abandon_alt_gain_ratio=knobs["abandon_alt_gain_ratio"],
    )

    filtered = ranked.copy()
    if sel_states:
        filtered = filtered[filtered["state"].isin(sel_states)]
    if modes:
        filtered = filtered[filtered["mode"].isin(modes)]
    filtered = filtered.reset_index(drop=True)

    sim = state_budget_simulation(
        filtered if len(filtered) else ranked,
        knobs["budget_m"] * 1_000_000,
        skip_abandon=True,
    )
    summary = chamber_majority_summary(filtered if len(filtered) else ranked)

    r_held = int((filtered["party_control"] == "R").sum()) if len(filtered) else 0
    n_ab = int((filtered["mode"] == "abandon").sum()) if len(filtered) else 0
    n_short = int((~summary["r_controls"]).sum()) if len(summary) else 0
    n_r_chambers = int(summary["r_controls"].sum()) if len(summary) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Seats shown (R-held)", f"{r_held} / {len(filtered)}")
    k2.metric("Chambers R controls", n_r_chambers)
    k3.metric("Chambers short of majority", n_short)
    k4.metric("Abandon seats", n_ab)
    k5.metric(
        f"Budget ${knobs['budget_m']:.1f}M → Δ seats",
        f"+{sim['total_prob_gain']:.2f}",
        help=f"Funds {sim['n_districts_funded']} seats across {sim.get('n_states_funded', 0)} states",
    )

    tab_map, tab_table, tab_chambers, tab_method = st.tabs(
        ["🗺️ Map & detail", "📋 Seats", "🏛️ Chamber scoreboard", "📖 Methodology"]
    )

    sel_key = f"selected_{prefix}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = (
            str(filtered.iloc[0]["district_id"]) if len(filtered) else None
        )

    with tab_map:
        left, right = st.columns([1.3, 1.0])
        with left:
            focus = None if map_state == "(centroids)" else map_state
            geo = load_state_geojson(chamber, state=focus) if focus else load_state_geojson(chamber)
            view = filtered if len(filtered) else ranked
            try:
                from streamlit_folium import st_folium

                if geo is not None and focus:
                    # Ensure district_id on features when possible
                    for feat in geo.get("features") or []:
                        props = feat.setdefault("properties", {})
                        if "district_id" not in props and focus:
                            dnum = props.get("district") or props.get("NAME") or props.get("SLDLST") or props.get("SLDUST")
                            if dnum is not None:
                                try:
                                    n = int(float(str(dnum).lstrip("0") or "0"))
                                    tag = "L" if chamber == "lower" else "U"
                                    props["district_id"] = f"{focus}-{tag}-{n:03d}"
                                except ValueError:
                                    pass
                    fmap = build_district_map(
                        view,
                        geo,
                        selected=st.session_state.get(sel_key),
                    )
                else:
                    fmap = build_simple_state_centroids_map(view)
                    st.caption(
                        "Polygon map: choose a **Map focus state** and place GeoJSON at "
                        f"`data/state/raw/{{ST}}_{chamber}.geojson` (or national sldl/sldu). "
                        "Showing state-level mean RIVS centroids."
                    )
                st_folium(fmap, width=None, height=480, returned_objects=[], key=f"{prefix}_folium")
            except ImportError:
                st.warning("streamlit-folium not installed")

            st.plotly_chart(top_rivs_bar(view, n=15), use_container_width=True, key=f"{prefix}_topbar")

        with right:
            ids = filtered["district_id"].tolist() if len(filtered) else ranked["district_id"].tolist()
            pick = st.selectbox("Select seat", ids, key=f"{prefix}_pick")
            st.session_state[sel_key] = pick
            row = ranked[ranked["district_id"] == pick]
            if len(row):
                r = row.iloc[0]
                st.markdown(f"### {r.get('seat_label', r['district_id'])}")
                m1, m2, m3 = st.columns(3)
                m1.metric("RIVS", f"{float(r['rivs']):.2f}")
                m2.metric("Mode", str(r["mode"]).title())
                m3.metric("PVI-like", f"{float(r['pvi']):+.1f}")
                if str(r["mode"]).lower() == "abandon":
                    st.warning("**Abandon** — redeploy $ to other seats in this chamber.")
                    if r.get("abandon_reason"):
                        st.caption(str(r["abandon_reason"]))
                st.write(
                    f"**Member:** {r.get('rep_name')} ({r.get('rep_party')}) · "
                    f"**Control:** {r.get('party_control')} · "
                    f"**Majority need (state):** {r.get('chamber_majority_threshold')} "
                    f"(R held ~{r.get('state_r_held')})"
                )
                cands = parse_state_candidates(r)
                if cands:
                    st.dataframe(pd.DataFrame(cands), hide_index=True, use_container_width=True)
                st.plotly_chart(_state_finance_bars(r, "2024"), use_container_width=True, key=f"{prefix}_fec")

    with tab_table:
        cols = [
            c
            for c in [
                "rivs_rank",
                "district_id",
                "seat_label",
                "state",
                "party_control",
                "mode",
                "rivs",
                "pvi",
                "baseline_win_prob_r",
                "expected_prob_gain",
                "incremental_cost",
                "seats_to_majority",
                "rep_name",
                "hist_cost_to_compete",
                "abandon_reason",
            ]
            if c in filtered.columns
        ]
        st.dataframe(filtered[cols], use_container_width=True, height=520)
        st.download_button(
            f"Download {chamber} CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"state_{chamber}_rivs.csv",
            mime="text/csv",
            key=f"{prefix}_dl",
        )

    with tab_chambers:
        st.markdown("##### Path to chamber majority by state")
        if len(summary):
            st.dataframe(
                summary.style.format(
                    {
                        "expected_r_seats": "{:.1f}",
                        "top_rivs": "{:.2f}",
                    }
                ),
                use_container_width=True,
                height=480,
            )
        else:
            st.write("No summary rows.")

    with tab_method:
        st.markdown(
            f"""
### State {chamber} RIVS

Same core formula as federal House, but **target seats = majority of that state's chamber**
(e.g. floor(N/2)+1), computed **separately per state**.

| Mode | Meaning |
|------|---------|
| attack | Opposition-held competitive seat |
| defend | Own-party vulnerable seat |
| abandon | Capital better spent on multiple other seats *in that state's competitive set* |
| safe | Deep seats |

**Data**
- Members / control: mock or OpenStates (`OPENSTATES_API_KEY`)
- Lean: synthetic PVI-like from state lean + district noise (not Cook)
- Finance: mock / FTM-ready columns (`state_raised_*`); OpenFEC is federal-only

**Geo**
- Place optional files in `data/state/raw/{{ST}}_{chamber}.geojson` or `sldl.geojson` / `sldu.geojson`

**Refresh**
```bash
python scripts/generate_state_mock_data.py
python scripts/build_state_masters.py --openstates --states AZ GA
```
            """
        )
