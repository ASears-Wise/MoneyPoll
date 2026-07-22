"""Streamlit UI for state House / Senate chamber tabs (snapshot-first)."""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import RIVS_BOUNDS, RIVS_DEFAULTS
from src.charts import top_rivs_bar
from src.formatters import (
    COLUMN_HELP,
    format_display_frame,
    fmt_int,
    fmt_money,
    fmt_num,
    fmt_pvi,
)
from src.map_builder import build_district_map, build_simple_state_centroids_map
from src.openstates_client import get_api_key as get_openstates_key
from src.state_data import list_states, load_state_geojson, load_state_master, parse_state_candidates
from src.state_pipeline import fetch_and_overlay_openstates
from src.state_rivs import chamber_majority_summary, state_budget_simulation
from src.ui_common import (
    PRIORITY_STATES,
    PRESETS,
    STATE_EXTRA_COLS,
    STATE_SLIM_COLS,
    cached_state_rivs,
    data_status_badge,
    write_refresh_meta,
)

Chamber = Literal["lower", "upper"]


@st.cache_data(
    ttl=12 * 3600,
    show_spinner="Fetching OpenStates (manual refresh)…",
)
def _cached_openstates_master(chamber: str, cache_bust: int):
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
) -> tuple[pd.DataFrame, str, dict | None]:
    """Snapshot-first. OpenStates only on force_refresh."""
    bust_key = f"os_cache_bust_{chamber}"
    if bust_key not in st.session_state:
        st.session_state[bust_key] = 0

    raw, label = load_state_master(chamber)
    stats = None
    if force_refresh and get_openstates_key():
        st.session_state[bust_key] += 1
        _cached_openstates_master.clear()
        try:
            raw, label, stats = _cached_openstates_master(
                chamber, st.session_state[bust_key]
            )
            try:
                write_refresh_meta(
                    {
                        "state_refreshed_at": __import__("datetime")
                        .datetime.now(__import__("datetime").timezone.utc)
                        .strftime("%Y-%m-%d %H:%M UTC"),
                        "state_source": "openstates_live",
                    },
                    state=True,
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            st.warning(f"OpenStates refresh failed — using snapshot. ({e})")
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
                text=[fmt_money(v) for v in vals],
                textposition="auto",
                hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"State campaign finance (proxy) — {c}",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        yaxis_tickformat="$,.0f",
    )
    return fig


def _default_state_scoring() -> dict[str, Any]:
    d = RIVS_DEFAULTS
    return {
        "modes": ["attack", "defend"],
        "sel_states": list(PRIORITY_STATES),
        "cost_sensitivity": d["cost_sensitivity"],
        "risk": d["risk_tolerance"],
        "w_attack": d["w_attack"],
        "w_defend": d["w_defend"],
        "long_term": d["long_term_multiplier"],
        "inc_bonus": d["incumbent_bonus"],
        "open_vol": d["open_seat_volatility"],
        "abandon_enabled": True,
        "abandon_cost_percentile": d["abandon_cost_percentile"],
        "abandon_min_cost_m": 1.0,
        "abandon_alt_races": d["abandon_alt_races"],
        "abandon_alt_gain_ratio": d["abandon_alt_gain_ratio"],
        "budget_m": 5.0,
        "top_n": 100,
        "show_extra": False,
        "show_map": False,
        "map_state": PRIORITY_STATES[0],
    }


def render_state_chamber_tab(chamber: Chamber) -> None:
    title = "State House (lower chamber)" if chamber == "lower" else "State Senate (upper chamber)"
    st.subheader(title)
    st.caption(
        "Chamber-majority RIVS per state. **Priority states** loaded by default for speed. "
        "OpenStates only when you click refresh."
    )
    if chamber == "upper":
        st.info("Nebraska is unicameral — no upper seats.")

    prefix = "L" if chamber == "lower" else "U"
    os_key = get_openstates_key() is not None
    score_key = f"{prefix}_scoring"

    # Bootstrap snapshot for filters
    try:
        bootstrap, _ = load_state_master(chamber)
    except Exception as e:
        st.error(f"Could not load state master: {e}")
        return
    if bootstrap.empty:
        st.warning("No seats in this chamber.")
        return

    states = list_states(bootstrap)

    with st.sidebar.expander(f"State controls ({prefix})", expanded=True):
        st.caption("Snapshot-first · Apply scores to recompute")
        preset = st.selectbox(
            f"Preset ({prefix})",
            list(PRESETS.keys()),
            key=f"{prefix}_preset",
            help="Load opinionated modes/weights, then Apply.",
        )
        if st.button(f"Load preset ({prefix})", key=f"{prefix}_load_p"):
            p = PRESETS[preset]
            st.session_state[f"{prefix}_modes"] = list(p["modes"])
            st.session_state[f"{prefix}_wa"] = p["w_attack"]
            st.session_state[f"{prefix}_wd"] = p["w_defend"]
            st.session_state[f"{prefix}_cost"] = p["cost_sensitivity"]
            st.session_state[f"{prefix}_risk"] = p["risk_tolerance"]
            st.session_state[f"{prefix}_ab_en"] = p["abandon_enabled"]

        with st.form(f"{prefix}_form"):
            pack = st.radio(
                "State pack",
                ["Priority states", "All states", "Custom"],
                horizontal=True,
                help="Priority = swing-ish pack for faster tables.",
                key=f"{prefix}_pack",
            )
            custom_states = st.multiselect(
                "Custom states",
                states,
                default=st.session_state.get(f"{prefix}_states", PRIORITY_STATES),
                key=f"{prefix}_custom_states",
                help="Used when pack = Custom.",
            )
            modes = st.multiselect(
                "Mode",
                ["attack", "defend", "abandon", "safe"],
                default=st.session_state.get(f"{prefix}_modes", ["attack", "defend"]),
                help="Default hides safe for a tighter list.",
            )
            b, d = RIVS_BOUNDS, RIVS_DEFAULTS
            cost_sensitivity = st.slider(
                "Cost sensitivity",
                b["cost_sensitivity"][0],
                b["cost_sensitivity"][1],
                float(st.session_state.get(f"{prefix}_cost", d["cost_sensitivity"])),
                0.05,
                help="Higher = penalize expensive seats.",
            )
            risk = st.slider(
                "Risk tolerance",
                b["risk_tolerance"][0],
                b["risk_tolerance"][1],
                float(st.session_state.get(f"{prefix}_risk", d["risk_tolerance"])),
                0.01,
            )
            w_attack = st.slider(
                "Attack weight",
                b["w_attack"][0],
                b["w_attack"][1],
                float(st.session_state.get(f"{prefix}_wa", d["w_attack"])),
                0.05,
            )
            w_defend = st.slider(
                "Defend weight",
                b["w_defend"][0],
                b["w_defend"][1],
                float(st.session_state.get(f"{prefix}_wd", d["w_defend"])),
                0.05,
            )
            budget_m = st.slider("Budget sim ($M)", 0.5, 50.0, 5.0, 0.5)
            top_n = st.slider("Table top N", 25, 500, 100, 25)
            show_extra = st.checkbox("Extra columns", False)
            show_map = st.checkbox(
                "Show map",
                False,
                help="Requires a focus state + optional GeoJSON.",
            )
            map_state = st.selectbox(
                "Map focus state",
                states,
                index=states.index(PRIORITY_STATES[0]) if PRIORITY_STATES[0] in states else 0,
                help="Polygons only for this state when GeoJSON is present.",
            )
            with st.expander("Advanced", expanded=False):
                abandon_enabled = st.checkbox(
                    "Enable abandon",
                    value=bool(st.session_state.get(f"{prefix}_ab_en", True)),
                )
                abandon_min = st.slider("Abandon min $M", 0.2, 5.0, 1.0, 0.1)
                long_term = st.slider(
                    "Long-term mult",
                    b["long_term_multiplier"][0],
                    b["long_term_multiplier"][1],
                    d["long_term_multiplier"],
                    0.05,
                )
            applied = st.form_submit_button("Apply scores", type="primary")

        if applied:
            if pack == "Priority states":
                sel = [s for s in PRIORITY_STATES if s in states]
            elif pack == "All states":
                sel = []
            else:
                sel = custom_states
            st.session_state[score_key] = {
                "modes": modes,
                "sel_states": sel,
                "cost_sensitivity": cost_sensitivity,
                "risk": risk,
                "w_attack": w_attack,
                "w_defend": w_defend,
                "long_term": long_term,
                "inc_bonus": d["incumbent_bonus"],
                "open_vol": d["open_seat_volatility"],
                "abandon_enabled": abandon_enabled,
                "abandon_cost_percentile": d["abandon_cost_percentile"],
                "abandon_min_cost_m": abandon_min,
                "abandon_alt_races": d["abandon_alt_races"],
                "abandon_alt_gain_ratio": d["abandon_alt_gain_ratio"],
                "budget_m": budget_m,
                "top_n": top_n,
                "show_extra": show_extra,
                "show_map": show_map,
                "map_state": map_state,
            }

        st.markdown("##### OpenStates (manual)")
        st.write("Key:", "✅ set" if os_key else "⚪ not set")
        force_os = st.button(
            f"Refresh OpenStates ({prefix})",
            disabled=not os_key,
            help="Live member overlay — slow. Snapshot used until then.",
            key=f"{prefix}_os_btn",
        )

    ctl = st.session_state.get(score_key) or _default_state_scoring()

    try:
        raw, source_label, os_stats = resolve_state_master(
            chamber, force_refresh=bool(force_os) if os_key else False
        )
    except Exception as e:
        st.error(str(e))
        return

    if os_stats:
        st.success(
            f"OpenStates · {fmt_int(os_stats.get('seats_matched', 0))}/"
            f"{fmt_int(os_stats.get('total_seats', 0))} seats matched"
        )

    st.caption(data_status_badge(source_label, raw))

    # Score only filtered states when possible (speed)
    score_df = raw
    if ctl.get("sel_states"):
        score_df = raw[raw["state"].isin(ctl["sel_states"])]
        if score_df.empty:
            score_df = raw

    ranked = cached_state_rivs(
        score_df,
        float(ctl["cost_sensitivity"]),
        float(ctl["long_term"]),
        float(ctl["risk"]),
        float(ctl["w_attack"]),
        float(ctl["w_defend"]),
        float(ctl["inc_bonus"]),
        float(ctl["open_vol"]),
        bool(ctl["abandon_enabled"]),
        float(ctl["abandon_cost_percentile"]),
        float(ctl["abandon_min_cost_m"]),
        int(ctl["abandon_alt_races"]),
        float(ctl["abandon_alt_gain_ratio"]),
    )

    filtered = ranked.copy()
    if ctl.get("modes"):
        filtered = filtered[filtered["mode"].isin(ctl["modes"])]
    filtered = filtered.sort_values("rivs_rank").head(int(ctl.get("top_n") or 100)).reset_index(
        drop=True
    )

    sim = state_budget_simulation(filtered if len(filtered) else ranked, float(ctl["budget_m"]) * 1e6)
    summary = chamber_majority_summary(ranked)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Seats shown (R)",
        f"{fmt_int((filtered['party_control']=='R').sum())} / {fmt_int(len(filtered))}",
        help="After filters + top N.",
    )
    k2.metric(
        "Chambers R controls",
        fmt_int(int(summary["r_controls"].sum()) if len(summary) else 0),
        help="Among scored states.",
    )
    k3.metric(
        "Chambers short",
        fmt_int(int((~summary["r_controls"]).sum()) if len(summary) else 0),
    )
    k4.metric("Abandon", fmt_int((filtered["mode"] == "abandon").sum()))
    k5.metric(
        f"Budget {fmt_money(float(ctl['budget_m'])*1e6)} Δ",
        f"+{fmt_num(sim['total_prob_gain'], decimals=2)}",
        help=f"Funds {fmt_int(sim['n_districts_funded'])} seats",
    )

    t1, t2, t3, t4 = st.tabs(["📋 Seats", "🗺️ Map", "🏛️ Scoreboard", "📖 Methodology"])

    with t1:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.plotly_chart(top_rivs_bar(filtered if len(filtered) else ranked, n=15), use_container_width=True)
            cols = list(STATE_SLIM_COLS)
            if ctl.get("show_extra"):
                cols += [c for c in STATE_EXTRA_COLS if c not in cols]
            cols = [c for c in cols if c in filtered.columns]
            st.caption(" · ".join(f"{c}: {COLUMN_HELP[c]}" for c in cols if c in COLUMN_HELP)[:600])
            st.dataframe(format_display_frame(filtered, cols), use_container_width=True, height=420)
            st.download_button(
                f"CSV {chamber}",
                filtered.to_csv(index=False).encode("utf-8"),
                f"state_{chamber}.csv",
                "text/csv",
                key=f"{prefix}_dl",
            )
        with right:
            ids = filtered["district_id"].tolist() if len(filtered) else ranked["district_id"].tolist()
            pick = st.selectbox("Seat detail", ids, key=f"{prefix}_pick")
            row = ranked[ranked["district_id"] == pick]
            if len(row):
                r = row.iloc[0]
                st.markdown(f"### {r.get('seat_label', r['district_id'])}")
                m1, m2, m3 = st.columns(3)
                m1.metric("RIVS", fmt_num(r["rivs"], decimals=2), help="Investment value score.")
                m2.metric("Mode", str(r["mode"]).title())
                m3.metric("Lean", fmt_pvi(r["pvi"]), help="Synthetic lean unless replaced.")
                if str(r["mode"]).lower() == "abandon":
                    st.warning("Abandon — redeploy $.")
                st.write(
                    f"**Member:** {r.get('rep_name')} ({r.get('rep_party')}) · "
                    f"**Majority need:** {fmt_int(r.get('chamber_majority_threshold'))}"
                )
                cands = parse_state_candidates(r)
                if cands:
                    st.dataframe(pd.DataFrame(cands), hide_index=True, use_container_width=True)
                st.plotly_chart(_state_finance_bars(r), use_container_width=True, key=f"{prefix}_fin")

    with t2:
        if not ctl.get("show_map"):
            st.info("Map off. Enable **Show map** in the form → **Apply scores**. Focus one state.")
        else:
            focus = ctl.get("map_state") or PRIORITY_STATES[0]
            geo = load_state_geojson(chamber, state=focus)
            view = filtered[filtered["state"] == focus] if focus else filtered
            if view.empty:
                view = ranked[ranked["state"] == focus] if focus else ranked
            try:
                from streamlit_folium import st_folium

                if geo is not None:
                    for feat in geo.get("features") or []:
                        props = feat.setdefault("properties", {})
                        if "district_id" not in props and focus:
                            dnum = (
                                props.get("district")
                                or props.get("SLDLST")
                                or props.get("SLDUST")
                            )
                            if dnum is not None:
                                try:
                                    n = int(float(str(dnum).lstrip("0") or "0"))
                                    tag = "L" if chamber == "lower" else "U"
                                    props["district_id"] = f"{focus}-{tag}-{n:03d}"
                                except ValueError:
                                    pass
                    fmap = build_district_map(view if len(view) else ranked, geo)
                    st.caption(f"Polygons for **{focus}** (when GeoJSON exists).")
                else:
                    fmap = build_simple_state_centroids_map(ranked)
                    st.caption(
                        f"No GeoJSON for {focus}. Place "
                        f"`data/state/raw/{focus}_{chamber}.geojson` for districts."
                    )
                st_folium(fmap, width=None, height=480, returned_objects=[], key=f"{prefix}_map")
            except ImportError:
                st.warning("streamlit-folium missing")

    with t3:
        st.caption("Majority path by state (scored set).")
        if len(summary):
            st.dataframe(format_display_frame(summary), use_container_width=True, height=480)
        else:
            st.write("No rows.")

    with t4:
        st.markdown(
            f"""
### State {chamber} performance notes
- Default **priority states**: {", ".join(PRIORITY_STATES)}
- Modes default to **attack + defend** (hides safe)
- OpenStates: **Refresh** button only
- Map: opt-in, **one state** at a time
- Nightly snapshot: GitHub Action `refresh-data.yml`
            """
        )
