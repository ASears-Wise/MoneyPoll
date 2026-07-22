"""Streamlit UI for state House / Senate (one state required; chamber-aware sidebar)."""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import RIVS_BOUNDS, RIVS_DEFAULTS, STATE_CHAMBER_SEATS
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
    PRESETS,
    STATE_EXTRA_COLS,
    STATE_SLIM_COLS,
    cached_state_rivs,
    data_status_badge,
    write_refresh_meta,
)

Chamber = Literal["lower", "upper"]

CHAMBER_LABEL = {
    "lower": "State House (lower chamber)",
    "upper": "State Senate (upper chamber)",
}
CHAMBER_SHORT = {"lower": "House", "upper": "Senate"}


def _majority(n: int) -> int:
    return n // 2 + 1 if n > 0 else 0


def _chamber_size(state: str, chamber: Chamber) -> int:
    return int(STATE_CHAMBER_SEATS.get(state, {}).get(chamber, 0) or 0)


@st.cache_data(ttl=12 * 3600, show_spinner="Fetching OpenStates for selected state…")
def _cached_openstates_state(chamber: str, state: str, cache_bust: int):
    return fetch_and_overlay_openstates(
        chamber,  # type: ignore[arg-type]
        states=[state],
        progress=False,
        persist=True,
    )


def resolve_state_master(
    chamber: Chamber,
    *,
    state: str | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str, dict | None]:
    """Load chamber master; optional single-state OpenStates refresh."""
    raw, label = load_state_master(chamber)
    stats = None
    if force_refresh and state and get_openstates_key():
        bust_key = f"os_bust_{chamber}_{state}"
        st.session_state[bust_key] = st.session_state.get(bust_key, 0) + 1
        _cached_openstates_state.clear()
        try:
            raw, label, stats = _cached_openstates_state(
                chamber, state, st.session_state[bust_key]
            )
            try:
                write_refresh_meta(
                    {
                        "state_refreshed_at": __import__("datetime")
                        .datetime.now(__import__("datetime").timezone.utc)
                        .strftime("%Y-%m-%d %H:%M UTC"),
                        "state_source": f"openstates:{state}",
                    },
                    state=True,
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            st.warning(f"OpenStates refresh failed — using snapshot. ({e})")
    if state:
        raw = raw[raw["state"] == state].copy()
        label = f"{label} · {state} only"
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


def render_state_chamber_tab(chamber: Chamber) -> None:
    """
    State House or Senate view.

    - Sidebar is chamber-specific (seat count, majority, target for that state).
    - User must pick a state before any seat data / scoring loads.
    """
    label = CHAMBER_LABEL[chamber]
    short = CHAMBER_SHORT[chamber]
    prefix = "L" if chamber == "lower" else "U"
    score_key = f"{prefix}_scoring"

    st.subheader(label)
    st.caption(
        f"Select a **state** first. Scores target that state's {short} majority "
        f"(not the federal 230-seat goal)."
    )

    # Lightweight master for state list only (no scoring yet)
    try:
        all_rows, _all_lab = load_state_master(chamber)
    except Exception as e:
        st.error(f"Could not load state master: {e}")
        return

    available = list_states(all_rows)
    if chamber == "upper":
        # Drop NE (unicameral)
        available = [s for s in available if _chamber_size(s, "upper") > 0]

    if not available:
        st.warning("No states available for this chamber.")
        return

    # ── Sidebar: state gate + chamber context ─────────────────────────
    st.sidebar.markdown(f"### 🏛️ {short} controls")
    st.sidebar.caption("Pick a state to load seats. Toolbar values match that chamber.")

    state_options = ["— Select a state —"] + available
    prev = st.session_state.get(f"{prefix}_selected_state")
    default_idx = 0
    if prev in available:
        default_idx = state_options.index(prev)

    selected = st.sidebar.selectbox(
        f"{short} — state",
        state_options,
        index=default_idx,
        key=f"{prefix}_state_picker",
        help="Required. Data and RIVS load only after you choose a state.",
    )

    if selected == "— Select a state —":
        st.session_state[f"{prefix}_selected_state"] = None
        st.sidebar.info("Choose a state above to configure scoring and load data.")
        st.warning(
            f"**Select a state** in the left sidebar to load {short} districts, "
            f"majority math, and RIVS."
        )
        st.markdown(
            f"""
### What you'll get after picking a state
- Total **{short} seats** in that state  
- **Majority threshold** (floor(N/2)+1)  
- Current **R-held** count and **seats short** of majority  
- Scoring knobs tuned for **state-level** spend (not federal House $)  
- Map and tables for **that state only**  
            """
        )
        return

    st.session_state[f"{prefix}_selected_state"] = selected
    state = selected

    # Chamber geometry for this state
    n_seats_cfg = _chamber_size(state, chamber)
    maj_cfg = _majority(n_seats_cfg)

    # Optional OpenStates refresh for this state only
    os_key = get_openstates_key() is not None
    force_os = st.sidebar.button(
        f"Refresh OpenStates ({state} {short})",
        disabled=not os_key,
        help=f"Pull live legislators for {state} {short} only (not all 50 states).",
        key=f"{prefix}_os_btn",
    )
    st.sidebar.write("OpenStates key:", "✅ set" if os_key else "⚪ not set")

    raw, source_label, os_stats = resolve_state_master(
        chamber,
        state=state,
        force_refresh=bool(force_os) and os_key,
    )
    if os_stats:
        st.success(
            f"OpenStates · {fmt_int(os_stats.get('seats_matched', 0))} seats updated for {state}"
        )

    if raw.empty:
        st.error(f"No {short} seats found for **{state}** in the snapshot.")
        return

    # Live counts from data (prefer data over static table if mismatch)
    n_seats = int(raw["chamber_total_seats"].iloc[0]) if "chamber_total_seats" in raw.columns else len(raw)
    if n_seats_cfg and n_seats_cfg != n_seats:
        n_seats = n_seats_cfg  # config is source of truth for majority math display
    maj = (
        int(raw["chamber_majority_threshold"].iloc[0])
        if "chamber_majority_threshold" in raw.columns
        else maj_cfg
    )
    if maj_cfg:
        maj = maj_cfg
    r_held_now = int((raw["party_control"] == "R").sum())
    seats_short = max(0, maj - r_held_now)

    # Sidebar chamber dashboard
    st.sidebar.markdown(f"##### {state} {short} at a glance")
    st.sidebar.metric(
        f"Total {short} seats",
        fmt_int(n_seats),
        help=f"Size of the {state} {short}.",
    )
    st.sidebar.metric(
        "Majority needed",
        fmt_int(maj),
        help=f"Seats required for majority control (usually {n_seats}//2+1).",
    )
    st.sidebar.metric(
        "R-held now",
        fmt_int(r_held_now),
        help="Seats coded Republican in the current snapshot.",
    )
    st.sidebar.metric(
        "Seats short of majority",
        fmt_int(seats_short),
        help="How many more R seats needed to control this chamber (0 = already majority).",
        delta="majority" if seats_short == 0 else f"-{seats_short} to go",
        delta_color="normal" if seats_short == 0 else "inverse",
    )

    # Presets
    preset = st.sidebar.selectbox(
        "Preset",
        list(PRESETS.keys()),
        key=f"{prefix}_preset",
        help="Opinionated scoring defaults for this chamber.",
    )
    if st.sidebar.button("Load preset", key=f"{prefix}_load_p"):
        p = PRESETS[preset]
        st.session_state[f"{prefix}_modes"] = list(p["modes"])
        st.session_state[f"{prefix}_wa"] = p["w_attack"]
        st.session_state[f"{prefix}_wd"] = p["w_defend"]
        st.session_state[f"{prefix}_cost"] = p["cost_sensitivity"]
        st.session_state[f"{prefix}_risk"] = p["risk_tolerance"]
        st.session_state[f"{prefix}_ab_en"] = p["abandon_enabled"]
        st.sidebar.success(PRESETS[preset]["description"])

    b, d = RIVS_BOUNDS, RIVS_DEFAULTS
    # State races: cheaper default budget (thousands–millions, not $50M federal)
    max_budget_m = 15.0 if n_seats >= 100 else 8.0
    default_budget = min(2.0, max_budget_m)

    with st.sidebar.form(f"{prefix}_form"):
        st.markdown(f"##### Score {state} {short}")
        modes = st.multiselect(
            "Mode",
            ["attack", "defend", "abandon", "safe"],
            default=st.session_state.get(f"{prefix}_modes", ["attack", "defend"]),
            help="attack=flip · defend=hold · abandon=redeploy · safe=deep",
        )
        target_seats = st.number_input(
            f"Target R seats ({state} {short})",
            min_value=1,
            max_value=max(n_seats, 1),
            value=int(st.session_state.get(f"{prefix}_target", maj)),
            step=1,
            help=(
                f"Strategic R seat goal for this chamber. "
                f"Default = majority ({maj} of {n_seats}). "
                f"Not the federal 230 target."
            ),
        )
        st.caption(f"Majority line is **{fmt_int(maj)}** of **{fmt_int(n_seats)}** seats.")

        cost_sensitivity = st.slider(
            "Cost sensitivity",
            b["cost_sensitivity"][0],
            b["cost_sensitivity"][1],
            float(st.session_state.get(f"{prefix}_cost", d["cost_sensitivity"])),
            0.05,
            help="Higher = penalize expensive seats more (state-scale costs).",
        )
        risk = st.slider(
            "Risk tolerance (P*)",
            b["risk_tolerance"][0],
            b["risk_tolerance"][1],
            float(st.session_state.get(f"{prefix}_risk", d["risk_tolerance"])),
            0.01,
            help="Target win probability that counts as enough investment.",
        )
        w_attack = st.slider(
            "Attack weight",
            b["w_attack"][0],
            b["w_attack"][1],
            float(st.session_state.get(f"{prefix}_wa", d["w_attack"])),
            0.05,
            help="Priority for flipping opposition seats in this chamber.",
        )
        w_defend = st.slider(
            "Defend weight",
            b["w_defend"][0],
            b["w_defend"][1],
            float(st.session_state.get(f"{prefix}_wd", d["w_defend"])),
            0.05,
            help="Priority for protecting vulnerable holds.",
        )
        budget_m = st.slider(
            f"Budget sim ($M) — {state}",
            0.1,
            max_budget_m,
            float(st.session_state.get(f"{prefix}_bud", default_budget)),
            0.1,
            help=(
                f"Simulated dollars for {state} {short} only. "
                f"State races are cheaper than federal House — scale is lower by default."
            ),
        )
        top_n = st.slider(
            "Table top N",
            10,
            max(n_seats, 10),
            min(50, max(n_seats, 10)),
            5,
            help=f"Show at most this many seats (chamber has {n_seats}).",
        )
        show_extra = st.checkbox("Extra table columns", False)
        show_map = st.checkbox(
            "Show district map",
            True,
            help=f"Map for {state} only (lighter than national).",
        )
        with st.expander("Advanced", expanded=False):
            abandon_enabled = st.checkbox(
                "Enable abandon",
                value=bool(st.session_state.get(f"{prefix}_ab_en", True)),
                help="Flag seats where $ is better spread within this state.",
            )
            abandon_min = st.slider(
                "Abandon min cost ($M)",
                0.05,
                3.0,
                0.4,
                0.05,
                help="State races: lower floor than federal House.",
            )
            long_term = st.slider(
                "Long-term mult",
                b["long_term_multiplier"][0],
                b["long_term_multiplier"][1],
                d["long_term_multiplier"],
                0.05,
            )
            skip_abandon_budget = st.checkbox("Budget: skip abandon", True)

        applied = st.form_submit_button("Apply scores", type="primary", use_container_width=True)

    if applied:
        st.session_state[score_key] = {
            "modes": modes,
            "target_seats": int(target_seats),
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
            "skip_abandon_budget": skip_abandon_budget,
        }
        st.session_state[f"{prefix}_target"] = int(target_seats)
        st.session_state[f"{prefix}_bud"] = budget_m

    ctl = st.session_state.get(score_key) or {
        "modes": ["attack", "defend"],
        "target_seats": maj,
        "cost_sensitivity": d["cost_sensitivity"],
        "risk": d["risk_tolerance"],
        "w_attack": d["w_attack"],
        "w_defend": d["w_defend"],
        "long_term": d["long_term_multiplier"],
        "inc_bonus": d["incumbent_bonus"],
        "open_vol": d["open_seat_volatility"],
        "abandon_enabled": True,
        "abandon_cost_percentile": d["abandon_cost_percentile"],
        "abandon_min_cost_m": 0.4,
        "abandon_alt_races": d["abandon_alt_races"],
        "abandon_alt_gain_ratio": d["abandon_alt_gain_ratio"],
        "budget_m": default_budget,
        "top_n": min(50, n_seats),
        "show_extra": False,
        "show_map": True,
        "skip_abandon_budget": True,
    }

    st.caption(data_status_badge(source_label, raw))

    # Force chamber majority columns for this state before RIVS
    work = raw.copy()
    work["chamber_total_seats"] = n_seats
    work["chamber_majority_threshold"] = int(ctl["target_seats"])

    ranked = cached_state_rivs(
        work,
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
        int(ctl["target_seats"]),
    )
    # Re-attach true majority for display (target may differ from majority)
    ranked["chamber_majority_threshold"] = maj
    ranked["chamber_target_seats"] = int(ctl["target_seats"])
    ranked["state_r_held"] = r_held_now
    ranked["seats_to_majority"] = seats_short

    filtered = ranked.copy()
    if ctl.get("modes"):
        filtered = filtered[filtered["mode"].isin(ctl["modes"])]
    filtered = (
        filtered.sort_values("rivs_rank")
        .head(int(ctl.get("top_n") or 50))
        .reset_index(drop=True)
    )

    sim = state_budget_simulation(
        filtered if len(filtered) else ranked,
        float(ctl["budget_m"]) * 1_000_000,
        skip_abandon=bool(ctl.get("skip_abandon_budget", True)),
    )
    summary = chamber_majority_summary(ranked)

    # Main KPIs — state chamber framed
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"{state} {short} seats", fmt_int(n_seats), help="Chamber size.")
    k2.metric("Majority", fmt_int(maj), help="Seats to control the chamber.")
    k3.metric("R-held", fmt_int(r_held_now), help="Current Republican seats.")
    k4.metric(
        "Short of majority",
        fmt_int(seats_short),
        help="Additional R seats needed (0 = R controls).",
    )
    k5.metric(
        "Score target",
        fmt_int(ctl["target_seats"]),
        help="RIVS path-to-majority target (sidebar).",
    )
    k6.metric(
        f"Budget {fmt_money(float(ctl['budget_m']) * 1e6)} Δ",
        f"+{fmt_num(sim['total_prob_gain'], decimals=2)}",
        help=f"Greedy sim funds {fmt_int(sim['n_districts_funded'])} seats in {state}.",
    )

    t1, t2, t3, t4 = st.tabs(["📋 Seats", "🗺️ Map", "🏛️ Majority board", "📖 Methodology"])

    with t1:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.plotly_chart(
                top_rivs_bar(filtered if len(filtered) else ranked, n=min(15, len(ranked))),
                use_container_width=True,
            )
            cols = list(STATE_SLIM_COLS)
            if ctl.get("show_extra"):
                cols += [c for c in STATE_EXTRA_COLS if c not in cols]
            cols = [c for c in cols if c in filtered.columns]
            st.caption(
                " · ".join(f"{c}: {COLUMN_HELP[c]}" for c in cols if c in COLUMN_HELP)[:600]
            )
            st.dataframe(
                format_display_frame(filtered, cols),
                use_container_width=True,
                height=420,
            )
            st.download_button(
                f"CSV {state} {short}",
                filtered.to_csv(index=False).encode("utf-8"),
                f"state_{chamber}_{state}.csv",
                "text/csv",
                key=f"{prefix}_dl",
            )
        with right:
            ids = (
                filtered["district_id"].tolist()
                if len(filtered)
                else ranked["district_id"].tolist()
            )
            pick = st.selectbox(
                f"{state} seat detail",
                ids,
                key=f"{prefix}_pick",
                help="District detail for this state chamber.",
            )
            row = ranked[ranked["district_id"] == pick]
            if len(row):
                r = row.iloc[0]
                st.markdown(f"### {r.get('seat_label', r['district_id'])}")
                m1, m2, m3 = st.columns(3)
                m1.metric("RIVS", fmt_num(r["rivs"], decimals=2), help="ROI score in this chamber.")
                m2.metric("Mode", str(r["mode"]).title())
                m3.metric("Lean", fmt_pvi(r["pvi"]), help="Synthetic lean unless replaced.")
                if str(r["mode"]).lower() == "abandon":
                    st.warning("Abandon — redeploy $ within this state.")
                    if r.get("abandon_reason"):
                        st.caption(str(r["abandon_reason"]))
                st.write(
                    f"**Member:** {r.get('rep_name')} ({r.get('rep_party')}) · "
                    f"**Chamber majority:** {fmt_int(maj)} · "
                    f"**R held:** {fmt_int(r_held_now)} · "
                    f"**Short:** {fmt_int(seats_short)}"
                )
                cands = parse_state_candidates(r)
                if cands:
                    st.dataframe(pd.DataFrame(cands), hide_index=True, use_container_width=True)
                st.plotly_chart(
                    _state_finance_bars(r), use_container_width=True, key=f"{prefix}_fin"
                )

    with t2:
        if not ctl.get("show_map"):
            st.info("Map off. Enable **Show district map** in the form → **Apply scores**.")
        else:
            geo = load_state_geojson(chamber, state=state)
            try:
                from streamlit_folium import st_folium

                view = filtered if len(filtered) else ranked
                if geo is not None:
                    for feat in geo.get("features") or []:
                        props = feat.setdefault("properties", {})
                        if "district_id" not in props:
                            dnum = (
                                props.get("district")
                                or props.get("SLDLST")
                                or props.get("SLDUST")
                            )
                            if dnum is not None:
                                try:
                                    n = int(float(str(dnum).lstrip("0") or "0"))
                                    tag = "L" if chamber == "lower" else "U"
                                    props["district_id"] = f"{state}-{tag}-{n:03d}"
                                except ValueError:
                                    pass
                    fmap = build_district_map(view, geo)
                    st.caption(f"District map — **{state} {short}**")
                else:
                    fmap = build_simple_state_centroids_map(view)
                    st.caption(
                        f"No GeoJSON for {state}. Add "
                        f"`data/state/raw/{state}_{chamber}.geojson` for district polygons."
                    )
                st_folium(fmap, width=None, height=480, returned_objects=[], key=f"{prefix}_map")
            except ImportError:
                st.warning("streamlit-folium missing")

    with t3:
        st.markdown(f"##### {state} {short} majority path")
        st.write(
            {
                "state": state,
                "chamber": short,
                "total_seats": n_seats,
                "majority": maj,
                "r_held": r_held_now,
                "seats_short": seats_short,
                "r_controls": seats_short == 0,
                "score_target": int(ctl["target_seats"]),
            }
        )
        if len(summary):
            st.dataframe(format_display_frame(summary), use_container_width=True, hide_index=True)

    with t4:
        st.markdown(
            f"""
### {state} {short}

| Field | Value |
|-------|--------|
| Chamber size | **{n_seats}** |
| Majority | **{maj}** |
| R-held (snapshot) | **{r_held_now}** |
| Seats short | **{seats_short}** |

RIVS uses **this chamber's target** (sidebar), not federal 230.  
Budget defaults are **state-scale** ($M, lower ceiling than US House).  
OpenStates refresh is **{state}-only** for speed.
            """
        )
