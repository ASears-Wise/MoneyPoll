"""Home / overview tab — one-screen story before full tools."""
from __future__ import annotations

import streamlit as st

from src.formatters import format_display_frame, fmt_int
from src.ui_common import (
    PRIORITY_STATES,
    attach_cost_to_compete,
    cached_federal_rivs,
    cached_state_rivs,
    data_status_badge,
)


def render_home(
    federal_df,
    federal_label: str,
    lower_df,
    lower_label: str,
    upper_df,
    upper_label: str,
) -> None:
    st.subheader("Overview")
    st.caption(
        "Snapshot-first: the app opens instantly from committed data. "
        "Use **Refresh** on each tab only when you want a live API pull."
    )

    st.info(
        data_status_badge(federal_label, federal_df)
        + " · State lower: `"
        + lower_label
        + "` · upper: `"
        + upper_label
        + "`"
    )

    # Fast default RIVS (cached)
    with st.spinner("Scoring snapshot…"):
        fed = cached_federal_rivs(
            federal_df,
            1.0,
            1.15,
            0.52,
            1.2,
            1.0,
            0.06,
            0.04,
            230,
            True,
            0.82,
            6.0,
            3,
            1.15,
        )
        # Priority states only for state overview speed
        low_sub = lower_df[lower_df["state"].isin(PRIORITY_STATES)] if len(lower_df) else lower_df
        up_sub = upper_df[upper_df["state"].isin(PRIORITY_STATES)] if len(upper_df) else upper_df
        low = cached_state_rivs(
            low_sub if len(low_sub) else lower_df.head(500),
            1.0,
            1.15,
            0.52,
            1.2,
            1.0,
            0.06,
            0.04,
            True,
            0.82,
            1.0,
            3,
            1.15,
            None,
        )
        up = cached_state_rivs(
            up_sub if len(up_sub) else upper_df.head(500),
            1.0,
            1.15,
            0.52,
            1.2,
            1.0,
            0.06,
            0.04,
            True,
            0.82,
            1.0,
            3,
            1.15,
            None,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("US House seats", fmt_int(len(fed)), help="Rows in federal master.")
    c2.metric(
        "R-held (House)",
        fmt_int((fed["party_control"] == "R").sum()),
        help="Seats coded Republican in snapshot.",
    )
    c3.metric(
        "House abandon flags",
        fmt_int((fed["mode"] == "abandon").sum()),
        help="High-cost races under default scoring.",
    )
    c4.metric(
        "State seats scored (sample)",
        fmt_int(len(low) + len(up)),
        help="Priority-state sample for overview speed.",
    )

    fed = attach_cost_to_compete(fed)
    low = attach_cost_to_compete(low)
    up = attach_cost_to_compete(up)

    st.markdown("##### Top 15 US House by RIVS")
    st.caption("Cost to be competitive shown for every district (2026-first finance).")
    top = fed.nsmallest(15, "rivs_rank")[
        [
            c
            for c in (
                "rivs_rank",
                "district_id",
                "mode",
                "rivs",
                "cost_to_be_competitive",
                "party_control",
                "rep_name",
            )
            if c in fed.columns
        ]
    ]
    st.dataframe(format_display_frame(top), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### House: abandon (do not double-down)")
        ab = fed[fed["mode"] == "abandon"].nsmallest(10, "rivs_rank")
        if len(ab):
            st.dataframe(
                format_display_frame(
                    ab[
                        [
                            c
                            for c in (
                                "district_id",
                                "mode",
                                "rivs",
                                "cost_to_be_competitive",
                                "abandon_reason",
                            )
                            if c in ab.columns
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No abandon seats under default parameters.")
    with col_b:
        st.markdown("##### State House — top RIVS (priority states)")
        st.caption(f"States: {', '.join(PRIORITY_STATES)}")
        st.dataframe(
            format_display_frame(
                low.nsmallest(12, "rivs_rank")[
                    [
                        c
                        for c in (
                            "rivs_rank",
                            "seat_label",
                            "state",
                            "mode",
                            "rivs",
                            "cost_to_be_competitive",
                            "party_control",
                        )
                        if c in low.columns
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("##### Go deeper")
    st.markdown(
        """
1. **US House** — full map (optional), filters, budget sim, FEC refresh  
2. **State House / Senate** — chamber majorities, OpenStates refresh  
3. Use **presets** (Flips only, Protect majority) then **Apply scores** so sliders do not recompute on every drag  
        """
    )
