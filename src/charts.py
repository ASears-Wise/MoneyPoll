"""Plotly charts for district detail and summaries."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fec_bars(row: pd.Series, cycle: str = "2024") -> go.Figure:
    """Raised/spent by party for a cycle."""
    c = str(cycle)
    labels = ["R Raised", "R Spent", "D Raised", "D Spent"]
    keys = [
        f"fec_raised_r_{c}",
        f"fec_spent_r_{c}",
        f"fec_raised_d_{c}",
        f"fec_spent_d_{c}",
    ]
    vals = []
    for k in keys:
        vals.append(float(row[k]) if k in row.index and pd.notna(row[k]) else 0.0)

    colors = ["#e81b23", "#a01218", "#00aef3", "#006b9a"]
    fig = go.Figure(
        data=[
            go.Bar(x=labels, y=vals, marker_color=colors, text=[f"${v/1e6:.2f}M" for v in vals], textposition="auto")
        ]
    )
    fig.update_layout(
        title=f"FEC Finance — {c} cycle",
        yaxis_title="USD",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    if c == "2024" and "fec_outside_2024" in row.index and pd.notna(row["fec_outside_2024"]):
        fig.add_annotation(
            text=f"Outside spending (2024): ${float(row['fec_outside_2024'])/1e6:.2f}M",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
        )
    return fig


def demographics_pie(row: pd.Series) -> go.Figure:
    labels = ["White", "Black", "Hispanic", "Asian", "Other"]
    w = float(row.get("pct_white", 0) or 0)
    b = float(row.get("pct_black", 0) or 0)
    h = float(row.get("pct_hispanic", 0) or 0)
    a = float(row.get("pct_asian", 0) or 0)
    other = max(0.0, 1.0 - w - b - h - a)
    fig = px.pie(
        names=labels,
        values=[w, b, h, a, other],
        title="Racial / ethnic share (ACS-style)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def top_rivs_bar(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.nsmallest(len(df), "rivs_rank").head(n).iloc[::-1]
    fig = px.bar(
        top,
        x="rivs",
        y="district_id",
        color="mode",
        orientation="h",
        title=f"Top {n} districts by RIVS",
        color_discrete_map={"attack": "#e81b23", "defend": "#3c3b6e", "safe": "#888888"},
    )
    fig.update_layout(height=420, margin=dict(l=60, r=20, t=50, b=40), template="plotly_white")
    return fig


def parse_candidates(row: pd.Series) -> list[dict[str, Any]]:
    raw = row.get("civic_candidates_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return list(data.get("candidates") or [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []
