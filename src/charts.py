"""Plotly charts for district detail and summaries."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fec_bars(row: pd.Series, cycle: str = "2026") -> go.Figure:
    """Raised/spent by party for a cycle (default: current 2026 cycle)."""
    c = str(cycle)
    labels = ["R Raised", "R Spent", "D Raised", "D Spent"]
    keys = [
        f"fec_raised_r_{c}",
        f"fec_spent_r_{c}",
        f"fec_raised_d_{c}",
        f"fec_spent_d_{c}",
    ]
    # Fall back to prior cycle columns if current-cycle fields are all empty
    vals = []
    for k in keys:
        vals.append(float(row[k]) if k in row.index and pd.notna(row[k]) else 0.0)
    if sum(vals) == 0 and c == "2026":
        return fec_bars(row, cycle="2024")

    colors = ["#e81b23", "#a01218", "#00aef3", "#006b9a"]

    def _money_label(v: float) -> str:
        if abs(v) >= 1_000_000:
            m = v / 1_000_000
            return f"${m:,.1f}M" if abs(m - round(m)) > 0.05 else f"${int(round(m)):,}M"
        if abs(v) >= 1_000:
            return f"${int(round(v / 1_000)):,}K"
        return f"${int(round(v)):,}"

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[_money_label(v) for v in vals],
                textposition="auto",
                hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"FEC Finance — {c} cycle",
        yaxis_title="USD",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        yaxis_tickformat="$,.0f",
    )
    # Outside spend: prefer matching cycle column, else 2024 IE aggregate
    outside_key = f"fec_outside_{c}"
    if outside_key not in row.index or pd.isna(row.get(outside_key)):
        outside_key = "fec_outside_2024"
    if outside_key in row.index and pd.notna(row.get(outside_key)) and float(row[outside_key] or 0) > 0:
        fig.add_annotation(
            text=f"Outside spending ({outside_key.split('_')[-1]}): {_money_label(float(row[outside_key]))}",
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
    top = df.nsmallest(len(df), "rivs_rank").head(n).copy()
    # Label with cost to be competitive when available
    cost_col = (
        "cost_to_be_competitive"
        if "cost_to_be_competitive" in top.columns
        else "hist_cost_to_compete"
        if "hist_cost_to_compete" in top.columns
        else "incremental_cost"
        if "incremental_cost" in top.columns
        else None
    )
    if cost_col:
        top = top.copy()
        top["y_label"] = top.apply(
            lambda r: f"{r['district_id']}  (${float(r[cost_col] or 0):,.0f} to compete)",
            axis=1,
        )
        y_col = "y_label"
    else:
        y_col = "district_id"
    top = top.iloc[::-1]
    fig = px.bar(
        top,
        x="rivs",
        y=y_col,
        color="mode",
        orientation="h",
        title=f"Top {n} districts by RIVS (label includes cost to be competitive)",
        color_discrete_map={
            "attack": "#e81b23",
            "defend": "#3c3b6e",
            "safe": "#888888",
            "abandon": "#f59e0b",  # amber — do not double-down
        },
    )
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=50, b=40),
        template="plotly_white",
        xaxis_title="RIVS",
        yaxis_title="District",
    )
    fig.update_traces(hovertemplate="%{y}<br>RIVS %{x:.2f}<extra></extra>")
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
