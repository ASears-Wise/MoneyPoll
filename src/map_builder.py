"""Folium choropleth map for House districts."""
from __future__ import annotations

from typing import Any

import branca.colormap as cm
import folium
import pandas as pd

from src.data_loader import normalize_geo_district_id


def _rivs_color(val: float, vmin: float, vmax: float) -> str:
    """Green (high RIVS) → yellow → red (low)."""
    if vmax <= vmin:
        return "#808080"
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # high t = green
    r = int(255 * (1 - t))
    g = int(180 + 75 * t)
    b = int(60 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def build_district_map(
    df: pd.DataFrame,
    geojson: dict[str, Any] | None,
    *,
    selected: str | None = None,
    value_col: str = "rivs",
    show_r: bool = True,
    show_d: bool = True,
    show_competitive_only: bool = False,
) -> folium.Map:
    """
    Build Folium map. If geojson is None, returns a simple continental US basemap
    with a marker note (caller should show table fallback).
    """
    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="cartodbpositron",
    )

    if geojson is None:
        folium.Marker(
            [39.5, -98.35],
            popup=(
                "GeoJSON not found. Run scripts/download_geojson.py "
                "or place boundaries at data/raw/cds.geojson"
            ),
            tooltip="No district boundaries loaded",
        ).add_to(m)
        return m

    # Index metrics
    idx = df.set_index("district_id", drop=False)
    vals = idx[value_col].astype(float)
    vmin, vmax = float(vals.quantile(0.05)), float(vals.quantile(0.95))
    if vmax <= vmin:
        vmin, vmax = float(vals.min()), float(vals.max() + 1e-6)

    colormap = cm.LinearColormap(
        colors=["#d73027", "#fee08b", "#1a9850"],
        vmin=vmin,
        vmax=vmax,
        caption="RIVS (high = better investment value)",
    )
    colormap.add_to(m)

    # Layer groups
    fg_all = folium.FeatureGroup(name="All (filtered)", show=True)
    fg_r = folium.FeatureGroup(name="R-held", show=False)
    fg_d = folium.FeatureGroup(name="D-held", show=False)
    fg_comp = folium.FeatureGroup(name="Competitive (mode ≠ safe)", show=False)

    features = geojson.get("features") or []
    for feat in features:
        props = dict(feat.get("properties") or {})
        did = normalize_geo_district_id(props)
        # Try geoid join
        row = None
        if did and did in idx.index:
            row = idx.loc[did]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        elif did:
            # At-large alt: XX-01 vs XX-00
            alt = did[:-2] + "00" if did.endswith("-01") else did[:-2] + "01"
            if alt in idx.index:
                row = idx.loc[alt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                did = alt
        if row is None and (props.get("GEOID") or props.get("geoid")):
            geoid = str(props.get("GEOID") or props.get("geoid"))
            match = df[df["geoid"].astype(str) == geoid]
            if len(match):
                row = match.iloc[0]
                did = str(row["district_id"])

        if row is None:
            # Unmatched polygon — light gray outline only
            folium.GeoJson(
                feat,
                style_function=lambda x: {
                    "fillColor": "#cccccc",
                    "color": "#666666",
                    "weight": 0.3,
                    "fillOpacity": 0.15,
                },
            ).add_to(fg_all)
            continue

        party = str(row["party_control"]).upper()
        mode = str(row.get("mode", "")).lower()
        rivs = float(row[value_col])
        color = _rivs_color(rivs, vmin, vmax)

        if not show_r and party == "R":
            continue
        if not show_d and party == "D":
            continue
        if show_competitive_only and mode == "safe":
            continue

        weight = 2.5 if selected and did == selected else 0.6
        fill_op = 0.75 if not selected or did == selected else 0.35

        tooltip = (
            f"<b>{did}</b><br>"
            f"Party: {party}<br>"
            f"Rep: {row.get('rep_name', '—')}<br>"
            f"PVI: {row.get('pvi', '—')}<br>"
            f"RIVS: {rivs:.2f}<br>"
            f"Mode: {mode}<br>"
            f"Rank: {row.get('rivs_rank', '—')}"
        )

        style = {
            "fillColor": color,
            "color": "#222222" if selected == did else "#444444",
            "weight": weight,
            "fillOpacity": fill_op,
        }

        gj = folium.GeoJson(
            feat,
            style_function=lambda x, s=style: s,
            tooltip=folium.Tooltip(tooltip),
        )
        gj.add_to(fg_all)
        if party == "R":
            folium.GeoJson(
                feat,
                style_function=lambda x, s=style: s,
                tooltip=folium.Tooltip(tooltip),
            ).add_to(fg_r)
        if party == "D":
            folium.GeoJson(
                feat,
                style_function=lambda x, s=style: s,
                tooltip=folium.Tooltip(tooltip),
            ).add_to(fg_d)
        if mode != "safe":
            folium.GeoJson(
                feat,
                style_function=lambda x, s=style: s,
                tooltip=folium.Tooltip(tooltip),
            ).add_to(fg_comp)

    fg_all.add_to(m)
    fg_r.add_to(m)
    fg_d.add_to(m)
    fg_comp.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_simple_state_centroids_map(df: pd.DataFrame) -> folium.Map:
    """
    Lightweight fallback: circle markers at approximate state centers
    sized/colored by mean RIVS — used when full GeoJSON missing.
    """
    # Approximate state centroids
    centroids = {
        "AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.3, -111.7), "AR": (34.9, -92.4),
        "CA": (37.2, -119.5), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
        "FL": (28.1, -81.7), "GA": (32.7, -83.4), "HI": (20.8, -156.3), "ID": (44.4, -114.6),
        "IL": (40.0, -89.2), "IN": (39.9, -86.3), "IA": (42.0, -93.5), "KS": (38.5, -98.3),
        "KY": (37.5, -85.3), "LA": (31.0, -92.0), "ME": (45.3, -69.2), "MD": (39.0, -76.7),
        "MA": (42.3, -71.8), "MI": (44.3, -85.4), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
        "MO": (38.4, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8), "NV": (39.3, -116.6),
        "NH": (43.7, -71.6), "NJ": (40.1, -74.6), "NM": (34.4, -106.1), "NY": (42.9, -75.5),
        "NC": (35.6, -79.4), "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5),
        "OR": (44.0, -120.5), "PA": (40.9, -77.8), "RI": (41.7, -71.5), "SC": (33.9, -80.9),
        "SD": (44.4, -100.2), "TN": (35.8, -86.3), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
        "VT": (44.1, -72.6), "VA": (37.5, -78.6), "WA": (47.4, -120.5), "WV": (38.6, -80.6),
        "WI": (44.6, -89.7), "WY": (43.0, -107.6),
    }
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron")
    g = df.groupby("state", as_index=False).agg(
        rivs=("rivs", "mean"),
        n=("district_id", "count"),
        r_seats=("party_control", lambda s: (s == "R").sum()),
    )
    vmin, vmax = g["rivs"].min(), g["rivs"].max()
    for _, row in g.iterrows():
        st = row["state"]
        if st not in centroids:
            continue
        lat, lon = centroids[st]
        t = 0.5 if vmax <= vmin else (row["rivs"] - vmin) / (vmax - vmin)
        color = _rivs_color(row["rivs"], vmin, vmax)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 + 3 * t + row["n"] * 0.15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=(
                f"{st}: mean RIVS={row['rivs']:.2f}, "
                f"districts={int(row['n'])}, R-held={int(row['r_seats'])}"
            ),
        ).add_to(m)
    return m
