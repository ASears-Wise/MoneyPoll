#!/usr/bin/env python3
"""
Download U.S. congressional district GeoJSON (boundaries valid through 119th Congress)
and merge into data/raw/cds.geojson for the Folium choropleth.

Source: Jeffrey B. Lewis congressional-district-boundaries (UCLA CDMaps lineage)
https://github.com/JeffreyBLewis/congressional-district-boundaries
https://cdmaps.polisci.ucla.edu/

Usage:
    python scripts/download_geojson.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import GEOJSON_PATH, RAW_DIR  # noqa: E402

API_CONTENTS = (
    "https://api.github.com/repos/JeffreyBLewis/congressional-district-boundaries"
    "/contents/GeoJson"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/JeffreyBLewis/"
    "congressional-district-boundaries/master/GeoJson/"
)

# Files whose district definitions run through congress 119
TO_119 = re.compile(r".+_to_119\.geojson$", re.I)

STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


def list_to_119_files() -> list[str]:
    print("Listing GeoJson files from JeffreyBLewis repo…")
    r = requests.get(API_CONTENTS, timeout=60)
    r.raise_for_status()
    names = []
    for item in r.json():
        name = item["name"]
        if not TO_119.match(name):
            continue
        # Skip non-state (e.g. DC)
        if name.lower().startswith("district of columbia"):
            continue
        names.append(name)
    names.sort()
    print(f"  Found {len(names)} files covering congress 119")
    return names


def _district_id(props: dict) -> str | None:
    sn = props.get("statename") or props.get("STATENAME") or props.get("state")
    cd = props.get("district") if props.get("district") is not None else props.get("DISTRICT")
    if not sn or cd is None:
        return None
    abbr = STATE_NAME_TO_ABBR.get(str(sn).title()) or STATE_NAME_TO_ABBR.get(str(sn))
    if not abbr and len(str(sn)) == 2:
        abbr = str(sn).upper()
    if not abbr:
        return None
    try:
        n = int(float(cd))
        return f"{abbr}-{n:02d}"
    except (TypeError, ValueError):
        return f"{abbr}-{cd}"


def download_and_merge(names: list[str], dest: Path) -> int:
    # Prefer later startcong when duplicate district_ids (e.g. RI dual files)
    by_id: dict[str, dict] = {}
    orphans: list[dict] = []

    for i, name in enumerate(names, 1):
        url = RAW_BASE + name.replace(" ", "%20")
        print(f"  [{i}/{len(names)}] {name}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        data = r.json()
        for f in data.get("features") or []:
            props = f.setdefault("properties", {})
            did = _district_id(props)
            if did:
                props["district_id"] = did
                if did.endswith("-01") and props.get("statename") in (
                    "Alaska", "Delaware", "North Dakota", "South Dakota",
                    "Vermont", "Wyoming",
                ):
                    props["district_id_alt"] = did[:-2] + "00"
                start = float(props.get("startcong") or 0)
                prev = by_id.get(did)
                if prev is None or start >= float((prev.get("properties") or {}).get("startcong") or 0):
                    by_id[did] = f
            else:
                orphans.append(f)
        time.sleep(0.08)

    features = list(by_id.values()) + orphans
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "source": "JeffreyBLewis/congressional-district-boundaries",
            "through_congress": 119,
            "n_features": len(features),
            "n_with_district_id": len(by_id),
        },
    }
    dest.write_text(json.dumps(out), encoding="utf-8")
    return len(features)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        names = list_to_119_files()
        if len(names) < 40:
            raise RuntimeError(f"Expected ~50 state files, got {len(names)}")
        n = download_and_merge(names, GEOJSON_PATH)
        print(f"Wrote {n} features → {GEOJSON_PATH}")
        if n < 400:
            print("WARNING: fewer than 400 features.")
            sys.exit(2)
        print("Done.")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print(
            "\nManual option: place a national CD FeatureCollection at:\n"
            f"  {GEOJSON_PATH}\n"
            "with properties.district_id like 'CA-12'.\n"
            "The app falls back to state-centroid markers without this file."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
