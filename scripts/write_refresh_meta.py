#!/usr/bin/env python3
"""Write data/refresh_meta.json timestamps (used by CI and local refresh)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ui_common import write_refresh_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--federal", action="store_true")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--source", default="manual")
    args = ap.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if args.federal:
        write_refresh_meta({"federal_refreshed_at": now, "federal_source": args.source})
        print("Wrote federal meta", now)
    if args.state:
        write_refresh_meta(
            {"state_refreshed_at": now, "state_source": args.source}, state=True
        )
        print("Wrote state meta", now)
    if not args.federal and not args.state:
        write_refresh_meta({"federal_refreshed_at": now, "federal_source": args.source})
        write_refresh_meta(
            {"state_refreshed_at": now, "state_source": args.source}, state=True
        )
        print("Wrote both", now)


if __name__ == "__main__":
    main()
