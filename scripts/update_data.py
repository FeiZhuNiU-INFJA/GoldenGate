#!/usr/bin/env python3
"""Incrementally refresh daily bars, benchmarks, and calendars.

For each symbol that already has a parquet, fetch from
(last trade_date - overlap days) through today and merge. Dates in the
overlap window are replaced by the new fetch; older rows stay on disk.
Symbols missing a file are downloaded from DOWNLOAD_START_DATE.

前复权 history can change outside that window after a split or dividend.
Rebuild with `scripts/download_data.py --force` when a full restatement is needed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DOWNLOAD_START_DATE, DOWNLOAD_WORKERS, MARKETS
from data.download import update_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Incrementally update CN/HK/US daily bars")
    p.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    p.add_argument("--end", default=None, help="YYYYMMDD inclusive end (default: local today)")
    p.add_argument(
        "--overlap-days",
        type=int,
        default=7,
        help="calendar days before the last stored bar to refetch (default: %(default)s)",
    )
    p.add_argument(
        "--start",
        default=DOWNLOAD_START_DATE,
        help="floor date for symbols that have no file yet (default: %(default)s)",
    )
    p.add_argument("--refresh-universe", action="store_true", help="refetch universe.csv before updating bars")
    p.add_argument("--max-symbols", type=int, default=None, help="limit per market (smoke tests)")
    p.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help="parallel workers (CN=processes, HK/US=threads). Default %(default)s",
    )
    args = p.parse_args()
    end = args.end or datetime.now().strftime("%Y%m%d")

    results = []
    for market in args.markets:
        results.append(
            update_market(
                market=market,
                end_date=end,
                overlap_days=args.overlap_days,
                floor_start=args.start,
                refresh_universe=args.refresh_universe,
                max_symbols=args.max_symbols,
                workers=args.workers,
            )
        )
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
