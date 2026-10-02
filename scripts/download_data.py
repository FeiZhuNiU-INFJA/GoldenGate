#!/usr/bin/env python3
"""Download market data (Baostock CN + yfinance/sina HK/US)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DOWNLOAD_START_DATE, DOWNLOAD_WORKERS, MARKETS
from data.download import download_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Download CN/HK/US daily bars")
    p.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    p.add_argument("--start", default=DOWNLOAD_START_DATE)
    p.add_argument("--end", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-symbols", type=int, default=None, help="limit per market (smoke tests)")
    p.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help="parallel workers (CN=processes, HK/US=threads). Default %(default)s",
    )
    args = p.parse_args()

    results = []
    for m in args.markets:
        results.append(
            download_market(
                market=m,
                start_date=args.start,
                end_date=args.end,
                force=args.force,
                max_symbols=args.max_symbols,
                workers=args.workers,
            )
        )
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
