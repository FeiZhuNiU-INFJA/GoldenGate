#!/usr/bin/env python3
"""Build 5d/20d excess-return labels."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import MARKETS
from labels.excess_return import label_market, summarize_label_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--summary", action="store_true", help="print label distribution after build")
    args = p.parse_args()

    for m in args.markets:
        print(label_market(m, force=args.force, max_symbols=args.max_symbols))
        if args.summary:
            dist = summarize_label_distribution(m, horizon=20)
            if not dist.empty:
                total = dist[["neutral", "buy", "sell"]].sum()
                print(f"{m} y_20d totals:\n{total}")


if __name__ == "__main__":
    main()
