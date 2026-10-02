#!/usr/bin/env python3
"""Resume bar downloads until universes are covered (or max rounds)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prefer sina for CN — baostock is currently blacklisting this IP after concurrent abuse.
os.environ.setdefault("EXTREME_QUANT_CN_SINA", "1")

from config.settings import DOWNLOAD_WORKERS, MARKETS  # noqa: E402
from data import store  # noqa: E402
from data.download import download_market, download_universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("resume_download")


def _coverage(market: str) -> tuple[int, int, int]:
    universe = store.read_universe(market)
    if universe is None:
        return 0, 0, 0
    have = {p.stem for p in store.bars_dir(market).glob("*.parquet")} if store.bars_dir(market).exists() else set()
    # Keep only symbols in universe
    n_have = sum(1 for s in universe["symbol"] if s in have)
    n_uni = len(universe)
    return n_have, n_uni - n_have, n_uni


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=list(MARKETS))
    p.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    p.add_argument("--rounds", type=int, default=8, help="max resume rounds")
    p.add_argument("--refresh-universe", action="store_true", help="refetch HK/US/CN universe files")
    args = p.parse_args()

    if args.refresh_universe:
        for m in args.markets:
            if m == "cn":
                # Keep existing CN universe if present (baostock list is richer).
                if store.universe_path("cn").exists():
                    logger.info("skip CN universe refresh (keeping existing file)")
                    continue
            download_universe(m, force=True)

    for round_i in range(1, args.rounds + 1):
        logger.info("===== resume round %s/%s =====", round_i, args.rounds)
        remaining = 0
        for m in args.markets:
            have, miss, total = _coverage(m)
            logger.info("%s coverage before: %s/%s (missing %s)", m, have, total, miss)
            if miss <= 0:
                continue
            result = download_market(market=m, workers=args.workers)
            logger.info("%s result: %s", m, result)
            have, miss, total = _coverage(m)
            logger.info("%s coverage after: %s/%s (missing %s)", m, have, total, miss)
            remaining += miss
        if remaining <= 0:
            logger.info("All markets fully covered.")
            break
        logger.info("remaining missing=%s; sleeping 5s before next round", remaining)
        time.sleep(5)
    else:
        logger.warning("Stopped after %s rounds with remaining gaps.", args.rounds)

    for m in args.markets:
        have, miss, total = _coverage(m)
        print({"market": m, "have": have, "missing": miss, "universe": total})


if __name__ == "__main__":
    main()
