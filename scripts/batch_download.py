#!/usr/bin/env python3
"""Robust batch downloader: small thread batches, never dies on single failure."""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EXTREME_QUANT_CN_SINA", "1")

from data import store  # noqa: E402
from data.download import _fetch_and_write_job, download_benchmark, download_universe  # noqa: E402
from config.settings import DOWNLOAD_START_DATE, MARKETS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_download")


def _pending(market: str, start: str, end: str) -> list[tuple]:
    universe = store.read_universe(market)
    assert universe is not None
    jobs = []
    for _, row in universe.iterrows():
        symbol = row["symbol"]
        if store.has_bars(market, symbol):
            continue
        jobs.append((market, symbol, str(row["raw_symbol"]), start, end))
    return jobs


def _run_batch(jobs: list[tuple], workers: int) -> tuple[int, int]:
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_and_write_job, job): job[1] for job in jobs}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                _s, success, err = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    if err and err != "empty":
                        logger.warning("fail %s: %s", sym, err)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                logger.warning("fail %s: %s", sym, exc)
    return ok, fail


def main() -> None:
    from datetime import datetime

    markets = sys.argv[1:] or list(MARKETS)
    start = DOWNLOAD_START_DATE
    end = datetime.utcnow().strftime("%Y%m%d")
    workers = int(os.environ.get("EQ_WORKERS", "6"))
    batch_size = int(os.environ.get("EQ_BATCH", "100"))

    for market in markets:
        download_universe(market, force=False)
        download_benchmark(market, start_date=start, end_date=end, force=False)

    round_i = 0
    while True:
        round_i += 1
        total_miss = 0
        for market in markets:
            jobs = _pending(market, start, end)
            total_miss += len(jobs)
            logger.info("round %s %s pending=%s", round_i, market, len(jobs))
            for i in range(0, len(jobs), batch_size):
                chunk = jobs[i : i + batch_size]
                t0 = time.time()
                ok, fail = _run_batch(chunk, workers=workers if market == "cn" else min(workers, 3))
                logger.info(
                    "%s batch %s-%s ok=%s fail=%s in %.1fs",
                    market,
                    i,
                    i + len(chunk),
                    ok,
                    fail,
                    time.time() - t0,
                )
                time.sleep(0.3)
        if total_miss == 0:
            logger.info("ALL DONE")
            break
        if round_i >= 30:
            logger.warning("stop after %s rounds; remaining gaps", round_i)
            break
        logger.info("round %s done; remaining≈%s; sleep 2s", round_i, total_miss)
        time.sleep(2)

    for market in markets:
        jobs = _pending(market, start, end)
        uni = store.read_universe(market)
        have = 0 if uni is None else len(uni) - len(jobs)
        total = 0 if uni is None else len(uni)
        print({"market": market, "have": have, "missing": len(jobs), "universe": total})


if __name__ == "__main__":
    main()
