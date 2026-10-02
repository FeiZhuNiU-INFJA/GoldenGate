"""Download universes, benchmarks, calendars, and daily bars."""
from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config.settings import DOWNLOAD_START_DATE, DOWNLOAD_WORKERS, MARKETS
from data import calendar as cal
from data import market_client as client
from data import store
from data.schema import normalize_bars

logger = logging.getLogger(__name__)

_CN_CHUNK_MULTIPLIER = 4
_HTTP_MAX_WORKERS = 3
_CHUNK_STALL_SEC = 120


def _end_date() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def download_universe(market: str, force: bool = False) -> pd.DataFrame:
    path_exists = store.universe_path(market).exists()
    if path_exists and not force:
        df = store.read_universe(market)
        assert df is not None
        return df
    fetcher = client.FETCH_UNIVERSE[market]
    df = fetcher()
    store.write_universe(market, df)
    logger.info("%s universe: %s symbols", market, len(df))
    return df


def download_benchmark(market: str, start_date: str, end_date: str, force: bool = False) -> pd.DataFrame:
    path = store.benchmark_path(market)
    if path.exists() and not force:
        df = store.read_parquet(path)
        assert df is not None
        return df
    fetcher = client.FETCH_BENCHMARK[market]
    df = fetcher(start_date=start_date, end_date=end_date)
    df = normalize_bars(df)
    store.write_parquet(df, path)
    calendar = cal.build_calendar_from_benchmark(df)
    cal.save_calendar(market, calendar)
    logger.info("%s benchmark rows=%s", market, len(df))
    return df


def download_symbol_bars(
    market: str,
    symbol: str,
    raw_symbol: str,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    path = store.bar_path(market, symbol)
    if path.exists() and not force:
        return store.read_parquet(path)
    fetcher = client.FETCH_BARS[market]
    try:
        df = fetcher(raw_symbol=raw_symbol, symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bars failed %s %s: %s", market, symbol, exc)
        return None
    if df is None or df.empty:
        logger.warning("empty bars %s %s", market, symbol)
        return None
    store.write_parquet(df, path)
    return df


def _init_cn_worker() -> None:
    """Warm worker; skip baostock when sina is forced."""
    logging.basicConfig(level=logging.WARNING)
    if client._cn_force_sina():
        return
    try:
        client._ensure_baostock(retries=1)
    except Exception as exc:  # noqa: BLE001
        logging.warning("cn worker using sina (%s)", exc)


def _fetch_and_write_job(args: tuple) -> tuple[str, bool, Optional[str]]:
    market, symbol, raw_symbol, start_date, end_date = args
    try:
        fetcher = client.FETCH_BARS[market]
        df = fetcher(raw_symbol=raw_symbol, symbol=symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return symbol, False, "empty"
        store.write_parquet(df, store.bar_path(market, symbol))
        return symbol, True, None
    except Exception as exc:  # noqa: BLE001
        return symbol, False, str(exc)


def _run_jobs(market: str, pending: list[tuple], workers: int) -> tuple[int, int]:
    if workers == 1:
        ok = fail = 0
        for job in tqdm(pending, desc=f"download-{market}"):
            symbol, success, err = _fetch_and_write_job(job)
            if success:
                ok += 1
            else:
                fail += 1
                if err and err != "empty":
                    logger.warning("bars failed %s %s: %s", market, symbol, err)
        return ok, fail
    # CN via sina is HTTP — prefer threads (process pool was dying under load).
    # Baostock would still need processes, but we force sina when blacklisted.
    if market == "cn" and client._cn_force_sina():
        return _run_thread_pool(market, pending, min(workers, 12))
    if market == "cn":
        return _run_cn_process_pool(pending, workers)
    return _run_thread_pool(market, pending, min(workers, _HTTP_MAX_WORKERS))


def _handle_future_result(market: str, fut, symbol: str) -> bool:
    try:
        _symbol, success, err = fut.result()
    except Exception as exc:  # noqa: BLE001
        logger.warning("bars failed %s %s: %s", market, symbol, exc)
        return False
    if success:
        return True
    if err and err != "empty":
        logger.warning("bars failed %s %s: %s", market, symbol, err)
    return False


def _consume_futures_with_stall(
    market: str,
    futures: dict,
    *,
    pbar: Optional[tqdm] = None,
    stall_sec: float = _CHUNK_STALL_SEC,
) -> tuple[int, int]:
    """Drain futures; on stall cancel leftovers and count them as fail."""
    ok = fail = 0
    own_bar = pbar is None
    bar = pbar or tqdm(total=len(futures), desc=f"download-{market}")
    pending = set(futures)
    try:
        while pending:
            done, pending = wait(pending, timeout=stall_sec, return_when=FIRST_COMPLETED)
            if not done:
                logger.error(
                    "%s download stalled (%ss, %s futures left); recycling workers",
                    market,
                    stall_sec,
                    len(pending),
                )
                for fut in list(pending):
                    fut.cancel()
                    fail += 1
                    bar.update(1)
                break
            for fut in done:
                symbol = futures[fut]
                if _handle_future_result(market, fut, symbol):
                    ok += 1
                else:
                    fail += 1
                bar.update(1)
    finally:
        if own_bar:
            bar.close()
    return ok, fail


def _shutdown_pool(ex) -> None:
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)


def _run_cn_process_pool(pending: list[tuple], workers: int) -> tuple[int, int]:
    """Chunked CN download; recreate process pool only after a stall."""
    ok = fail = 0
    chunk_size = max(workers * _CN_CHUNK_MULTIPLIER, workers)
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_cn_worker)
    try:
        with tqdm(total=len(pending), desc="download-cn") as pbar:
            i = 0
            while i < len(pending):
                chunk = pending[i : i + chunk_size]
                try:
                    futures = {ex.submit(_fetch_and_write_job, job): job[1] for job in chunk}
                    c_ok, c_fail = _consume_futures_with_stall("cn", futures, pbar=pbar)
                    ok += c_ok
                    fail += c_fail
                    # If many cancels (stall), recycle the pool then continue.
                    if c_fail and c_fail == len(chunk) - c_ok and c_ok == 0:
                        logger.warning("recycling CN process pool after empty/stalled chunk")
                        _shutdown_pool(ex)
                        ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_cn_worker)
                    i += chunk_size
                except Exception as exc:  # noqa: BLE001
                    logger.error("CN process pool error at %s: %s; recycling", i, exc)
                    for job in chunk:
                        if store.bar_path("cn", job[1]).exists():
                            ok += 1
                        else:
                            fail += 1
                        pbar.update(1)
                    _shutdown_pool(ex)
                    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_cn_worker)
                    i += chunk_size
    finally:
        _shutdown_pool(ex)
    return ok, fail


def _run_thread_pool(market: str, pending: list[tuple], workers: int) -> tuple[int, int]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_and_write_job, job): job[1] for job in pending}
        return _consume_futures_with_stall(market, futures, stall_sec=180)


def download_market(
    market: str,
    start_date: str = DOWNLOAD_START_DATE,
    end_date: Optional[str] = None,
    force: bool = False,
    max_symbols: Optional[int] = None,
    symbols: Optional[list[str]] = None,
    workers: int = DOWNLOAD_WORKERS,
) -> dict:
    end_date = end_date or _end_date()
    universe = download_universe(market, force=force)
    download_benchmark(market, start_date=start_date, end_date=end_date, force=force)

    rows = universe
    if symbols:
        sym_set = set(symbols)
        rows = universe[universe["symbol"].isin(sym_set)]
    if max_symbols is not None:
        rows = rows.head(max_symbols)

    pending: list[tuple] = []
    ok, fail, skipped = 0, 0, 0
    for _, row in rows.iterrows():
        symbol = row["symbol"]
        path = store.bar_path(market, symbol)
        if path.exists() and not force:
            skipped += 1
            ok += 1
            continue
        pending.append((market, symbol, str(row["raw_symbol"]), start_date, end_date))

    if not pending:
        logger.info("%s: nothing to download (skipped existing=%s)", market, skipped)
        return {"market": market, "ok": ok, "fail": fail, "skipped": skipped, "universe": len(universe)}

    workers = max(1, int(workers))
    logger.info("%s: fetch %s symbols with workers=%s (skipped existing=%s)", market, len(pending), workers, skipped)

    try:
        c_ok, c_fail = _run_jobs(market, pending, workers)
        ok += c_ok
        fail += c_fail
    finally:
        if market == "cn" and workers == 1:
            client.baostock_logout()

    return {
        "market": market,
        "ok": ok,
        "fail": fail,
        "skipped": skipped,
        "universe": len(universe),
        "workers": workers,
    }


def download_all(
    markets: tuple[str, ...] = MARKETS,
    start_date: str = DOWNLOAD_START_DATE,
    end_date: Optional[str] = None,
    force: bool = False,
    max_symbols: Optional[int] = None,
    workers: int = DOWNLOAD_WORKERS,
) -> list[dict]:
    results = []
    for market in markets:
        results.append(
            download_market(
                market=market,
                start_date=start_date,
                end_date=end_date,
                force=force,
                max_symbols=max_symbols,
                workers=workers,
            )
        )
    return results
