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


def _as_timestamp(value) -> pd.Timestamp:
    text = str(value).replace("-", "")[:8]
    return pd.Timestamp(f"{text[:4]}-{text[4:6]}-{text[6:8]}")


def incremental_start_date(
    last_date: Optional[pd.Timestamp],
    *,
    end_date: str,
    overlap_days: int,
    floor_start: str,
) -> Optional[str]:
    """YYYYMMDD start for an incremental fetch.

    Returns None when last_date is already after end_date (nothing to request).
    Otherwise starts `overlap_days` calendar days before the last stored bar so
    recent rows can be revised, clamped to `floor_start`.
    """
    end = _as_timestamp(end_date)
    floor = _as_timestamp(floor_start)
    if last_date is None or pd.isna(last_date):
        start = floor
    else:
        start = pd.Timestamp(last_date).normalize() - pd.Timedelta(days=max(0, int(overlap_days)))
        if start < floor:
            start = floor
    if start > end:
        return None
    return start.strftime("%Y%m%d")


def merge_bars(existing: Optional[pd.DataFrame], incoming: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Concat stored bars with a newer fetch. Duplicate dates keep the incoming row."""
    frames = [df for df in (existing, incoming) if df is not None and not df.empty]
    if not frames:
        return normalize_bars(None)
    return normalize_bars(pd.concat(frames, ignore_index=True))


def _new_row_count(existing: Optional[pd.DataFrame], incoming: Optional[pd.DataFrame]) -> int:
    if incoming is None or incoming.empty:
        return 0
    if existing is None or existing.empty or "trade_date" not in existing.columns:
        return len(incoming)
    prev = pd.to_datetime(existing["trade_date"]).max()
    return int((pd.to_datetime(incoming["trade_date"]) > prev).sum())


def _fetch_and_merge_job(args: tuple) -> tuple[str, bool, Optional[str], int]:
    """Fetch a short window and merge it into the existing parquet."""
    market, symbol, raw_symbol, floor_start, end_date, overlap_days = args
    path = store.bar_path(market, symbol)
    try:
        existing = store.read_parquet(path) if path.exists() else None
        last = None
        if existing is not None and not existing.empty and "trade_date" in existing.columns:
            last = pd.to_datetime(existing["trade_date"]).max()
        start_date = incremental_start_date(
            last,
            end_date=end_date,
            overlap_days=overlap_days,
            floor_start=floor_start,
        )
        if start_date is None:
            return symbol, True, "up_to_date", 0
        fetcher = client.FETCH_BARS[market]
        fresh = fetcher(raw_symbol=raw_symbol, symbol=symbol, start_date=start_date, end_date=end_date)
        if fresh is None or fresh.empty:
            if existing is not None and not existing.empty:
                return symbol, True, "unchanged", 0
            return symbol, False, "empty", 0
        added = _new_row_count(existing, fresh)
        store.write_parquet(merge_bars(existing, fresh), path)
        return symbol, True, None, added
    except Exception as exc:  # noqa: BLE001
        return symbol, False, str(exc), 0


def _job_stats(result: tuple) -> tuple[str, bool, Optional[str], int]:
    symbol = result[0]
    success = bool(result[1])
    err = result[2] if len(result) > 2 else None
    added = int(result[3]) if len(result) > 3 and result[3] else 0
    return symbol, success, err, added


def _run_jobs(
    market: str,
    pending: list[tuple],
    workers: int,
    job_fn=_fetch_and_write_job,
    desc: Optional[str] = None,
) -> tuple[int, int, int]:
    label = desc or f"download-{market}"
    if workers == 1:
        ok = fail = added = 0
        for job in tqdm(pending, desc=label):
            symbol, success, err, n_new = _job_stats(job_fn(job))
            if success:
                ok += 1
                added += n_new
            else:
                fail += 1
                if err and err != "empty":
                    logger.warning("bars failed %s %s: %s", market, symbol, err)
        return ok, fail, added
    # CN via sina is HTTP — prefer threads (process pool was dying under load).
    # Baostock would still need processes, but we force sina when blacklisted.
    if market == "cn" and client._cn_force_sina():
        return _run_thread_pool(market, pending, min(workers, 12), job_fn=job_fn, desc=label)
    if market == "cn":
        return _run_cn_process_pool(pending, workers, job_fn=job_fn, desc=label)
    return _run_thread_pool(market, pending, min(workers, _HTTP_MAX_WORKERS), job_fn=job_fn, desc=label)


def _handle_future_result(market: str, fut, symbol: str) -> tuple[bool, int]:
    try:
        _symbol, success, err, added = _job_stats(fut.result())
    except Exception as exc:  # noqa: BLE001
        logger.warning("bars failed %s %s: %s", market, symbol, exc)
        return False, 0
    if success:
        return True, added
    if err and err != "empty":
        logger.warning("bars failed %s %s: %s", market, symbol, err)
    return False, 0


def _consume_futures_with_stall(
    market: str,
    futures: dict,
    *,
    pbar: Optional[tqdm] = None,
    stall_sec: float = _CHUNK_STALL_SEC,
) -> tuple[int, int, int]:
    """Drain futures; on stall cancel leftovers and count them as fail."""
    ok = fail = added = 0
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
                success, n_new = _handle_future_result(market, fut, symbol)
                if success:
                    ok += 1
                    added += n_new
                else:
                    fail += 1
                bar.update(1)
    finally:
        if own_bar:
            bar.close()
    return ok, fail, added


def _shutdown_pool(ex) -> None:
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)


def _run_cn_process_pool(
    pending: list[tuple],
    workers: int,
    job_fn=_fetch_and_write_job,
    desc: str = "download-cn",
) -> tuple[int, int, int]:
    """Chunked CN download; recreate process pool only after a stall."""
    ok = fail = added = 0
    chunk_size = max(workers * _CN_CHUNK_MULTIPLIER, workers)
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_cn_worker)
    try:
        with tqdm(total=len(pending), desc=desc) as pbar:
            i = 0
            while i < len(pending):
                chunk = pending[i : i + chunk_size]
                try:
                    futures = {ex.submit(job_fn, job): job[1] for job in chunk}
                    c_ok, c_fail, c_added = _consume_futures_with_stall("cn", futures, pbar=pbar)
                    ok += c_ok
                    fail += c_fail
                    added += c_added
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
    return ok, fail, added


def _run_thread_pool(
    market: str,
    pending: list[tuple],
    workers: int,
    job_fn=_fetch_and_write_job,
    desc: Optional[str] = None,
) -> tuple[int, int, int]:
    label = desc or f"download-{market}"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(job_fn, job): job[1] for job in pending}
        with tqdm(total=len(pending), desc=label) as pbar:
            return _consume_futures_with_stall(market, futures, stall_sec=180, pbar=pbar)


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
        if store.has_bars(market, symbol) and not force:
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
        c_ok, c_fail, _added = _run_jobs(market, pending, workers)
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


def update_benchmark(
    market: str,
    end_date: str,
    overlap_days: int = 7,
    floor_start: str = DOWNLOAD_START_DATE,
) -> pd.DataFrame:
    """Append recent benchmark bars and rebuild the trading calendar."""
    path = store.benchmark_path(market)
    existing = store.read_parquet(path) if path.exists() else None
    last = None
    if existing is not None and not existing.empty and "trade_date" in existing.columns:
        last = pd.to_datetime(existing["trade_date"]).max()
    start = incremental_start_date(
        last,
        end_date=end_date,
        overlap_days=overlap_days,
        floor_start=floor_start,
    )
    if start is None:
        assert existing is not None
        return existing
    fresh = client.FETCH_BENCHMARK[market](start_date=start, end_date=end_date)
    merged = merge_bars(existing, fresh)
    if merged.empty:
        if existing is not None and not existing.empty:
            return existing
        raise RuntimeError(f"{market} benchmark empty")
    store.write_parquet(merged, path)
    cal.save_calendar(market, cal.build_calendar_from_benchmark(merged))
    logger.info("%s benchmark rows=%s (+%s)", market, len(merged), _new_row_count(existing, fresh))
    return merged


def update_market(
    market: str,
    end_date: Optional[str] = None,
    overlap_days: int = 7,
    floor_start: str = DOWNLOAD_START_DATE,
    refresh_universe: bool = False,
    max_symbols: Optional[int] = None,
    symbols: Optional[list[str]] = None,
    workers: int = DOWNLOAD_WORKERS,
) -> dict:
    """Incrementally refresh benchmark, calendar, and per-symbol daily bars.

    Existing files are extended from (last trade_date - overlap_days). Symbols
    with no file are fetched from `floor_start`. Duplicate dates keep the newer
    fetch so the overlap window can revise the latest bars.
    """
    end_date = end_date or _end_date()
    universe = download_universe(market, force=refresh_universe)
    update_benchmark(
        market,
        end_date=end_date,
        overlap_days=overlap_days,
        floor_start=floor_start,
    )

    rows = universe
    if symbols:
        rows = universe[universe["symbol"].isin(set(symbols))]
    if max_symbols is not None:
        rows = rows.head(max_symbols)

    pending = []
    for _, row in rows.iterrows():
        symbol = row["symbol"]
        # Names shared with the S&P 500 are refreshed by the us update.
        if (
            market == "ndx"
            and not store.bar_path(market, symbol).exists()
            and store.has_bars(market, symbol)
        ):
            continue
        pending.append(
            (market, symbol, str(row["raw_symbol"]), floor_start, end_date, int(overlap_days))
        )
    if not pending:
        logger.info("%s: no symbols to update", market)
        return {
            "market": market,
            "ok": 0,
            "fail": 0,
            "added": 0,
            "universe": len(universe),
            "workers": workers,
        }

    workers = max(1, int(workers))
    logger.info(
        "%s: incremental update %s symbols overlap=%sd workers=%s end=%s",
        market,
        len(pending),
        overlap_days,
        workers,
        end_date,
    )
    try:
        ok, fail, added = _run_jobs(
            market,
            pending,
            workers,
            job_fn=_fetch_and_merge_job,
            desc=f"update-{market}",
        )
    finally:
        if market == "cn" and workers == 1:
            client.baostock_logout()

    logger.info("%s: updated ok=%s fail=%s new_rows=%s", market, ok, fail, added)
    return {
        "market": market,
        "ok": ok,
        "fail": fail,
        "added": added,
        "universe": len(universe),
        "workers": workers,
    }


def update_all(
    markets: tuple[str, ...] = MARKETS,
    end_date: Optional[str] = None,
    overlap_days: int = 7,
    floor_start: str = DOWNLOAD_START_DATE,
    refresh_universe: bool = False,
    max_symbols: Optional[int] = None,
    workers: int = DOWNLOAD_WORKERS,
) -> list[dict]:
    results = []
    for market in markets:
        results.append(
            update_market(
                market=market,
                end_date=end_date,
                overlap_days=overlap_days,
                floor_start=floor_start,
                refresh_universe=refresh_universe,
                max_symbols=max_symbols,
                workers=workers,
            )
        )
    return results
