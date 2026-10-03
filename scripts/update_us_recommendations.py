#!/usr/bin/env python3
"""Refresh the S&P 500 and Nasdaq-100 intersection logs.

Each ledger starts at its first stored signal, or at ``start`` when the list
is still empty. Earlier sessions are not backfilled. A new session is recorded
only when that date has a full cross-section.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_DATASET
from data import store
from data.akshare_client import _force_requests_direct, _maybe_clear_proxies
from eval.hold_returns import top_k
from eval.recommendation_html import render_html
from eval.recommendation_log import render_log, session_path
from train.rank_features import RANK_FEATURES, _usable_close, rank_frame

SIGNS = {5: 1, 10: 1}
BOOKS = (
    {
        "key": "us",
        "min_names": 450,
        "models": {5: ROOT / "checkpoints" / "ranker_us_h5.txt", 10: ROOT / "checkpoints" / "ranker_us_h10.txt"},
        "ledger": ROOT / "docs" / "live" / "us-intersection.json",
        "page": ROOT / "docs" / "live" / "us-intersection.md",
        "html": ROOT / "docs" / "live" / "us-intersection.html",
        "benchmark": DIR_DATASET / "us" / "benchmark.parquet",
        "copy": None,
        "html_title": "美股交集推荐",
        "kicker": "标普 500 · 5 日模型 ∩ 10 日模型 · 各取前 5",
        "bench_label": "标普 500",
    },
    {
        "key": "ndx",
        "min_names": 80,
        "models": {5: ROOT / "checkpoints" / "ranker_ndx_h5.txt", 10: ROOT / "checkpoints" / "ranker_ndx_h10.txt"},
        "ledger": ROOT / "docs" / "live" / "ndx-intersection.json",
        "page": ROOT / "docs" / "live" / "ndx-intersection.md",
        "html": ROOT / "docs" / "live" / "ndx-intersection.html",
        "benchmark": DIR_DATASET / "ndx" / "benchmark.parquet",
        "copy": {
            "title": "纳斯达克推荐跟踪：5 日 ∩ 10 日 Top 5",
            "since": "2026-10-02",
            "models": "`checkpoints/ranker_ndx_h5.txt`、`checkpoints/ranker_ndx_h10.txt`",
            "bench": "纳斯达克 100",
            "ledger": "docs/live/ndx-intersection.json",
            "html": "docs/live/ndx-intersection.html",
        },
        "html_title": "纳斯达克交集推荐",
        "kicker": "纳斯达克 100 · 5 日模型 ∩ 10 日模型 · 各取前 5",
        "bench_label": "纳斯达克 100",
    },
)


def main() -> None:
    os.environ["EQ_SKIP_YFINANCE"] = "1"
    _maybe_clear_proxies()
    _force_requests_direct()
    _refresh_ndx()
    for book in BOOKS:
        _update_book(book)


def _update_book(book: dict) -> None:
    ledger = json.loads(book["ledger"].read_text())
    bench = _benchmark(book["benchmark"])
    added = _record_new(book, ledger, bench)
    paths, entries = _paths(book, ledger, bench)
    names = _names(book["key"])
    book["page"].write_text(render_log(ledger["signals"], paths, names, entries, book=book["copy"]))
    book["html"].write_text(
        render_html(
            ledger["signals"],
            paths,
            names,
            entries,
            page_title=book["html_title"],
            kicker=book["kicker"],
            bench=book["bench_label"],
        )
    )
    if added:
        book["ledger"].write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {book['page'].relative_to(ROOT)} and {book['html'].relative_to(ROOT)} "
        f"signals={len(ledger['signals'])} added={added}"
    )


def _record_new(book: dict, ledger: dict, bench: pd.Series) -> int:
    pending = _pending_dates(ledger, bench)
    if not pending:
        return 0
    import lightgbm as lgb

    models = {horizon: lgb.Booster(model_file=str(path)) for horizon, path in book["models"].items()}
    panel = _feature_panel(book, pending, bench)
    added = 0
    for day in pending:
        scored = panel[panel["trade_date"] == day] if not panel.empty else panel
        if len(scored) < book["min_names"]:
            print(f"{book['key']} skip {day.date()}: only {len(scored)} names, left unrecorded")
            break
        ledger["signals"].append(_signal(day, scored, models))
        added += 1
        print(f"{book['key']} recorded {day.date()} picks={ledger['signals'][-1]['picks']}")
    return added


def _pending_dates(ledger: dict, bench: pd.Series) -> list[pd.Timestamp]:
    signals = ledger.get("signals") or []
    latest = bench.index.max()
    have = {pd.Timestamp(item["date"]).normalize() for item in signals}
    if signals:
        start = min(have)
        return [day for day in bench.index if start < day <= latest and day not in have]
    start = ledger.get("start")
    if not start:
        return []
    start_day = pd.Timestamp(start).normalize()
    return [day for day in bench.index if start_day <= day <= latest and day not in have]


def _feature_panel(book: dict, dates: list[pd.Timestamp], bench: pd.Series) -> pd.DataFrame:
    wanted = set(dates)
    frames = []
    for symbol in _symbols(book["key"]):
        raw = _read_bars(book["key"], symbol)
        if raw is None or raw.empty:
            continue
        raw = raw.copy()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
        raw["market"] = book["key"]
        raw["bench_close"] = raw["trade_date"].map(bench)
        feat = rank_frame(raw, horizons=(5, 10, 20))
        feat = feat[feat["trade_date"].isin(wanted)].dropna(subset=list(RANK_FEATURES))
        if not feat.empty:
            frames.append(feat)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", *RANK_FEATURES])
    return pd.concat(frames, ignore_index=True)


def _signal(day: pd.Timestamp, scored: pd.DataFrame, models: dict) -> dict:
    frames = []
    for horizon, model in models.items():
        part = scored.sort_values("symbol").copy()
        part["score"] = model.predict(part[list(RANK_FEATURES)]) * SIGNS[horizon]
        frames.append(part)
    boards = {}
    picked = None
    for horizon, frame in zip(models, frames):
        board = []
        for symbol in top_k(frame, 5)["symbol"]:
            score = float(frame.loc[frame["symbol"] == symbol, "score"].iloc[0])
            board.append({"symbol": symbol, "score": round(score, 6)})
        boards[str(horizon)] = board
        symbols = {item["symbol"] for item in board}
        picked = symbols if picked is None else picked & symbols
    return {"date": day.strftime("%Y-%m-%d"), "note": "", "picks": sorted(picked or []), "top5": boards}


def _paths(book: dict, ledger: dict, bench: pd.Series) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    needed = {symbol for signal in ledger["signals"] for symbol in signal.get("picks") or []}
    closes = {symbol: _masked_close(book["key"], symbol) for symbol in sorted(needed)}
    paths = {}
    entries = {}
    for signal in ledger["signals"]:
        symbols = list(signal.get("picks") or [])
        path = session_path({symbol: closes[symbol] for symbol in symbols}, bench, signal["date"], symbols)
        paths[signal["date"]] = path
        signal_day = pd.Timestamp(signal["date"]).normalize()
        entries[signal["date"]] = {}
        for symbol in symbols:
            series = closes[symbol]
            series = series[np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)]
            if signal_day in series.index:
                entries[signal["date"]][symbol] = float(series.loc[signal_day])
    return paths, entries


def _masked_close(market: str, symbol: str) -> pd.Series:
    raw = _read_bars(market, symbol)
    if raw is None or raw.empty:
        return pd.Series(dtype="float64")
    raw = raw.sort_values("trade_date").reset_index(drop=True)
    close = raw["close"].astype(np.float64)
    usable = _usable_close(close).to_numpy()
    dates = pd.to_datetime(raw["trade_date"]).dt.normalize()
    return pd.Series(np.where(usable, close.to_numpy(), np.nan), index=dates)


def _benchmark(path: Path) -> pd.Series:
    raw = pd.read_parquet(path)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    series = raw.sort_values("trade_date").set_index("trade_date")["close"].astype(np.float64)
    series = series[~series.index.duplicated(keep="last")]
    return series.where(np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)).dropna()


def _names(market: str) -> dict[str, str]:
    if market == "ndx":
        frame = pd.read_csv(DIR_DATASET / "ndx" / "universe.csv")
    else:
        frame = store.read_universe("us")
    if frame is None:
        return {}
    return dict(zip(frame["symbol"], frame["name"]))


def _symbols(market: str) -> list[str]:
    if market == "ndx":
        frame = pd.read_csv(DIR_DATASET / "ndx" / "universe.csv")
        return [symbol for symbol in frame["symbol"].astype(str) if not symbol.startswith("SYN")]
    return [symbol for symbol in store.list_bar_symbols("us") if not symbol.startswith("SYN")]


def _read_bars(market: str, symbol: str) -> pd.DataFrame | None:
    path = store.resolve_bar_path(market, symbol)
    if path is None:
        return None
    return store.read_parquet(path)


def _refresh_ndx() -> None:
    """Refresh the Nasdaq-100 index and names that are not already S&P files."""
    from data.download import update_benchmark
    from data.market_client import fetch_ndx_bars

    end = _benchmark(store.benchmark_path("us")).index.max()
    end_s = end.strftime("%Y%m%d")
    update_benchmark("ndx", end_date=end_s)
    frame = store.read_universe("ndx")
    if frame is None:
        return
    for raw_symbol, symbol in zip(frame["raw_symbol"], frame["symbol"].astype(str)):
        if store.has_bars("ndx", symbol):
            continue
        fresh = fetch_ndx_bars(str(raw_symbol), symbol, "20140101", end_s)
        if fresh is None or fresh.empty:
            continue
        store.write_parquet(fresh, store.bar_path("ndx", symbol))


if __name__ == "__main__":
    main()
