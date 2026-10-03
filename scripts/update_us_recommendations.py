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
from eval.recommendation_html import render_data
from eval.recommendation_log import page_data, render_log, session_path
from train.rank_features import RANK_FEATURES, _usable_close, rank_frame

STRATEGY = "grades-5-15-seeds-3-top5"
SEEDS = (0, 1, 2)
HORIZONS = (5, 10, 20)
TOP_K = 5
MODEL_DIR = ROOT / "checkpoints" / "ensemble"
DATA_PAGE = ROOT / "docs" / "live" / "intersection-data.js"
BOOKS = (
    {
        "key": "us",
        "start": "2026-10-01",
        "min_names": 450,
        "model_dir": MODEL_DIR,
        "ledger": ROOT / "docs" / "live" / "us-intersection.json",
        "page": ROOT / "docs" / "live" / "us-intersection.md",
        "benchmark": DIR_DATASET / "us" / "benchmark.parquet",
        "copy": None,
        "kicker": "标普 500 · 前 5 / 第 6–15 / 其余 · 三个模型前 5 交集",
        "bench_label": "标普 500",
    },
    {
        "key": "ndx",
        "start": "2026-10-02",
        "min_names": 80,
        "model_dir": MODEL_DIR,
        "ledger": ROOT / "docs" / "live" / "ndx-intersection.json",
        "page": ROOT / "docs" / "live" / "ndx-intersection.md",
        "benchmark": DIR_DATASET / "ndx" / "benchmark.parquet",
        "copy": {
            "title": "纳斯达克推荐跟踪：三个模型前 5 名交集",
            "since": "2026-10-02",
            "models": "`checkpoints/ensemble/ranker_ndx_h{5,10,20}_s{0,1,2}.txt`",
            "bench": "纳斯达克 100",
            "ledger": "docs/live/ndx-intersection.json",
            "html": "docs/live/intersection.html",
        },
        "kicker": "纳斯达克 100 · 前 5 / 第 6–15 / 其余 · 三个模型前 5 交集",
        "bench_label": "纳斯达克 100",
    },
)


def main() -> None:
    os.environ["EQ_SKIP_YFINANCE"] = "1"
    _maybe_clear_proxies()
    _force_requests_direct()
    _refresh_ndx()
    markets = [_update_book(book) for book in BOOKS]
    DATA_PAGE.write_text(render_data(markets))
    for stale in (DATA_PAGE.parent / "us-intersection.html", DATA_PAGE.parent / "ndx-intersection.html"):
        if stale.exists():
            stale.unlink()
    print(f"wrote {DATA_PAGE.relative_to(ROOT)}")


def _update_book(book: dict) -> dict:
    ledger = json.loads(book["ledger"].read_text())
    notes = {item["date"]: item.get("note") or "" for item in ledger.get("signals") or []}
    rebuilt = ledger.get("strategy") != STRATEGY
    if rebuilt:
        ledger = {
            "market": book["key"],
            "strategy": STRATEGY,
            "k": TOP_K,
            "seeds": list(SEEDS),
            "grade_cuts": [5, 15],
            "horizons": list(HORIZONS),
            "start": book["start"],
            "signals": [],
        }
    bench = _benchmark(book["benchmark"])
    added = _record_new(book, ledger, bench, notes)
    paths, entries = _paths(book, ledger, bench)
    names = _names(book["key"])
    book["page"].write_text(render_log(ledger["signals"], paths, names, entries, book=book["copy"]))
    if rebuilt or added:
        book["ledger"].write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {book['page'].relative_to(ROOT)} signals={len(ledger['signals'])} added={added}")
    payload = page_data(ledger["signals"], paths, names, entries)
    return {
        "key": book["key"],
        "label": book["bench_label"],
        "bench": book["bench_label"],
        "kicker": book["kicker"],
        "as_of": payload["as_of"],
        "horizons": payload["horizons"],
        "groups": payload["groups"],
    }


def _record_new(book: dict, ledger: dict, bench: pd.Series, notes: dict[str, str]) -> int:
    pending = _pending_dates(ledger, bench)
    if not pending:
        return 0
    models = _load_models(book)
    panel = _feature_panel(book, pending, bench)
    added = 0
    for day in pending:
        scored = panel[panel["trade_date"] == day] if not panel.empty else panel
        if len(scored) < book["min_names"]:
            print(f"{book['key']} skip {day.date()}: only {len(scored)} names, left unrecorded")
            break
        signal = _signal(day, scored, models)
        signal["note"] = notes.get(signal["date"], "")
        ledger["signals"].append(signal)
        added += 1
        picks = {horizon: signal["horizons"][str(horizon)]["picks"] for horizon in HORIZONS}
        print(f"{book['key']} recorded {day.date()} picks={picks}")
    return added


def _load_models(book: dict) -> dict:
    import lightgbm as lgb

    signs = json.loads((book["model_dir"] / "manifest.json").read_text())["signs"]
    models = {}
    for horizon in HORIZONS:
        for seed in SEEDS:
            key = f"{book['key']}_h{horizon}_s{seed}"
            path = book["model_dir"] / f"ranker_{key}.txt"
            models[(horizon, seed)] = (lgb.Booster(model_file=str(path)), int(signs[key]))
    return models


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
    ordered = scored.sort_values("symbol")
    features = ordered[list(RANK_FEATURES)]
    books = {}
    for horizon in HORIZONS:
        boards = {}
        picked = None
        for seed in SEEDS:
            model, sign = models[(horizon, seed)]
            part = ordered[["trade_date", "symbol"]].copy()
            part["score"] = model.predict(features) * sign
            board = []
            for symbol in top_k(part, TOP_K)["symbol"]:
                score = float(part.loc[part["symbol"] == symbol, "score"].iloc[0])
                board.append({"symbol": symbol, "score": round(score, 6)})
            boards[str(seed)] = board
            symbols = {item["symbol"] for item in board}
            picked = symbols if picked is None else picked & symbols
        books[str(horizon)] = {"picks": sorted(picked or []), "seeds": boards}
    return {"date": day.strftime("%Y-%m-%d"), "note": "", "horizons": books}


def _paths(book: dict, ledger: dict, bench: pd.Series) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    needed = {
        symbol
        for signal in ledger["signals"]
        for horizon in HORIZONS
        for symbol in (signal.get("horizons") or {}).get(str(horizon), {}).get("picks") or []
    }
    closes = {symbol: _masked_close(book["key"], symbol) for symbol in sorted(needed)}
    paths = {}
    entries = {}
    for signal in ledger["signals"]:
        paths[signal["date"]] = {}
        entries[signal["date"]] = {}
        signal_day = pd.Timestamp(signal["date"]).normalize()
        for horizon in HORIZONS:
            symbols = list((signal.get("horizons") or {}).get(str(horizon), {}).get("picks") or [])
            paths[signal["date"]][horizon] = session_path(
                {symbol: closes[symbol] for symbol in symbols}, bench, signal["date"], symbols
            )
            entries[signal["date"]][horizon] = {}
            for symbol in symbols:
                series = closes[symbol]
                series = series[np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)]
                if signal_day in series.index:
                    entries[signal["date"]][horizon][symbol] = float(series.loc[signal_day])
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
