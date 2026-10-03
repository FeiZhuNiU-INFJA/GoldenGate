#!/usr/bin/env python3
"""Append new US intersection recommendations and refresh the return log.

The ledger starts at the first date already stored in
``docs/live/us-intersection.json``. Earlier sessions are not backfilled.
A new session is recorded only when that date has a full US cross-section,
so a partial download does not become a recommendation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import store
from eval.hold_returns import top_k
from eval.recommendation_html import render_html
from eval.recommendation_log import render_log, session_path
from train.rank_features import RANK_FEATURES, _usable_close, rank_frame

LEDGER = ROOT / "docs" / "live" / "us-intersection.json"
PAGE = ROOT / "docs" / "live" / "us-intersection.md"
HTML = ROOT / "docs" / "live" / "us-intersection.html"
MIN_NAMES = 450
SIGNS = {5: 1, 10: 1}
MODELS = {5: ROOT / "checkpoints" / "ranker_us_h5.txt", 10: ROOT / "checkpoints" / "ranker_us_h10.txt"}


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    bench = _benchmark()
    added = _record_new(ledger, bench)
    paths, entries = _paths(ledger, bench)
    names = _names()
    PAGE.write_text(render_log(ledger["signals"], paths, names, entries))
    HTML.write_text(render_html(ledger["signals"], paths, names, entries))
    if added:
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {PAGE.relative_to(ROOT)} and {HTML.relative_to(ROOT)} "
        f"signals={len(ledger['signals'])} added={added}"
    )


def _record_new(ledger: dict, bench: pd.Series) -> int:
    pending = _pending_dates(ledger["signals"], bench)
    if not pending:
        return 0
    import lightgbm as lgb

    models = {horizon: lgb.Booster(model_file=str(path)) for horizon, path in MODELS.items()}
    panel = _feature_panel(pending)
    added = 0
    for day in pending:
        scored = panel[panel["trade_date"] == day] if not panel.empty else panel
        if len(scored) < MIN_NAMES:
            print(f"skip {day.date()}: only {len(scored)} names, left unrecorded")
            break
        ledger["signals"].append(_signal(day, scored, models))
        added += 1
        print(f"recorded {day.date()} picks={ledger['signals'][-1]['picks']}")
    return added


def _pending_dates(signals: list[dict], bench: pd.Series) -> list[pd.Timestamp]:
    if not signals:
        return []
    start = min(pd.Timestamp(item["date"]).normalize() for item in signals)
    have = {pd.Timestamp(item["date"]).normalize() for item in signals}
    latest = bench.index.max()
    return [day for day in bench.index if start < day <= latest and day not in have]


def _feature_panel(dates: list[pd.Timestamp]) -> pd.DataFrame:
    wanted = set(dates)
    bench = store.read_parquet(store.benchmark_path("us"))
    bench["trade_date"] = pd.to_datetime(bench["trade_date"]).dt.normalize()
    bench_close = bench.set_index("trade_date")["close"]
    frames = []
    for symbol in store.list_bar_symbols("us"):
        if symbol.startswith("SYN"):
            continue
        raw = store.read_parquet(store.bar_path("us", symbol))
        if raw is None or raw.empty:
            continue
        raw = raw.copy()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
        raw["bench_close"] = raw["trade_date"].map(bench_close)
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


def _paths(ledger: dict, bench: pd.Series) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    needed = {symbol for signal in ledger["signals"] for symbol in signal.get("picks") or []}
    closes = {symbol: _masked_close(symbol) for symbol in sorted(needed)}
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


def _masked_close(symbol: str) -> pd.Series:
    raw = store.read_parquet(store.bar_path("us", symbol))
    raw = raw.sort_values("trade_date").reset_index(drop=True)
    close = raw["close"].astype(np.float64)
    usable = _usable_close(close).to_numpy()
    dates = pd.to_datetime(raw["trade_date"]).dt.normalize()
    return pd.Series(np.where(usable, close.to_numpy(), np.nan), index=dates)


def _benchmark() -> pd.Series:
    raw = store.read_parquet(store.benchmark_path("us"))
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    series = raw.sort_values("trade_date").set_index("trade_date")["close"].astype(np.float64)
    series = series[~series.index.duplicated(keep="last")]
    return series.where(np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)).dropna()


def _names() -> dict[str, str]:
    universe = store.read_universe("us")
    if universe is None:
        return {}
    return dict(zip(universe["symbol"], universe["name"]))


if __name__ == "__main__":
    main()
