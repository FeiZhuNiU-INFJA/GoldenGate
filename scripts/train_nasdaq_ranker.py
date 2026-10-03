#!/usr/bin/env python3
"""Train Nasdaq-100 rankers with the same price features as the S&P models.

Step-1 only: ret_5, ret_20, ret_60, vol_20, activity_ratio_20, range_pct, close_loc.
The universe is the current Nasdaq-100. Excess is versus the Nasdaq-100 index,
not the S&P 500. Checkpoints are ranker_ndx_h{horizon}.txt.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_DATASET, DIR_REPORTS, DOWNLOAD_START_DATE
from data.akshare_client import _force_requests_direct, _maybe_clear_proxies
from data.schema import normalize_bars
from train.rank_features import RANK_FEATURES, rank_frame

_spec = importlib.util.spec_from_file_location(
    "compare_rank_horizons", ROOT / "scripts" / "compare_rank_horizons.py"
)
_horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_horizons)
_jsonable = _horizons._jsonable
_print_tables = _horizons._print_tables
train_market = _horizons.train_market
from train.rank_protocol import RANK_HORIZONS, REPORT_START, TUNE_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_nasdaq_ranker")

NDX_DIR = DIR_DATASET / "ndx"
US_BARS = DIR_DATASET / "us" / "bars"
LIST_HTML = Path("/tmp/ndx_list.html")
WIKI = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
END = "20261002"


def main() -> None:
    os.environ["EQ_SKIP_YFINANCE"] = "1"
    _maybe_clear_proxies()
    _force_requests_direct()
    universe = _universe()
    benchmark = _benchmark()
    panel = _panel(universe, benchmark)
    summary = {
        "market": "ndx",
        "benchmark": "Nasdaq-100",
        "features": list(RANK_FEATURES),
        "tune_start": TUNE_START,
        "report_start": REPORT_START,
        "symbols": int(universe["symbol"].nunique()),
        "horizons": list(RANK_HORIZONS),
        "markets": {},
    }
    summary["markets"]["ndx"] = train_market(
        panel,
        "ndx",
        RANK_HORIZONS,
        TUNE_START,
        REPORT_START,
        min_names=20,
        n_top=5,
        intersect_k=10,
        features=RANK_FEATURES,
        checkpoint_pattern="ranker_ndx_h{horizon}.txt",
        benchmark=benchmark,
    )
    out = DIR_REPORTS / "rank_ndx_summary.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2))
    _print_tables(summary)
    print(f"\nsummary: {out}")


def _universe() -> pd.DataFrame:
    if not LIST_HTML.exists():
        subprocess.run(
            ["curl", "-fsSL", "-A", "Mozilla/5.0", "-o", str(LIST_HTML), WIKI],
            check=True,
        )
    table = pd.read_html(LIST_HTML)[0]
    raw = table["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    frame = pd.DataFrame(
        {
            "symbol": raw.map(lambda ticker: f"{ticker}.US"),
            "name": table["Company"].astype(str),
            "market": "ndx",
            "raw_symbol": raw,
        }
    ).drop_duplicates("symbol")
    NDX_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(NDX_DIR / "universe.csv", index=False)
    logger.info("nasdaq-100 symbols=%s", len(frame))
    return frame


def _benchmark() -> pd.DataFrame:
    import akshare as ak

    raw = ak.index_us_stock_sina(symbol=".NDX")
    out = raw.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = "NDX.US"
    out["market"] = "ndx"
    out = normalize_bars(out)
    start, end = pd.Timestamp("2014-01-01"), pd.Timestamp("2026-10-02")
    out = out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("empty Nasdaq-100 benchmark")
    out.to_parquet(NDX_DIR / "benchmark.parquet", index=False)
    logger.info("benchmark rows=%s max=%s", len(out), out["trade_date"].max().date())
    return out


def _bars(raw_symbol: str, symbol: str) -> pd.DataFrame | None:
    shared = US_BARS / f"{symbol}.parquet"
    if shared.exists():
        return pd.read_parquet(shared)
    from data.market_client import fetch_us_bars

    try:
        fresh = fetch_us_bars(raw_symbol, symbol, DOWNLOAD_START_DATE, END)
    except Exception as exc:  # noqa: BLE001
        logger.warning("download failed %s: %s", symbol, exc)
        return None
    if fresh is None or fresh.empty:
        logger.warning("no bars %s", symbol)
        return None
    path = NDX_DIR / "bars" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(path, index=False)
    logger.info("downloaded %s rows=%s", symbol, len(fresh))
    return fresh


def _panel(universe: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    bench_close = benchmark.set_index("trade_date")["close"]
    frames = []
    for raw_symbol, symbol in zip(universe["raw_symbol"], universe["symbol"]):
        bars = _bars(raw_symbol, symbol)
        if bars is None or bars.empty:
            continue
        bars = bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        bars["symbol"] = symbol
        bars["market"] = "ndx"
        bars["bench_close"] = bars["trade_date"].map(bench_close)
        feat = rank_frame(bars, horizons=RANK_HORIZONS)
        feat = feat.dropna(subset=list(RANK_FEATURES))
        if not feat.empty:
            frames.append(feat)
    if not frames:
        raise RuntimeError("no nasdaq rows")
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    logger.info("panel symbols=%s rows=%s", panel["symbol"].nunique(), len(panel))
    return panel


if __name__ == "__main__":
    main()
