"""Forward excess-return labels for 5d / 20d horizons."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import HORIZONS, LABEL_THRESHOLDS, MARKETS
from data import store
from data.schema import normalize_bars

logger = logging.getLogger(__name__)


def _forward_return(close: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-horizon)
    return future / close - 1.0


def excess_to_class(exret: pd.Series, threshold: float) -> pd.Series:
    y = pd.Series(0, index=exret.index, dtype=np.int64)
    y = y.mask(exret >= threshold, 1)
    y = y.mask(exret <= -threshold, 2)
    y = y.where(exret.notna(), other=pd.NA)
    return y


def label_symbol_frame(bars: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Attach ret/exret/y columns. Rows near the end without future prices get NA labels."""
    df = normalize_bars(bars)
    bench = normalize_bars(benchmark)[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    out = df.merge(bench, on="trade_date", how="left")
    out = out.sort_values("trade_date").reset_index(drop=True)

    # Align benchmark forward returns on the stock's own trading calendar by reindexing bench to stock dates
    # using merge-asof style: bench already merged on trade_date; forward return uses stock row index.
    stock_close = out["close"]
    # For benchmark, forward along the merged rows (same dates present for both).
    # If bench_close is missing on a stock date, forward-fill from last known bench level first.
    out["bench_close"] = out["bench_close"].ffill()
    bench_close = out["bench_close"]

    for h in HORIZONS:
        thr = LABEL_THRESHOLDS[h]
        ret = _forward_return(stock_close, h)
        bret = _forward_return(bench_close, h)
        exret = ret - bret
        out[f"ret_{h}d"] = ret
        out[f"exret_{h}d"] = exret
        out[f"y_{h}d"] = excess_to_class(exret, thr)
    return out


def label_market(market: str, force: bool = False, max_symbols: Optional[int] = None) -> dict:
    bench = store.read_parquet(store.benchmark_path(market))
    if bench is None or bench.empty:
        raise FileNotFoundError(f"missing benchmark for {market}; run download first")

    symbols = store.list_bar_symbols(market)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    ok, fail, skipped = 0, 0, 0
    for sym in tqdm(symbols, desc=f"label-{market}"):
        out_path = store.labeled_path(market, sym)
        if out_path.exists() and not force:
            skipped += 1
            continue
        bars = store.read_parquet(store.bar_path(market, sym))
        if bars is None or bars.empty:
            fail += 1
            continue
        try:
            labeled = label_symbol_frame(bars, bench)
            store.write_parquet(labeled, out_path)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("label failed %s %s: %s", market, sym, exc)
            fail += 1
    return {"market": market, "ok": ok, "fail": fail, "skipped": skipped}


def label_all(markets: tuple[str, ...] = MARKETS, force: bool = False, max_symbols: Optional[int] = None) -> list[dict]:
    return [label_market(m, force=force, max_symbols=max_symbols) for m in markets]


def summarize_label_distribution(market: str, horizon: int = 20) -> pd.DataFrame:
    col = f"y_{horizon}d"
    rows = []
    for sym in store.list_labeled_symbols(market):
        df = store.read_parquet(store.labeled_path(market, sym))
        if df is None or col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=True)
        rows.append(
            {
                "symbol": sym,
                "n": int(vc.sum()),
                "neutral": int(vc.get(0, 0)),
                "buy": int(vc.get(1, 0)),
                "sell": int(vc.get(2, 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "n", "neutral", "buy", "sell"])
    return pd.DataFrame(rows)
