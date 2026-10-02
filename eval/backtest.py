"""Rough weekly Top-N long/short backtest vs market benchmark."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import BACKTEST_REBALANCE, BACKTEST_TOP_N, SCORE_WEIGHT_5D, SCORE_WEIGHT_20D
from data import store


def cross_sectional_scores_on_date(
    market: str,
    trade_date: pd.Timestamp,
    scores: dict[str, float],
) -> pd.DataFrame:
    rows = [{"symbol": s, "score": v} for s, v in scores.items()]
    return pd.DataFrame(rows)


def pick_long_short(scores: pd.DataFrame, top_n: int = BACKTEST_TOP_N) -> tuple[list[str], list[str]]:
    s = scores.dropna().sort_values("score", ascending=False)
    longs = s.head(top_n)["symbol"].tolist()
    shorts = s.tail(top_n)["symbol"].tolist()
    return longs, shorts


def next_open_return(bars: pd.DataFrame, date: pd.Timestamp, horizon_days: int = 5) -> Optional[float]:
    """Forward return from date close over next horizon trading rows."""
    df = bars.sort_values("trade_date").reset_index(drop=True)
    dates = pd.to_datetime(df["trade_date"])
    idx = dates.searchsorted(pd.Timestamp(date))
    if idx >= len(df) or dates.iloc[idx] != pd.Timestamp(date):
        # not an exact trading day for this symbol
        return None
    j = idx + horizon_days
    if j >= len(df):
        return None
    return float(df.loc[j, "close"] / df.loc[idx, "close"] - 1.0)


def run_topn_backtest(
    market: str,
    date_to_scores: dict[pd.Timestamp, dict[str, float]],
    top_n: int = BACKTEST_TOP_N,
    hold_days: int = 5,
) -> pd.DataFrame:
    """
    date_to_scores: mapping rebalance date -> {symbol: score}
    Portfolio: equal-weight long top_n, short bottom_n, hold `hold_days` trading days.
    Also computes benchmark buy-hold over same horizon when possible.
    """
    bench = store.read_parquet(store.benchmark_path(market))
    records = []
    for dt, scores in sorted(date_to_scores.items(), key=lambda x: x[0]):
        df_scores = pd.DataFrame([{"symbol": k, "score": v} for k, v in scores.items()])
        if df_scores.empty:
            continue
        longs, shorts = pick_long_short(df_scores, top_n=top_n)
        long_rets, short_rets = [], []
        for sym in longs:
            bars = store.read_parquet(store.bar_path(market, sym))
            if bars is None:
                continue
            r = next_open_return(bars, dt, horizon_days=hold_days)
            if r is not None:
                long_rets.append(r)
        for sym in shorts:
            bars = store.read_parquet(store.bar_path(market, sym))
            if bars is None:
                continue
            r = next_open_return(bars, dt, horizon_days=hold_days)
            if r is not None:
                short_rets.append(r)
        if not long_rets and not short_rets:
            continue
        port = 0.0
        if long_rets:
            port += float(np.mean(long_rets))
        if short_rets:
            port -= float(np.mean(short_rets))
        # If both sides present, average the two legs contribution already summed; scale
        if long_rets and short_rets:
            port *= 0.5

        bench_ret = None
        if bench is not None and not bench.empty:
            bench_ret = next_open_return(bench, dt, horizon_days=hold_days)

        records.append(
            {
                "trade_date": dt,
                "port_ret": port,
                "bench_ret": bench_ret,
                "excess": None if bench_ret is None else port - bench_ret,
                "n_long": len(long_rets),
                "n_short": len(short_rets),
            }
        )
    return pd.DataFrame(records)


def summarize_backtest(bt: pd.DataFrame) -> dict:
    if bt is None or bt.empty:
        return {"n": 0}
    rets = bt["port_ret"].dropna().to_numpy()
    excess = bt["excess"].dropna().to_numpy() if "excess" in bt.columns else np.array([])
    return {
        "n": int(len(bt)),
        "mean_port_ret": float(np.mean(rets)) if len(rets) else None,
        "cum_port_ret": float(np.prod(1 + rets) - 1) if len(rets) else None,
        "mean_excess": float(np.mean(excess)) if len(excess) else None,
        "hit_rate": float(np.mean(rets > 0)) if len(rets) else None,
    }
