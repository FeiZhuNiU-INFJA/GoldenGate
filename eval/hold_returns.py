"""Top-k and intersection holding-period returns versus the market benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_k(df: pd.DataFrame, k: int, score_col: str = "score") -> pd.DataFrame:
    """Highest ``k`` scores per trade date. Ties break toward the smaller symbol."""
    if k < 1:
        raise ValueError("k must be positive")
    ordered = df.dropna(subset=[score_col]).sort_values(
        ["trade_date", score_col, "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ordered.groupby("trade_date", sort=False).head(k)[["trade_date", "symbol"]].reset_index(drop=True)


def intersect_top_k(frames: list[pd.DataFrame], k: int, score_col: str = "score") -> pd.DataFrame:
    """Symbols that land in every frame's daily top ``k``."""
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol"])
    picked = top_k(frames[0], k, score_col=score_col)
    for frame in frames[1:]:
        picked = picked.merge(top_k(frame, k, score_col=score_col), on=["trade_date", "symbol"], how="inner")
    return picked.reset_index(drop=True)


def signal_dates(frames: list[pd.DataFrame]) -> pd.DatetimeIndex:
    """Trade dates present in every score frame."""
    if not frames:
        return pd.DatetimeIndex([])
    common = None
    for frame in frames:
        days = pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]).dt.normalize().unique())
        common = days if common is None else common.intersection(days)
    return pd.DatetimeIndex([] if common is None else common).sort_values()


def intersection_coverage(frames: list[pd.DataFrame], picks: pd.DataFrame) -> dict:
    """How often the intersection is non-empty, and how many names it holds."""
    days = signal_dates(frames)
    if picks.empty or "trade_date" not in picks.columns:
        sizes = pd.Series(dtype="int64")
    else:
        sizes = picks.groupby(pd.to_datetime(picks["trade_date"]).dt.normalize()).size()
    nonempty = int(sizes.shape[0])
    n_days = int(len(days))
    return {
        "days": n_days,
        "days_nonempty": nonempty,
        "coverage": float(nonempty / n_days) if n_days else None,
        "mean_names": float(sizes.mean()) if nonempty else None,
    }


def book_daily(
    picks: pd.DataFrame,
    returns: pd.DataFrame,
    bench: pd.Series,
    ret_col: str,
    min_names: int = 1,
) -> pd.DataFrame:
    """Equal-weight close-to-close return, benchmark return, and excess.

    A day is kept only when at least ``min_names`` picks exist and every pick
    has a finite holding-period return. Excess is the portfolio return minus
    that day's benchmark return.
    """
    if picks.empty:
        return pd.DataFrame(columns=["trade_date", "port", "bench", "excess", "n"])
    cols = ["trade_date", "symbol", ret_col]
    merged = picks.merge(returns[cols], on=["trade_date", "symbol"], how="left")
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.normalize()
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index).normalize()
    rows = []
    for trade_date, group in merged.groupby("trade_date", sort=True):
        if len(group) < min_names or group[ret_col].isna().any():
            continue
        bench_ret = bench.get(trade_date, np.nan)
        if not np.isfinite(bench_ret):
            continue
        port = float(group[ret_col].mean())
        bench_ret = float(bench_ret)
        rows.append(
            {
                "trade_date": trade_date,
                "port": port,
                "bench": bench_ret,
                "excess": port - bench_ret,
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_hold(daily: pd.DataFrame, step: int) -> dict:
    """Mean portfolio, benchmark, and excess return, plus a non-overlapping subsample."""
    empty = {
        "days": 0,
        "port": None,
        "bench": None,
        "excess": None,
        "port_nonoverlap": None,
        "bench_nonoverlap": None,
        "excess_nonoverlap": None,
    }
    if daily is None or daily.empty:
        return empty
    ordered = daily.sort_values("trade_date")
    spaced = ordered.iloc[:: max(int(step), 1)]
    return {
        "days": int(len(ordered)),
        "port": float(ordered["port"].mean()),
        "bench": float(ordered["bench"].mean()),
        "excess": float(ordered["excess"].mean()),
        "port_nonoverlap": float(spaced["port"].mean()),
        "bench_nonoverlap": float(spaced["bench"].mean()),
        "excess_nonoverlap": float(spaced["excess"].mean()),
    }
