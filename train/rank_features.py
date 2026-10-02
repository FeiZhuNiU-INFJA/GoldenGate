"""Causal tabular features from daily bars, for a cross-sectional ranker."""
from __future__ import annotations

import numpy as np
import pandas as pd

# All windows end at t. Nothing here reads exret_* or future prices.
RANK_FEATURES = (
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "activity_ratio_20",
    "range_pct",
    "close_loc",
)


def features_from_labeled(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """One row per bar. ``exret_{horizon}d`` is copied through as the raw target."""
    out = df.sort_values("trade_date").reset_index(drop=True)
    close = out["close"].astype(np.float64)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)
    amount = out["amount"].astype(np.float64)
    volume = out["volume"].astype(np.float64)
    activity = amount if float(amount.fillna(0).abs().sum()) > 0 else volume
    ret1 = close.pct_change()
    span = (high - low).replace(0, np.nan)
    label_col = f"exret_{horizon}d"
    frame = pd.DataFrame(
        {
            "trade_date": out["trade_date"],
            "symbol": out["symbol"],
            "market": out["market"],
            label_col: out[label_col].astype(np.float64),
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "ret_60": close.pct_change(60),
            "vol_20": ret1.rolling(20).std(),
            "activity_ratio_20": activity / activity.rolling(20).mean(),
            "range_pct": span / close.replace(0, np.nan),
            "close_loc": (close - low) / span,
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _usable_close(close: pd.Series) -> pd.Series:
    """True when a close is positive and not a one-day print above 2.5x.

    A centered 11-day median also rejects prices that sit far from their neighbors,
    including negative adjusted closes. The mask is only applied to labels and to
    whether a row is kept, not to the feature formulas themselves.
    """
    close = close.astype(np.float64)
    med = close.rolling(11, center=True, min_periods=5).median()
    ratio = close / close.shift(1)
    level_ok = (close > 0) & close.ge(0.2 * med) & close.le(5 * med)
    jump_ok = ratio.isna() | (ratio.ge(0.4) & ratio.le(2.5))
    return level_ok & jump_ok


def rank_frame(df: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Features plus forward close returns and excess returns for each horizon.

    Forward columns use the stock's own rows for ``ret_{h}d`` and the benchmark
    aligned onto those same rows for the excess. Feature columns still end at t.
    """
    work = df.sort_values("trade_date").reset_index(drop=True)
    label_col = f"exret_{horizons[-1]}d"
    if label_col not in work.columns:
        work[label_col] = np.nan
    feat = features_from_labeled(work, horizon=horizons[-1])
    close = work["close"].astype(np.float64)
    bench = work["bench_close"].astype(np.float64)
    usable = _usable_close(close).to_numpy()
    feat.loc[~usable, list(RANK_FEATURES)] = np.nan
    bench_ok = ((bench > 0).to_numpy())
    for h in horizons:
        future = close.shift(-h)
        bench_future = bench.shift(-h)
        future_ok = np.zeros(len(usable), dtype=bool)
        if h < len(usable):
            future_ok[: len(usable) - h] = usable[h:]
        ret = (future / close - 1.0).to_numpy()
        bret = (bench_future / bench - 1.0).to_numpy()
        bad = ~usable | ~future_ok | ~bench_ok | ~(bench_future.to_numpy() > 0)
        ret = np.where(bad, np.nan, ret)
        feat[f"ret_{h}d"] = ret
        feat[f"exret_{h}d"] = np.where(bad, np.nan, ret - bret)
    return feat.replace([np.inf, -np.inf], np.nan)
