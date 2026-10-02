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
