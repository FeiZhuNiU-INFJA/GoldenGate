"""Cross-sectional labels: within each market-day, rank forward excess return."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_cross_section(
    df: pd.DataFrame,
    value_col: str = "exret_20d",
    min_names: int = 20,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Add ``z`` and integer ``relevance`` (0 = worst, n_bins-1 = best).

    Groups smaller than ``min_names`` are dropped. Relevance is the
    within-group percentile bin, so a fixed excess threshold is not used.
    """
    if df.empty:
        out = df.copy()
        out["z"] = pd.Series(dtype="float64")
        out["relevance"] = pd.Series(dtype="int64")
        return out

    keys = ["market", "trade_date"]
    counts = df.groupby(keys, sort=False)[value_col].transform("size")
    out = df.loc[counts >= min_names].copy()
    if out.empty:
        out["z"] = pd.Series(dtype="float64")
        out["relevance"] = pd.Series(dtype="int64")
        return out

    grouped = out.groupby(keys, sort=False)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    out["z"] = (out[value_col] - mean) / std
    pct = grouped.rank(method="average", pct=True)
    out["relevance"] = np.minimum((pct * n_bins).astype(np.int64), n_bins - 1)
    out = out.dropna(subset=["z"]).reset_index(drop=True)
    out["relevance"] = out["relevance"].astype(np.int64)
    return out
