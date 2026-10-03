"""Industry-relative factors on top of the existing price features.

Peer groups are the same market, the same day, and the same industry label.
A group with fewer than ``min_peers`` finite values falls back to the
market-day median, as does a symbol with no industry label. The inputs are
already causal, and the median uses only that same day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PEER_COLUMNS = ("ret_5", "ret_20", "ret_60", "vol_20")
INDUSTRY_FEATURES = (
    "xs_ret_5",
    "xs_ret_20",
    "xs_ret_60",
    "xs_vol_20",
    "ind_ret_20",
)


def add_industry_relative(
    df: pd.DataFrame,
    industry: pd.DataFrame,
    min_peers: int = 5,
) -> pd.DataFrame:
    """Add residual-versus-peers columns and the peer median of ``ret_20``.

    ``industry`` has ``symbol`` and ``industry``. ``industry_peers`` is how many
    finite peers were available; it is not a model feature.
    """
    if min_peers < 1:
        raise ValueError("min_peers must be positive")
    labels = industry.loc[:, ["symbol", "industry"]].drop_duplicates("symbol")
    work = df.merge(labels, on="symbol", how="left")
    work["industry_peers"] = 0
    for col in PEER_COLUMNS:
        peer_med, peer_n = _peer_stat(work, col, min_peers)
        work[f"xs_{col}"] = work[col] - peer_med
        if col == "ret_20":
            work["ind_ret_20"] = peer_med
            work["industry_peers"] = peer_n
    return work.replace([np.inf, -np.inf], np.nan)


def _peer_stat(work: pd.DataFrame, col: str, min_peers: int) -> tuple[pd.Series, pd.Series]:
    market_med = work.groupby(["market", "trade_date"], sort=False)[col].transform("median")
    has = work["industry"].notna() & work["industry"].ne("")
    empty_n = pd.Series(0, index=work.index, dtype="int64")
    if not has.any():
        return market_med, empty_n
    grouped = work.loc[has, ["market", "trade_date", "industry", col]].copy()
    grouped["_hit"] = grouped[col].notna().astype(np.int64)
    stats = (
        grouped.groupby(["market", "trade_date", "industry"], sort=False)
        .agg(med=(col, "median"), n=("_hit", "sum"))
        .reset_index()
    )
    merged = work.merge(stats, on=["market", "trade_date", "industry"], how="left")
    use_industry = merged["n"].fillna(0).ge(min_peers).to_numpy()
    peer_med = merged["med"].where(use_industry, market_med.to_numpy())
    peer_n = merged["n"].fillna(0).astype("int64").where(use_industry, 0)
    peer_med.index = work.index
    peer_n.index = work.index
    return peer_med, peer_n
