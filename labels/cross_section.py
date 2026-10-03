"""Cross-sectional labels: within each market-day, rank forward excess return."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ranks 1–10, 11–20, and the rest. Gain stays 2^r - 1: 0, 1, 3.
GRADE_CUTS = (10, 20)
LABEL_GAIN = [2**i - 1 for i in range(len(GRADE_CUTS) + 1)]
RELEVANCE_BINS = 5


def assign_cross_section(
    df: pd.DataFrame,
    value_col: str = "exret_20d",
    min_names: int = 20,
    n_bins: int = RELEVANCE_BINS,
    top_names: int | None = None,
    grade_cuts: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Add ``z`` and integer ``relevance`` (higher is better).

    Groups smaller than ``min_names`` are dropped. ``grade_cuts`` are
    inclusive rank cutoffs from the best name, so ``(10, 20)`` makes
    ranks 1–10 the top grade, 11–20 the middle grade, and the rest 0.
    Ties at a cutoff stay in the higher grade. ``top_names`` keeps the
    best names in one grade and percentile-bins the rest. With neither,
    relevance is an equal-width percentile bin.
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
    if grade_cuts is not None:
        if tuple(sorted(grade_cuts)) != tuple(grade_cuts) or any(cut < 1 for cut in grade_cuts):
            raise ValueError("grade_cuts must be positive and increasing")
        rank = grouped.rank(method="min", ascending=False)
        out["relevance"] = _headcount_grades(rank, grade_cuts)
    elif top_names is None:
        pct = grouped.rank(method="average", pct=True)
        out["relevance"] = _bins(pct, n_bins)
    else:
        if top_names < 1 or n_bins < 2:
            raise ValueError("top_names and n_bins must leave a lower grade")
        rank = grouped.rank(method="min", ascending=False)
        out["relevance"] = np.int64(0)
        top = rank <= top_names
        out.loc[top, "relevance"] = n_bins - 1
        if (~top).any():
            rest = out.loc[~top]
            pct = rest.groupby(keys, sort=False)[value_col].rank(method="average", pct=True)
            out.loc[pct.index, "relevance"] = _bins(pct, n_bins - 1)
    out = out.dropna(subset=["z"]).reset_index(drop=True)
    out["relevance"] = out["relevance"].astype(np.int64)
    return out


def _bins(pct: pd.Series, n_bins: int) -> np.ndarray:
    return np.minimum((pct.to_numpy() * n_bins).astype(np.int64), n_bins - 1)


def _headcount_grades(rank: pd.Series, cuts: tuple[int, ...]) -> np.ndarray:
    grades = np.zeros(len(rank), dtype=np.int64)
    order = rank.to_numpy()
    for relevance, cut in enumerate(reversed(cuts), start=1):
        grades[order <= cut] = relevance
    return grades
