"""Cross-sectional labels: within each market-day, rank forward excess return."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ranks 1–5, 6–15, and the rest. Gain stays 2^r - 1: 0, 1, 3.
GRADE_CUTS = (5, 15)
LABEL_GAIN = [2**i - 1 for i in range(len(GRADE_CUTS) + 1)]


def assign_cross_section(
    df: pd.DataFrame,
    value_col: str = "exret_20d",
    min_names: int = 20,
    grade_cuts: tuple[int, ...] = GRADE_CUTS,
) -> pd.DataFrame:
    """Add ``z`` and integer ``relevance`` (higher is better).

    Groups smaller than ``min_names`` are dropped. ``grade_cuts`` are
    inclusive rank cutoffs from the best name, so ``(5, 15)`` makes
    ranks 1–5 relevance 2, ranks 6–15 relevance 1, and the rest 0.
    Ties at a cutoff stay in the higher grade.
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
    if tuple(sorted(grade_cuts)) != tuple(grade_cuts) or any(cut < 1 for cut in grade_cuts):
        raise ValueError("grade_cuts must be positive and increasing")
    rank = grouped.rank(method="min", ascending=False)
    out["relevance"] = _headcount_grades(rank, grade_cuts)
    out = out.dropna(subset=["z"]).reset_index(drop=True)
    out["relevance"] = out["relevance"].astype(np.int64)
    return out


def _headcount_grades(rank: pd.Series, cuts: tuple[int, ...]) -> np.ndarray:
    grades = np.zeros(len(rank), dtype=np.int64)
    order = rank.to_numpy()
    for relevance, cut in enumerate(reversed(cuts), start=1):
        grades[order <= cut] = relevance
    return grades
