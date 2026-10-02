"""Daily cross-sectional rank metrics. IC is computed per trade date, then averaged."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def _tail_spread(g: pd.DataFrame, score_col: str, label_col: str, quantile: float) -> float:
    n = max(int(len(g) * quantile), 1)
    ordered = g.sort_values(score_col)
    short = ordered.iloc[:n][label_col].mean()
    long = ordered.iloc[-n:][label_col].mean()
    return float(long - short)


def summarize_scores(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "exret_20d",
    quantile: float = 0.2,
    step: int = 20,
) -> dict:
    """Mean daily RankIC, ICIR, and top-minus-bottom forward excess.

    ``step`` also reports IC on every n-th date so overlapping labels are not
    the only number on the page. ICIR is mean(IC) / std(IC), not annualized.
    """
    rows = []
    for trade_date, g in df.groupby("trade_date", sort=True):
        g = g.dropna(subset=[score_col, label_col])
        if len(g) < 20 or g[score_col].nunique() < 2 or g[label_col].nunique() < 2:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "ic": _spearman(g[score_col], g[label_col]),
                "spread": _tail_spread(g, score_col, label_col, quantile),
                "n": len(g),
            }
        )
    if not rows:
        return {"days": 0, "ic": None, "icir": None, "spread": None, "ic_nonoverlap": None, "pos_rate": None}
    daily = pd.DataFrame(rows)
    ic = daily["ic"]
    std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    nonoverlap = daily.iloc[::step]["ic"]
    return {
        "days": int(len(daily)),
        "ic": float(ic.mean()),
        "icir": float(ic.mean() / std) if std and np.isfinite(std) else None,
        "spread": float(daily["spread"].mean()),
        "ic_nonoverlap": float(nonoverlap.mean()),
        "pos_rate": float((ic > 0).mean()),
    }
