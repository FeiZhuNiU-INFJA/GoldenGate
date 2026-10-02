"""Cross-sectional labels and causal rank features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.rank_metrics import summarize_scores
from labels.cross_section import assign_cross_section
from train.rank_features import features_from_labeled


def _bars(n: int = 80) -> pd.DataFrame:
    close = np.linspace(10, 12, n) + np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "symbol": ["AAA.SH"] * n,
            "market": ["cn"] * n,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 1000.0),
            "amount": np.full(n, 10000.0),
            "exret_20d": np.linspace(-0.05, 0.05, n),
        }
    )


def test_cross_section_ranks_winners_higher():
    rows = []
    for symbol, exret in (("A", 0.01), ("B", 0.20), ("C", -0.10), ("D", 0.00), ("E", 0.05)):
        rows.append({"market": "cn", "trade_date": "2023-01-03", "symbol": symbol, "exret_20d": exret})
    # pad to min_names
    for i in range(20):
        rows.append({"market": "cn", "trade_date": "2023-01-03", "symbol": f"P{i}", "exret_20d": -0.01 + i * 0.001})
    out = assign_cross_section(pd.DataFrame(rows), min_names=20, n_bins=5)
    by_sym = out.set_index("symbol")
    assert by_sym.loc["B", "z"] > by_sym.loc["A", "z"] > by_sym.loc["C", "z"]
    assert by_sym.loc["B", "relevance"] >= by_sym.loc["A", "relevance"]
    assert by_sym.loc["B", "relevance"] > by_sym.loc["C", "relevance"]
    assert abs(out["z"].mean()) < 1e-8


def test_features_use_only_past_closes():
    raw = _bars()
    feat = features_from_labeled(raw, horizon=20)
    i = 60
    close = raw["close"].to_numpy()
    assert np.isclose(feat.loc[i, "ret_5"], close[i] / close[i - 5] - 1.0)
    future = raw.copy()
    future.loc[i + 1 :, "close"] = 999.0
    feat2 = features_from_labeled(future, horizon=20)
    assert np.isclose(feat.loc[i, "ret_20"], feat2.loc[i, "ret_20"])
    zero_amount = raw.copy()
    zero_amount["amount"] = 0.0
    zero_amount.loc[20:, "volume"] = 2000.0
    backed = features_from_labeled(zero_amount, horizon=20)
    assert backed["activity_ratio_20"].iloc[30] > 1.0


def test_rank_ic_perfect_order():
    df = pd.DataFrame(
        {
            "trade_date": ["2023-01-02"] * 20 + ["2023-01-03"] * 20,
            "score": list(range(20)) + list(range(20)),
            "exret_20d": list(range(20)) + list(range(20)),
        }
    )
    stats = summarize_scores(df, step=1)
    assert stats["days"] == 2
    assert stats["ic"] > 0.99
    assert stats["spread"] > 0
