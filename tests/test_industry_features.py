"""Industry residuals use same-day peers and fall back when the group is thin."""
from __future__ import annotations

import numpy as np
import pandas as pd

from train.industry_features import add_industry_relative


def _panel() -> pd.DataFrame:
    rows = []
    for symbol, industry, ret in (
        ("A", "bank", 0.10),
        ("B", "bank", 0.00),
        ("C", "bank", -0.10),
        ("D", "tech", 0.40),
        ("E", "tech", 0.20),
    ):
        rows.append(
            {
                "market": "cn",
                "trade_date": "2024-06-03",
                "symbol": symbol,
                "ret_5": ret,
                "ret_20": ret,
                "ret_60": ret,
                "vol_20": abs(ret),
            }
        )
    return pd.DataFrame(rows)


def test_residual_is_versus_industry_median_not_the_whole_market():
    industry = pd.DataFrame(
        {"symbol": ["A", "B", "C", "D", "E"], "industry": ["bank", "bank", "bank", "tech", "tech"]}
    )
    # tech has only 2 names, so it falls back to the market median
    out = add_industry_relative(_panel(), industry, min_peers=3).set_index("symbol")
    assert np.isclose(out.loc["A", "xs_ret_20"], 0.10)
    assert np.isclose(out.loc["B", "xs_ret_20"], 0.0)
    assert np.isclose(out.loc["A", "ind_ret_20"], 0.0)
    assert out.loc["A", "industry_peers"] == 3
    market_median = 0.10
    assert np.isclose(out.loc["D", "xs_ret_20"], 0.40 - market_median)
    assert out.loc["D", "industry_peers"] == 0


def test_missing_industry_uses_the_market_median():
    industry = pd.DataFrame({"symbol": ["A", "B", "C"], "industry": ["bank", "bank", "bank"]})
    out = add_industry_relative(_panel(), industry, min_peers=3).set_index("symbol")
    assert out.loc["D", "industry_peers"] == 0
    assert np.isfinite(out.loc["D", "xs_ret_20"])


def test_later_day_does_not_change_an_earlier_residual():
    day = _panel()
    later = day.copy()
    later["trade_date"] = "2024-06-04"
    later["ret_20"] = 5.0
    industry = pd.DataFrame(
        {"symbol": list("ABCDE"), "industry": ["bank", "bank", "bank", "tech", "tech"]}
    )
    both = add_industry_relative(pd.concat([day, later], ignore_index=True), industry, min_peers=3)
    first = both.loc[both["trade_date"] == "2024-06-03"].set_index("symbol")
    assert np.isclose(first.loc["A", "xs_ret_20"], 0.10)
