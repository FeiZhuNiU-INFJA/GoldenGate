"""Holding-period books and the pre-report tuning split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.hold_returns import book_daily, intersect_top_k, intersection_coverage, summarize_hold, top_k
from train.rank_features import rank_frame
from train.rank_protocol import assign_split


def _scores(model_rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(model_rows, columns=["trade_date", "symbol", "score"])


def test_split_keeps_report_window_out_of_training():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-05-31", "2024-06-03"])
    roles = assign_split(pd.Series(dates))
    assert list(roles["fit"]) == [True, False, False, False]
    assert list(roles["tune"]) == [False, True, True, False]
    assert list(roles["train"]) == [True, True, True, False]
    assert list(roles["report"]) == [False, False, False, True]
    assert not (roles["train"] & roles["report"]).any()


def test_rank_frame_forward_returns_match_shifted_closes():
    n = 30
    close = np.linspace(10.0, 13.0, n)
    bench = np.linspace(100.0, 110.0, n)
    raw = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "symbol": ["AAA.SH"] * n,
            "market": ["cn"] * n,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 1000.0),
            "amount": np.full(n, 10000.0),
            "bench_close": bench,
        }
    )
    feat = rank_frame(raw, horizons=(5, 10, 20))
    i = 5
    assert np.isclose(feat.loc[i, "ret_10d"], close[i + 10] / close[i] - 1.0)
    bench_ret = bench[i + 10] / bench[i] - 1.0
    assert np.isclose(feat.loc[i, "exret_10d"], feat.loc[i, "ret_10d"] - bench_ret)
    assert np.isclose(feat.loc[i, "ret_5"], close[i] / close[i - 5] - 1.0)
    broken = raw.copy()
    broken.loc[i + 10, "close"] = -1.0
    broken_feat = rank_frame(broken, horizons=(5, 10, 20))
    assert np.isnan(broken_feat.loc[i, "ret_10d"])
    assert np.isnan(broken_feat.loc[i, "exret_10d"])
    spiked = raw.copy()
    spiked.loc[20, "close"] = float(raw.loc[20, "close"] * 20)
    spiked_feat = rank_frame(spiked, horizons=(5, 10, 20))
    assert np.isnan(spiked_feat.loc[10, "ret_10d"])
    assert np.isnan(spiked_feat.loc[20, "ret_5"])


def test_top_k_and_intersection():
    day = "2024-06-03"
    other = "2024-06-04"
    m5 = _scores([(day, "A", 5), (day, "B", 4), (day, "C", 1), (other, "A", 3), (other, "C", 1)])
    m10 = _scores([(day, "A", 9), (day, "C", 8), (day, "B", 0), (other, "D", 3), (other, "E", 2)])
    m20 = _scores([(day, "A", 2), (day, "D", 1), (day, "B", 0), (other, "F", 4), (other, "G", 3)])
    picks = intersect_top_k([m5, m10, m20], k=2)
    got = set(picks.loc[picks["trade_date"] == day, "symbol"])
    assert got == {"A"}
    assert picks.loc[picks["trade_date"] == other].empty
    cover = intersection_coverage([m5, m10, m20], picks)
    assert cover["days"] == 2
    assert cover["days_nonempty"] == 1
    assert cover["coverage"] == 0.5
    assert cover["mean_names"] == 1.0


def test_book_requires_full_top_and_subtracts_benchmark():
    picks = pd.DataFrame(
        {
            "trade_date": ["2024-06-03", "2024-06-03", "2024-06-04", "2024-06-04"],
            "symbol": ["A", "B", "A", "B"],
        }
    )
    returns = pd.DataFrame(
        {
            "trade_date": ["2024-06-03", "2024-06-03", "2024-06-04", "2024-06-04"],
            "symbol": ["A", "B", "A", "B"],
            "ret_5d": [0.10, 0.20, 0.05, np.nan],
        }
    )
    bench = pd.Series({pd.Timestamp("2024-06-03"): 0.04, pd.Timestamp("2024-06-04"): 0.01})
    daily = book_daily(picks, returns, bench, ret_col="ret_5d", min_names=2)
    assert list(pd.to_datetime(daily["trade_date"]).dt.strftime("%Y-%m-%d")) == ["2024-06-03"]
    assert np.isclose(daily.iloc[0]["port"], 0.15)
    assert np.isclose(daily.iloc[0]["bench"], 0.04)
    assert np.isclose(daily.iloc[0]["excess"], 0.11)
    stats = summarize_hold(daily, step=5)
    assert stats["days"] == 1
    assert np.isclose(stats["port"], 0.15)
    assert np.isclose(stats["excess_nonoverlap"], 0.11)
