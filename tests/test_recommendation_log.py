"""Cumulative recommendation returns counted on the benchmark calendar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.recommendation_log import horizon_state, render_log, session_path


def test_session_path_is_equal_weight_and_keeps_hold_when_a_name_is_missing():
    idx = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"])
    prices = {
        "A": pd.Series([10.0, 11.0, 12.0], index=idx),
        "B": pd.Series([20.0, np.nan, 22.0], index=idx),
    }
    bench = pd.Series([100.0, 101.0, 102.0], index=idx)
    path = session_path(prices, bench, "2026-10-01", ["A", "B"])
    assert list(path["hold"]) == [0, 2]
    assert np.isclose(path.iloc[0]["port"], 0.0)
    assert np.isclose(path.iloc[0]["excess"], 0.0)
    port = ((12.0 / 10.0 - 1.0) + (22.0 / 20.0 - 1.0)) / 2.0
    assert np.isclose(path.iloc[1]["port"], port)
    assert np.isclose(path.iloc[1]["bench"], 102.0 / 100.0 - 1.0)
    assert np.isclose(path.iloc[1]["excess"], port - (102.0 / 100.0 - 1.0))
    assert horizon_state(path, 5, flat=False) == "未到期"
    assert horizon_state(path, 5, flat=True) == "—"


def test_horizon_row_matches_the_nth_benchmark_close():
    idx = pd.bdate_range("2026-10-01", periods=8)
    close = pd.Series(np.arange(10.0, 18.0), index=idx)
    bench = pd.Series(np.arange(100.0, 108.0), index=idx)
    path = session_path({"A": close}, bench, idx[0], ["A"], max_hold=20)
    state = horizon_state(path, 5, flat=False)
    assert np.isclose(state["port"], close.iloc[5] / close.iloc[0] - 1.0)
    assert np.isclose(state["bench"], bench.iloc[5] / bench.iloc[0] - 1.0)
    assert int(path["hold"].max()) == 7


def test_path_stops_at_hold_20_and_ignores_sessions_off_the_benchmark():
    idx = pd.bdate_range("2026-10-01", periods=25)
    close = pd.Series(np.linspace(10.0, 20.0, 25), index=idx)
    extra = close.copy()
    extra.loc[pd.Timestamp("2026-10-03")] = 99.0  # Saturday, not a benchmark session
    bench = pd.Series(np.linspace(100.0, 120.0, 25), index=idx)
    path = session_path({"A": extra}, bench, idx[0], ["A"])
    assert int(path["hold"].max()) == 20
    assert len(path) == 21
    assert pd.Timestamp("2026-10-03") not in set(path["trade_date"])


def test_render_marks_an_open_recommendation_as_not_yet_due():
    signal = {
        "date": "2026-10-01",
        "picks": ["APP.US", "BR.US"],
        "top5": {
            "5": [{"symbol": "APP.US", "score": 0.09}, {"symbol": "BR.US", "score": 0.11}],
            "10": [{"symbol": "APP.US", "score": 0.12}, {"symbol": "BR.US", "score": 0.10}],
        },
    }
    idx = pd.to_datetime(["2026-10-01"])
    path = session_path(
        {"APP.US": pd.Series([100.0], index=idx), "BR.US": pd.Series([50.0], index=idx)},
        pd.Series([1000.0], index=idx),
        idx[0],
        ["APP.US", "BR.US"],
    )
    text = render_log(
        [signal],
        {"2026-10-01": path},
        {"APP.US": "AppLovin", "BR.US": "Broadridge"},
        {"2026-10-01": {"APP.US": 100.0, "BR.US": 50.0}},
    )
    assert "未到期" in text
    assert "APP" in text
    assert "AppLovin" in text
    assert "+0.00%" in text
