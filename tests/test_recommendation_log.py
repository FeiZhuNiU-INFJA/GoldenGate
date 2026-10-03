"""Cumulative recommendation returns counted on the benchmark calendar."""
from __future__ import annotations

import numpy as np
import pandas as pd

import json

from eval.recommendation_html import render_html
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
        "horizons": {
            "5": {
                "picks": ["APP.US", "BR.US"],
                "seeds": {
                    "0": [{"symbol": "APP.US", "score": 0.09}, {"symbol": "BR.US", "score": 0.11}],
                    "1": [{"symbol": "APP.US", "score": 0.08}, {"symbol": "BR.US", "score": 0.10}],
                    "2": [{"symbol": "APP.US", "score": 0.07}, {"symbol": "BR.US", "score": 0.12}],
                },
            }
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
        {"2026-10-01": {5: path}},
        {"APP.US": "AppLovin", "BR.US": "Broadridge"},
        {"2026-10-01": {5: {"APP.US": 100.0, "BR.US": 50.0}}},
    )
    assert "未到期" in text
    assert "APP" in text
    assert "AppLovin" in text
    assert "+0.00%" in text
    assert "intersection.html" in text


def _page(script: str) -> dict:
    prefix = "window.INTERSECTION = "
    assert script.startswith(prefix)
    assert script.endswith(";\n")
    blob = script[len(prefix) : -2]
    assert "<" not in blob
    return json.loads(blob)


def test_html_page_carries_the_open_path_and_escapes_names():
    signal = {
        "date": "2026-10-01",
        "note": "",
        "horizons": {
            "5": {
                "picks": ["APP.US"],
                "seeds": {"0": [{"symbol": "APP.US", "score": 0.09}]},
            }
        },
    }
    flat = {"date": "2026-10-02", "picks": [], "horizons": {}}
    idx = pd.to_datetime(["2026-10-01", "2026-10-02"])
    path = session_path(
        {"APP.US": pd.Series([100.0, 90.0], index=idx)},
        pd.Series([1000.0, 1010.0], index=idx),
        idx[0],
        ["APP.US"],
    )
    html = render_html(
        [signal, flat],
        {"2026-10-01": {5: path}, "2026-10-02": {}},
        {"APP.US": "A<B&C>"},
        {"2026-10-01": {5: {"APP.US": 100.0}}},
    )
    page = _page(html)["markets"][0]
    assert page["as_of"] == "2026-10-02"
    book = page["groups"]["5"][0]
    assert book["picks"][0]["name"] == "A<B&C>"
    assert book["picks"][0]["symbol"] == "APP"
    assert book["held"] == 1
    assert book["horizons"]["5"]["status"] == "未到期"
    assert book["path"][1]["port"] == round(90.0 / 100.0 - 1.0, 8)
    assert book["path"][1]["excess"] == round((90.0 / 100.0 - 1.0) - (1010.0 / 1000.0 - 1.0), 8)
    assert page["groups"]["5"][1]["flat"] is True
    assert page["groups"]["5"][1]["horizons"]["5"]["status"] == "—"
