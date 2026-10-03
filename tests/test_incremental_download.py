"""Incremental bar merge without network."""
from __future__ import annotations

import pandas as pd

from data import store
from data.download import incremental_start_date, merge_bars, update_market
from data.schema import BAR_COLS


def _bars(dates, closes, symbol="AAA.US", market="us") -> pd.DataFrame:
    rows = []
    for date, close in zip(dates, closes):
        rows.append(
            {
                "trade_date": date,
                "symbol": symbol,
                "market": market,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
                "amount": 1.0,
            }
        )
    return pd.DataFrame(rows)[BAR_COLS]


def test_incremental_start_overlaps_and_clamps():
    start = incremental_start_date(
        pd.Timestamp("2024-06-10"),
        end_date="20240610",
        overlap_days=7,
        floor_start="20140101",
    )
    assert start == "20240603"

    clamped = incremental_start_date(
        pd.Timestamp("2010-01-01"),
        end_date="20240610",
        overlap_days=7,
        floor_start="20140101",
    )
    assert clamped == "20140101"

    assert (
        incremental_start_date(
            pd.Timestamp("2024-06-12"),
            end_date="20240610",
            overlap_days=0,
            floor_start="20140101",
        )
        is None
    )
    assert (
        incremental_start_date(None, end_date="20240610", overlap_days=7, floor_start="20140101")
        == "20140101"
    )


def test_merge_bars_keeps_newer_overlap_and_appends():
    existing = _bars(["2024-06-01", "2024-06-03"], [10.0, 11.0])
    incoming = _bars(["2024-06-03", "2024-06-04"], [11.5, 12.0])
    out = merge_bars(existing, incoming)
    assert list(out["trade_date"].dt.strftime("%Y-%m-%d")) == ["2024-06-01", "2024-06-03", "2024-06-04"]
    assert out.loc[out["trade_date"] == "2024-06-03", "close"].iloc[0] == 11.5
    assert merge_bars(None, None).empty


def test_update_market_appends_without_rewriting_history(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DIR_DATASET", tmp_path)
    store.write_universe(
        "us",
        pd.DataFrame(
            [{"symbol": "AAA.US", "name": "AAA", "market": "us", "raw_symbol": "AAA"}]
        ),
    )
    store.write_parquet(_bars(["2024-06-01", "2024-06-03"], [10.0, 11.0]), store.bar_path("us", "AAA.US"))
    store.write_parquet(
        _bars(["2024-06-01"], [100.0], symbol="SPX.US"),
        store.benchmark_path("us"),
    )

    calls = {}

    def fake_bars(raw_symbol, symbol, start_date, end_date):
        calls["bars"] = (raw_symbol, symbol, start_date, end_date)
        return _bars(["2024-06-03", "2024-06-04"], [11.5, 12.0], symbol=symbol)

    def fake_benchmark(start_date, end_date):
        calls["bench"] = (start_date, end_date)
        return _bars(["2024-06-01", "2024-06-04"], [100.0, 101.0], symbol="SPX.US")

    import data.market_client as client

    monkeypatch.setitem(client.FETCH_BARS, "us", fake_bars)
    monkeypatch.setitem(client.FETCH_BENCHMARK, "us", fake_benchmark)

    result = update_market("us", end_date="20240604", overlap_days=5, workers=1)

    assert calls["bars"][2] == "20240529"
    assert calls["bars"][3] == "20240604"
    assert result["ok"] == 1
    assert result["fail"] == 0
    assert result["added"] == 1

    bars = store.read_parquet(store.bar_path("us", "AAA.US"))
    assert list(bars["trade_date"].dt.strftime("%Y-%m-%d")) == ["2024-06-01", "2024-06-03", "2024-06-04"]
    assert bars.loc[bars["trade_date"] == "2024-06-01", "close"].iloc[0] == 10.0
    assert bars.loc[bars["trade_date"] == "2024-06-03", "close"].iloc[0] == 11.5

    calendar = store.read_parquet(store.calendar_path("us"))
    assert "2024-06-04" in set(calendar["trade_date"].dt.strftime("%Y-%m-%d"))
