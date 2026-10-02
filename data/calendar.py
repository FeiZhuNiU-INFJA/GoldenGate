"""Trading calendar helpers per market."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from data import store


def build_calendar_from_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    if benchmark is None or benchmark.empty:
        return pd.DataFrame(columns=["trade_date"])
    dates = (
        pd.to_datetime(benchmark["trade_date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"trade_date": dates})


def load_calendar(market: str) -> Optional[pd.DataFrame]:
    return store.read_parquet(store.calendar_path(market))


def save_calendar(market: str, calendar: pd.DataFrame) -> None:
    store.write_parquet(calendar, store.calendar_path(market))


def trading_days_between(
    calendar: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    cal = pd.to_datetime(calendar["trade_date"]).sort_values()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (cal >= start_ts) & (cal <= end_ts)
    return pd.DatetimeIndex(cal.loc[mask].values)
