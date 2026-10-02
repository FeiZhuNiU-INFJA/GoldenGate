"""Parquet / CSV persistence helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DIR_DATASET, MARKETS


def market_dir(market: str) -> Path:
    if market not in MARKETS:
        raise ValueError(f"unknown market: {market}")
    path = DIR_DATASET / market
    path.mkdir(parents=True, exist_ok=True)
    return path


def bars_dir(market: str) -> Path:
    path = market_dir(market) / "bars"
    path.mkdir(parents=True, exist_ok=True)
    return path


def labeled_dir(market: str) -> Path:
    path = market_dir(market) / "labeled"
    path.mkdir(parents=True, exist_ok=True)
    return path


def universe_path(market: str) -> Path:
    return market_dir(market) / "universe.csv"


def calendar_path(market: str) -> Path:
    return market_dir(market) / "calendar.parquet"


def benchmark_path(market: str) -> Path:
    return market_dir(market) / "benchmark.parquet"


def bar_path(market: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return bars_dir(market) / f"{safe}.parquet"


def labeled_path(market: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return labeled_dir(market) / f"{safe}.parquet"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_universe(market: str, df: pd.DataFrame) -> None:
    path = universe_path(market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_universe(market: str) -> Optional[pd.DataFrame]:
    path = universe_path(market)
    if not path.exists():
        return None
    return pd.read_csv(path)


def list_bar_symbols(market: str) -> list[str]:
    paths = sorted(bars_dir(market).glob("*.parquet"))
    return [p.stem for p in paths]


def list_labeled_symbols(market: str) -> list[str]:
    paths = sorted(labeled_dir(market).glob("*.parquet"))
    return [p.stem for p in paths]
