"""Create synthetic multi-market bars for offline smoke tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import MARKETS
from data import calendar as cal
from data import store
from data.schema import normalize_bars


def _make_price_series(n: int, seed: int, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0005, 0.015, size=n)
    return start * np.cumprod(1.0 + shocks)


def generate_market(market: str, n_symbols: int = 5, n_days: int = 2600, seed: int = 0) -> None:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    bench_close = _make_price_series(n_days, seed=seed + 99)
    bench = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": f"BENCH.{market.upper()}",
            "market": market,
            "open": bench_close,
            "high": bench_close * 1.01,
            "low": bench_close * 0.99,
            "close": bench_close,
            "volume": np.full(n_days, 1e6),
            "amount": np.full(n_days, 1e8),
        }
    )
    bench = normalize_bars(bench)
    store.write_parquet(bench, store.benchmark_path(market))
    cal.save_calendar(market, cal.build_calendar_from_benchmark(bench))

    rows = []
    for i in range(n_symbols):
        sym = f"SYN{i:03d}.{market.upper()}"
        close = _make_price_series(n_days, seed=seed + i + 1, start=50 + i)
        bars = pd.DataFrame(
            {
                "trade_date": dates,
                "symbol": sym,
                "market": market,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(n_days, 1e5 + i),
                "amount": np.full(n_days, 1e7 + i),
            }
        )
        bars = normalize_bars(bars)
        store.write_parquet(bars, store.bar_path(market, sym))
        rows.append({"symbol": sym, "name": sym, "market": market, "raw_symbol": sym.split(".")[0]})
    store.write_universe(market, pd.DataFrame(rows))


def generate_all(n_symbols: int = 5, n_days: int = 2600) -> None:
    for i, m in enumerate(MARKETS):
        generate_market(m, n_symbols=n_symbols, n_days=n_days, seed=100 * i)


if __name__ == "__main__":
    from config.settings import DIR_DATASET

    generate_all()
    print(f"synthetic dataset written under {DIR_DATASET}")
