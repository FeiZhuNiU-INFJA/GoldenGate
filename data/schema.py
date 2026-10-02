"""Unified bar schema and column helpers."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

BAR_COLS = [
    "trade_date",
    "symbol",
    "market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount"]


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLS)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns, sorted dates, and dtypes."""
    if df is None or df.empty:
        return empty_bars()
    out = df.copy()
    missing = [c for c in BAR_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    out = out[BAR_COLS].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    out["market"] = out["market"].astype(str)
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["trade_date", "close"])
    out = out.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def rename_zh_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    cols = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=cols)


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out
