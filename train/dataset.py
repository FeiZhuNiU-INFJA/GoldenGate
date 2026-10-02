"""Window dataset over labeled parquet bars."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.settings import (
    FEATURE_COLS,
    HORIZONS,
    MARKET_TO_ID,
    MARKETS,
    SEQ_LEN,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from data import store

Split = Literal["train", "val", "test"]

SPLIT_RANGES = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}


def preprocess_window(x: np.ndarray) -> np.ndarray:
    """Normalize OHLCV+amount by first-day open / first-day volume / first-day amount."""
    out = x.astype(np.float64).copy()
    base_open = out[0, 0]
    if base_open == 0 or np.isnan(base_open):
        base_open = 1.0
    out[:, 0:4] /= base_open
    vol0 = out[0, 4] if out[0, 4] not in (0, np.nan) else 1.0
    amt0 = out[0, 5] if out[0, 5] not in (0, np.nan) else 1.0
    out[:, 4] /= vol0
    out[:, 5] /= amt0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


class SymbolWindowIndex:
    def __init__(self, market: str, symbol: str, df: pd.DataFrame, seq_len: int, start: str, end: str):
        self.market = market
        self.symbol = symbol
        self.market_id = MARKET_TO_ID[market]
        self.seq_len = seq_len
        self.df = df
        self.features = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        dates = pd.to_datetime(df["trade_date"])
        y5 = df["y_5d"].to_numpy()
        y20 = df["y_20d"].to_numpy()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        n = len(df)
        if n < seq_len:
            self.indices = []
        else:
            idx = np.arange(seq_len - 1, n)
            in_range = (dates.iloc[idx].to_numpy() >= start_ts) & (dates.iloc[idx].to_numpy() <= end_ts)
            labeled = (~pd.isna(y5[idx])) & (~pd.isna(y20[idx]))
            self.indices = idx[in_range & labeled].tolist()
        self.y5 = y5
        self.y20 = y20
        self.dates = dates

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, local_idx: int):
        i = self.indices[local_idx]
        window = self.features[i - self.seq_len + 1 : i + 1]
        x = preprocess_window(window)
        return (
            x,
            int(self.y5[i]),
            int(self.y20[i]),
            self.market_id,
            str(self.symbol),
            str(self.dates.iloc[i].date()),
        )


class MultiMarketDataset(Dataset):
    def __init__(
        self,
        split: Split,
        markets: Sequence[str] = MARKETS,
        seq_len: int = SEQ_LEN,
        max_symbols_per_market: Optional[int] = None,
        with_meta: bool = False,
    ) -> None:
        self.with_meta = with_meta
        start, end = SPLIT_RANGES[split]
        self.entries: list[SymbolWindowIndex] = []
        self._cumlen: list[int] = []
        total = 0
        for market in markets:
            symbols = store.list_labeled_symbols(market)
            if max_symbols_per_market is not None:
                symbols = symbols[:max_symbols_per_market]
            for sym in symbols:
                df = store.read_parquet(store.labeled_path(market, sym))
                if df is None or df.empty:
                    continue
                needed = FEATURE_COLS + ["y_5d", "y_20d", "trade_date"]
                if any(c not in df.columns for c in needed):
                    continue
                entry = SymbolWindowIndex(market, sym, df, seq_len, start, end)
                if len(entry) == 0:
                    continue
                self.entries.append(entry)
                total += len(entry)
                self._cumlen.append(total)

    def __len__(self) -> int:
        return self._cumlen[-1] if self._cumlen else 0

    def _locate(self, idx: int) -> tuple[SymbolWindowIndex, int]:
        # binary search cumulative lengths
        lo, hi = 0, len(self._cumlen) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumlen[mid]:
                hi = mid
            else:
                lo = mid + 1
        entry = self.entries[lo]
        prev = self._cumlen[lo - 1] if lo > 0 else 0
        return entry, idx - prev

    def __getitem__(self, idx: int):
        entry, local = self._locate(idx)
        x, y5, y20, mid, symbol, date = entry.get(local)
        x_t = torch.from_numpy(x)
        y5_t = torch.tensor(y5, dtype=torch.long)
        y20_t = torch.tensor(y20, dtype=torch.long)
        mid_t = torch.tensor(mid, dtype=torch.long)
        if self.with_meta:
            return x_t, y5_t, y20_t, mid_t, symbol, date
        return x_t, y5_t, y20_t, mid_t


def estimate_class_weights(dataset: MultiMarketDataset, horizon: int = 20) -> torch.Tensor:
    """Inverse-frequency weights for CE loss, shape (3,)."""
    counts = np.zeros(3, dtype=np.float64)
    # Sample up to 50k for speed
    n = min(len(dataset), 50000)
    step = max(len(dataset) // n, 1) if len(dataset) else 1
    for i in range(0, len(dataset), step):
        _, y5, y20, _ = dataset[i]
        y = y5 if horizon == 5 else y20
        counts[int(y.item())] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
