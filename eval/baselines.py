"""Simple baselines: always-neutral and momentum."""
from __future__ import annotations

import numpy as np


def always_neutral(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def momentum_labels(closes: np.ndarray, lookback: int = 20, threshold: float = 0.04) -> np.ndarray:
    """
    For each index i (>= lookback), label by past lookback return.
    Returns array same length as closes with leading zeros where undefined -> neutral.
    """
    y = np.zeros(len(closes), dtype=np.int64)
    for i in range(lookback, len(closes)):
        ret = closes[i] / closes[i - lookback] - 1.0
        if ret >= threshold:
            y[i] = 1
        elif ret <= -threshold:
            y[i] = 2
    return y


def momentum_from_features(x_batch: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """
    x_batch: (N, T, F) with close at feature index 3, already window-normalized.
    Use last/first close ratio in the window as momentum proxy.
    """
    first = x_batch[:, 0, 3]
    last = x_batch[:, -1, 3]
    ret = last / np.maximum(first, 1e-6) - 1.0
    y = np.zeros(len(ret), dtype=np.int64)
    y[ret >= threshold] = 1
    y[ret <= -threshold] = 2
    return y
