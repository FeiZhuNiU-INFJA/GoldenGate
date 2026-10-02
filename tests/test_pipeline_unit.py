"""Unit tests that do not require network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.schema import normalize_bars
from eval.baselines import always_neutral, momentum_from_features
from labels.excess_return import excess_to_class, label_symbol_frame
from models.multi_head import MultiMarketModel
from train.dataset import preprocess_window


def test_normalize_bars_sorts_and_dedups():
    df = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-01", "2020-01-02"],
            "symbol": ["AAA.SH"] * 3,
            "market": ["cn"] * 3,
            "open": [10, 9, 11],
            "high": [11, 10, 12],
            "low": [9, 8, 10],
            "close": [10.5, 9.5, 11.5],
            "volume": [1, 1, 1],
            "amount": [1, 1, 1],
        }
    )
    out = normalize_bars(df)
    assert len(out) == 2
    assert str(out.iloc[0]["trade_date"].date()) == "2020-01-01"
    assert out.iloc[-1]["close"] == 11.5


def test_excess_to_class_thresholds():
    s = pd.Series([0.02, 0.0, -0.02, np.nan])
    y = excess_to_class(s, 0.015)
    assert list(y.iloc[:3].astype(int)) == [1, 0, 2]
    assert pd.isna(y.iloc[3])


def test_label_symbol_frame_shapes():
    dates = pd.bdate_range("2020-01-01", periods=40)
    close = np.linspace(100, 120, len(dates))
    bars = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["T.SH"] * len(dates),
            "market": ["cn"] * len(dates),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(len(dates)) * 1000,
            "amount": np.ones(len(dates)) * 1e6,
        }
    )
    bench = bars.copy()
    bench["symbol"] = "000300.SH"
    labeled = label_symbol_frame(bars, bench)
    assert "y_5d" in labeled.columns and "y_20d" in labeled.columns
    # last 20 rows should have NA for 20d label
    assert labeled["y_20d"].isna().sum() >= 20


def test_preprocess_window():
    x = np.ones((8, 6), dtype=np.float64)
    x[:, 0] = 10
    x[:, 3] = 12
    out = preprocess_window(x)
    assert np.isclose(out[0, 0], 1.0)
    assert np.isclose(out[0, 3], 1.2)


def test_model_forward_shapes():
    model = MultiMarketModel()
    x = torch.randn(4, 128, 6)
    mid = torch.tensor([0, 1, 2, 0])
    out = model(x, mid)
    assert out["5d"].shape == (4, 3)
    assert out["20d"].shape == (4, 3)
    score = model.score(x, mid)
    assert score.shape == (4,)


def test_baselines():
    y = always_neutral(10)
    assert y.sum() == 0
    x = np.ones((5, 20, 6), dtype=np.float32)
    x[:, -1, 3] = 1.1
    mom = momentum_from_features(x, threshold=0.04)
    assert (mom == 1).all()
