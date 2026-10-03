"""Nasdaq-100 is a fourth market on the shared download and model path."""
from __future__ import annotations

import torch

from config.settings import MARKETS
from data import store
from data.market_client import FETCH_BARS, FETCH_BENCHMARK, FETCH_UNIVERSE
from models.multi_head import MultiMarketModel
from train.trainer import fit_checkpoint_state


def test_market_registry_includes_nasdaq():
    assert MARKETS == ("cn", "hk", "us", "ndx")
    assert set(FETCH_UNIVERSE) == set(MARKETS)
    assert set(FETCH_BARS) == set(MARKETS)
    assert set(FETCH_BENCHMARK) == set(MARKETS)


def test_shared_sp500_bar_resolves_for_nasdaq():
    path = store.resolve_bar_path("ndx", "AAPL.US")
    assert path is not None and path.exists()
    assert path.parent.parent.name == "us"
    assert store.has_bars("ndx", "AAPL.US")
    assert "AAPL.US" in store.list_bar_symbols("ndx")


def test_old_three_market_checkpoint_loads():
    torch.manual_seed(0)
    trained = MultiMarketModel()
    saved = {key: value.clone() for key, value in trained.state_dict().items()}
    saved["market_emb.weight"] = saved["market_emb.weight"][:3].clone()
    saved = {key: value for key, value in saved.items() if not key.startswith("heads.ndx.")}

    torch.manual_seed(1)
    fresh = MultiMarketModel()
    fit_checkpoint_state(fresh, saved)

    assert fresh.market_emb.weight.shape[0] == 4
    assert torch.equal(fresh.state_dict()["heads.cn.5d.weight"], saved["heads.cn.5d.weight"])
    assert fresh.market_emb.weight.shape[0] == 4
    x = torch.randn(1, 128, 6)
    out = fresh(x, torch.tensor([3]))
    assert out["5d"].shape == (1, 3)
