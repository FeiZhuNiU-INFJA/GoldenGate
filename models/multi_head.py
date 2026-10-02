"""Shared encoder + per-market dual-horizon classification heads."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import (
    EMBED_DIM,
    HORIZONS,
    MARKET_TO_ID,
    MARKETS,
    N_CLASSES,
    N_FEATURES,
    N_HEADS,
    N_LAYERS,
    DROPOUT,
    SEQ_LEN,
)
from models.encoder_base import EncoderBase
from models.transformer_encoder import TransformerEncoder


class MultiMarketModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderBase | None = None,
        n_features: int = N_FEATURES,
        seq_len: int = SEQ_LEN,
        embed_dim: int = EMBED_DIM,
        n_markets: int = len(MARKETS),
        market_emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder or TransformerEncoder(
            n_features=n_features,
            seq_len=seq_len,
            embed_dim=embed_dim,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            dropout=DROPOUT,
        )
        self.market_emb = nn.Embedding(n_markets, market_emb_dim)
        fused = self.encoder.embed_dim + market_emb_dim
        self.heads = nn.ModuleDict()
        for market in MARKETS:
            self.heads[market] = nn.ModuleDict(
                {f"{h}d": nn.Linear(fused, N_CLASSES) for h in HORIZONS}
            )

    def encode(self, x: torch.Tensor, market_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        m = self.market_emb(market_ids)
        return torch.cat([h, m], dim=-1)

    def forward(self, x: torch.Tensor, market_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys '5d' and '20d', each (B, 3) logits.
        Routes each sample to its market-specific head.
        """
        fused = self.encode(x, market_ids)
        device = x.device
        out: Dict[str, torch.Tensor] = {}
        for h in HORIZONS:
            key = f"{h}d"
            logits = torch.zeros(x.size(0), N_CLASSES, device=device, dtype=fused.dtype)
            for market, mid in MARKET_TO_ID.items():
                mask = market_ids == mid
                if not mask.any():
                    continue
                logits[mask] = self.heads[market][key](fused[mask])
            out[key] = logits
        return out

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, market_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        logits = self.forward(x, market_ids)
        return {k: F.softmax(v, dim=-1) for k, v in logits.items()}

    @torch.no_grad()
    def score(self, x: torch.Tensor, market_ids: torch.Tensor, w5: float = 0.4, w20: float = 0.6) -> torch.Tensor:
        """Long-short score = P(buy) - P(sell), blended across horizons."""
        probs = self.predict_proba(x, market_ids)
        s5 = probs["5d"][:, 1] - probs["5d"][:, 2]
        s20 = probs["20d"][:, 1] - probs["20d"][:, 2]
        return w5 * s5 + w20 * s20
