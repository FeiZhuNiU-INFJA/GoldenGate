"""Lightweight Pre-Norm Transformer encoder for daily bars."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.encoder_base import EncoderBase


class TransformerEncoder(EncoderBase):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 128,
        embed_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self._reset_linear(self.input_proj)

    @staticmethod
    def _reset_linear(m: nn.Linear) -> None:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, _ = x.shape
        if t > self.seq_len:
            x = x[:, -self.seq_len :, :]
            t = self.seq_len
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :t, :]
        h = self.encoder(h)
        h = self.norm(h)
        return h[:, -1, :]
