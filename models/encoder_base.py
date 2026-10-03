"""Encoder protocol for swappable backbones."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class EncoderBase(nn.Module, ABC):
    """Maps (batch, seq, features) -> (batch, embed_dim)."""

    embed_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
