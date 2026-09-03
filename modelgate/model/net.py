"""ETA regressor: a small MLP over the encoded trip features."""

from __future__ import annotations

import torch
from torch import nn

from modelgate.model.features import FEATURE_DIM


class EtaNet(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 2, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.ReLU())
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softplus keeps predicted minutes positive without clipping gradients.
        return nn.functional.softplus(self.body(x)).squeeze(-1)
