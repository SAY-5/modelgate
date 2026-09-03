"""ETA regressor: a small MLP over the encoded trip features."""

from __future__ import annotations

from pathlib import Path

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


def save_model(model: EtaNet, path: Path, hidden: int, depth: int) -> None:
    torch.save({"hidden": hidden, "depth": depth, "state_dict": model.state_dict()}, path)


def load_model(path: Path) -> EtaNet:
    """Rebuild an EtaNet from a saved artifact. Only tensors are unpickled."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = EtaNet(hidden=int(payload["hidden"]), depth=int(payload["depth"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
