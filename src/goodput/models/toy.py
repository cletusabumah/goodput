"""Tiny MLP for synthetic regression (disposable — systems focus, not accuracy)."""

from __future__ import annotations

import torch
from torch import nn


class ToyMLP(nn.Module):
    """Two-layer MLP: input → hidden → 1 (regression)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
