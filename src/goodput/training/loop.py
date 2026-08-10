"""Single-process toy training loop (ticket 1.3)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import torch
from torch import nn

from goodput.config import Settings
from goodput.data import Batch, SyntheticDataLoader, load_batch_fixture
from goodput.models import ToyMLP


@dataclass
class TrainResult:
    """Outcome of a single-process training run."""

    steps_completed: int
    losses: list[float] = field(default_factory=list)
    final_loss: float = float("nan")
    device: str = "cpu"

    @property
    def ok(self) -> bool:
        return self.steps_completed > 0 and math.isfinite(self.final_loss)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _batch_stream(batches: Iterable[Batch]) -> Iterator[Batch]:
    """Cycle batches so short loaders can feed many steps."""
    return cycle(batches)


def train_steps(
    *,
    model: nn.Module,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device | str = "cpu",
) -> TrainResult:
    """Run ``steps`` SGD updates over a cycling batch stream."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    device_t = torch.device(device)
    model = model.to(device_t)
    model.train()
    criterion = nn.MSELoss()
    stream = _batch_stream(batches)

    losses: list[float] = []
    for step in range(steps):
        batch = next(stream).to(device_t)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch.inputs)
        loss = criterion(preds, batch.targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return TrainResult(
        steps_completed=len(losses),
        losses=losses,
        final_loss=losses[-1],
        device=str(device_t),
    )


def train_from_settings(settings: Settings) -> TrainResult:
    """Build model/loader/optimizer from settings and run ``settings.steps``."""
    device = _resolve_device(settings.device)
    torch.manual_seed(settings.seed)

    model = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    loader = SyntheticDataLoader(
        num_batches=max(1, min(settings.steps, 16)),
        batch_size=settings.batch_size,
        input_size=settings.input_size,
        seed=settings.seed,
        device=device,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    return train_steps(
        model=model,
        batches=loader,
        optimizer=optimizer,
        steps=settings.steps,
        device=device,
    )


def train_on_fixture(
    fixture_path: str | Path,
    *,
    steps: int = 20,
    hidden_size: int = 32,
    learning_rate: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
) -> TrainResult:
    """Smoke helper: train repeatedly on one committed fixture batch."""
    torch.manual_seed(seed)
    device_t = _resolve_device(device)
    batch = load_batch_fixture(fixture_path, device=device_t)
    model = ToyMLP(input_size=batch.input_size, hidden_size=hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return train_steps(
        model=model,
        batches=[batch],
        optimizer=optimizer,
        steps=steps,
        device=device_t,
    )
