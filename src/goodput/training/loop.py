"""Single-process toy training loop (tickets 1.3 + 1.5)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import torch
from torch import nn

from goodput.checkpointing import capture_training_state, restore_training_state
from goodput.config import Settings
from goodput.data import Batch, SyntheticDataLoader, load_batch_fixture
from goodput.models import ToyMLP
from goodput.providers.base import CheckpointStore


@dataclass
class TrainResult:
    """Outcome of a single-process training run."""

    steps_completed: int
    losses: list[float] = field(default_factory=list)
    final_loss: float = float("nan")
    device: str = "cpu"
    last_checkpoint_step: int | None = None
    resumed_from_step: int | None = None

    @property
    def ok(self) -> bool:
        return self.steps_completed > 0 and math.isfinite(self.final_loss)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _batch_stream(batches: Iterable[Batch], *, skip: int = 0) -> Iterator[Batch]:
    """Cycle batches so short loaders can feed many steps.

    ``skip`` advances past batches already consumed before ``start_step`` so
    resume continues at the same position in the cycle as an uninterrupted run.
    """
    stream = cycle(batches)
    for _ in range(skip):
        next(stream)
    return stream


def train_steps(
    *,
    model: nn.Module,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device | str = "cpu",
    start_step: int = 0,
    checkpoint_store: CheckpointStore | None = None,
    ckpt_interval: int = 0,
) -> TrainResult:
    """
    Run training from ``start_step`` for ``steps`` updates.

    Global step index goes ``start_step .. start_step+steps-1``.
    When ``ckpt_interval > 0`` and a store is provided, saves after every
    N completed steps (and always after the final step).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if start_step < 0:
        raise ValueError("start_step must be >= 0")

    device_t = torch.device(device)
    model = model.to(device_t)
    model.train()
    criterion = nn.MSELoss()
    stream = _batch_stream(batches, skip=start_step)

    losses: list[float] = []
    last_ckpt: int | None = None
    end_step = start_step + steps

    for step in range(start_step, end_step):
        batch = next(stream).to(device_t)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch.inputs)
        loss = criterion(preds, batch.targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        completed = step + 1  # steps done so far (1-based count of updates)
        should_ckpt = checkpoint_store is not None and ckpt_interval > 0 and (
            completed % ckpt_interval == 0 or completed == end_step
        )
        if should_ckpt:
            assert checkpoint_store is not None
            payload = capture_training_state(model, optimizer, step=completed)
            checkpoint_store.save(payload)
            last_ckpt = completed

    return TrainResult(
        steps_completed=len(losses),
        losses=losses,
        final_loss=losses[-1],
        device=str(device_t),
        last_checkpoint_step=last_ckpt,
        resumed_from_step=start_step if start_step > 0 else None,
    )


def train_from_settings(
    settings: Settings,
    *,
    checkpoint_store: CheckpointStore | None = None,
) -> TrainResult:
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

    start_step = 0
    if checkpoint_store is not None and checkpoint_store.latest() is not None:
        payload = checkpoint_store.load()
        start_step = restore_training_state(model, optimizer, payload, device=device)

    return train_steps(
        model=model,
        batches=loader,
        optimizer=optimizer,
        steps=settings.steps,
        device=device,
        start_step=start_step,
        checkpoint_store=checkpoint_store,
        ckpt_interval=settings.ckpt_interval,
    )


def resume_after_crash(
    *,
    settings: Settings,
    checkpoint_store: CheckpointStore,
    remaining_steps: int,
) -> TrainResult:
    """
    Simulate a process death: discard in-memory state, reload latest checkpoint,
    continue for ``remaining_steps``.

    This is the Done-when for ticket 1.5 (real SIGKILL is ticket 1.6).
    """
    if checkpoint_store.latest() is None:
        raise FileNotFoundError("no checkpoint to resume from")

    device = _resolve_device(settings.device)
    torch.manual_seed(settings.seed)  # rebuild graph; weights come from ckpt
    model = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    payload = checkpoint_store.load()
    ckpt_step = restore_training_state(model, optimizer, payload, device=device)

    # Keep the same cycling batch pool an uninterrupted run would have used,
    # not a shorter pool sized only to ``remaining_steps``.
    pool = max(settings.steps, ckpt_step + remaining_steps)
    loader = SyntheticDataLoader(
        num_batches=max(1, min(pool, 16)),
        batch_size=settings.batch_size,
        input_size=settings.input_size,
        seed=settings.seed,
        device=device,
    )
    return train_steps(
        model=model,
        batches=loader,
        optimizer=optimizer,
        steps=remaining_steps,
        device=device,
        start_step=ckpt_step,
        checkpoint_store=checkpoint_store,
        ckpt_interval=settings.ckpt_interval,
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
