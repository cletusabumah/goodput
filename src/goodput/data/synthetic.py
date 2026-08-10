"""Synthetic batch types and loaders (no real datasets)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    """One mini-batch of synthetic regression data."""

    inputs: Tensor  # (batch_size, input_size)
    targets: Tensor  # (batch_size, 1)

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2:
            raise ValueError(f"inputs must be 2D, got shape {tuple(self.inputs.shape)}")
        if self.targets.ndim != 2 or self.targets.shape[-1] != 1:
            raise ValueError(f"targets must be (B, 1), got shape {tuple(self.targets.shape)}")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("inputs and targets batch sizes must match")

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def input_size(self) -> int:
        return int(self.inputs.shape[1])

    def to(self, device: torch.device | str) -> Batch:
        return Batch(inputs=self.inputs.to(device), targets=self.targets.to(device))


def generate_synthetic_batch(
    *,
    batch_size: int,
    input_size: int,
    seed: int,
    noise_std: float = 0.1,
    device: torch.device | str = "cpu",
) -> Batch:
    """
    Deterministic Gaussian features with a fixed linear target + noise.

    y = x @ w + b + eps, with w/b derived from ``seed`` so the same seed
    always yields the same batch (reproducible CI fixtures).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if input_size < 1:
        raise ValueError("input_size must be >= 1")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    inputs = torch.randn(batch_size, input_size, generator=gen)
    weights = torch.randn(input_size, 1, generator=gen)
    bias = torch.randn(1, generator=gen)
    noise = noise_std * torch.randn(batch_size, 1, generator=gen)
    targets = inputs @ weights + bias + noise

    return Batch(inputs=inputs.to(device), targets=targets.to(device))


def iter_synthetic_batches(
    *,
    num_batches: int,
    batch_size: int,
    input_size: int,
    seed: int,
    noise_std: float = 0.1,
    device: torch.device | str = "cpu",
) -> Iterator[Batch]:
    """Yield ``num_batches`` deterministic batches (seed, seed+1, ...)."""
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    for i in range(num_batches):
        yield generate_synthetic_batch(
            batch_size=batch_size,
            input_size=input_size,
            seed=seed + i,
            noise_std=noise_std,
            device=device,
        )


class SyntheticDataLoader:
    """Minimal iterable loader for the toy trainer (no Dataset/Sampler stack)."""

    def __init__(
        self,
        *,
        num_batches: int,
        batch_size: int,
        input_size: int,
        seed: int,
        noise_std: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if input_size < 1:
            raise ValueError("input_size must be >= 1")
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed
        self.noise_std = noise_std
        self.device = device

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Batch]:
        return iter_synthetic_batches(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            input_size=self.input_size,
            seed=self.seed,
            noise_std=self.noise_std,
            device=self.device,
        )


def save_batch_fixture(batch: Batch, path: Path | str, *, seed: int | None = None) -> Path:
    """Write a tiny torch fixture (git-safe KB scale)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": batch.inputs.detach().cpu(),
        "targets": batch.targets.detach().cpu(),
        "meta": {
            "batch_size": batch.batch_size,
            "input_size": batch.input_size,
            "seed": seed,
            "synthetic": True,
        },
    }
    torch.save(payload, path)
    return path


def load_batch_fixture(path: Path | str, *, device: torch.device | str = "cpu") -> Batch:
    """Load a batch fixture written by ``save_batch_fixture``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "inputs" not in payload or "targets" not in payload:
        raise ValueError(f"Invalid fixture format: {path}")
    return Batch(inputs=payload["inputs"], targets=payload["targets"]).to(device)


def train_val_split(
    batch: Batch,
    *,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[Batch, Batch]:
    """Split one batch into train/val along the batch dimension (deterministic)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n = batch.batch_size
    if n < 2:
        raise ValueError("need batch_size >= 2 to split")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train = Batch(inputs=batch.inputs[train_idx], targets=batch.targets[train_idx])
    val = Batch(inputs=batch.inputs[val_idx], targets=batch.targets[val_idx])
    return train, val
