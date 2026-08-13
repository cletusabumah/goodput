"""Swappable provider interfaces — mocks keep CI GPU-free and download-free."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointPayload:
    """Model/optimizer snapshot at a training step."""

    step: int
    state: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class CheckpointStore(ABC):
    name: str

    @abstractmethod
    def save(self, payload: CheckpointPayload) -> Path | str:
        """Persist checkpoint; return locator."""

    @abstractmethod
    def load(self, locator: Path | str | None = None) -> CheckpointPayload:
        """Load latest or specified checkpoint."""

    @abstractmethod
    def latest(self) -> Path | str | None:
        """Return locator of the newest checkpoint, or None."""

    def load_at_step(self, step: int) -> CheckpointPayload:
        """Load the checkpoint written for a specific global step (ticket 2.1)."""
        raise NotImplementedError(f"{type(self).__name__} does not support load_at_step")


class FaultInjector(ABC):
    name: str

    @abstractmethod
    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        """Return fault type if injected at this step, else None."""


class MetricsSink(ABC):
    name: str

    @abstractmethod
    def emit(self, metrics: dict[str, Any]) -> None:
        """Write or print a metrics payload."""
