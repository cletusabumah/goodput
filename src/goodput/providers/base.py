"""Swappable provider interfaces — mocks keep CI GPU-free and download-free."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointPayload:
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
        ...


class FaultInjector(ABC):
    name: str

    @abstractmethod
    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        """Return fault type if injected at this step, else None."""


class MetricsSink(ABC):
    name: str

    @abstractmethod
    def emit(self, metrics: dict[str, Any]) -> None:
        ...


class MockCheckpointStore(CheckpointStore):
    name = "mock"

    def __init__(self) -> None:
        self._latest: CheckpointPayload | None = None
        self._locator = "memory://latest"

    def save(self, payload: CheckpointPayload) -> Path | str:
        self._latest = payload
        return self._locator

    def load(self, locator: Path | str | None = None) -> CheckpointPayload:
        if self._latest is None:
            raise FileNotFoundError("No checkpoint in mock store")
        return self._latest

    def latest(self) -> Path | str | None:
        return self._locator if self._latest is not None else None


class MockFaultInjector(FaultInjector):
    """Records intent without signaling real processes (CI-safe)."""

    name = "mock"

    def __init__(self, inject_at: int | None = None, fault: str = "kill") -> None:
        self.inject_at = inject_at
        self.fault = fault
        self.injected: list[tuple[int, int, str]] = []

    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        if self.inject_at is not None and step == self.inject_at:
            self.injected.append((step, worker_id, self.fault))
            return self.fault
        return None


class StdoutMetricsSink(MetricsSink):
    name = "stdout"

    def emit(self, metrics: dict[str, Any]) -> None:
        print(metrics)


class JsonFileMetricsSink(MetricsSink):
    name = "json_file"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, metrics: dict[str, Any]) -> None:
        import json

        self.path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
