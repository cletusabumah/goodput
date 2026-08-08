"""Metrics sink implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodput.providers.base import MetricsSink


class StdoutMetricsSink(MetricsSink):
    name = "stdout"

    def emit(self, metrics: dict[str, Any]) -> None:
        print(json.dumps(metrics, sort_keys=True))


class JsonFileMetricsSink(MetricsSink):
    name = "json_file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, metrics: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MockMetricsSink(MetricsSink):
    """In-memory sink for unit tests."""

    name = "mock"

    def __init__(self) -> None:
        self.emitted: list[dict[str, Any]] = []

    def emit(self, metrics: dict[str, Any]) -> None:
        self.emitted.append(dict(metrics))
