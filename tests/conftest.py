"""Shared fixtures — mock providers, temp artifact dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from goodput.providers import (
    MockCheckpointStore,
    MockFaultInjector,
    MockMetricsSink,
)


@pytest.fixture
def mock_checkpoint_store() -> MockCheckpointStore:
    return MockCheckpointStore()


@pytest.fixture
def mock_fault_injector() -> MockFaultInjector:
    return MockFaultInjector(inject_at=5, fault="kill")


@pytest.fixture
def mock_metrics_sink() -> MockMetricsSink:
    return MockMetricsSink()


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts"
    path.mkdir()
    return path
