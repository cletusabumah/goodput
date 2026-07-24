"""Shared fixtures — mock providers, temp artifact dirs."""

from __future__ import annotations

import pytest

from goodput.providers import MockCheckpointStore, MockFaultInjector


@pytest.fixture
def mock_checkpoint_store() -> MockCheckpointStore:
    return MockCheckpointStore()


@pytest.fixture
def mock_fault_injector() -> MockFaultInjector:
    return MockFaultInjector(inject_at=5, fault="kill")
