"""Unit tests for provider ABCs, mocks, LocalFs, and factory (ticket 1.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.providers import (
    CheckpointPayload,
    JsonFileMetricsSink,
    LocalFsCheckpointStore,
    MockCheckpointStore,
    MockFaultInjector,
    MockMetricsSink,
    ProcessFaultInjector,
    build_providers,
)


def test_mock_checkpoint_roundtrip_and_history() -> None:
    store = MockCheckpointStore()
    assert store.latest() is None
    with pytest.raises(FileNotFoundError):
        store.load()

    store.save(CheckpointPayload(step=1, state={"w": 0.1}))
    store.save(CheckpointPayload(step=2, state={"w": 0.2}, meta={"lr": 1e-3}))
    assert store.save_count == 2
    assert store.latest() == "memory://latest"

    loaded = store.load()
    assert loaded.step == 2
    assert loaded.state["w"] == 0.2
    assert loaded.meta["lr"] == 1e-3


def test_local_fs_checkpoint_roundtrip(tmp_path: Path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    loc = store.save(CheckpointPayload(step=7, state={"bias": 1.5}, meta={"epoch": 0}))
    assert Path(loc).exists()
    assert store.latest() == str(loc)

    loaded = store.load()
    assert loaded.step == 7
    assert loaded.state["bias"] == 1.5
    assert loaded.meta["epoch"] == 0

    again = store.load(loc)
    assert again.step == 7


def test_local_fs_load_missing_raises(tmp_path: Path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        store.load()


def test_mock_fault_injector_records_without_side_effects() -> None:
    inj = MockFaultInjector(inject_at=3, fault="kill")
    assert inj.maybe_inject(1, worker_id=0) is None
    assert inj.maybe_inject(3, worker_id=1) == "kill"
    assert inj.injected == [(3, 1, "kill")]
    assert inj.maybe_inject(3, worker_id=0) == "kill"
    assert len(inj.injected) == 2


def test_process_fault_injector_dry_run_does_not_require_pid() -> None:
    inj = ProcessFaultInjector(inject_at=2, fault="kill", dry_run=True)
    assert inj.maybe_inject(2, worker_id=0) == "kill"
    assert inj.injected == [(2, 0, "kill")]


def test_process_fault_injector_live_kill_requires_pid() -> None:
    inj = ProcessFaultInjector(inject_at=1, fault="kill", dry_run=False)
    with pytest.raises(RuntimeError, match="No PID"):
        inj.maybe_inject(1, worker_id=0)


def test_json_file_metrics_sink(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    sink = JsonFileMetricsSink(path)
    sink.emit({"goodput": 0.9, "ckpt_save_s": 0.01})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["goodput"] == 0.9
    assert data["ckpt_save_s"] == 0.01


def test_mock_metrics_sink() -> None:
    sink = MockMetricsSink()
    sink.emit({"a": 1})
    sink.emit({"b": 2})
    assert sink.emitted == [{"a": 1}, {"b": 2}]


def test_build_providers_ci_mode_forces_mocks(tmp_path: Path) -> None:
    settings = Settings(
        ci_mode=True,
        checkpoint_provider="local_fs",
        fault_provider="process",
        metrics_provider="mock",
        artifacts_dir=tmp_path,
        run_name="ci",
        fault_mode="kill",
        fault_at="5",
    )
    providers = build_providers(settings, artifacts_dir=tmp_path)
    assert providers.checkpoint.name == "mock"
    assert providers.fault.name == "mock"
    assert providers.metrics.name == "mock"
    assert providers.fault.maybe_inject(5, 0) == "kill"


def test_build_providers_local_fs_and_json(tmp_path: Path) -> None:
    settings = Settings(
        ci_mode=False,
        checkpoint_provider="local_fs",
        fault_provider="mock",
        metrics_provider="json_file",
        ckpt_dir=tmp_path / "ckpts",
        artifacts_dir=tmp_path,
        run_name="unit",
        fault_mode="none",
    )
    providers = build_providers(settings, artifacts_dir=tmp_path)
    assert providers.checkpoint.name == "local_fs"
    assert providers.fault.name == "mock"
    assert providers.metrics.name == "json_file"

    providers.checkpoint.save(CheckpointPayload(step=1, state={"x": 1}))
    assert providers.checkpoint.load().step == 1
    assert providers.fault.maybe_inject(99, 0) is None

    providers.metrics.emit({"goodput": 1.0})
    report = tmp_path / "reports" / "unit" / "report.json"
    assert report.exists()
