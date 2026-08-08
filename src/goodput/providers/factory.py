"""Build providers from settings (CI defaults to mocks)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from goodput.config import Settings
from goodput.providers.base import CheckpointStore, FaultInjector, MetricsSink
from goodput.providers.checkpoint import LocalFsCheckpointStore, MockCheckpointStore
from goodput.providers.faults import FaultType, MockFaultInjector, ProcessFaultInjector
from goodput.providers.metrics import JsonFileMetricsSink, MockMetricsSink, StdoutMetricsSink


@dataclass(frozen=True)
class Providers:
    checkpoint: CheckpointStore
    fault: FaultInjector
    metrics: MetricsSink


def _parse_fault_at(fault_at: str) -> int | None:
    if fault_at in {"", "none", "random"}:
        return None
    return int(fault_at)


def _fault_type(settings: Settings) -> FaultType:
    if settings.fault_mode == "none":
        return "kill"
    return cast(FaultType, settings.fault_mode)


def build_providers(settings: Settings, *, artifacts_dir: Path | None = None) -> Providers:
    """
    Construct the provider trio from settings.

    When ``settings.ci_mode`` is true, checkpoint and fault providers force to mocks
    and process kills stay dry-run — CI never SIGKILLs or depends on GPU/disk layout.
    """
    root = artifacts_dir if artifacts_dir is not None else settings.artifacts_dir
    inject_at = None if settings.fault_mode == "none" else _parse_fault_at(settings.fault_at)
    fault_name = _fault_type(settings)

    # Checkpoint
    ckpt_kind = "mock" if settings.ci_mode else settings.checkpoint_provider
    if ckpt_kind == "mock":
        checkpoint: CheckpointStore = MockCheckpointStore()
    elif ckpt_kind == "local_fs":
        checkpoint = LocalFsCheckpointStore(Path(settings.ckpt_dir))
    else:
        raise ValueError(f"Unknown checkpoint_provider: {ckpt_kind}")

    # Fault
    fault_kind = "mock" if settings.ci_mode else settings.fault_provider
    if fault_kind == "mock":
        fault: FaultInjector = MockFaultInjector(inject_at=inject_at, fault=fault_name)
    elif fault_kind == "process":
        fault = ProcessFaultInjector(
            inject_at=inject_at,
            fault=fault_name,
            dry_run=bool(settings.ci_mode),
        )
    else:
        raise ValueError(f"Unknown fault_provider: {fault_kind}")

    # Metrics
    if settings.metrics_provider == "stdout":
        metrics: MetricsSink = StdoutMetricsSink()
    elif settings.metrics_provider == "json_file":
        report_path = Path(root) / "reports" / settings.run_name / "report.json"
        metrics = JsonFileMetricsSink(report_path)
    elif settings.metrics_provider == "mock":
        metrics = MockMetricsSink()
    else:
        raise ValueError(f"Unknown metrics_provider: {settings.metrics_provider}")

    return Providers(checkpoint=checkpoint, fault=fault, metrics=metrics)
