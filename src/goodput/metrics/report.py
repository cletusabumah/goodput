"""Run report schema + builders (tickets 1.7 / 3.4).

MVP fields match README / ml-strategy: goodput, ckpt save/restore latency,
a wasted GPU-hours proxy, plus git SHA / config hash / package versions.
"""

from __future__ import annotations

from typing import Any

from goodput.config import Settings
from goodput.metrics.goodput import compute_goodput, compute_wasted_gpu_hours
from goodput.metrics.reproducibility import reproducibility_fields
from goodput.providers.base import MetricsSink

# Keys that must appear in every emitted report (tickets 1.7 / 3.4).
REQUIRED_REPORT_FIELDS: tuple[str, ...] = (
    "goodput",
    "ckpt_save_s",
    "ckpt_restore_s",
    "wasted_gpu_hours",
    "useful_seconds",
    "wall_seconds",
    "num_workers",
    "steps_completed",
    "seed",
    "run_name",
    "git_sha",
    "config_hash",
    "package_versions",
)

SCHEMA_VERSION = "1.1"


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_run_report(
    *,
    settings: Settings,
    wall_seconds: float,
    useful_seconds: float,
    steps_completed: int,
    ckpt_save_seconds: list[float] | float = 0.0,
    ckpt_restore_seconds: float = 0.0,
    final_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble a machine-readable metrics payload.

    ``ckpt_save_s`` is the mean save latency (0 if no checkpoints were timed).
    ``useful_seconds`` should exclude discarded post-failure work; callers decide.
    """
    if isinstance(ckpt_save_seconds, list):
        save_s = mean_or_zero(ckpt_save_seconds)
        save_count = len(ckpt_save_seconds)
    else:
        save_s = float(ckpt_save_seconds)
        save_count = 1 if save_s > 0 else 0

    wall = float(wall_seconds)
    useful = max(0.0, float(useful_seconds))
    # Clamp useful to wall so noisy clocks cannot push goodput above 1 via math.
    if wall > 0:
        useful = min(useful, wall)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_name": settings.run_name,
        "seed": settings.seed,
        "num_workers": settings.num_workers,
        "steps_completed": int(steps_completed),
        "wall_seconds": wall,
        "useful_seconds": useful,
        "goodput": compute_goodput(useful, wall) if wall > 0 else 0.0,
        "ckpt_save_s": save_s,
        "ckpt_save_count": save_count,
        "ckpt_restore_s": float(ckpt_restore_seconds),
        "wasted_gpu_hours": compute_wasted_gpu_hours(wall, useful, settings.num_workers),
        "ckpt_mode": settings.ckpt_mode,
        "device": settings.device,
    }
    if final_loss is not None:
        report["final_loss"] = float(final_loss)
    if extra:
        report.update(extra)
    # Ticket 3.4 — SHA / versions / config hash win over extra so reports stay complete.
    report.update(reproducibility_fields(settings))
    return report


def assert_required_fields(report: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_REPORT_FIELDS if k not in report]
    if missing:
        raise ValueError(f"report missing required fields: {missing}")


def emit_run_report(sink: MetricsSink, report: dict[str, Any]) -> dict[str, Any]:
    """Validate required fields then write via the configured MetricsSink."""
    assert_required_fields(report)
    sink.emit(report)
    return report
