"""Experiment trackers (ticket 4.3).

Default is ``none`` (no network, no extra deps). ``mlflow`` logs a run to a
local file store when the optional extra is installed; otherwise it writes the
same payload under ``artifacts/mlflow/`` so ``GOODPUT_TRACKER=mlflow`` still
records a run in CI. WandB is the same shape: real SDK if installed, else JSON.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_METRIC_KEYS = (
    "goodput",
    "ckpt_save_s",
    "ckpt_restore_s",
    "wasted_gpu_hours",
    "wall_seconds",
    "useful_seconds",
    "steps_completed",
    "final_loss",
)

_PARAM_KEYS = (
    "run_name",
    "seed",
    "ckpt_mode",
    "device",
    "num_workers",
    "schema_version",
    "git_sha",
    "config_hash",
)


class Tracker(ABC):
    """Log a finished run report to an experiment tracker (or no-op)."""

    name: str

    @abstractmethod
    def log_run(self, report: dict[str, Any]) -> str | None:
        """Persist params/metrics. Return a run id, or None if nothing was logged."""


class NullTracker(Tracker):
    name = "none"

    def log_run(self, report: dict[str, Any]) -> str | None:
        return None


def _params(report: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _PARAM_KEYS:
        if key in report and report[key] is not None:
            out[key] = str(report[key])
    return out


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _METRIC_KEYS:
        if key not in report or report[key] is None:
            continue
        try:
            out[key] = float(report[key])
        except (TypeError, ValueError):
            continue
    return out


def _import_mlflow() -> Any:
    import mlflow

    return mlflow


def _import_wandb() -> Any:
    import wandb

    return wandb


def _write_json_run(directory: Path, report: dict[str, Any], *, backend: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = str(report.get("run_name") or "run")
    run_id = f"{run_name}-{stamp}"
    payload = {
        "backend": backend,
        "run_id": run_id,
        "params": _params(report),
        "metrics": _metrics(report),
    }
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return run_id


class MLflowTracker(Tracker):
    """MLflow Tracking API, or a local JSON file if ``mlflow`` is not installed."""

    name = "mlflow"

    def __init__(self, *, tracking_uri: str, artifacts_dir: Path) -> None:
        self.tracking_uri = tracking_uri
        self.artifacts_dir = Path(artifacts_dir)

    def log_run(self, report: dict[str, Any]) -> str | None:
        try:
            mlflow = _import_mlflow()
        except ImportError:
            return _write_json_run(
                self.artifacts_dir / "mlflow",
                report,
                backend="mlflow_json",
            )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("goodput")
        with mlflow.start_run(run_name=str(report.get("run_name") or "goodput")) as run:
            mlflow.log_params(_params(report))
            metrics = _metrics(report)
            if metrics:
                mlflow.log_metrics(metrics)
            return str(run.info.run_id)


class WandbTracker(Tracker):
    """Weights & Biases, or a local JSON file if ``wandb`` is not installed."""

    name = "wandb"

    def __init__(self, *, project: str, artifacts_dir: Path) -> None:
        self.project = project
        self.artifacts_dir = Path(artifacts_dir)

    def log_run(self, report: dict[str, Any]) -> str | None:
        try:
            wandb = _import_wandb()
        except ImportError:
            return _write_json_run(
                self.artifacts_dir / "wandb",
                report,
                backend="wandb_json",
            )
        run = wandb.init(
            project=self.project,
            name=str(report.get("run_name") or "goodput"),
            config=_params(report),
            reinit=True,
        )
        metrics = _metrics(report)
        if metrics:
            wandb.log(metrics)
        run_id = str(run.id)
        wandb.finish()
        return run_id
