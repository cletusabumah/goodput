"""Failure-rate × checkpoint-mode sweep (ticket 2.2).

Each cell is one training job under a (ckpt_mode, failure_rate) pair.
``failure_rate=0`` is uninterrupted. ``failure_rate>0`` injects one soft crash
at a durable checkpoint (mean gap ≈ 1/rate steps, snapped to ckpt_interval)
then resumes — the 1.5 path, so CI never SIGKILLs a matrix of workers.

Writes comparison JSON + CSV for the Phase 2 goodput-vs-rate plot.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from goodput.config import Settings
from goodput.data import SyntheticDataLoader
from goodput.metrics import build_run_report
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore
from goodput.training import TrainResult, resume_after_crash, train_from_settings, train_steps
from goodput.training.loop import _resolve_device

CkptMode = Literal["naive", "incremental"]

COMPARISON_FIELDS: tuple[str, ...] = (
    "ckpt_mode",
    "failure_rate",
    "kill_at",
    "goodput",
    "ckpt_save_s",
    "ckpt_restore_s",
    "wasted_gpu_hours",
    "wall_seconds",
    "useful_seconds",
    "steps_completed",
    "run_name",
)


@dataclass(frozen=True)
class SweepSpec:
    """Parsed sweep YAML: shared settings + matrix axes."""

    path: Path
    name: str
    settings: Settings
    ckpt_modes: tuple[CkptMode, ...]
    failure_rates: tuple[float, ...]
    output_dir: Path


@dataclass
class SweepResult:
    spec: SweepSpec
    rows: list[dict[str, Any]] = field(default_factory=list)
    json_path: Path | None = None
    csv_path: Path | None = None


def kill_at_for_rate(*, steps: int, ckpt_interval: int, failure_rate: float) -> int | None:
    """
    Map a failure rate to a single durable crash step, or None for no crash.

    ``failure_rate`` is crashes-per-step (so mean gap is ``1/rate``). The kill
    lands on a positive multiple of ``ckpt_interval`` strictly before ``steps``
    so resume has remaining work.
    """
    if failure_rate <= 0:
        return None
    if ckpt_interval < 1:
        raise ValueError("ckpt_interval must be >= 1 when failure_rate > 0")
    if steps <= ckpt_interval:
        raise ValueError("steps must exceed ckpt_interval to inject a crash and resume")

    mean_gap = max(ckpt_interval, int(round(1.0 / failure_rate)))
    kill_at = (mean_gap // ckpt_interval) * ckpt_interval
    last_safe = (steps // ckpt_interval) * ckpt_interval
    if last_safe >= steps:
        last_safe -= ckpt_interval
    kill_at = min(max(ckpt_interval, kill_at), last_safe)
    if kill_at < ckpt_interval or kill_at >= steps:
        return None
    return kill_at


def load_sweep_yaml(path: str | Path, *, base: Settings | None = None) -> SweepSpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"sweep config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"sweep YAML must be a mapping: {config_path}")

    data: dict[str, Any] = dict(raw)
    name = str(data.pop("name", config_path.stem))
    mode = data.pop("mode", "sweep")
    if mode != "sweep":
        raise ValueError(f"sweep YAML mode must be 'sweep', got {mode!r}")

    modes_raw = data.pop("ckpt_modes", ["naive", "incremental"])
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ValueError("ckpt_modes must be a non-empty list")
    ckpt_modes: list[CkptMode] = []
    for m in modes_raw:
        if m not in ("naive", "incremental"):
            raise ValueError(f"unknown ckpt_mode in sweep: {m!r}")
        ckpt_modes.append(m)

    rates_raw = data.pop("failure_rates", [0.0])
    if not isinstance(rates_raw, list) or not rates_raw:
        raise ValueError("failure_rates must be a non-empty list")
    failure_rates = tuple(float(r) for r in rates_raw)
    if any(r < 0 for r in failure_rates):
        raise ValueError("failure_rates must be >= 0")

    output_dir_raw = data.pop("output_dir", None)
    unknown = sorted(k for k in data if k not in Settings.model_fields)
    if unknown:
        raise ValueError(f"unknown sweep keys in {config_path}: {unknown}")

    base_settings = base if base is not None else Settings()
    settings = base_settings.model_copy(update=data)
    artifacts = Path(settings.artifacts_dir)
    output_dir = Path(output_dir_raw) if output_dir_raw else artifacts / "sweeps" / name

    return SweepSpec(
        path=config_path.resolve(),
        name=name,
        settings=settings,
        ckpt_modes=tuple(ckpt_modes),
        failure_rates=failure_rates,
        output_dir=output_dir,
    )


def _train_prefix(settings: Settings, store: LocalFsCheckpointStore, steps: int) -> TrainResult:
    """Run the first ``steps`` updates with the uninterrupted loader pool size."""
    device = _resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    model = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    loader = SyntheticDataLoader(
        num_batches=max(1, min(settings.steps, 16)),
        batch_size=settings.batch_size,
        input_size=settings.input_size,
        seed=settings.seed,
        device=device,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    return train_steps(
        model=model,
        batches=loader,
        optimizer=optimizer,
        steps=steps,
        device=device,
        start_step=0,
        checkpoint_store=store,
        ckpt_interval=settings.ckpt_interval,
        ckpt_mode=settings.ckpt_mode,
        ckpt_full_every=settings.ckpt_full_every,
    )


def _reset_ckpt_dir(ckpt_dir: Path) -> None:
    """Drop leftover checkpoints so a rerun is a fresh job, not a resume."""
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


def run_sweep_cell(
    settings: Settings,
    *,
    ckpt_mode: CkptMode,
    failure_rate: float,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """Train (and optionally soft-crash + resume) one matrix cell."""
    cell_name = f"{settings.run_name}-{ckpt_mode}-r{failure_rate:g}"
    cell_settings = settings.model_copy(
        update={
            "ckpt_mode": ckpt_mode,
            "run_name": cell_name,
            "ckpt_dir": ckpt_dir,
            "num_workers": 1,
        }
    )
    _reset_ckpt_dir(Path(ckpt_dir))
    store = LocalFsCheckpointStore(ckpt_dir)
    kill_at = kill_at_for_rate(
        steps=cell_settings.steps,
        ckpt_interval=cell_settings.ckpt_interval,
        failure_rate=failure_rate,
    )

    if kill_at is None:
        result = train_from_settings(cell_settings, checkpoint_store=store)
        report = build_run_report(
            settings=cell_settings,
            wall_seconds=result.wall_seconds,
            useful_seconds=result.useful_seconds,
            steps_completed=result.steps_completed,
            ckpt_save_seconds=result.ckpt_save_seconds,
            ckpt_restore_seconds=result.ckpt_restore_seconds,
            final_loss=result.final_loss,
            extra={
                "mode": "sweep_train",
                "failure_rate": failure_rate,
                "kill_at": None,
                "ckpt_mode": ckpt_mode,
            },
        )
        return report

    first = _train_prefix(cell_settings, store, kill_at)
    remaining = cell_settings.steps - kill_at
    resumed = resume_after_crash(
        settings=cell_settings,
        checkpoint_store=store,
        remaining_steps=remaining,
    )
    wall = first.wall_seconds + resumed.wall_seconds
    useful = first.useful_seconds + resumed.useful_seconds
    saves = list(first.ckpt_save_seconds) + list(resumed.ckpt_save_seconds)
    report = build_run_report(
        settings=cell_settings,
        wall_seconds=wall,
        useful_seconds=useful,
        steps_completed=first.steps_completed + resumed.steps_completed,
        ckpt_save_seconds=saves,
        ckpt_restore_seconds=resumed.ckpt_restore_seconds,
        final_loss=resumed.final_loss,
        extra={
            "mode": "sweep_crash",
            "failure_rate": failure_rate,
            "kill_at": kill_at,
            "ckpt_mode": ckpt_mode,
            "resumed_from_step": resumed.resumed_from_step,
        },
    )
    return report


def write_comparison(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: row.get(k) for k in COMPARISON_FIELDS} for row in rows]
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    json_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COMPARISON_FIELDS))
        writer.writeheader()
        writer.writerows(slim)
    return json_path, csv_path


def run_sweep(spec: SweepSpec) -> SweepResult:
    rows: list[dict[str, Any]] = []
    ckpt_root = Path(spec.settings.ckpt_dir)
    for mode in spec.ckpt_modes:
        for rate in spec.failure_rates:
            cell_dir = ckpt_root / spec.name / f"{mode}-r{rate:g}"
            report = run_sweep_cell(
                spec.settings,
                ckpt_mode=mode,
                failure_rate=rate,
                ckpt_dir=cell_dir,
            )
            rows.append(report)

    json_path, csv_path = write_comparison(rows, spec.output_dir)
    return SweepResult(spec=spec, rows=rows, json_path=json_path, csv_path=csv_path)
