"""Checkpoint/restore latency vs worker count (ticket 2.5).

Each cell trains ``steps`` with periodic checkpoints at a given ``num_workers``,
then loads ``latest.pt`` on the 1.5 resume path so restore is timed without
SIGKILL. Rank 0 dumps the same payload at every N, so save/restore should stay
roughly flat while train wall grows with spawn + barriers.

Writes JSON + CSV + a markdown table under ``artifacts/sweeps/<name>/``.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from goodput.config import Settings
from goodput.metrics import build_run_report
from goodput.providers import LocalFsCheckpointStore
from goodput.training import launch_workers, resume_after_crash, train_from_settings

LATENCY_FIELDS: tuple[str, ...] = (
    "num_workers",
    "ckpt_mode",
    "ckpt_save_s",
    "ckpt_restore_s",
    "ckpt_save_count",
    "goodput",
    "wasted_gpu_hours",
    "wall_seconds",
    "useful_seconds",
    "steps_completed",
    "run_name",
)


@dataclass(frozen=True)
class LatencySpec:
    """Parsed latency YAML: shared settings + worker-count axis."""

    path: Path
    name: str
    settings: Settings
    worker_counts: tuple[int, ...]
    output_dir: Path


@dataclass
class LatencyResult:
    spec: LatencySpec
    rows: list[dict[str, Any]] = field(default_factory=list)
    json_path: Path | None = None
    csv_path: Path | None = None
    table_path: Path | None = None


def load_latency_yaml(path: str | Path, *, base: Settings | None = None) -> LatencySpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"latency config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"latency YAML must be a mapping: {config_path}")

    data: dict[str, Any] = dict(raw)
    name = str(data.pop("name", config_path.stem))
    mode = data.pop("mode", "latency")
    if mode != "latency":
        raise ValueError(f"latency YAML mode must be 'latency', got {mode!r}")

    counts_raw = data.pop("worker_counts", None)
    if not isinstance(counts_raw, list) or not counts_raw:
        raise ValueError("worker_counts must be a non-empty list")
    worker_counts: list[int] = []
    seen: set[int] = set()
    for n in counts_raw:
        count = int(n)
        if count < 1:
            raise ValueError("worker_counts must be >= 1")
        if count not in seen:
            worker_counts.append(count)
            seen.add(count)

    output_dir_raw = data.pop("output_dir", None)
    unknown = sorted(k for k in data if k not in Settings.model_fields)
    if unknown:
        raise ValueError(f"unknown latency keys in {config_path}: {unknown}")

    base_settings = base if base is not None else Settings()
    settings = base_settings.model_copy(update=data)
    if settings.ckpt_interval < 1:
        raise ValueError("ckpt_interval must be >= 1 to measure save/restore latency")
    if settings.steps < settings.ckpt_interval:
        raise ValueError("steps must be >= ckpt_interval so at least one checkpoint is written")

    artifacts = Path(settings.artifacts_dir)
    output_dir = Path(output_dir_raw) if output_dir_raw else artifacts / "sweeps" / name

    return LatencySpec(
        path=config_path.resolve(),
        name=name,
        settings=settings,
        worker_counts=tuple(worker_counts),
        output_dir=output_dir,
    )


def _reset_ckpt_dir(ckpt_dir: Path) -> None:
    """Drop leftover checkpoints so a rerun is a fresh job, not a resume."""
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


def _format_cell(value: Any, field: str) -> str:
    if value is None:
        return ""
    if field in {
        "ckpt_save_s",
        "ckpt_restore_s",
        "wall_seconds",
        "useful_seconds",
        "wasted_gpu_hours",
    }:
        return f"{float(value):.6f}"
    if field == "goodput":
        return f"{float(value):.4f}"
    return str(value)


def render_latency_table(rows: list[dict[str, Any]]) -> str:
    """Markdown evaluation report: header + one row per worker count."""
    lines = [
        "# Checkpoint/restore latency vs worker count",
        "",
        "Each row trains with periodic checkpoints at `num_workers`, then loads",
        "`latest.pt` on the 1.5 resume path (no SIGKILL). Save time is the mean",
        "rank-0 dump; restore is load + `restore_training_state`. Naive dumps are",
        "not sharded, so save/restore should stay roughly flat while train wall",
        "grows with spawn and barriers.",
        "",
        "| " + " | ".join(LATENCY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in LATENCY_FIELDS) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(key), key) for key in LATENCY_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_latency_table(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: row.get(k) for k in LATENCY_FIELDS} for row in rows]
    json_path = output_dir / "latency.json"
    csv_path = output_dir / "latency.csv"
    table_path = output_dir / "table.md"
    json_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LATENCY_FIELDS))
        writer.writeheader()
        writer.writerows(slim)
    table_path.write_text(render_latency_table(slim), encoding="utf-8")
    return json_path, csv_path, table_path


def run_latency_cell(
    settings: Settings,
    *,
    num_workers: int,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """Train at ``num_workers``, then probe restore latency from the written ckpt."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    cell_name = f"{settings.run_name}-n{num_workers}"
    cell_settings = settings.model_copy(
        update={
            "num_workers": num_workers,
            "run_name": cell_name,
            "ckpt_dir": ckpt_dir,
        }
    )
    _reset_ckpt_dir(Path(ckpt_dir))
    store = LocalFsCheckpointStore(ckpt_dir)

    train_resumed_from: int | None = None
    if num_workers <= 1:
        result = train_from_settings(cell_settings, checkpoint_store=store)
        wall = result.wall_seconds
        useful = result.useful_seconds
        saves = list(result.ckpt_save_seconds)
        steps_completed = result.steps_completed
        last_ckpt = result.last_checkpoint_step
        final_loss = result.final_loss
        train_resumed_from = result.resumed_from_step
    else:
        wall_t0 = time.perf_counter()
        mp_result = launch_workers(
            world_size=num_workers,
            steps=cell_settings.steps,
            settings=cell_settings,
            ckpt_dir=ckpt_dir,
        )
        wall = max(time.perf_counter() - wall_t0, 1e-9)
        useful = wall
        saves = list(mp_result.ckpt_save_seconds)
        steps_completed = mp_result.steps
        last_ckpt = mp_result.last_checkpoint_step
        final_loss = mp_result.final_loss
        if not mp_result.ok:
            errors = [w.error for w in mp_result.workers if w.error]
            raise RuntimeError(f"multiprocess latency cell failed: {errors}")

    if store.latest() is None:
        raise RuntimeError(f"no checkpoint written for num_workers={num_workers}")

    resumed = resume_after_crash(
        settings=cell_settings,
        checkpoint_store=store,
        remaining_steps=1,
    )
    return build_run_report(
        settings=cell_settings,
        wall_seconds=wall,
        useful_seconds=useful,
        steps_completed=steps_completed,
        ckpt_save_seconds=saves,
        ckpt_restore_seconds=resumed.ckpt_restore_seconds,
        final_loss=final_loss,
        extra={
            "mode": "latency_cell",
            "last_checkpoint_step": last_ckpt,
            "train_resumed_from_step": train_resumed_from,
            "resumed_from_step": resumed.resumed_from_step,
        },
    )


def run_latency(spec: LatencySpec) -> LatencyResult:
    rows: list[dict[str, Any]] = []
    ckpt_root = Path(spec.settings.ckpt_dir)
    for n in spec.worker_counts:
        cell_dir = ckpt_root / spec.name / f"n{n}"
        report = run_latency_cell(spec.settings, num_workers=n, ckpt_dir=cell_dir)
        rows.append(report)

    json_path, csv_path, table_path = write_latency_table(rows, spec.output_dir)
    return LatencyResult(
        spec=spec,
        rows=rows,
        json_path=json_path,
        csv_path=csv_path,
        table_path=table_path,
    )
