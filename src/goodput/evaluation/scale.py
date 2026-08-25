"""Goodput vs worker count — MTBF-at-scale illustration (ticket 4.1).

Independent GPU failures make cluster interruption rate grow with N:

    cluster_failure_rate ≈ num_workers × per_gpu_failure_rate

Each cell maps that rate to one soft crash + resume (the 2.2 path) so CI never
SIGKILLs a worker matrix. Training stays single-process; ``num_workers`` is the
*simulated cluster size* that (1) sets the crash schedule and (2) scales wasted
GPU-hours. That matches the interview story (more GPUs → shorter MTBF) without
pretending we ran a 16k-GPU job.

Writes JSON + CSV + markdown write-up under ``artifacts/sweeps/<name>/``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from goodput.config import Settings
from goodput.evaluation.sweep import CkptMode, kill_at_for_rate, run_sweep_cell
from goodput.metrics import compute_wasted_gpu_hours

SCALE_ROW_FIELDS: tuple[str, ...] = (
    "num_workers",
    "ckpt_mode",
    "per_gpu_failure_rate",
    "cluster_failure_rate",
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

DEFAULT_PER_GPU_FAILURE_RATE = 0.125

WRITEUP = (
    "Illustration, not a cluster trace. Independent GPU failures make cluster "
    "MTBF shrink as ~1/N, so interruption rate grows with worker count. Each "
    "cell maps `cluster_failure_rate = N × per_gpu_failure_rate` onto the same "
    "soft-crash + resume path as the Phase 2 sweep (one durable kill, then "
    "resume). Training is single-process; N only sets the crash schedule and "
    "the wasted-GPU-hours multiplier. Toy millisecond walls are not Meta-scale "
    "MTBF — the shape of the curve is the claim, not the absolute goodput."
)


@dataclass(frozen=True)
class ScaleSpec:
    """Parsed scale YAML: worker-count axis + per-GPU failure rate."""

    path: Path
    name: str
    settings: Settings
    worker_counts: tuple[int, ...]
    ckpt_modes: tuple[CkptMode, ...]
    per_gpu_failure_rate: float
    output_dir: Path


@dataclass
class ScaleResult:
    spec: ScaleSpec
    rows: list[dict[str, Any]] = field(default_factory=list)
    json_path: Path | None = None
    csv_path: Path | None = None
    table_path: Path | None = None
    plot_path: Path | None = None


def cluster_failure_rate(num_workers: int, per_gpu_failure_rate: float) -> float:
    """Cluster crashes-per-step if each GPU fails independently at ``per_gpu_failure_rate``."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if per_gpu_failure_rate < 0:
        raise ValueError("per_gpu_failure_rate must be >= 0")
    return float(num_workers) * float(per_gpu_failure_rate)


def load_scale_yaml(path: str | Path, *, base: Settings | None = None) -> ScaleSpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"scale config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scale YAML must be a mapping: {config_path}")

    data: dict[str, Any] = dict(raw)
    name = str(data.pop("name", config_path.stem))
    mode = data.pop("mode", "scale")
    if mode != "scale":
        raise ValueError(f"scale YAML mode must be 'scale', got {mode!r}")

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

    modes_raw = data.pop("ckpt_modes", ["naive", "incremental"])
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ValueError("ckpt_modes must be a non-empty list")
    ckpt_modes: list[CkptMode] = []
    for m in modes_raw:
        if m not in ("naive", "incremental"):
            raise ValueError(f"unknown ckpt_mode in scale: {m!r}")
        ckpt_modes.append(m)

    per_gpu = float(data.pop("per_gpu_failure_rate", DEFAULT_PER_GPU_FAILURE_RATE))
    if per_gpu < 0:
        raise ValueError("per_gpu_failure_rate must be >= 0")

    output_dir_raw = data.pop("output_dir", None)
    unknown = sorted(k for k in data if k not in Settings.model_fields)
    if unknown:
        raise ValueError(f"unknown scale keys in {config_path}: {unknown}")

    base_settings = base if base is not None else Settings()
    settings = base_settings.model_copy(update=data)
    artifacts = Path(settings.artifacts_dir)
    output_dir = Path(output_dir_raw) if output_dir_raw else artifacts / "sweeps" / name

    return ScaleSpec(
        path=config_path.resolve(),
        name=name,
        settings=settings,
        worker_counts=tuple(worker_counts),
        ckpt_modes=tuple(ckpt_modes),
        per_gpu_failure_rate=per_gpu,
        output_dir=output_dir,
    )


def run_scale_cell(
    settings: Settings,
    *,
    num_workers: int,
    ckpt_mode: CkptMode,
    per_gpu_failure_rate: float,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """One (N, ckpt_mode) cell: cluster rate ∝ N, then the 2.2 crash/resume cell."""
    rate = cluster_failure_rate(num_workers, per_gpu_failure_rate)
    kill_at = kill_at_for_rate(
        steps=settings.steps,
        ckpt_interval=settings.ckpt_interval,
        failure_rate=rate,
    )
    cell_settings = settings.model_copy(
        update={
            "num_workers": num_workers,
            "ckpt_mode": ckpt_mode,
            "run_name": f"{settings.run_name}-{ckpt_mode}-n{num_workers}",
        }
    )
    report = run_sweep_cell(
        cell_settings,
        ckpt_mode=ckpt_mode,
        failure_rate=rate,
        ckpt_dir=ckpt_dir,
    )
    report["num_workers"] = num_workers
    report["per_gpu_failure_rate"] = per_gpu_failure_rate
    report["cluster_failure_rate"] = rate
    report["kill_at"] = kill_at
    report["wasted_gpu_hours"] = compute_wasted_gpu_hours(
        float(report["wall_seconds"]),
        float(report["useful_seconds"]),
        num_workers,
    )
    report["mode"] = "scale_cell"
    return report


def _format_cell(value: Any, field: str) -> str:
    if value is None:
        return ""
    if field in {"cluster_failure_rate", "per_gpu_failure_rate"}:
        return f"{float(value):g}"
    if field == "goodput":
        return f"{float(value):.4f}"
    if field in {"ckpt_save_s", "ckpt_restore_s", "wall_seconds", "useful_seconds"}:
        return f"{float(value):.6f}"
    if field == "wasted_gpu_hours":
        return f"{float(value):.6f}"
    return str(value)


def render_scale_table(rows: list[dict[str, Any]], spec: ScaleSpec) -> str:
    """Markdown write-up: MTBF model + measured goodput vs N."""
    lines = [
        "# Goodput vs worker count (MTBF-at-scale)",
        "",
        WRITEUP,
        "",
        f"- **Per-GPU failure rate:** {spec.per_gpu_failure_rate:g} crashes/step "
        "(toy, not hardware MTBF)",
        f"- **Cluster rate:** `N × {spec.per_gpu_failure_rate:g}`",
        f"- **Worker counts:** {', '.join(str(n) for n in spec.worker_counts)}",
        f"- **Checkpoint modes:** {', '.join(spec.ckpt_modes)}",
        "",
        "Lower goodput at larger N means interruptions land earlier in the job "
        "(shorter implied MTBF). Incremental vs naive is the same trade-off as "
        "the Phase 2 sweep, now sliced by cluster size instead of a fixed rate.",
        "",
        "| " + " | ".join(SCALE_ROW_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SCALE_ROW_FIELDS) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(key), key) for key in SCALE_ROW_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_scale_table(rows: list[dict[str, Any]], spec: ScaleSpec) -> tuple[Path, Path, Path]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: row.get(k) for k in SCALE_ROW_FIELDS} for row in rows]
    payload = {
        "mode": "scale",
        "name": spec.name,
        "writeup": WRITEUP,
        "per_gpu_failure_rate": spec.per_gpu_failure_rate,
        "worker_counts": list(spec.worker_counts),
        "ckpt_modes": list(spec.ckpt_modes),
        "rows": slim,
    }
    json_path = output_dir / "scale.json"
    csv_path = output_dir / "scale.csv"
    table_path = output_dir / "table.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCALE_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(slim)
    table_path.write_text(render_scale_table(slim, spec), encoding="utf-8")
    return json_path, csv_path, table_path


def run_scale(spec: ScaleSpec) -> ScaleResult:
    rows: list[dict[str, Any]] = []
    ckpt_root = Path(spec.settings.ckpt_dir)
    for mode in spec.ckpt_modes:
        for n in spec.worker_counts:
            cell_dir = ckpt_root / spec.name / f"{mode}-n{n}"
            rows.append(
                run_scale_cell(
                    spec.settings,
                    num_workers=n,
                    ckpt_mode=mode,
                    per_gpu_failure_rate=spec.per_gpu_failure_rate,
                    ckpt_dir=cell_dir,
                )
            )

    json_path, csv_path, table_path = write_scale_table(rows, spec)
    plot_path: Path | None = None
    try:
        from goodput.evaluation.plot import plot_goodput_vs_workers

        plot_path = plot_goodput_vs_workers(
            slim_rows(rows), spec.output_dir / "goodput_vs_workers.png"
        )
    except ImportError:
        plot_path = None

    return ScaleResult(
        spec=spec,
        rows=rows,
        json_path=json_path,
        csv_path=csv_path,
        table_path=table_path,
        plot_path=plot_path,
    )


def slim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: row.get(k) for k in SCALE_ROW_FIELDS} for row in rows]
