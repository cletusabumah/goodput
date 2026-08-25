"""Naive / incremental vs torch.distributed.checkpoint latency note (ticket 4.4).

Side-by-side **note**, not a swap of the training loop onto DCP. Our stores pickle
one unsharded blob; DCP is built for sharded multi-rank state. On a 1-process
toy MLP the timings can go either way — the point is the API and fidelity gap.

Writes JSON + markdown under ``artifacts/sweeps/<name>/``.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from goodput.checkpointing import IncrementalCheckpointer, capture_training_state
from goodput.checkpointing.incremental import restore_training_state as restore_incremental
from goodput.checkpointing.naive import restore_training_state as restore_naive
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore

DCP_ROW_FIELDS: tuple[str, ...] = (
    "path",
    "available",
    "save_s",
    "restore_s",
    "bytes",
    "notes",
)

WRITEUP = """\
# Naive vs incremental vs torch.distributed.checkpoint

Ticket **4.4** is a latency **note**, not a production DCP integration.

| Path | What it writes | Rank model |
|------|----------------|------------|
| **Naive** (`torch.save`) | Full model+optimizer `.pt` | Single process |
| **Incremental** | Full base every N; model-only between | Same store |
| **DCP** | Directory of shards + metadata | FSDP / multi-rank |

**When DCP wins:** many ranks, sharded params, coordinated filesystem. **When a naive
dump is enough (and often faster here):** one rank, tiny unsharded state, CI CPU.

Times below are median over a few CPU repeats on `ToyMLP` + Adam. Do not treat
them as a cluster-scale ranking. Regenerable via:

```bash
goodput-run --dcp-compare experiments/dcp-compare.yaml
```
"""


@dataclass(frozen=True)
class DcpCompareSpec:
    path: Path
    name: str
    repeats: int
    input_size: int
    hidden_size: int
    output_dir: Path
    work_dir: Path


@dataclass
class DcpCompareResult:
    spec: DcpCompareSpec
    rows: list[dict[str, Any]] = field(default_factory=list)
    json_path: Path | None = None
    table_path: Path | None = None


def dcp_available() -> bool:
    try:
        import torch.distributed.checkpoint as dcp  # noqa: F401
    except ImportError:
        return False
    return True


def load_dcp_compare_yaml(path: str | Path) -> DcpCompareSpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"dcp-compare config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"dcp-compare YAML must be a mapping: {config_path}")
    data = dict(raw)
    name = str(data.pop("name", config_path.stem))
    mode = data.pop("mode", "dcp_compare")
    if mode != "dcp_compare":
        raise ValueError(f"dcp-compare YAML mode must be 'dcp_compare', got {mode!r}")
    repeats = int(data.pop("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    input_size = int(data.pop("input_size", 64))
    hidden_size = int(data.pop("hidden_size", 256))
    artifacts = Path(str(data.pop("artifacts_dir", "artifacts")))
    output_dir = Path(data.pop("output_dir", artifacts / "sweeps" / name))
    work_dir = Path(data.pop("work_dir", artifacts / "checkpoints" / name))
    unknown = sorted(data)
    if unknown:
        raise ValueError(f"unknown dcp-compare keys in {config_path}: {unknown}")
    return DcpCompareSpec(
        path=config_path.resolve(),
        name=name,
        repeats=repeats,
        input_size=input_size,
        hidden_size=hidden_size,
        output_dir=output_dir,
        work_dir=work_dir,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return total


def _fresh_model(spec: DcpCompareSpec) -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = ToyMLP(input_size=spec.input_size, hidden_size=spec.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dummy = torch.randn(4, spec.input_size)
    loss = model(dummy).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def _time_naive(spec: DcpCompareSpec, work: Path) -> dict[str, Any]:
    saves: list[float] = []
    restores: list[float] = []
    nbytes = 0
    for i in range(spec.repeats):
        cell = work / "naive" / f"r{i}"
        if cell.exists():
            shutil.rmtree(cell)
        store = LocalFsCheckpointStore(cell)
        model, opt = _fresh_model(spec)
        payload = capture_training_state(model, opt, step=1)
        t0 = time.perf_counter()
        store.save(payload)
        saves.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        loaded = store.load()
        restore_naive(model, opt, loaded, device="cpu")
        restores.append(time.perf_counter() - t1)
        latest = store.latest()
        nbytes = _dir_bytes(Path(str(latest))) if latest else 0
    return {
        "path": "naive_torch_save",
        "available": True,
        "save_s": _median(saves),
        "restore_s": _median(restores),
        "bytes": nbytes,
        "notes": "LocalFsCheckpointStore: one .pt pickle of model+optimizer",
    }


def _time_incremental(spec: DcpCompareSpec, work: Path) -> dict[str, Any]:
    saves: list[float] = []
    restores: list[float] = []
    nbytes = 0
    for i in range(spec.repeats):
        cell = work / "incremental" / f"r{i}"
        if cell.exists():
            shutil.rmtree(cell)
        store = LocalFsCheckpointStore(cell)
        model, opt = _fresh_model(spec)
        ckpt = IncrementalCheckpointer(full_every=4)
        store.save(ckpt.capture(model, opt, step=1))
        inc = ckpt.capture(model, opt, step=2)
        t0 = time.perf_counter()
        store.save(inc)
        saves.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        loaded = store.load()
        restore_incremental(model, opt, loaded, device="cpu", store=store)
        restores.append(time.perf_counter() - t1)
        latest = store.latest()
        nbytes = _dir_bytes(Path(str(latest))) if latest else 0
    return {
        "path": "incremental_model_only",
        "available": True,
        "save_s": _median(saves),
        "restore_s": _median(restores),
        "bytes": nbytes,
        "notes": "Second save is model-only; restore reattaches last full optimizer base",
    }


def _time_dcp(spec: DcpCompareSpec, work: Path) -> dict[str, Any]:
    if not dcp_available():
        return {
            "path": "torch_distributed_checkpoint",
            "available": False,
            "save_s": None,
            "restore_s": None,
            "bytes": None,
            "notes": "torch.distributed.checkpoint not importable in this torch build",
        }
    import torch.distributed.checkpoint as dcp

    saves: list[float] = []
    restores: list[float] = []
    nbytes = 0
    for i in range(spec.repeats):
        cell = work / "dcp" / f"r{i}"
        if cell.exists():
            shutil.rmtree(cell)
        cell.mkdir(parents=True, exist_ok=True)
        model, opt = _fresh_model(spec)
        state = {"model": model.state_dict(), "optim": opt.state_dict()}
        try:
            t0 = time.perf_counter()
            dcp.save(state, checkpoint_id=str(cell))
            saves.append(time.perf_counter() - t0)
            load_state = {"model": model.state_dict(), "optim": opt.state_dict()}
            t1 = time.perf_counter()
            dcp.load(load_state, checkpoint_id=str(cell))
            model.load_state_dict(load_state["model"])
            restores.append(time.perf_counter() - t1)
            nbytes = _dir_bytes(cell)
        except Exception as exc:  # noqa: BLE001 — skip, don't fail the note
            return {
                "path": "torch_distributed_checkpoint",
                "available": False,
                "save_s": None,
                "restore_s": None,
                "bytes": None,
                "notes": (
                    f"DCP save/load skipped ({type(exc).__name__}: {exc}). "
                    "Some torch builds need a process group; our loop stays on torch.save."
                ),
            }
    return {
        "path": "torch_distributed_checkpoint",
        "available": True,
        "save_s": _median(saves),
        "restore_s": _median(restores),
        "bytes": nbytes,
        "notes": "FileSystem checkpoint_id directory (unsharded 1-rank dict)",
    }


def render_dcp_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        WRITEUP.rstrip(),
        "",
        "| " + " | ".join(DCP_ROW_FIELDS) + " |",
        "| " + " | ".join("---" for _ in DCP_ROW_FIELDS) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for key in DCP_ROW_FIELDS:
            val = row.get(key)
            if val is None:
                cells.append("—")
            elif key in {"save_s", "restore_s"}:
                cells.append(f"{float(val):.6f}")
            elif key == "bytes":
                cells.append(str(int(val)))
            elif key == "available":
                cells.append("yes" if val else "no")
            else:
                cells.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def run_dcp_compare(spec: DcpCompareSpec) -> DcpCompareResult:
    work = spec.work_dir
    work.mkdir(parents=True, exist_ok=True)
    rows = [
        _time_naive(spec, work),
        _time_incremental(spec, work),
        _time_dcp(spec, work),
    ]
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec.output_dir / "dcp_compare.json"
    table_path = spec.output_dir / "table.md"
    payload = {
        "mode": "dcp_compare",
        "name": spec.name,
        "torch": torch.__version__,
        "dcp_importable": dcp_available(),
        "repeats": spec.repeats,
        "input_size": spec.input_size,
        "hidden_size": spec.hidden_size,
        "rows": rows,
        "writeup": WRITEUP,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table_path.write_text(render_dcp_table(rows), encoding="utf-8")
    return DcpCompareResult(
        spec=spec, rows=rows, json_path=json_path, table_path=table_path
    )
