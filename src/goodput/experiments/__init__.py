"""Load committed experiment YAML into Settings (ticket 1.8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from goodput.config import Settings, get_settings

RunMode = Literal["train", "fault_kill"]


@dataclass(frozen=True)
class ExperimentSpec:
    """Parsed experiment file: settings + how the CLI should run it."""

    path: Path
    settings: Settings
    mode: RunMode
    fault_at: int | None
    fault_rank: int
    name: str


def load_experiment_yaml(
    path: str | Path,
    *,
    base: Settings | None = None,
) -> ExperimentSpec:
    """
    Read ``experiments/*.yaml`` and merge into a Settings copy.

    ``name`` maps to ``run_name`` (report path). ``mode`` selects train vs
    fault-kill. Unknown keys raise so typos fail loudly.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment YAML must be a mapping: {config_path}")

    data: dict[str, Any] = dict(raw)
    name = str(data.pop("name", config_path.stem))
    mode_raw = data.pop("mode", "train")
    if mode_raw not in ("train", "fault_kill"):
        raise ValueError(f"mode must be 'train' or 'fault_kill', got {mode_raw!r}")
    mode: RunMode = mode_raw  # type: ignore[assignment]
    fault_rank = int(data.pop("fault_rank", 1))

    # Optional kill step: YAML may use int; Settings.fault_at stays a string.
    fault_at: int | None = None
    if "fault_at" in data:
        fa = data["fault_at"]
        if fa is not None and fa != "random":
            fault_at = int(fa)
            data["fault_at"] = str(fa)

    unknown = sorted(k for k in data if k not in Settings.model_fields)
    if unknown:
        raise ValueError(f"unknown experiment keys in {config_path}: {unknown}")

    updates: dict[str, Any] = dict(data)
    updates["run_name"] = name

    base_settings = base if base is not None else get_settings()
    settings = base_settings.model_copy(update=updates)

    if mode == "fault_kill" and fault_at is None:
        raise ValueError(f"{config_path}: mode=fault_kill requires fault_at")

    return ExperimentSpec(
        path=config_path.resolve(),
        settings=settings,
        mode=mode,
        fault_at=fault_at,
        fault_rank=fault_rank,
        name=name,
    )
