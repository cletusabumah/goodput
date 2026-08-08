"""Checkpoint store implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodput.providers.base import CheckpointPayload, CheckpointStore


class MockCheckpointStore(CheckpointStore):
    """In-memory store for CI and unit tests (no disk I/O)."""

    name = "mock"

    def __init__(self) -> None:
        self._history: list[CheckpointPayload] = []
        self._locator = "memory://latest"

    def save(self, payload: CheckpointPayload) -> Path | str:
        self._history.append(
            CheckpointPayload(step=payload.step, state=dict(payload.state), meta=dict(payload.meta))
        )
        return self._locator

    def load(self, locator: Path | str | None = None) -> CheckpointPayload:
        if not self._history:
            raise FileNotFoundError("No checkpoint in mock store")
        latest = self._history[-1]
        return CheckpointPayload(step=latest.step, state=dict(latest.state), meta=dict(latest.meta))

    def latest(self) -> Path | str | None:
        return self._locator if self._history else None

    @property
    def save_count(self) -> int:
        return len(self._history)


class LocalFsCheckpointStore(CheckpointStore):
    """JSON checkpoint files under a directory (toy state only — Phase 1)."""

    name = "local_fs"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._latest_path = self.root / "latest.json"

    def _step_path(self, step: int) -> Path:
        return self.root / f"step_{step:06d}.json"

    def save(self, payload: CheckpointPayload) -> Path | str:
        path = self._step_path(payload.step)
        body: dict[str, Any] = {
            "step": payload.step,
            "state": payload.state,
            "meta": payload.meta,
        }
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        self._latest_path.write_text(
            json.dumps({"locator": str(path), "step": payload.step}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, locator: Path | str | None = None) -> CheckpointPayload:
        path = Path(locator) if locator is not None else None
        if path is None:
            latest = self.latest()
            if latest is None:
                raise FileNotFoundError(f"No checkpoint under {self.root}")
            path = Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return CheckpointPayload(
            step=int(data["step"]),
            state=dict(data.get("state", {})),
            meta=dict(data.get("meta", {})),
        )

    def latest(self) -> Path | str | None:
        if not self._latest_path.exists():
            return None
        pointer = json.loads(self._latest_path.read_text(encoding="utf-8"))
        locator = pointer.get("locator")
        return locator
