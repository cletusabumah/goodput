"""Checkpoint store implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from goodput.providers.base import CheckpointPayload, CheckpointStore


class MockCheckpointStore(CheckpointStore):
    """In-memory store for CI and unit tests (no disk I/O)."""

    name = "mock"

    def __init__(self) -> None:
        self._history: list[CheckpointPayload] = []
        self._locator = "memory://latest"

    def save(self, payload: CheckpointPayload) -> Path | str:
        # Deep-enough copy so later mutations don't rewrite history.
        state = _clone_state(payload.state)
        self._history.append(
            CheckpointPayload(step=payload.step, state=state, meta=dict(payload.meta))
        )
        return self._locator

    def load(self, locator: Path | str | None = None) -> CheckpointPayload:
        if not self._history:
            raise FileNotFoundError("No checkpoint in mock store")
        latest = self._history[-1]
        return CheckpointPayload(
            step=latest.step,
            state=_clone_state(latest.state),
            meta=dict(latest.meta),
        )

    def latest(self) -> Path | str | None:
        return self._locator if self._history else None

    def load_at_step(self, step: int) -> CheckpointPayload:
        for payload in reversed(self._history):
            if payload.step == step:
                return CheckpointPayload(
                    step=payload.step,
                    state=_clone_state(payload.state),
                    meta=dict(payload.meta),
                )
        raise FileNotFoundError(f"No mock checkpoint at step {step}")

    @property
    def save_count(self) -> int:
        return len(self._history)


class LocalFsCheckpointStore(CheckpointStore):
    """
    Full checkpoint files under a directory via ``torch.save`` (.pt).

    Replaces the earlier JSON store so model/optimizer tensors round-trip (ticket 1.5).
    """

    name = "local_fs"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._latest_path = self.root / "latest.pt"

    def _step_path(self, step: int) -> Path:
        return self.root / f"step_{step:06d}.pt"

    def save(self, payload: CheckpointPayload) -> Path | str:
        path = self._step_path(payload.step)
        body: dict[str, Any] = {
            "step": payload.step,
            "state": payload.state,
            "meta": payload.meta,
        }
        torch.save(body, path)
        torch.save({"locator": str(path), "step": payload.step}, self._latest_path)
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
        data = torch.load(path, map_location="cpu", weights_only=False)
        return CheckpointPayload(
            step=int(data["step"]),
            state=dict(data.get("state", {})),
            meta=dict(data.get("meta", {})),
        )

    def latest(self) -> Path | str | None:
        if not self._latest_path.exists():
            return None
        pointer = torch.load(self._latest_path, map_location="cpu", weights_only=False)
        return pointer.get("locator")

    def load_at_step(self, step: int) -> CheckpointPayload:
        path = self._step_path(step)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found for step {step}: {path}")
        return self.load(path)


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    """Clone nested state; clone tensors so history is immutable."""
    out: dict[str, Any] = {}
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu().clone()
        elif isinstance(val, dict):
            out[key] = _clone_state(val)
        else:
            out[key] = val
    return out
