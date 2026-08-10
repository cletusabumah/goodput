"""Naive full-state checkpoint pack/unpack (ticket 1.5).

"Naive" means we dump the entire model + optimizer every time — simple and correct,
not fast. Incremental/faster paths come later (Phase 2).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from goodput.providers.base import CheckpointPayload

CHECKPOINT_FORMAT = "naive_v1"


def capture_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    meta: dict[str, Any] | None = None,
) -> CheckpointPayload:
    """Snapshot model + optimizer + step into a CheckpointPayload."""
    if step < 0:
        raise ValueError("step must be >= 0")
    payload_meta = {"format": CHECKPOINT_FORMAT}
    if meta:
        payload_meta.update(meta)
    return CheckpointPayload(
        step=step,
        state={
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": _cpu_clone_optimizer_state(optimizer.state_dict()),
        },
        meta=payload_meta,
    )


def restore_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: CheckpointPayload,
    *,
    device: torch.device | str = "cpu",
) -> int:
    """
    Load model + optimizer from ``payload``.

    Returns the checkpointed step (caller resumes at ``step + 1``).
    """
    if payload.meta.get("format") not in {None, CHECKPOINT_FORMAT}:
        raise ValueError(f"unsupported checkpoint format: {payload.meta.get('format')}")
    if "model" not in payload.state or "optimizer" not in payload.state:
        raise ValueError("checkpoint state must include 'model' and 'optimizer'")

    device_t = torch.device(device)
    model.load_state_dict(payload.state["model"])
    model.to(device_t)
    optimizer.load_state_dict(payload.state["optimizer"])
    # Move optimizer state tensors to the target device
    for state in optimizer.state.values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                state[key] = val.to(device_t)
    return int(payload.step)


def _cpu_clone_optimizer_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy optimizer state_dict with tensors on CPU (picklable / torch.save-safe)."""
    out: dict[str, Any] = {"state": {}, "param_groups": state_dict["param_groups"]}
    for pid, state in state_dict.get("state", {}).items():
        cloned: dict[str, Any] = {}
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                cloned[key] = val.detach().cpu().clone()
            else:
                cloned[key] = val
        out["state"][pid] = cloned
    return out
