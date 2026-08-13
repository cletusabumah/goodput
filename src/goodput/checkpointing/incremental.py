"""Incremental / async-friendly checkpoint path (ticket 2.1).

Strategy (simplified but honest):
- Every ``full_every`` capture (and the first) writes a **full** naive blob
  (model + optimizer) — the durable base.
- Intervening captures write **model weights only** + a pointer to that base.
  Skipping the optimizer clone/serialize is what makes save time drop on a
  realistic fixture (e.g. Adam), which is the Done-when for 2.1.

Restore loads the latest payload; if it is incremental, the store is asked for
the base step so optimizer state can be reattached. Training can treat the
lighter captures as the async-friendly hot path (less work on the critical
section before returning to the step loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from goodput.checkpointing.naive import (
    CHECKPOINT_FORMAT as NAIVE_FORMAT,
)
from goodput.checkpointing.naive import (
    capture_training_state,
)
from goodput.checkpointing.naive import (
    restore_training_state as restore_naive,
)
from goodput.providers.base import CheckpointPayload, CheckpointStore

INCREMENTAL_FORMAT = "incremental_v1"
INCREMENTAL_FULL_FORMAT = "incremental_full_v1"


def _clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@dataclass
class IncrementalCheckpointer:
    """Stateful capturer: full base every N saves, model-only otherwise."""

    full_every: int = 4
    _capture_index: int = 0
    _last_full_step: int | None = None
    last_kinds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.full_every < 1:
            raise ValueError("full_every must be >= 1")

    def capture(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CheckpointPayload:
        if step < 0:
            raise ValueError("step must be >= 0")
        self._capture_index += 1
        # Captures 1, 1+full_every, 1+2*full_every, ... are full bases.
        want_full = (
            self._last_full_step is None
            or (self._capture_index - 1) % self.full_every == 0
        )

        if want_full:
            payload = capture_training_state(model, optimizer, step=step, meta=meta)
            payload.meta["format"] = INCREMENTAL_FULL_FORMAT
            payload.meta["kind"] = "full"
            self._last_full_step = step
            self.last_kinds.append("full")
            return payload

        assert self._last_full_step is not None
        payload_meta: dict[str, Any] = {
            "format": INCREMENTAL_FORMAT,
            "kind": "delta",
            "base_step": self._last_full_step,
        }
        if meta:
            payload_meta.update(meta)
        self.last_kinds.append("delta")
        return CheckpointPayload(
            step=step,
            state={"model": _clone_model_state(model)},
            meta=payload_meta,
        )


def materialize_full_payload(
    payload: CheckpointPayload,
    store: CheckpointStore,
) -> CheckpointPayload:
    """
    Turn an incremental (model-only) payload into a naive-shaped full payload
    by loading optimizer state from ``base_step``.
    """
    fmt = payload.meta.get("format")
    if fmt in {None, NAIVE_FORMAT, INCREMENTAL_FULL_FORMAT}:
        return payload
    if fmt != INCREMENTAL_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {fmt}")

    base_step = payload.meta.get("base_step")
    if base_step is None:
        raise ValueError("incremental checkpoint missing base_step")
    if "model" not in payload.state:
        raise ValueError("incremental checkpoint missing model weights")

    if not hasattr(store, "load_at_step"):
        raise TypeError(f"store {type(store).__name__} cannot load_at_step for incremental restore")
    base = store.load_at_step(int(base_step))  # type: ignore[attr-defined]
    if "optimizer" not in base.state:
        raise ValueError(f"base checkpoint at step {base_step} has no optimizer")

    return CheckpointPayload(
        step=payload.step,
        state={
            "model": payload.state["model"],
            "optimizer": base.state["optimizer"],
        },
        meta={
            "format": NAIVE_FORMAT,
            "kind": "materialized",
            "base_step": int(base_step),
            "from_format": INCREMENTAL_FORMAT,
        },
    )


def restore_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: CheckpointPayload,
    *,
    device: torch.device | str = "cpu",
    store: CheckpointStore | None = None,
) -> int:
    """Restore naive or incremental payloads (incremental needs ``store``)."""
    fmt = payload.meta.get("format")
    if fmt == INCREMENTAL_FORMAT:
        if store is None:
            raise ValueError("incremental restore requires a CheckpointStore to load the base")
        payload = materialize_full_payload(payload, store)
    elif fmt not in {None, NAIVE_FORMAT, INCREMENTAL_FULL_FORMAT}:
        raise ValueError(f"unsupported checkpoint format: {fmt}")
    # Treat incremental_full like naive for the load path.
    if payload.meta.get("format") == INCREMENTAL_FULL_FORMAT:
        payload = CheckpointPayload(
            step=payload.step,
            state=payload.state,
            meta={**payload.meta, "format": NAIVE_FORMAT},
        )
    return restore_naive(model, optimizer, payload, device=device)
