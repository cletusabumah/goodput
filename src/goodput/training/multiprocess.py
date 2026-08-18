"""Multi-process toy trainer with a barrier + gradient all-reduce stub (ticket 1.4).

This is intentionally *not* full PyTorch DDP. Workers are real OS processes that:
1. train on sharded synthetic batches,
2. wait on a barrier each step,
3. average gradients via a shared CPU tensor (toy all-reduce).

That is enough to simulate lockstep failure modes later (kill one worker → others stall).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from torch import nn

from goodput.config import Settings
from goodput.data import SyntheticDataLoader
from goodput.models import ToyMLP
from goodput.training.loop import TrainResult, _resolve_device


@dataclass
class WorkerResult:
    rank: int
    steps_completed: int
    final_loss: float
    ok: bool
    error: str | None = None
    ckpt_save_seconds: list[float] = field(default_factory=list)
    last_checkpoint_step: int | None = None
    losses: list[float] = field(default_factory=list)
    bitflip_applied: bool = False
    corruption_detected: bool = False
    corruption_detected_at: int | None = None


@dataclass
class MultiProcessResult:
    """Aggregated outcome of a multi-worker run."""

    world_size: int
    steps: int
    workers: list[WorkerResult] = field(default_factory=list)
    ckpt_save_seconds: list[float] = field(default_factory=list)
    last_checkpoint_step: int | None = None

    @property
    def ok(self) -> bool:
        return bool(self.workers) and all(w.ok for w in self.workers)

    @property
    def final_loss(self) -> float:
        finite = [w.final_loss for w in self.workers if math.isfinite(w.final_loss)]
        if not finite:
            return float("nan")
        return sum(finite) / len(finite)


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("no gradients to all-reduce")
    return torch.cat(parts)


def _unflatten_grads(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        n = p.grad.numel()
        p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
        offset += n


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _worker_should_hang(hang_rank: Any | None, rank: int) -> bool:
    if hang_rank is None:
        return False
    with hang_rank.get_lock():
        return hang_rank.value == rank


def _worker_hang_if_requested(hang_rank: Any | None, rank: int) -> None:
    if _worker_should_hang(hang_rank, rank):
        while True:
            time.sleep(0.5)


def _worker_entry(
    rank: int,
    world_size: int,
    steps: int,
    settings_dict: dict[str, Any],
    grad_bucket: torch.Tensor,
    barrier: mp.Barrier,
    result_queue: mp.Queue,
    progress: Any | None = None,
    ckpt_dir: str | None = None,
    ckpt_interval: int = 0,
    hang_rank: Any | None = None,
    bitflip_rank: Any | None = None,
    bitflip_at_step: Any | None = None,
    corruption_detected: Any | None = None,
    corruption_at_step: Any | None = None,
) -> None:
    """Child process target — must be top-level for spawn pickling."""
    try:
        from itertools import cycle

        from goodput.checkpointing import IncrementalCheckpointer, capture_training_state
        from goodput.faults.bitflip import (
            BITFLIP_RANK_IDLE,
            BITFLIP_STEP_IDLE,
            detect_gradient_outlier,
            flip_float32_bit,
        )
        from goodput.providers import LocalFsCheckpointStore

        settings = Settings(**settings_dict)
        device = _resolve_device(settings.device)
        # Same init seed → identical starting weights on every rank (like DDP broadcast).
        torch.manual_seed(settings.seed)
        model = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
        criterion = nn.MSELoss()

        store = None
        checkpointer = None
        if ckpt_dir and rank == 0 and ckpt_interval > 0:
            store = LocalFsCheckpointStore(Path(ckpt_dir))
            if settings.ckpt_mode == "incremental":
                checkpointer = IncrementalCheckpointer(full_every=settings.ckpt_full_every)

        # Bit-flip demos reuse the same seed so local grads match until corruption.
        data_seed = settings.seed if bitflip_at_step is not None else settings.seed + rank * 10_000
        loader = SyntheticDataLoader(
            num_batches=max(1, min(steps, 16)),
            batch_size=settings.batch_size,
            input_size=settings.input_size,
            seed=data_seed,
            device=device,
        )
        stream = cycle(loader)
        model.train()
        last_loss = float("nan")
        ckpt_save_seconds: list[float] = []
        last_ckpt: int | None = None
        losses: list[float] = []
        flipped = False
        saw_corruption = False
        corruption_step: int | None = None

        for step in range(steps):
            _worker_hang_if_requested(hang_rank, rank)

            batch = next(stream).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.inputs), batch.targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"rank {rank}: non-finite loss at step {step}")
            loss.backward()

            # --- toy all-reduce: write local grads → barrier → average → barrier ---
            flat = _flatten_grads(model).detach().cpu()
            completed = step + 1
            if (
                not flipped
                and bitflip_rank is not None
                and bitflip_at_step is not None
            ):
                with bitflip_rank.get_lock():
                    target_rank = bitflip_rank.value
                with bitflip_at_step.get_lock():
                    target_step = bitflip_at_step.value
                if (
                    target_rank == rank
                    and target_step == completed
                    and target_rank != BITFLIP_RANK_IDLE
                    and target_step != BITFLIP_STEP_IDLE
                ):
                    flip_float32_bit(flat)
                    flipped = True

            grad_bucket[rank].copy_(flat)
            _worker_hang_if_requested(hang_rank, rank)
            barrier.wait()

            if (
                rank == 0
                and settings.bitflip_detect
                and world_size > 1
                and corruption_detected is not None
            ):
                if detect_gradient_outlier(
                    grad_bucket,
                    ratio_threshold=settings.bitflip_grad_ratio_threshold,
                ):
                    saw_corruption = True
                    corruption_step = completed
                    with corruption_detected.get_lock():
                        corruption_detected.value = 1
                    if corruption_at_step is not None:
                        with corruption_at_step.get_lock():
                            corruption_at_step.value = completed

            averaged = grad_bucket.mean(dim=0)
            _worker_hang_if_requested(hang_rank, rank)
            barrier.wait()
            _unflatten_grads(model, averaged.to(device))

            optimizer.step()
            last_loss = float(loss.item())
            losses.append(last_loss)

            if store is not None and (
                completed % ckpt_interval == 0 or completed == steps
            ):
                save_t0 = time.perf_counter()
                if checkpointer is not None:
                    payload = checkpointer.capture(model, optimizer, step=completed)
                else:
                    payload = capture_training_state(model, optimizer, step=completed)
                store.save(payload)
                ckpt_save_seconds.append(time.perf_counter() - save_t0)
                last_ckpt = completed

            # Rank 0 publishes progress so the parent can schedule SIGKILL after a ckpt.
            if progress is not None and rank == 0:
                with progress.get_lock():
                    progress.value = completed
                # Yield after publishing a checkpointed step so the parent can
                # SIGKILL before the next all-reduce advances "latest".
                if store is not None and completed % max(ckpt_interval, 1) == 0:
                    time.sleep(0.25)

        result_queue.put(
            WorkerResult(
                rank=rank,
                steps_completed=steps,
                final_loss=last_loss,
                ok=math.isfinite(last_loss),
                ckpt_save_seconds=ckpt_save_seconds,
                last_checkpoint_step=last_ckpt,
                losses=losses,
                bitflip_applied=flipped,
                corruption_detected=saw_corruption,
                corruption_detected_at=corruption_step,
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface to parent as WorkerResult
        result_queue.put(
            WorkerResult(
                rank=rank,
                steps_completed=0,
                final_loss=float("nan"),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )


def launch_workers(
    *,
    world_size: int,
    steps: int,
    settings: Settings,
    ckpt_dir: Path | str | None = None,
) -> MultiProcessResult:
    """
    Spawn ``world_size`` workers that run ``steps`` synchronized train steps.

    When ``ckpt_dir`` is set and ``settings.ckpt_interval > 0``, rank 0 writes
    naive checkpoints (same path the SIGKILL recovery demo uses).

    Uses the ``spawn`` start method so this stays safe on macOS and in pytest.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    # Probe param count on the parent (tiny model) to size the shared grad bucket.
    probe = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    n_params = _param_count(probe)

    ctx = mp.get_context("spawn")
    grad_bucket = torch.zeros(world_size, n_params, dtype=torch.float32).share_memory_()
    barrier = ctx.Barrier(world_size)
    result_queue: mp.Queue = ctx.Queue()

    # Settings must be picklable — pass a plain dict into children.
    settings_dict = settings.model_dump(mode="json")
    ckpt_interval = settings.ckpt_interval if ckpt_dir is not None else 0
    if ckpt_dir is not None:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    processes: list[mp.Process] = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_worker_entry,
            args=(rank, world_size, steps, settings_dict, grad_bucket, barrier, result_queue),
            kwargs={
                "ckpt_dir": str(ckpt_dir) if (ckpt_dir is not None and rank == 0) else None,
                "ckpt_interval": ckpt_interval,
            },
            name=f"goodput-worker-{rank}",
        )
        proc.start()
        processes.append(proc)

    workers: list[WorkerResult] = []
    for _ in range(world_size):
        workers.append(result_queue.get(timeout=120))

    for proc in processes:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

    workers.sort(key=lambda w: w.rank)
    rank0 = next((w for w in workers if w.rank == 0), None)
    return MultiProcessResult(
        world_size=world_size,
        steps=steps,
        workers=workers,
        ckpt_save_seconds=list(rank0.ckpt_save_seconds) if rank0 else [],
        last_checkpoint_step=rank0.last_checkpoint_step if rank0 else None,
    )


def train_multiprocess_from_settings(
    settings: Settings,
    *,
    checkpoint: bool = False,
) -> MultiProcessResult | TrainResult:
    """Use multi-process when num_workers > 1; otherwise single-process TrainResult."""
    from goodput.training.loop import train_from_settings

    if settings.num_workers <= 1:
        store = None
        if checkpoint and settings.ckpt_interval > 0:
            from goodput.providers import LocalFsCheckpointStore

            store = LocalFsCheckpointStore(settings.ckpt_dir)
        return train_from_settings(settings, checkpoint_store=store)

    ckpt_dir = None
    if checkpoint and settings.ckpt_interval > 0:
        ckpt_dir = settings.ckpt_dir
    return launch_workers(
        world_size=settings.num_workers,
        steps=settings.steps,
        settings=settings,
        ckpt_dir=ckpt_dir,
    )
