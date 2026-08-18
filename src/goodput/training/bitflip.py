"""Gradient bit-flip mid-run (ticket 3.2).

Unlike kill/hang, training **continues** with corrupted gradients. The parent
arms shared memory when rank-0 reaches ``flip_at_step``; the target worker XORs
one float32 gradient bit before the all-reduce. Rank 0 may flag an outlier via
``detect_gradient_outlier`` (optional, ``settings.bitflip_detect``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp

from goodput.config import Settings
from goodput.faults.bitflip import BITFLIP_STEP_IDLE
from goodput.models import ToyMLP
from goodput.training.multiprocess import WorkerResult, _param_count, _worker_entry
from goodput.training.recovery import _wait_for_progress


@dataclass
class BitflipResult:
    """Outcome of a bit-flip corruption run (no checkpoint recovery)."""

    world_size: int
    flip_at_step: int
    flip_rank: int
    workers: list[WorkerResult] = field(default_factory=list)
    injected: list[tuple[int, int, str]] = field(default_factory=list)
    corruption_detected: bool = False
    corruption_detected_at: int | None = None
    wall_seconds: float = 0.0
    useful_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        if not self.workers or not all(w.ok for w in self.workers):
            return False
        flipped = any(w.bitflip_applied for w in self.workers)
        if not flipped:
            return False
        rank0 = next((w for w in self.workers if w.rank == 0), None)
        if rank0 is None or len(rank0.losses) < self.flip_at_step:
            return False
        return True

    @property
    def rank0_losses(self) -> list[float]:
        rank0 = next((w for w in self.workers if w.rank == 0), None)
        return list(rank0.losses) if rank0 else []


def _validate_bitflip_step(settings: Settings, flip_at_step: int, flip_rank: int) -> int:
    world_size = settings.num_workers
    if world_size < 2:
        raise ValueError("bitflip demo requires num_workers >= 2")
    if flip_at_step < 1 or flip_at_step > settings.steps:
        raise ValueError("flip_at_step must be in [1, settings.steps]")
    if not (0 <= flip_rank < world_size):
        raise ValueError(f"flip_rank must be in [0, {world_size})")
    return world_size


def run_bitflip_train(
    settings: Settings,
    *,
    flip_at_step: int,
    flip_rank: int = 1,
) -> BitflipResult:
    """Multiprocess train with one scripted gradient bit-flip at ``flip_at_step``."""
    world_size = _validate_bitflip_step(settings, flip_at_step, flip_rank)

    probe = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    n_params = _param_count(probe)

    ctx = mp.get_context("spawn")
    grad_bucket = torch.zeros(world_size, n_params, dtype=torch.float32).share_memory_()
    barrier = ctx.Barrier(world_size)
    result_queue: mp.Queue = ctx.Queue()
    progress = ctx.Value("i", 0)
    bitflip_rank = ctx.Value("i", flip_rank)
    bitflip_at_step = ctx.Value("i", flip_at_step)
    corruption_detected = ctx.Value("i", 0)
    corruption_at_step = ctx.Value("i", BITFLIP_STEP_IDLE)

    settings_dict = settings.model_dump(mode="json")
    processes: list[mp.Process] = []
    wall_t0 = time.perf_counter()
    injected: list[tuple[int, int, str]] = [(flip_at_step, flip_rank, "bitflip")]
    try:
        for rank in range(world_size):
            proc = ctx.Process(
                target=_worker_entry,
                args=(
                    rank,
                    world_size,
                    settings.steps,
                    settings_dict,
                    grad_bucket,
                    barrier,
                    result_queue,
                ),
                kwargs={
                    "progress": progress,
                    "bitflip_rank": bitflip_rank,
                    "bitflip_at_step": bitflip_at_step,
                    "corruption_detected": corruption_detected,
                    "corruption_at_step": corruption_at_step,
                },
                name=f"goodput-worker-{rank}",
            )
            proc.start()
            processes.append(proc)

        _wait_for_progress(progress, settings.steps, processes)

        workers: list[WorkerResult] = []
        for _ in range(world_size):
            workers.append(result_queue.get(timeout=120))

        for proc in processes:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

        workers.sort(key=lambda w: w.rank)
        wall_s = time.perf_counter() - wall_t0
        detected = corruption_detected.value == 1
        detected_at = (
            int(corruption_at_step.value)
            if corruption_at_step.value != BITFLIP_STEP_IDLE
            else None
        )
        return BitflipResult(
            world_size=world_size,
            flip_at_step=flip_at_step,
            flip_rank=flip_rank,
            workers=workers,
            injected=injected,
            corruption_detected=detected or any(w.corruption_detected for w in workers),
            corruption_detected_at=detected_at,
            wall_seconds=wall_s,
            useful_seconds=wall_s,
        )
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
