"""Fault injection mid-run + checkpoint recovery (tickets 1.6 / 3.1).

Stories encoded:
- **kill:** SIGKILL one worker after a durable checkpoint; reap barrier-stuck peers; resume.
- **hang:** Block one worker alive after a durable checkpoint; parent detects stall via
  rank-0 progress timeout (not process exit); reap; resume.

CI may run real SIGKILL / hang on short-lived child processes we own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.multiprocessing as mp

from goodput.config import Settings
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore
from goodput.providers.faults import HANG_RANK_IDLE, ProcessFaultInjector
from goodput.training.loop import TrainResult, resume_after_crash
from goodput.training.multiprocess import _param_count, _worker_entry

FaultKind = Literal["kill", "hang"]


@dataclass
class FaultRecoveryResult:
    """Outcome of a fault-then-recover experiment."""

    world_size: int
    kill_rank: int
    kill_at_step: int
    checkpoint_step: int
    killed_pid: int | None
    pre_kill_workers_reaped: int
    recovered: TrainResult
    injected: list[tuple[int, int, str]]
    fault_type: FaultKind = "kill"
    detection_latency_s: float = 0.0
    hang_detected: bool = False
    # Ticket 1.7 — experiment-level timings for the JSON report.
    wall_seconds: float = 0.0
    useful_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        base = (
            self.checkpoint_step == self.kill_at_step
            and self.recovered.ok
            and self.recovered.resumed_from_step == self.checkpoint_step
        )
        if self.fault_type == "hang":
            return base and self.hang_detected and self.detection_latency_s > 0
        return base


def _validate_fault_step(settings: Settings, fault_at_step: int, fault_rank: int) -> int:
    world_size = settings.num_workers
    if world_size < 2:
        raise ValueError("fault recovery demo requires num_workers >= 2")
    if settings.ckpt_interval < 1:
        raise ValueError("ckpt_interval must be >= 1 for fault recovery")
    if fault_at_step < 1 or fault_at_step % settings.ckpt_interval != 0:
        raise ValueError(
            "fault_at_step must be a positive multiple of ckpt_interval "
            f"(got fault_at_step={fault_at_step}, ckpt_interval={settings.ckpt_interval})"
        )
    if fault_at_step > settings.steps:
        raise ValueError("fault_at_step cannot exceed settings.steps")
    if not (0 <= fault_rank < world_size):
        raise ValueError(f"fault_rank must be in [0, {world_size})")
    return world_size


def _wait_for_progress(
    progress: mp.Value,
    target: int,
    processes: list[mp.Process],
    *,
    timeout_s: float = 60.0,
) -> None:
    """Block until rank-0 reports ``target`` completed steps (or timeout / early death)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if progress.value >= target:
            return
        if processes and all(not p.is_alive() for p in processes):
            raise RuntimeError(
                f"all workers exited before reaching step {target} (progress={progress.value})"
            )
        time.sleep(0.02)
    raise TimeoutError(f"progress stuck at {progress.value}, wanted {target}")


def wait_for_hang_detection(
    progress: mp.Value,
    stalled_at: int,
    processes: list[mp.Process],
    *,
    timeout_s: float,
) -> float:
    """
    Detect a hang when rank-0 progress stays at ``stalled_at`` for ``timeout_s``.

    Returns detection latency in seconds. Requires at least one worker still alive —
    detection does **not** rely on process exit (ticket 3.1 Done-when).
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    while time.monotonic() < deadline:
        if progress.value > stalled_at:
            raise RuntimeError(
                f"progress advanced to {progress.value} during hang detection "
                f"(expected stall at {stalled_at})"
            )
        if processes and all(not p.is_alive() for p in processes):
            raise RuntimeError("all workers exited during hang detection")
        time.sleep(0.05)
    if not any(p.is_alive() for p in processes):
        raise RuntimeError("hang detection expected at least one live worker")
    return time.monotonic() - t0


def _reap_all(processes: list[mp.Process], *, grace_s: float = 2.0) -> int:
    """SIGKILL any still-alive workers (barrier hang after a peer dies is expected)."""
    reaped = 0
    deadline = time.monotonic() + grace_s
    for proc in processes:
        if proc.is_alive():
            proc.kill()
            reaped += 1
    for proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        proc.join(timeout=remaining)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
    return reaped


def _run_fault_and_recover(
    settings: Settings,
    *,
    ckpt_dir: Path,
    fault_at_step: int,
    fault_rank: int,
    fault_type: FaultKind,
    remaining_steps: int | None = None,
    health_check_timeout_s: float | None = None,
) -> FaultRecoveryResult:
    world_size = _validate_fault_step(settings, fault_at_step, fault_rank)

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    store = LocalFsCheckpointStore(ckpt_dir)

    probe = ToyMLP(input_size=settings.input_size, hidden_size=settings.hidden_size)
    n_params = _param_count(probe)

    ctx = mp.get_context("spawn")
    grad_bucket = torch.zeros(world_size, n_params, dtype=torch.float32).share_memory_()
    barrier = ctx.Barrier(world_size)
    result_queue: mp.Queue = ctx.Queue()
    progress = ctx.Value("i", 0)
    hang_rank = ctx.Value("i", HANG_RANK_IDLE)

    settings_dict = settings.model_dump(mode="json")
    processes: list[mp.Process] = []
    wall_t0 = time.perf_counter()
    detection_latency_s = 0.0
    hang_detected = False
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
                    "ckpt_dir": str(ckpt_dir) if rank == 0 else None,
                    "ckpt_interval": settings.ckpt_interval,
                    "hang_rank": hang_rank,
                },
                name=f"goodput-worker-{rank}",
            )
            proc.start()
            processes.append(proc)

        injector = ProcessFaultInjector(
            inject_at=fault_at_step,
            fault=fault_type,
            dry_run=False,
            hang_rank=hang_rank,
        )
        for rank, proc in enumerate(processes):
            if proc.pid is not None:
                injector.register_worker(rank, proc.pid)

        _wait_for_progress(progress, fault_at_step, processes)

        if fault_type == "hang":
            # Set hang before rank 0 can advance past the fault step (progress publisher).
            injector.maybe_inject(fault_at_step, fault_rank)

        step_path = ckpt_dir / f"step_{fault_at_step:06d}.pt"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if step_path.exists():
                break
            time.sleep(0.02)
        else:
            raise TimeoutError(f"checkpoint missing: {step_path}")

        torch.save({"locator": str(step_path), "step": fault_at_step}, ckpt_dir / "latest.pt")
        useful_pre_fault = time.perf_counter() - wall_t0
        checkpoint_step = fault_at_step
        fault_pid = processes[fault_rank].pid
        if fault_type == "kill":
            injector.maybe_inject(fault_at_step, fault_rank)

        if fault_type == "hang":
            timeout = (
                health_check_timeout_s
                if health_check_timeout_s is not None
                else settings.health_check_timeout_s
            )
            detection_latency_s = wait_for_hang_detection(
                progress,
                fault_at_step,
                processes,
                timeout_s=timeout,
            )
            hang_detected = True
            assert processes[fault_rank].is_alive(), "hung worker exited before detection"

        reaped = _reap_all(processes)

        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:  # noqa: BLE001
                break

        remain = (
            remaining_steps
            if remaining_steps is not None
            else max(1, settings.steps - checkpoint_step)
        )
        recovered = resume_after_crash(
            settings=settings,
            checkpoint_store=store,
            remaining_steps=remain,
        )
        wall_seconds = time.perf_counter() - wall_t0
        useful_seconds = useful_pre_fault + recovered.useful_seconds

        return FaultRecoveryResult(
            world_size=world_size,
            kill_rank=fault_rank,
            kill_at_step=fault_at_step,
            checkpoint_step=checkpoint_step,
            killed_pid=fault_pid,
            pre_kill_workers_reaped=reaped,
            recovered=recovered,
            injected=list(injector.injected),
            fault_type=fault_type,
            detection_latency_s=detection_latency_s,
            hang_detected=hang_detected,
            wall_seconds=wall_seconds,
            useful_seconds=useful_seconds,
        )
    finally:
        _reap_all(processes)


def run_sigkill_and_recover(
    settings: Settings,
    *,
    ckpt_dir: Path,
    kill_at_step: int,
    kill_rank: int = 0,
    remaining_steps: int | None = None,
) -> FaultRecoveryResult:
    """
    Multiprocess train → SIGKILL one rank at ``kill_at_step`` → resume from ckpt.

    ``kill_at_step`` must be a positive multiple of ``settings.ckpt_interval`` so a
    durable checkpoint exists before the kill.
    """
    return _run_fault_and_recover(
        settings,
        ckpt_dir=ckpt_dir,
        fault_at_step=kill_at_step,
        fault_rank=kill_rank,
        fault_type="kill",
        remaining_steps=remaining_steps,
    )


def run_hang_and_recover(
    settings: Settings,
    *,
    ckpt_dir: Path,
    hang_at_step: int,
    hang_rank: int = 1,
    remaining_steps: int | None = None,
    health_check_timeout_s: float | None = None,
) -> FaultRecoveryResult:
    """
    Multiprocess train → hang one rank at ``hang_at_step`` → detect stall → resume.

    Detection polls rank-0 progress for ``health_check_timeout_s`` without requiring
    the hung worker to exit (ticket 3.1).
    """
    return _run_fault_and_recover(
        settings,
        ckpt_dir=ckpt_dir,
        fault_at_step=hang_at_step,
        fault_rank=hang_rank,
        fault_type="hang",
        remaining_steps=remaining_steps,
        health_check_timeout_s=health_check_timeout_s,
    )
