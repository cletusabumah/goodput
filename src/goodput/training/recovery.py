"""SIGKILL mid-run + checkpoint recovery (ticket 1.6).

Story this encodes:
1. Launch N synchronized workers that checkpoint periodically.
2. After a chosen completed step (and a durable ckpt), SIGKILL one worker.
3. Other workers may be stuck on the next barrier — parent tears them down.
4. Restart training from the last checkpoint (job-level recovery).

CI may run this with real SIGKILL on short-lived child processes we own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

from goodput.config import Settings
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore
from goodput.providers.faults import ProcessFaultInjector
from goodput.training.loop import TrainResult, resume_after_crash
from goodput.training.multiprocess import _param_count, _worker_entry


@dataclass
class FaultRecoveryResult:
    """Outcome of a kill-then-recover experiment."""

    world_size: int
    kill_rank: int
    kill_at_step: int
    checkpoint_step: int
    killed_pid: int | None
    pre_kill_workers_reaped: int
    recovered: TrainResult
    injected: list[tuple[int, int, str]]
    # Ticket 1.7 — experiment-level timings for the JSON report.
    wall_seconds: float = 0.0
    useful_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.checkpoint_step == self.kill_at_step
            and self.recovered.ok
            and self.recovered.resumed_from_step == self.checkpoint_step
        )


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
    world_size = settings.num_workers
    if world_size < 2:
        raise ValueError("sigkill recovery demo requires num_workers >= 2")
    if settings.ckpt_interval < 1:
        raise ValueError("ckpt_interval must be >= 1 for fault recovery")
    if kill_at_step < 1 or kill_at_step % settings.ckpt_interval != 0:
        raise ValueError(
            "kill_at_step must be a positive multiple of ckpt_interval "
            f"(got kill_at_step={kill_at_step}, ckpt_interval={settings.ckpt_interval})"
        )
    if kill_at_step > settings.steps:
        raise ValueError("kill_at_step cannot exceed settings.steps")
    if not (0 <= kill_rank < world_size):
        raise ValueError(f"kill_rank must be in [0, {world_size})")

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

    settings_dict = settings.model_dump(mode="json")
    processes: list[mp.Process] = []
    wall_t0 = time.perf_counter()
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
                },
                name=f"goodput-worker-{rank}",
            )
            proc.start()
            processes.append(proc)

        injector = ProcessFaultInjector(
            inject_at=kill_at_step,
            fault="kill",
            dry_run=False,
        )
        for rank, proc in enumerate(processes):
            if proc.pid is not None:
                injector.register_worker(rank, proc.pid)

        _wait_for_progress(progress, kill_at_step, processes)

        # Wait for the *specific* kill-at checkpoint file (not merely "latest",
        # which can race ahead if rank 0 keeps training).
        step_path = ckpt_dir / f"step_{kill_at_step:06d}.pt"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if step_path.exists():
                break
            time.sleep(0.02)
        else:
            raise TimeoutError(f"checkpoint missing: {step_path}")

        # Pin latest → kill-at step so resume does not pick a newer racing ckpt.
        torch.save({"locator": str(step_path), "step": kill_at_step}, ckpt_dir / "latest.pt")
        # Durable progress through kill_at counts as useful; kill→resume is waste.
        useful_pre_kill = time.perf_counter() - wall_t0
        checkpoint_step = kill_at_step
        killed_pid = processes[kill_rank].pid
        injector.maybe_inject(kill_at_step, kill_rank)

        # Peer workers are likely blocked on the next barrier — reap the job.
        reaped = _reap_all(processes)

        # Drain any late queue messages (best-effort).
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
        useful_seconds = useful_pre_kill + recovered.useful_seconds

        return FaultRecoveryResult(
            world_size=world_size,
            kill_rank=kill_rank,
            kill_at_step=kill_at_step,
            checkpoint_step=checkpoint_step,
            killed_pid=killed_pid,
            pre_kill_workers_reaped=reaped,
            recovered=recovered,
            injected=list(injector.injected),
            wall_seconds=wall_seconds,
            useful_seconds=useful_seconds,
        )
    finally:
        _reap_all(processes)
