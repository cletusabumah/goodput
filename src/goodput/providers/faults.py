"""Fault injector implementations — SIGKILL (1.6) and hang (3.1)."""

from __future__ import annotations

import os
import signal
from typing import Any, Literal

from goodput.providers.base import FaultInjector

FaultType = Literal["kill", "hang", "bitflip"]

# hang_rank shared Value uses -1 for "no hang"; injected rank id when hanging.
HANG_RANK_IDLE = -1


class MockFaultInjector(FaultInjector):
    """Records injection intent without signaling processes (CI-safe)."""

    name = "mock"

    def __init__(
        self,
        inject_at: int | None = None,
        fault: FaultType = "kill",
    ) -> None:
        self.inject_at = inject_at
        self.fault: FaultType = fault
        self.injected: list[tuple[int, int, str]] = []

    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        if self.inject_at is not None and step == self.inject_at:
            self.injected.append((step, worker_id, self.fault))
            return self.fault
        return None


class ProcessFaultInjector(FaultInjector):
    """
    Process-aware injector.

    By default ``dry_run=True``: records intent only (safe for CI).
    When ``dry_run=False``:
    - ``kill``: SIGKILL a registered worker PID
    - ``hang``: set ``hang_rank`` shared memory so the worker blocks before the next step
    - ``bitflip``: set ``bitflip_rank`` / ``bitflip_at_step`` for gradient corruption
    """

    name = "process"

    def __init__(
        self,
        inject_at: int | None = None,
        fault: FaultType = "kill",
        *,
        dry_run: bool = True,
        hang_rank: Any | None = None,
        bitflip_rank: Any | None = None,
        bitflip_at_step: Any | None = None,
    ) -> None:
        self.inject_at = inject_at
        self.fault: FaultType = fault
        self.dry_run = dry_run
        self.hang_rank = hang_rank
        self.bitflip_rank = bitflip_rank
        self.bitflip_at_step = bitflip_at_step
        self.worker_pids: dict[int, int] = {}
        self.injected: list[tuple[int, int, str]] = []

    def register_worker(self, worker_id: int, pid: int) -> None:
        self.worker_pids[worker_id] = pid

    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        if self.inject_at is None or step != self.inject_at:
            return None

        self.injected.append((step, worker_id, self.fault))

        if self.dry_run:
            return self.fault

        if self.fault == "kill":
            pid = self.worker_pids.get(worker_id)
            if pid is None:
                raise RuntimeError(f"No PID registered for worker_id={worker_id}")
            os.kill(pid, signal.SIGKILL)
            return self.fault

        if self.fault == "hang" and self.hang_rank is not None:
            with self.hang_rank.get_lock():
                self.hang_rank.value = worker_id
            return self.fault

        if (
            self.fault == "bitflip"
            and self.bitflip_rank is not None
            and self.bitflip_at_step is not None
        ):
            with self.bitflip_rank.get_lock():
                self.bitflip_rank.value = worker_id
            with self.bitflip_at_step.get_lock():
                self.bitflip_at_step.value = step
            return self.fault

        return self.fault
