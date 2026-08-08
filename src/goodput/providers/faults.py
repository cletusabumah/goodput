"""Fault injector implementations — real SIGKILL is gated (ticket 1.6)."""

from __future__ import annotations

import os
import signal
from typing import Literal

from goodput.providers.base import FaultInjector

FaultType = Literal["kill", "hang", "bitflip"]


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
    When ``dry_run=False`` and a worker PID is registered, sends SIGKILL on kill faults.
    Hang/bitflip remain record-only until tickets 3.1 / 3.2.
    """

    name = "process"

    def __init__(
        self,
        inject_at: int | None = None,
        fault: FaultType = "kill",
        *,
        dry_run: bool = True,
    ) -> None:
        self.inject_at = inject_at
        self.fault: FaultType = fault
        self.dry_run = dry_run
        self.worker_pids: dict[int, int] = {}
        self.injected: list[tuple[int, int, str]] = []

    def register_worker(self, worker_id: int, pid: int) -> None:
        self.worker_pids[worker_id] = pid

    def maybe_inject(self, step: int, worker_id: int) -> str | None:
        if self.inject_at is None or step != self.inject_at:
            return None

        self.injected.append((step, worker_id, self.fault))

        if self.dry_run or self.fault != "kill":
            return self.fault

        pid = self.worker_pids.get(worker_id)
        if pid is None:
            raise RuntimeError(f"No PID registered for worker_id={worker_id}")
        os.kill(pid, signal.SIGKILL)
        return self.fault
