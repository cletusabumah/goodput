"""CLI entrypoint — train, checkpoints, SIGKILL recovery, metrics report (1.3–1.7)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from goodput import __version__
from goodput.config import Settings, get_settings
from goodput.metrics import build_run_report, emit_run_report
from goodput.providers import LocalFsCheckpointStore, build_providers
from goodput.providers.metrics import JsonFileMetricsSink
from goodput.training import (
    MultiProcessResult,
    TrainResult,
    run_sigkill_and_recover,
    train_from_settings,
    train_multiprocess_from_settings,
)
from goodput.training.recovery import FaultRecoveryResult


def _emit_and_announce(settings: Settings, report: dict[str, Any]) -> Path | None:
    """Write report via configured sink; return path when using json_file."""
    providers = build_providers(settings)
    emit_run_report(providers.metrics, report)
    if isinstance(providers.metrics, JsonFileMetricsSink):
        path = providers.metrics.path
        print(f"report={path}")
        return path
    print("report=emitted")
    return None


def _report_from_train(settings: Settings, result: TrainResult) -> dict[str, Any]:
    return build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=result.steps_completed,
        ckpt_save_seconds=result.ckpt_save_seconds,
        ckpt_restore_seconds=result.ckpt_restore_seconds,
        final_loss=result.final_loss,
        extra={
            "mode": "train",
            "last_checkpoint_step": result.last_checkpoint_step,
            "resumed_from_step": result.resumed_from_step,
        },
    )


def _report_from_fault(settings: Settings, result: FaultRecoveryResult) -> dict[str, Any]:
    recovered = result.recovered
    return build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=result.checkpoint_step + recovered.steps_completed,
        ckpt_save_seconds=recovered.ckpt_save_seconds,
        ckpt_restore_seconds=recovered.ckpt_restore_seconds,
        final_loss=recovered.final_loss,
        extra={
            "mode": "fault_kill",
            "kill_rank": result.kill_rank,
            "kill_at_step": result.kill_at_step,
            "checkpoint_step": result.checkpoint_step,
            "failures_injected": len(result.injected),
            "recoveries_succeeded": int(result.ok),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goodput-run", description="Goodput simulator")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML (ticket 1.8)")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override GOODPUT_STEPS for a short local run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override GOODPUT_NUM_WORKERS (1=single-process, >=2=multiprocess)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (required for --fault-kill)",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=None,
        help="Override GOODPUT_CKPT_INTERVAL (steps between checkpoints)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override GOODPUT_RUN_NAME (report path artifacts/reports/<name>/)",
    )
    parser.add_argument(
        "--fault-kill",
        action="store_true",
        help="SIGKILL one worker mid-run then resume from checkpoint (ticket 1.6)",
    )
    parser.add_argument(
        "--fault-at",
        type=int,
        default=None,
        help="Completed step at which to SIGKILL (must be a multiple of ckpt_interval)",
    )
    parser.add_argument(
        "--fault-rank",
        type=int,
        default=1,
        help="Worker rank to kill (default: 1, leave rank 0 as checkpointer)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run toy training (single- or multi-process)",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    settings = get_settings()
    updates: dict = {}
    if args.steps is not None:
        updates["steps"] = args.steps
    if args.workers is not None:
        updates["num_workers"] = args.workers
    if args.ckpt_dir is not None:
        updates["ckpt_dir"] = args.ckpt_dir
    if args.ckpt_interval is not None:
        updates["ckpt_interval"] = args.ckpt_interval
    if args.run_name is not None:
        updates["run_name"] = args.run_name
    if updates:
        settings = settings.model_copy(update=updates)

    print(
        f"goodput {__version__} | device={settings.device} "
        f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
    )

    if args.config:
        print(f"config={args.config} (YAML runner lands in ticket 1.8)")
        return 0

    if args.fault_kill:
        if args.ckpt_dir is None:
            print("--fault-kill requires --ckpt-dir")
            return 2
        if args.fault_at is None:
            print("--fault-kill requires --fault-at")
            return 2
        if settings.num_workers < 2:
            settings = settings.model_copy(update={"num_workers": 2})
        result = run_sigkill_and_recover(
            settings,
            ckpt_dir=args.ckpt_dir,
            kill_at_step=args.fault_at,
            kill_rank=args.fault_rank,
        )
        print(
            f"fault-kill ok={result.ok} kill_rank={result.kill_rank} "
            f"kill_at={result.kill_at_step} ckpt_step={result.checkpoint_step} "
            f"recovered_steps={result.recovered.steps_completed} "
            f"resumed_from={result.recovered.resumed_from_step}"
        )
        report = _report_from_fault(settings, result)
        _emit_and_announce(settings, report)
        print(
            f"goodput={report['goodput']:.4f} "
            f"ckpt_save_s={report['ckpt_save_s']:.6f} "
            f"ckpt_restore_s={report['ckpt_restore_s']:.6f} "
            f"wasted_gpu_hours={report['wasted_gpu_hours']:.6f}"
        )
        return 0 if result.ok else 1

    if args.train:
        if settings.num_workers <= 1:
            store = None
            if args.ckpt_dir is not None:
                store = LocalFsCheckpointStore(settings.ckpt_dir)
            result = train_from_settings(settings, checkpoint_store=store)
            ckpt_msg = (
                f" last_ckpt={result.last_checkpoint_step}"
                if result.last_checkpoint_step is not None
                else ""
            )
            resume_msg = (
                f" resumed_from={result.resumed_from_step}"
                if result.resumed_from_step is not None
                else ""
            )
            print(
                f"train ok={result.ok} steps={result.steps_completed} "
                f"final_loss={result.final_loss:.6f} device={result.device}"
                f"{ckpt_msg}{resume_msg}"
            )
            report = _report_from_train(settings, result)
            _emit_and_announce(settings, report)
            print(
                f"goodput={report['goodput']:.4f} "
                f"ckpt_save_s={report['ckpt_save_s']:.6f} "
                f"ckpt_restore_s={report['ckpt_restore_s']:.6f} "
                f"wasted_gpu_hours={report['wasted_gpu_hours']:.6f}"
            )
            return 0 if result.ok else 1

        wall_t0 = time.perf_counter()
        mp_result = train_multiprocess_from_settings(settings)
        wall_s = time.perf_counter() - wall_t0
        assert isinstance(mp_result, MultiProcessResult)
        print(
            f"train ok={mp_result.ok} workers={mp_result.world_size} "
            f"steps={mp_result.steps} final_loss={mp_result.final_loss:.6f}"
        )
        for w in mp_result.workers:
            err = f" error={w.error}" if w.error else ""
            print(
                f"  rank={w.rank} ok={w.ok} steps={w.steps_completed} "
                f"final_loss={w.final_loss:.6f}{err}"
            )
        # Uninterrupted multi-process: treat wall ≈ useful (no failure window).
        # Fine-grained ckpt timers land when workers report them; fault-kill path
        # already measures restore + post-resume useful time.
        report = build_run_report(
            settings=settings,
            wall_seconds=max(wall_s, 1e-9),
            useful_seconds=max(wall_s, 1e-9),
            steps_completed=mp_result.steps,
            final_loss=mp_result.final_loss,
            extra={"mode": "train_multiprocess"},
        )
        _emit_and_announce(settings, report)
        print(
            f"goodput={report['goodput']:.4f} "
            f"ckpt_save_s={report['ckpt_save_s']:.6f} "
            f"ckpt_restore_s={report['ckpt_restore_s']:.6f} "
            f"wasted_gpu_hours={report['wasted_gpu_hours']:.6f}"
        )
        return 0 if mp_result.ok else 1

    print("Pass --train or --fault-kill, or see docs/phase-1-tickets.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
