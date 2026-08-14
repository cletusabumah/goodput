"""CLI entrypoint — train, YAML experiments, sweeps, plots, Compose (2.4)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from goodput import __version__
from goodput.config import Settings, get_settings
from goodput.evaluation.plot import (
    default_plot_path,
    load_comparison,
    plot_goodput_vs_failure_rate,
)
from goodput.evaluation.sweep import load_sweep_yaml, run_sweep
from goodput.experiments import load_experiment_yaml
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


def _print_metrics(report: dict[str, Any]) -> None:
    print(
        f"goodput={report['goodput']:.4f} "
        f"ckpt_save_s={report['ckpt_save_s']:.6f} "
        f"ckpt_restore_s={report['ckpt_restore_s']:.6f} "
        f"wasted_gpu_hours={report['wasted_gpu_hours']:.6f}"
    )


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


def _run_train(settings: Settings, *, use_checkpoint_store: bool) -> int:
    # Compose nodes share a volume; only rank 0 persists so latest.pt is not a race.
    use_store = use_checkpoint_store and (settings.num_workers > 1 or settings.rank == 0)
    if settings.num_workers <= 1:
        store = LocalFsCheckpointStore(settings.ckpt_dir) if use_store else None
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
            f"train ok={result.ok} rank={settings.rank} steps={result.steps_completed} "
            f"final_loss={result.final_loss:.6f} device={result.device}"
            f"{ckpt_msg}{resume_msg}"
        )
        report = _report_from_train(settings, result)
        _emit_and_announce(settings, report)
        _print_metrics(report)
        return 0 if result.ok else 1

    wall_t0 = time.perf_counter()
    mp_result = train_multiprocess_from_settings(
        settings,
        checkpoint=use_checkpoint_store,
    )
    wall_s = time.perf_counter() - wall_t0
    assert isinstance(mp_result, MultiProcessResult)
    print(
        f"train ok={mp_result.ok} workers={mp_result.world_size} "
        f"steps={mp_result.steps} final_loss={mp_result.final_loss:.6f}"
        + (
            f" last_ckpt={mp_result.last_checkpoint_step}"
            if mp_result.last_checkpoint_step is not None
            else ""
        )
    )
    for w in mp_result.workers:
        err = f" error={w.error}" if w.error else ""
        print(
            f"  rank={w.rank} ok={w.ok} steps={w.steps_completed} "
            f"final_loss={w.final_loss:.6f}{err}"
        )
    report = build_run_report(
        settings=settings,
        wall_seconds=max(wall_s, 1e-9),
        useful_seconds=max(wall_s, 1e-9),
        steps_completed=mp_result.steps,
        ckpt_save_seconds=mp_result.ckpt_save_seconds,
        final_loss=mp_result.final_loss,
        extra={
            "mode": "train_multiprocess",
            "last_checkpoint_step": mp_result.last_checkpoint_step,
        },
    )
    _emit_and_announce(settings, report)
    _print_metrics(report)
    return 0 if mp_result.ok else 1


def _run_fault_kill(
    settings: Settings,
    *,
    kill_at_step: int,
    kill_rank: int,
) -> int:
    if settings.num_workers < 2:
        settings = settings.model_copy(update={"num_workers": 2})
    result = run_sigkill_and_recover(
        settings,
        ckpt_dir=settings.ckpt_dir,
        kill_at_step=kill_at_step,
        kill_rank=kill_rank,
    )
    print(
        f"fault-kill ok={result.ok} kill_rank={result.kill_rank} "
        f"kill_at={result.kill_at_step} ckpt_step={result.checkpoint_step} "
        f"recovered_steps={result.recovered.steps_completed} "
        f"resumed_from={result.recovered.resumed_from_step}"
    )
    report = _report_from_fault(settings, result)
    _emit_and_announce(settings, report)
    _print_metrics(report)
    return 0 if result.ok else 1


def _run_plot(source: Path, output: Path | None) -> int:
    try:
        rows = load_comparison(source)
        out = output if output is not None else default_plot_path()
        written = plot_goodput_vs_failure_rate(rows, out)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    except ValueError as exc:
        print(f"plot error: {exc}")
        return 2
    except ImportError as exc:
        print(exc)
        return 2
    print(f"plot={written}")
    return 0


def _apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    updates: dict[str, Any] = {}
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
        return settings.model_copy(update=updates)
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goodput-run", description="Goodput simulator")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML (ticket 1.8)")
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Sweep YAML: failure rate × ckpt mode → comparison JSON/CSV (ticket 2.2)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Plot goodput vs failure rate from comparison JSON/CSV (ticket 2.3). "
            "Pass a path, or use with --sweep to plot the sweep result."
        ),
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="PNG output path (default: artifacts/plots/goodput_vs_failure_rate.png)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override GOODPUT_STEPS / YAML steps",
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
        help="Checkpoint directory (required for --fault-kill without YAML)",
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
        default=None,
        help="Worker rank to kill (default: 1, or YAML fault_rank)",
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

    if args.sweep:
        try:
            spec = load_sweep_yaml(args.sweep)
        except (OSError, ValueError, TypeError) as exc:
            print(f"sweep config error: {exc}")
            return 2
        print(
            f"goodput {__version__} | sweep={spec.name} "
            f"modes={list(spec.ckpt_modes)} rates={list(spec.failure_rates)}"
        )
        print(f"config={spec.path}")
        result = run_sweep(spec)
        print(f"sweep cells={len(result.rows)} json={result.json_path} csv={result.csv_path}")
        for row in result.rows:
            print(
                f"  mode={row['ckpt_mode']} rate={row['failure_rate']} "
                f"kill_at={row['kill_at']} goodput={row['goodput']:.4f}"
            )
        if args.plot is not None:
            source = result.json_path if args.plot == "auto" else Path(args.plot)
            return _run_plot(source, args.plot_out)
        return 0

    if args.plot is not None:
        source = (
            Path("artifacts/sweeps/phase2-sweep/comparison.json")
            if args.plot == "auto"
            else Path(args.plot)
        )
        return _run_plot(source, args.plot_out)

    if args.config:
        try:
            spec = load_experiment_yaml(args.config)
        except (OSError, ValueError, TypeError) as exc:
            print(f"config error: {exc}")
            return 2
        settings = _apply_cli_overrides(spec.settings, args)
        print(
            f"goodput {__version__} | device={settings.device} "
            f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
        )
        print(f"config={spec.path}")
        if spec.mode == "fault_kill":
            kill_at = args.fault_at if args.fault_at is not None else spec.fault_at
            if kill_at is None:
                print("fault_kill requires fault_at in YAML or --fault-at")
                return 2
            kill_rank = args.fault_rank if args.fault_rank is not None else spec.fault_rank
            return _run_fault_kill(settings, kill_at_step=kill_at, kill_rank=kill_rank)
        # YAML train path always checkpoints when ckpt_interval > 0.
        return _run_train(settings, use_checkpoint_store=settings.ckpt_interval > 0)

    settings = _apply_cli_overrides(get_settings(), args)
    print(
        f"goodput {__version__} | device={settings.device} "
        f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
    )

    if args.fault_kill:
        if args.ckpt_dir is None:
            print("--fault-kill requires --ckpt-dir")
            return 2
        if args.fault_at is None:
            print("--fault-kill requires --fault-at")
            return 2
        kill_rank = args.fault_rank if args.fault_rank is not None else 1
        return _run_fault_kill(
            settings,
            kill_at_step=args.fault_at,
            kill_rank=kill_rank,
        )

    if args.train:
        use_store = args.ckpt_dir is not None
        return _run_train(settings, use_checkpoint_store=use_store)

    print(
        "Pass --train, --fault-kill, --config, --sweep, or --plot; "
        "see docs/phase-1-tickets.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
