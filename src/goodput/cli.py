"""CLI entrypoint — train, YAML experiments, sweeps, plots, latency, dollar, Compose."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from goodput import __version__
from goodput.config import Settings, get_settings
from goodput.evaluation.dollar import load_dollar_yaml, run_dollar
from goodput.evaluation.latency import load_latency_yaml, run_latency
from goodput.evaluation.plot import (
    default_plot_path,
    default_scale_plot_path,
    load_comparison,
    plot_goodput_vs_failure_rate,
    plot_goodput_vs_workers,
)
from goodput.evaluation.scale import load_scale_yaml, run_scale
from goodput.evaluation.sweep import load_sweep_yaml, run_sweep
from goodput.experiments import load_experiment_yaml
from goodput.metrics import build_run_report, emit_run_report
from goodput.providers import LocalFsCheckpointStore, build_providers
from goodput.providers.metrics import JsonFileMetricsSink
from goodput.training import (
    BitflipResult,
    MultiProcessResult,
    TrainResult,
    run_bitflip_train,
    run_hang_and_recover,
    run_sigkill_and_recover,
    train_from_settings,
    train_multiprocess_from_settings,
)
from goodput.training.recovery import FaultRecoveryResult


def _emit_and_announce(settings: Settings, report: dict[str, Any]) -> Path | None:
    """Write report via configured sink; return path when using json_file."""
    providers = build_providers(settings)
    emit_run_report(providers.metrics, report)
    run_id = providers.tracker.log_run(report)
    if run_id is not None:
        print(f"tracker={providers.tracker.name} run={run_id}")
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
    git = report.get("git_sha") or "unknown"
    short = git[:12] if git != "unknown" else git
    dirty = " dirty" if report.get("git_dirty") else ""
    print(f"repro git={short}{dirty} config={report.get('config_hash', '')}")


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
    mode = f"fault_{result.fault_type}"
    extra: dict[str, Any] = {
        "mode": mode,
        "fault_type": result.fault_type,
        "kill_rank": result.kill_rank,
        "kill_at_step": result.kill_at_step,
        "checkpoint_step": result.checkpoint_step,
        "failures_injected": len(result.injected),
        "recoveries_succeeded": int(result.ok),
    }
    if result.fault_type == "hang":
        extra["hang_rank"] = result.kill_rank
        extra["hang_at_step"] = result.kill_at_step
        extra["hang_detected"] = int(result.hang_detected)
        extra["detection_latency_s"] = result.detection_latency_s
    return build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=result.checkpoint_step + recovered.steps_completed,
        ckpt_save_seconds=recovered.ckpt_save_seconds,
        ckpt_restore_seconds=recovered.ckpt_restore_seconds,
        final_loss=recovered.final_loss,
        extra=extra,
    )


def _report_from_bitflip(settings: Settings, result: BitflipResult) -> dict[str, Any]:
    rank0 = next((w for w in result.workers if w.rank == 0), None)
    final_loss = rank0.final_loss if rank0 else float("nan")
    return build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=settings.steps,
        final_loss=final_loss,
        extra={
            "mode": "fault_bitflip",
            "fault_type": "bitflip",
            "flip_rank": result.flip_rank,
            "flip_at_step": result.flip_at_step,
            "failures_injected": len(result.injected),
            "bitflip_applied": int(any(w.bitflip_applied for w in result.workers)),
            "corruption_detected": int(result.corruption_detected),
            "corruption_detected_at": result.corruption_detected_at,
            "rank0_losses": result.rank0_losses,
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


def _run_fault_hang(
    settings: Settings,
    *,
    hang_at_step: int,
    hang_rank: int,
) -> int:
    if settings.num_workers < 2:
        settings = settings.model_copy(update={"num_workers": 2})
    result = run_hang_and_recover(
        settings,
        ckpt_dir=settings.ckpt_dir,
        hang_at_step=hang_at_step,
        hang_rank=hang_rank,
        health_check_timeout_s=settings.health_check_timeout_s,
    )
    print(
        f"fault-hang ok={result.ok} hang_rank={result.kill_rank} "
        f"hang_at={result.kill_at_step} ckpt_step={result.checkpoint_step} "
        f"detected={result.hang_detected} detection_s={result.detection_latency_s:.3f} "
        f"recovered_steps={result.recovered.steps_completed} "
        f"resumed_from={result.recovered.resumed_from_step}"
    )
    report = _report_from_fault(settings, result)
    _emit_and_announce(settings, report)
    _print_metrics(report)
    return 0 if result.ok else 1


def _run_fault_bitflip(
    settings: Settings,
    *,
    flip_at_step: int,
    flip_rank: int,
) -> int:
    if settings.num_workers < 2:
        settings = settings.model_copy(update={"num_workers": 2})
    result = run_bitflip_train(
        settings,
        flip_at_step=flip_at_step,
        flip_rank=flip_rank,
    )
    rank0 = next((w for w in result.workers if w.rank == 0), None)
    print(
        f"fault-bitflip ok={result.ok} flip_rank={result.flip_rank} "
        f"flip_at={result.flip_at_step} applied={any(w.bitflip_applied for w in result.workers)} "
        f"detected={result.corruption_detected} "
        f"detected_at={result.corruption_detected_at} "
        f"final_loss={rank0.final_loss if rank0 else float('nan'):.6f}"
    )
    report = _report_from_bitflip(settings, result)
    _emit_and_announce(settings, report)
    _print_metrics(report)
    return 0 if result.ok else 1


def _run_plot(source: Path, output: Path | None) -> int:
    try:
        rows = load_comparison(source)
        is_scale = bool(rows) and "num_workers" in rows[0] and "cluster_failure_rate" in rows[0]
        if is_scale:
            out = output if output is not None else default_scale_plot_path()
            written = plot_goodput_vs_workers(rows, out)
        else:
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
        "--latency",
        type=str,
        default=None,
        help="Latency YAML: worker count → ckpt save/restore table (ticket 2.5)",
    )
    parser.add_argument(
        "--dollar",
        type=str,
        default=None,
        help="Dollar YAML: public $/GPU-hr × measured goodput delta (ticket 3.3)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Scale YAML: goodput vs worker count (MTBF-at-scale, ticket 4.1)",
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
        "--fault-hang",
        action="store_true",
        help="Hang one worker mid-run; detect via progress timeout; resume (ticket 3.1)",
    )
    parser.add_argument(
        "--fault-bitflip",
        action="store_true",
        help="Flip one gradient bit on a worker; optional cross-rank detector (ticket 3.2)",
    )
    parser.add_argument(
        "--fault-at",
        type=int,
        default=None,
        help="Step at which to inject fault (kill/hang/bitflip; kill/hang need ckpt alignment)",
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

    if args.latency:
        try:
            spec = load_latency_yaml(args.latency)
        except (OSError, ValueError, TypeError) as exc:
            print(f"latency config error: {exc}")
            return 2
        print(
            f"goodput {__version__} | latency={spec.name} "
            f"workers={list(spec.worker_counts)}"
        )
        print(f"config={spec.path}")
        result = run_latency(spec)
        print(
            f"latency cells={len(result.rows)} json={result.json_path} "
            f"csv={result.csv_path} table={result.table_path}"
        )
        for row in result.rows:
            print(
                f"  workers={row['num_workers']} "
                f"ckpt_save_s={row['ckpt_save_s']:.6f} "
                f"ckpt_restore_s={row['ckpt_restore_s']:.6f} "
                f"wall_s={row['wall_seconds']:.6f}"
            )
        return 0

    if args.dollar:
        try:
            spec = load_dollar_yaml(args.dollar)
        except (OSError, ValueError, TypeError) as exc:
            print(f"dollar config error: {exc}")
            return 2
        print(
            f"goodput {__version__} | dollar={spec.name} "
            f"cluster={spec.cluster_size} gpu=${spec.usd_per_gpu_hour:.2f}/hr "
            f"hours={spec.hours:g}"
        )
        print(f"config={spec.path}")
        try:
            result = run_dollar(spec)
        except (OSError, ValueError, TypeError) as exc:
            print(f"dollar error: {exc}")
            return 2
        print(
            f"dollar rows={len(result.rows)} json={result.json_path} "
            f"csv={result.csv_path} table={result.table_path}"
        )
        for row in result.rows:
            print(
                f"  rate={row['failure_rate']:g} "
                f"delta={row['goodput_delta']:+.4f} "
                f"${row['dollar_delta']:,.2f}"
            )
        return 0

    if args.scale:
        try:
            spec = load_scale_yaml(args.scale)
        except (OSError, ValueError, TypeError) as exc:
            print(f"scale config error: {exc}")
            return 2
        print(
            f"goodput {__version__} | scale={spec.name} "
            f"workers={list(spec.worker_counts)} modes={list(spec.ckpt_modes)} "
            f"per_gpu_rate={spec.per_gpu_failure_rate:g}"
        )
        print(f"config={spec.path}")
        try:
            result = run_scale(spec)
        except (OSError, ValueError, TypeError) as exc:
            print(f"scale error: {exc}")
            return 2
        print(
            f"scale cells={len(result.rows)} json={result.json_path} "
            f"csv={result.csv_path} table={result.table_path}"
        )
        if result.plot_path is not None:
            print(f"plot={result.plot_path}")
        for row in result.rows:
            print(
                f"  workers={row['num_workers']} mode={row['ckpt_mode']} "
                f"cluster_rate={row['cluster_failure_rate']:g} "
                f"kill_at={row['kill_at']} goodput={row['goodput']:.4f}"
            )
        if args.plot is not None and result.json_path is not None:
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
        if spec.mode == "fault_hang":
            hang_at = args.fault_at if args.fault_at is not None else spec.fault_at
            if hang_at is None:
                print("fault_hang requires fault_at in YAML or --fault-at")
                return 2
            hang_rank = args.fault_rank if args.fault_rank is not None else spec.fault_rank
            return _run_fault_hang(settings, hang_at_step=hang_at, hang_rank=hang_rank)
        if spec.mode == "fault_bitflip":
            flip_at = args.fault_at if args.fault_at is not None else spec.fault_at
            if flip_at is None:
                print("fault_bitflip requires fault_at in YAML or --fault-at")
                return 2
            flip_rank = args.fault_rank if args.fault_rank is not None else spec.fault_rank
            return _run_fault_bitflip(settings, flip_at_step=flip_at, flip_rank=flip_rank)
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

    if args.fault_hang:
        if args.ckpt_dir is None:
            print("--fault-hang requires --ckpt-dir")
            return 2
        if args.fault_at is None:
            print("--fault-hang requires --fault-at")
            return 2
        hang_rank = args.fault_rank if args.fault_rank is not None else 1
        return _run_fault_hang(
            settings,
            hang_at_step=args.fault_at,
            hang_rank=hang_rank,
        )

    if args.fault_bitflip:
        if args.fault_at is None:
            print("--fault-bitflip requires --fault-at")
            return 2
        flip_rank = args.fault_rank if args.fault_rank is not None else 1
        return _run_fault_bitflip(
            settings,
            flip_at_step=args.fault_at,
            flip_rank=flip_rank,
        )

    if args.train:
        use_store = args.ckpt_dir is not None
        return _run_train(settings, use_checkpoint_store=use_store)

    print(
        "Pass --train, --fault-kill, --fault-hang, --fault-bitflip, --config, "
        "--sweep, --latency, --dollar, --scale, or --plot; see docs/phase-1-tickets.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
