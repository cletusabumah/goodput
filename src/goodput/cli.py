"""CLI entrypoint — train + naive checkpoints (tickets 1.3–1.5)."""

from __future__ import annotations

import argparse
from pathlib import Path

from goodput import __version__
from goodput.config import get_settings
from goodput.providers import LocalFsCheckpointStore
from goodput.training import (
    MultiProcessResult,
    train_from_settings,
    train_multiprocess_from_settings,
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
        help="Enable naive checkpoints under this directory (single-process)",
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
    if updates:
        settings = settings.model_copy(update=updates)

    print(
        f"goodput {__version__} | device={settings.device} "
        f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
    )

    if args.config:
        print(f"config={args.config} (YAML runner lands in ticket 1.8)")
        return 0

    if args.train:
        if settings.num_workers <= 1:
            store = None
            if args.ckpt_dir is not None or settings.ckpt_interval > 0:
                # Only persist when user asks via --ckpt-dir (avoids cluttering cwd).
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
            return 0 if result.ok else 1

        mp_result = train_multiprocess_from_settings(settings)
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
        return 0 if mp_result.ok else 1

    print("Pass --train for a train run, or see docs/phase-1-tickets.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
