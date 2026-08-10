"""CLI entrypoint — single-process train smoke (ticket 1.3); full YAML in 1.8."""

from __future__ import annotations

import argparse

from goodput import __version__
from goodput.config import get_settings
from goodput.training import train_from_settings


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
        "--train",
        action="store_true",
        help="Run single-process toy training (ticket 1.3)",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    settings = get_settings()
    if args.steps is not None:
        settings = settings.model_copy(update={"steps": args.steps})

    print(
        f"goodput {__version__} | device={settings.device} "
        f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
    )

    if args.config:
        print(f"config={args.config} (YAML runner lands in ticket 1.8)")
        return 0

    if args.train:
        result = train_from_settings(settings)
        print(
            f"train ok={result.ok} steps={result.steps_completed} "
            f"final_loss={result.final_loss:.6f} device={result.device}"
        )
        return 0 if result.ok else 1

    print("Pass --train for a single-process smoke run, or see docs/phase-1-tickets.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
