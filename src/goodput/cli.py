"""CLI entrypoint stub — Phase 1 wires real runs."""

from __future__ import annotations

import argparse

from goodput import __version__
from goodput.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goodput-run", description="Goodput simulator")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML (Phase 1+)")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    settings = get_settings()
    print(
        f"goodput {__version__} | device={settings.device} "
        f"workers={settings.num_workers} ckpt_mode={settings.ckpt_mode}"
    )
    if args.config:
        print(f"config={args.config} (run loop not implemented until Phase 1)")
    else:
        print("Phase 0 hello-world OK. Implement train loop in Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
