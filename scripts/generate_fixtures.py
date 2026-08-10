#!/usr/bin/env python3
"""Regenerate tiny synthetic fixtures under test-fixtures/ (ticket 1.2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from goodput.data import generate_synthetic_batch, save_batch_fixture

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "test-fixtures" / "synthetic_gaussian_batch_n8.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    batch = generate_synthetic_batch(
        batch_size=args.batch_size,
        input_size=args.input_size,
        seed=args.seed,
    )
    path = save_batch_fixture(batch, args.out, seed=args.seed)
    print(f"wrote {path} shape=({batch.batch_size}, {batch.input_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
