"""Data package — synthetic loaders only (no raw datasets in git)."""

from goodput.data.synthetic import (
    Batch,
    SyntheticDataLoader,
    generate_synthetic_batch,
    iter_synthetic_batches,
    load_batch_fixture,
    save_batch_fixture,
    train_val_split,
)

__all__ = [
    "Batch",
    "SyntheticDataLoader",
    "generate_synthetic_batch",
    "iter_synthetic_batches",
    "load_batch_fixture",
    "save_batch_fixture",
    "train_val_split",
]
