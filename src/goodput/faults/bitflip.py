"""Simulated gradient bit-flip + optional cross-rank detector (ticket 3.2)."""

from __future__ import annotations

import torch

# Shared-memory sentinels (same pattern as hang_rank idle).
BITFLIP_RANK_IDLE = -1
BITFLIP_STEP_IDLE = -1


def flip_float32_bit(flat: torch.Tensor, *, index: int = -1) -> int:
    """
    Flip one bit in a float32 gradient vector (simulated silent corruption).

    XORs the IEEE-754 sign bit of ``flat[index]`` so the update direction changes
    at one parameter while magnitude stays the same, then propagates through the
    all-reduce into every worker's step. Returns the flat index that was flipped.
    """
    if flat.numel() == 0:
        raise ValueError("cannot flip bit on empty gradient")
    idx = int(index) % flat.numel()
    view = flat.detach().view(torch.int32)
    # Flip the sign bit — one IEEE-754 bit flip with a visible training effect.
    view[idx] = view[idx] ^ (1 << 31)
    return idx


def detect_gradient_outlier(
    grad_bucket: torch.Tensor,
    *,
    ratio_threshold: float = 8.0,
) -> bool:
    """
    Optional detector for gradient corruption before all-reduce averaging.

    Bit-flip demos use identical batches on every rank so local grads match until
    one rank is corrupted; a sign-bit flip desyncs that rank without changing its
    L2 norm. We therefore check peer desync from rank 0 first, then fall back to
    the classic per-rank norm outlier rule for sharded or synthetic spikes.
    """
    if grad_bucket.shape[0] < 2:
        return False
    if ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must be > 1")

    ref = grad_bucket[0]
    peer_diffs = (grad_bucket[1:] - ref).norm(dim=1)
    max_diff = float(peer_diffs.max()) if peer_diffs.numel() else 0.0

    norms = grad_bucket.norm(dim=1)
    max_norm = float(norms.max())
    min_norm = float(norms.min())
    norms_match = max_norm > 0 and (max_norm - min_norm) / max_norm <= 1e-5

    # Sign-bit corruption keeps every rank's L2 norm identical but desyncs one row.
    if norms_match and max_diff > 1e-5:
        return True

    median = float(norms.median())
    if median <= 0:
        return bool(norms.max() > 0)
    return bool((norms > ratio_threshold * median).any())
