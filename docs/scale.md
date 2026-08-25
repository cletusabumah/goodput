# Goodput vs worker count (ticket 4.1)

**MTBF-at-scale illustration.** Independent GPU failures make a synchronized job fail more often as N grows: cluster interruption rate ≈ `N × per_gpu_failure_rate`, so implied MTBF shrinks as ~1/N. This is the same story as “a 16k-GPU job interrupts every few hours even if one GPU is reliable.”

## What we measure

```bash
goodput-run --scale experiments/scale.yaml
```

Each cell is `(ckpt_mode, N)`:

1. `cluster_failure_rate = N × per_gpu_failure_rate` (default `0.125` crashes/step — **toy**, not hardware MTBF).
2. That rate maps to one durable crash step via the Phase 2 sweep helper, then soft-crash + resume.
3. Training stays **single-process**. N only (a) sets when the crash lands and (b) multiplies wasted GPU-hours. We do not spawn N ranks here — that is the latency table (ticket 2.5).

## What to look at

| File | Role |
|------|------|
| `artifacts/sweeps/goodput-vs-workers/table.md` | Write-up + numbers |
| `scale.json` / `scale.csv` | Machine-readable rows |
| `goodput_vs_workers.png` | Plot (needs `pip install -e '.[viz]'`) |

Larger N should crash **earlier** (`kill_at` drops) and goodput should not rise. Incremental vs naive is the same checkpoint trade-off as the Phase 2 curve, sliced by cluster size.

## Honesty

Toy millisecond walls are not Meta-scale MTBF. The **shape** (more workers → more frequent interruptions → lower goodput unless restore is cheap) is the claim. Dollar scaling of these deltas is ticket 3.3, still labeled not-a-quote.
