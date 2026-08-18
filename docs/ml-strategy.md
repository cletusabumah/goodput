# ML Strategy

## Problem framing

This is **systems ML**, not predictive modeling. The “model” is a disposable toy used to stress checkpointing and recovery. Success is measured in **goodput and recovery latency**, not validation accuracy.

## Baseline

| Piece | Choice |
|-------|--------|
| Model | Tiny MLP (or linear layer) on synthetic Gaussian data |
| Parallelism | Multi-process workers; start with local spawn, then Compose |
| Checkpoint | **Naive full state dump** every K steps |
| Failure | Single SIGKILL mid-run |
| Recovery | Restart workers from last checkpoint |

Baseline answers: *What goodput do we get with dumb checkpointing under failure rate r?*

## Primary metrics

| Metric | Formula / definition | Must for MVP? |
|--------|----------------------|---------------|
| **Goodput** | `useful_compute_time / wall_clock_time` | Yes |
| **Ckpt save time** | wall seconds to persist state | Yes |
| **Ckpt restore time** | detect → first resumed step | Yes |
| **Wasted GPU-hours** | `(wall_clock - useful) * num_workers` (CPU-time proxy when no GPU) | Yes |

### Useful compute time (working definition)

Sum of time spent in successful training steps whose gradients are **not discarded** by a subsequent failure before the next durable checkpoint. Time spent in failed windows after the last checkpoint does **not** count as useful.

Edge cases (confirm in Phase 1):

- Warm-up / init: exclude from both numerator and denominator, or document inclusion.
- Hang time before timeout: counts as wall-clock, not useful.
- Bit-flip without detection: progress may be “useful but wrong” — track separately as **corrupted steps**.

## Secondary metrics

- Steps completed / steps attempted
- Failures injected vs recoveries succeeded
- Detection latency (hang mode)
- Bytes written per checkpoint
- Loss curve sanity (NaN → fail CI smoke)

## Ablation plan

1. Checkpoint interval K ∈ {5, 10, 20, 50}
2. Checkpoint mode: `naive` vs `incremental`
3. Failure rate / mean interval
4. Worker count N ∈ {2, 4, 8} (stretch: higher)
5. Failure mode: kill vs hang vs bitflip

## Data leakage guards

N/A for predictive leakage — data is synthetic. Guards that **do** apply:

- Never leak real secrets into reports
- Never commit full run artifacts
- Keep CI fixtures tiny and deterministic

## Reproducibility

- `GOODPUT_SEED` + per-worker seed derivation
- Log: git SHA, package versions (+ hash of that map), config hash of training knobs
- Commit experiment YAMLs under `experiments/`
- Reports stay at `artifacts/reports/<run_name>/report.json` with the hash **inside** the JSON (the `{run_name}/{config_hash}/{timestamp}/` layout is still a naming convention for later)

## Model registry / artifacts

| What | Where |
|------|-------|
| Checkpoints | `artifacts/checkpoints/` (gitignored) |
| Reports | `artifacts/reports/` (gitignored) |
| Committed configs | `experiments/*.yaml` |
| Tiny fixtures | `test-fixtures/` |

No external model registry for MVP. Naming convention ready for later: `toy-mlp/v{n}/{config_hash}`.

## When to retrain vs fine-tune

Not applicable. We **re-run experiments** when code or config changes; we do not fine-tune a production model.

## Failure modes & monitoring hooks (placeholders)

| Hook | Intent | MVP status |
|------|--------|------------|
| Worker heartbeat | Detect hangs | Stub → Phase 3 |
| Loss NaN guard | Abort bad runs | Smoke assert in CI |
| Ckpt checksum | Detect corruption | Stretch |
| Drift / latency | N/A serving | Document only |

## Feature / training / serving skew

Serving is out of scope. Documented anyway: all transforms used in training live in `src/goodput/features/` and must be imported by any future inference path — notebooks must not define one-off preprocessing.
