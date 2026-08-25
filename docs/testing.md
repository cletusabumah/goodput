# Testing

## How to run

```bash
source .venv/bin/activate
pytest -q
pytest tests/test_smoke.py -v
ruff check src tests
```

## Test layers

| Layer | Purpose | Runs in CI? |
|-------|---------|-------------|
| Unit | Metrics math, config parsing, provider mocks, transforms | Yes |
| Integration | Loader → batch → one train step; ckpt round-trip | Yes (smoke) |
| API | FastAPI endpoints | **N/A (MVP)** |
| Evaluation | Full metric suite + sweeps | Nightly or manual |
| E2E | Multi-worker + real SIGKILL on Compose | Weekly / manual |
| Migrations | Alembic | **N/A (no DB)** |

## Fixtures

- Tiny synthetic tensors / JSON under `test-fixtures/`
- Temp dirs via `tmp_path` in pytest
- Mock providers in `conftest.py` — **no GPU, no downloads, no SIGKILL in default CI**

## Evaluation harness

- Each run writes `artifacts/reports/<run>/report.json`
- Required keys documented in `ml-strategy.md` (goodput fields **and** ticket 3.4: `git_sha`, `config_hash`, `package_versions`)
- CI smoke asserts: report exists, goodput in `[0, 1]`, loss not NaN

## Manual / weekly

- Docker Compose kill script (ticket 2.4) — see [`docker/README.md`](../docker/README.md):

```bash
rm -rf artifacts/checkpoints/compose artifacts/reports/compose-worker-0 artifacts/reports/compose-worker-1
mkdir -p artifacts/checkpoints/compose
docker compose -f docker/compose.yaml up --build
# other terminal — after both reports exist (not on first step_*.pt):
./docker/kill-worker.sh --dry-run
./docker/kill-worker.sh
```

Default pytest does **not** start Compose.
- Plot regeneration from committed experiment configs (ticket 2.3):

```bash
pip install -e ".[viz]"
goodput-run --sweep experiments/sweep.yaml --plot
# PNG: artifacts/plots/goodput_vs_failure_rate.png  (gitignored)
# Re-plot an existing table:
goodput-run --plot artifacts/sweeps/phase2-sweep/comparison.json
```

- Checkpoint/restore latency vs worker count (ticket 2.5):

```bash
goodput-run --latency experiments/latency.yaml
```

Table: `artifacts/sweeps/latency-table/table.md` (also `latency.json` / `latency.csv`, gitignored). Default pytest does **not** spawn the N=4 recipe; tests use N=1–2.

- Dollar impact from measured goodput deltas (ticket 3.3):

```bash
goodput-run --dollar experiments/dollar.yaml
```

Estimate: `artifacts/sweeps/dollar-impact/table.md` (also `dollar.json` / `dollar.csv`, gitignored). Formula is `cluster_size × public $/GPU-hr × hours × Δgoodput`. Default pytest does **not** run the full sweep; tests feed a tiny comparison fixture. Missing comparison JSON triggers `experiments/sweep.yaml`.

- Reproducibility pack (ticket 3.4): every `report.json` includes `git_sha`, `config_hash`, and `package_versions`. Two same-seed trains must match loss within tolerance (`tests/test_reproducibility.py`).

- Goodput vs worker count (ticket 4.1):

```bash
goodput-run --scale experiments/scale.yaml
```

Write-up: `artifacts/sweeps/goodput-vs-workers/table.md` (also `scale.json` / `scale.csv`, gitignored). Default pytest uses N=1–2, not the N=4 recipe. Model notes: [`docs/scale.md`](scale.md).

- Portfolio demo (ticket 3.5): `./scripts/portfolio-demo.sh` regenerates smoke + sweep + latency + dollar; `./scripts/portfolio-demo.sh --dry-run` for CI.

- Colab GPU demo (ticket 4.2): [`notebooks/colab_gpu_demo.ipynb`](../notebooks/colab_gpu_demo.ipynb). CI does **not** start Colab; `tests/test_colab_demo.py` checks YAML ↔ notebook knobs and a CPU train of the same settings.

- Tracker provider (ticket 4.3): `GOODPUT_TRACKER=mlflow goodput-run --config experiments/tracker.yaml`. Default CI has no MLflow SDK — the tracker writes `artifacts/mlflow/*.json`. Optional: `pip install -e ".[tracker]"` for a real local `file:./artifacts/mlruns` store.

- DCP comparison (ticket 4.4): `goodput-run --dcp-compare experiments/dcp-compare.yaml`. Writes `artifacts/sweeps/dcp-compare/table.md`. Tests use tiny hidden sizes; DCP may be `available: no` if this torch cannot save without a process group.

- Fresh-clone test after setup-affecting PRs

## Anti-patterns

- Notebooks as the only training logic
- Skipping tests because “it’s research”
- Downloading datasets/weights in CI
- Bundling unrelated commands into one opaque CI step
