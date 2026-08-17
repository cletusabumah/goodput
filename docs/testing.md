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
- Required keys documented in `ml-strategy.md`
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

- Fresh-clone test after setup-affecting PRs

## Anti-patterns

- Notebooks as the only training logic
- Skipping tests because “it’s research”
- Downloading datasets/weights in CI
- Bundling unrelated commands into one opaque CI step
