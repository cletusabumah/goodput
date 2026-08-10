# Test data protocol

## What may be committed

| Allowed in git | Examples |
|----------------|----------|
| Tiny synthetic fixtures (KB, not GB) | `test-fixtures/tiny_batch.pt`, small JSON configs |
| Experiment YAML/JSON configs | `experiments/baseline.yaml` |
| Docs, diagrams, scripts | Everything under `docs/`, `scripts/` |
| Golden **metric schemas** / tiny expected JSON snippets | e.g. shape of report, not full run dumps |

## What must stay local

| Keep local / gitignored | Why |
|-------------------------|-----|
| `artifacts/`, `checkpoints/`, `runs/` | Large, regenerable |
| Real cloud logs, host inventories | May contain sensitive infra info |
| `.env` | Secrets |
| WandB/MLflow caches | Vendor local state |
| Any non-synthetic dataset | Out of scope and policy |

## Naming

- Fixtures: descriptive, e.g. `synthetic_gaussian_batch_n8.pt`
- Never use real employee, customer, or cluster names in fixture filenames

## Regenerating fixtures

```bash
source .venv/bin/activate
python scripts/generate_fixtures.py
pytest -q tests/test_data_loaders.py
```

See `test-fixtures/README.md` for filenames and defaults.

## Consent

N/A — no human data. If that ever changes, stop and update this doc + `privacy-security.md` before committing anything.
