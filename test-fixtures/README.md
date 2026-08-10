# Test fixtures

Tiny **synthetic** artifacts only (KB scale). No real datasets, weights, or cluster logs.

| File | Description |
|------|-------------|
| `synthetic_gaussian_batch_n8.pt` | 8×16 Gaussian batch, seed=42 (regression targets) |

## Regenerate

From the repo root (venv active, package installed):

```bash
python scripts/generate_fixtures.py
# optional overrides:
python scripts/generate_fixtures.py --batch-size 8 --input-size 16 --seed 42
```

Then re-run:

```bash
pytest -q tests/test_data_loaders.py
```
