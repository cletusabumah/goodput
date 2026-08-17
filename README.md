# Goodput

**Distributed training fault-injection & checkpoint simulator**

A solo, portfolio-grade ML infrastructure project that measures **goodput** — the fraction of GPU-hours that produce useful, un-lost training progress — under injected worker failures, with and without fast incremental checkpointing.

Inspired by Meta’s Llama cluster interruption story and Google’s TPU “goodput” problem: at large GPU counts, hardware failure is routine, and a naive restart-from-scratch job wastes enormous compute.

## One-paragraph stack rationale

**PyTorch** (CPU-first, optional CUDA) provides a realistic DDP-style training loop without requiring a production cluster. **Multi-process workers** (local spawn + Docker Compose “nodes”) simulate sharded training so we can SIGKILL / hang / corrupt a worker mid-run. **pydantic-settings** keeps every knob env-driven. **JSON experiment configs + artifact reports** replace a database and frontend for MVP — the deliverable is measured goodput curves, not a SaaS UI. **CI runs on CPU with mock providers and tiny fixtures**; no weight downloads, no GPUs required for green PRs. Local laptop for development; **Google Colab** for optional GPU demos; cloud VMs only if multi-node timing fidelity becomes a stretch goal.

## Core loop

```
config → spawn N workers → train + periodic checkpoint
       → inject failure (kill / hang / bit-flip)
       → detect → restore → resume
       → emit goodput + ckpt latency + wasted GPU-hours report
```

## MVP metrics (must ship)

| Metric | Definition |
|--------|------------|
| **Goodput %** | useful compute time ÷ wall-clock time |
| **Checkpoint save time** | wall time to persist model + optimizer state |
| **Checkpoint restore time** | wall time from failure detect → resumed step |
| **Wasted GPU-hours** | (wall-clock − useful) × worker count, at each injected failure rate |

## What we intentionally skip (MVP)

- **No database** — results land in `artifacts/` as JSON/CSV.
- **No frontend / FastAPI serving** — CLI + report files; plot with `goodput-run --plot`.
- **No real datasets or PII** — synthetic toy tensors only.

See [`docs/mvp-spec.md`](docs/mvp-spec.md) and [`docs/vision.md`](docs/vision.md).

## Fresh clone (< 30 minutes)

Requires **Python 3.11+** (CI uses 3.11; local may use 3.12/3.13).

```bash
git clone git@github.com:cletusabumah/goodput.git
cd goodput
python3 -m venv .venv   # or: python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
cp .env.example .env
pytest -q
python -c "from goodput import __version__; print(__version__)"
```

Optional shell alias for weekly todos:

```bash
./scripts/setup-shell.sh
exec zsh
gdone who cletus
gdone status
```

## Plots

Goodput vs injected failure rate, one line per checkpoint mode (ticket 2.3):

```bash
pip install -e ".[viz]"
goodput-run --sweep experiments/sweep.yaml --plot
```

PNG lands at `artifacts/plots/goodput_vs_failure_rate.png` (gitignored). Full recipe: [`docs/testing.md`](docs/testing.md).

## Compose cluster

Two CPU worker containers + a host SIGKILL script (ticket 2.4). Recipe: [`docker/README.md`](docker/README.md).

```bash
docker compose -f docker/compose.yaml up --build
./docker/kill-worker.sh
```

## Docs

Start here: [`docs/README.md`](docs/README.md) → [`docs/master-plan.md`](docs/master-plan.md).

## Status

**Phase 0 — Foundation.** Docs + skeleton + smoke test + CI. Feature work begins after Phase 0 confirmation.

## License / visibility

Private repository. Owner: **Cletus Abumah** (`cletusabumah`).
