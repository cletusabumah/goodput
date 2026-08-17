# Docker Compose simulated nodes (ticket 2.4)

Two worker containers by default (`worker-0`, `worker-1`), optional four via profile `four`. Each container is **one single-process rank**. They share `./artifacts` so rank 0 can checkpoint while a host script SIGKILLs another node.

This is **not** cross-container DDP. The lockstep barrier + toy all-reduce (tickets 1.4 / 1.6) still runs as local processes. Compose is the visual cluster: named nodes, a shared volume, and a documented kill.

Default CI and `pytest` do **not** start Compose.

## Prerequisites

The `docker` CLI must be on your `PATH`. `zsh: command not found: docker` means no engine is installed, not that Compose is misconfigured.

Pick **one** (Apple Silicon / Homebrew) — not both:

```bash
# Docker Desktop (GUI + `docker` + `docker compose`)
brew install --cask docker
open /Applications/Docker.app   # wait until the whale is idle, then:
docker info

# Lighter: Colima VM + CLI plugins
brew install colima docker docker-compose
colima start
docker info
```

Then retry the recipe. `./docker/kill-worker.sh --dry-run` works without Docker; `up --build` does not.

## Recipe

From the repo root (Docker Desktop / Engine required):

```bash
cd /path/to/goodput
rm -rf artifacts/checkpoints/compose artifacts/reports/compose-worker-0 artifacts/reports/compose-worker-1
mkdir -p artifacts/checkpoints/compose
docker compose -f docker/compose.yaml up --build
```

In another terminal, after **both** workers finish training (reports appear under
`artifacts/reports/compose-worker-*/`):

```bash
./docker/kill-worker.sh
# preview only:
./docker/kill-worker.sh --dry-run
```

`kill-worker.sh` waits for those reports before SIGKILL. Do not kill on the first
`step_*.pt` — that lands at step 10 of 400 and would interrupt worker-1 mid-run.

Four nodes:

```bash
docker compose -f docker/compose.yaml --profile four up --build
```

Tear down:

```bash
docker compose -f docker/compose.yaml --profile four down
```

## What you should see

- Rank 0 writes `artifacts/checkpoints/compose/step_*.pt` (and `latest.pt`).
- Rank ≥ 1 trains but does not write checkpoints (avoids clobbering `latest.pt` on the shared volume); its log shows `ckpt_save_s=0`.
- Each rank writes `artifacts/reports/compose-worker-<rank>/report.json` when train finishes.
- After train, workers `sleep infinity` so they stay up for the kill demo.
- `./docker/kill-worker.sh` waits for **both** reports, then SIGKILLs `worker-1` (status **137**). `worker-0` stays **Up**.

In-container command: wipe this rank’s report dir (and rank-0 ckpts), run
`goodput-run --config experiments/compose.yaml --run-name compose-worker-$GOODPUT_RANK`,
then `sleep infinity`.

## Files

| Path | Role |
|------|------|
| `Dockerfile` | CPU image; `pip install` the package |
| `docker/compose.yaml` | `worker-0` … `worker-3` (2 and 3 behind profile `four`) |
| `docker/kill-worker.sh` | Wait for ckpt, SIGKILL a service |
| `experiments/compose.yaml` | Short CPU train; `num_workers: 1` |
