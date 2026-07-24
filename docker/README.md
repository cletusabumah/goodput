# Docker Compose simulated nodes (Phase 2.4)

Placeholder. Planned services:

- `worker-0` … `worker-N` — run `goodput-run` with rank env vars
- Shared volume for checkpoints
- Sidecar or host script that SIGKILLs a worker on a schedule

Do not require Compose for unit tests or default CI.
