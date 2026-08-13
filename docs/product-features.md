# Product features by persona

This is an **infra simulator**, not a multi-sided product. Personas below keep scope honest.

## Persona A — Cletus (operator)

| Feature | Priority | Notes |
|---------|----------|-------|
| One-command short run | Must | `goodput-run --config …` |
| Kill a worker mid-run | Must | Scripted SIGKILL |
| JSON goodput report | Must | Machine-readable |
| Compose multi-node demo | Must | Portfolio realism |
| Plot recipes | Must | `goodput-run --plot` (matplotlib extra) |
| Colab GPU demo | Stretch | Optional |

## Persona B — Interviewer (consumer of evidence)

| Feature | Priority | Notes |
|---------|----------|-------|
| Goodput vs failure rate chart | Must | Before/after ckpt modes |
| Ckpt latency vs workers | Must | Shows trade-off |
| Dollar impact estimate | Stretch → Phase 3 must | Back-of-envelope, labeled |
| Failure mode variety | Stretch | Hang + bitflip |

## Persona C — Future collaborator

| Feature | Priority | Notes |
|---------|----------|-------|
| Fresh-clone <30 min | Must | Phase 0 |
| Mock providers for CI | Must | No GPU required |
| Numbered tickets + DoD | Must | Discipline |
| Weekly learned log | Must | Portfolio narrative |

## Non-features (explicitly not building)

- User accounts, billing, multi-tenant SaaS
- Mobile app / React dashboard (MVP)
- Production scheduler integration (Slurm/K8s) beyond docs mentions
