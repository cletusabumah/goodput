# MVP Spec — Goodput Simulator

**Timeline:** ~6–8 weeks (shortest schedule that still yields the three portfolio charts).  
**Owner:** Cletus Abumah (solo).  
**Start:** Phase 0 complete → confirm → Phase 1.

## One-line core loop

```
config → multi-worker toy train → checkpoint → inject fault → recover → goodput report
```

---

## Must-haves (ship blockers)

1. **Multi-process toy trainer** — N workers, synthetic batches, synchronized-enough steps (PyTorch DDP or a from-scratch barrier/all-reduce stub that is honest about what it simulates).
2. **Periodic checkpointing** — save model + optimizer (+ step) to disk; restore and resume after failure.
3. **Fault injection script** — at minimum **SIGKILL** a random worker on a schedule; stubs for hang + bit-flip.
4. **Primary metrics JSON per run** — goodput %, ckpt save time, ckpt restore time, wasted GPU-hours proxy.
5. **A/B experiment configs** — same workload with `ckpt_mode=naive` vs `ckpt_mode=incremental` (incremental may start as a faster mock path, then become real).
6. **Docker Compose “nodes”** — at least 2–4 containers acting as workers for demo realism (local processes acceptable for unit tests).
7. **CI** — lint + unit + smoke train (≤ few steps, CPU, mocks) under 5 minutes.
8. **Reproducibility** — seed, config hash, git SHA in every report.

## Stretch (if ahead)

- Hang failure mode with active health-check / timeout (not just crash).
- Silent **bit-flip** in a gradient + detection story (checksum / redundant compare).
- Goodput vs **worker count** curve (MTBF-at-scale narrative).
- Dollar-impact notebook/script using public GPU rental rates.
- Optional Colab notebook that runs a GPU demo end-to-end.
- Experiment tracker stub (MLflow or WandB interface; default `none`).

## Post-MVP

- Real multi-node cloud run (2+ VMs) for timing fidelity.
- Asynchronous / pipeline-parallel failure modes.
- Frontend results explorer (only if CLI reports become painful).
- Database for experiment catalog (only if file-based runs become unmanageable).
- Integration with a real checkpoint library (e.g. torch.distributed.checkpoint) for comparison.

---

## Explicit non-goals (MVP)

| Discarded from SaaS reference | Why |
|-------------------------------|-----|
| Postgres / Alembic | No multi-user state; artifacts are files. |
| React frontend | Deliverable is metrics, not UI. |
| FastAPI prediction API | This is a batch simulator, not an inference service. |
| Real datasets / model zoo downloads | CI must stay fast and offline-friendly. |
| User auth / PII pipelines | No human data. |

---

## Decisions locked for Phase 0

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data | Synthetic tensors only | No PII, no consent issues, CI-safe |
| Deployment | Local + Docker Compose; Colab optional | Reproducible demos without cloud spend |
| GPU access | CPU default; CUDA/Colab optional | CI + laptops work; GPU is a demo accelerator |
| Database | **None (MVP)** | JSON reports suffice |
| Frontend | **None (MVP)** | Charts via script/notebook |
| Serving API | **None (MVP)** | CLI entrypoint `goodput-run` |
| Tracker | Interface stub; default `none` | Avoid vendor lock-in early |

## Open decisions (need confirmation)

See response summary — mainly: DDP vs toy from-scratch for Phase 1, and exact goodput formula edge cases (how to count hang time, warm-up, etc.).
