# Goodput Master Plan — Phase 0 → Ship

We work through these steps in order. This is the build plan from first commit to shipped metrics.

**Start date:** Week of Phase 0 merge  
**Target duration:** 6–8 weeks  
**Team:** Cletus Abumah (solo), ~10–15 hrs/week  
**Goal:** Measurable goodput before/after fast checkpointing, plus a reproducible evaluation story

**Owner in all tables:** **Cletus**

Track Week 1 in `todos/week-01.json` — run `gdone <task-id>` when finished. See [`todos.md`](todos.md).

Do not skip Phase 0. Parallel work inside a week is fine when dependencies allow.

---

## Phase 0 — Foundation (Week 1)

| Step | Task | Owner | Done when |
|------|------|-------|-----------|
| 0.1 | Initialize private GitHub repo (`cletusabumah/goodput`); protect `main` | Cletus | Repo live, 2FA on, branch protection checklist started |
| 0.2 | Stack + scope decisions written (no DB/UI; CPU-first PyTorch; Compose) | Cletus | Documented in README + `mvp-spec.md` with rationale |
| 0.3 | Scaffold `src/goodput`, tests, experiments, docs, todos | Cletus | Package importable; layout matches README |
| 0.4 | `.env.example`, fresh-clone section, pydantic-settings config | Cletus | New clone productive in <30 minutes |
| 0.5 | CI: ruff lint + pytest on every PR | Cletus | Green check on a test PR |
| 0.6 | Test-data + privacy protocol (synthetic only) | Cletus | `test-data-protocol.md` + `privacy-security.md` populated |
| 0.7 | Phase 1 tickets + Week 1 todos written | Cletus | `phase-1-tickets.md` + `todos/week-01.json` |

**Exit criteria:** Private repo exists, docs populated, smoke pytest green, CI green on a PR, Phase 1 tickets ready. **No feature training code required beyond hello-world.**

---

## Phase 1 — Toy trainer + naive checkpoint + kill injection (Weeks 2–3)

*Prove: multi-worker run survives a SIGKILL via restore-from-checkpoint, and emits a goodput number.*

| Step | Task | Owner | Done when |
|------|------|-------|-----------|
| 1.1 | Provider interfaces: CheckpointStore, FaultInjector, MetricsSink + mocks | Cletus | Unit tests pass with mocks (no GPU) |
| 1.2 | Synthetic data loader + tiny fixture tensors | Cletus | `test_data_loaders.py` green on fixtures |
| 1.3 | Toy model + single-process train loop (N steps) | Cletus | Loss decreases on fixture; smoke test |
| 1.4 | Multi-process worker launcher (local spawn) | Cletus | N=2 workers complete a short run |
| 1.5 | Naive full checkpoint save/restore | Cletus | Kill process, restore, resume step matches |
| 1.6 | Fault injector: SIGKILL schedule | Cletus | Scripted kill mid-run; exit code / recovery path tested |
| 1.7 | Goodput + ckpt timing metrics writer (JSON) | Cletus | Report file contains required fields |
| 1.8 | Experiment YAML baseline (`experiments/baseline.yaml`) | Cletus | `goodput-run --config ...` produces report |
| 1.9 | Integration smoke in CI (≤10 steps, mock faults optional) | Cletus | `ml-ci` or job step finishes <2 min |

**Exit criteria:** Demo: 2 workers, kill one, resume from checkpoint, print goodput %.

---

## Phase 2 — Fast incremental checkpoint + A/B goodput curve (Weeks 4–5)

*Prove: faster checkpointing improves measured goodput under the same failure schedule.*

| Step | Task | Owner | Done when |
|------|------|-------|-----------|
| 2.1 | Incremental / async-friendly checkpoint path (even if simplified) | Cletus | Save time measurably lower than naive on fixture |
| 2.2 | Sweep runner: failure rate × ckpt mode | Cletus | Matrix of runs writes comparison CSV/JSON |
| 2.3 | Plot: goodput vs failure rate (with/without fast ckpt) | Cletus | Figure saved under `artifacts/` (gitignored) + recipe in docs |
| 2.4 | Docker Compose multi-node simulation (2–4 services) | Cletus | Compose up + kill script works on documented path |
| 2.5 | Checkpoint/restore latency vs worker count table | Cletus | Table in evaluation report |

**Exit criteria:** Before/after goodput chart exists and is reproducible from configs.

---

## Phase 3 — Extra failure modes + dollar narrative (Weeks 6–7)

| Step | Task | Owner | Done when |
|------|------|-------|-----------|
| 3.1 | Hang injector + health-check timeout | Cletus | Hang detected without relying on process exit |
| 3.2 | Bit-flip injector (gradient corruption) stub + optional detector | Cletus | Injected corruption changes loss trajectory in test |
| 3.3 | Dollar-impact script (public $/GPU-hr × goodput delta) | Cletus | Markdown/JSON estimate from measured deltas |
| 3.4 | Reproducibility pack: SHA, versions, config hash in reports | Cletus | Two runs with same seed match within tolerance |
| 3.5 | Portfolio polish: architecture diagram, README demo script, learned log | Cletus | Ready for a public demo walkthrough |

**Exit criteria:** Three-story portfolio: goodput curve, scale latency, dollar model.

---

## Phase 4 — Stretch (Week 8+, optional)

| Step | Task | Owner | Done when |
|------|------|-------|-----------|
| 4.1 | Goodput vs worker count (MTBF-at-scale illustration) | Cletus | Plot + short write-up |
| 4.2 | Colab GPU demo notebook | Cletus | Notebook runs with documented settings |
| 4.3 | Tracker provider (MLflow or WandB) behind interface | Cletus | `GOODPUT_TRACKER=mlflow` logs a run |
| 4.4 | Compare against `torch.distributed.checkpoint` | Cletus | Side-by-side latency note |

**Exit criteria:** At least one stretch item shipped or explicitly deferred with reason.

---

## Phase map summary

| Phase | Weeks | Outcome |
|-------|-------|---------|
| 0 | 1 | Repo + docs + CI |
| 1 | 2–3 | Kill → restore → goodput number |
| 2 | 4–5 | Fast ckpt A/B chart |
| 3 | 6–7 | Hang/bitflip + dollar story |
| 4 | 8+ | Stretch scale / Colab / tracker |

## Dependency rule

Every PR maps to a numbered step (`feat/1.5-naive-checkpoint`). No “misc refactor” PRs without a ticket.
