# What I learned

Weekly portfolio log. Fill at end of each week. Keep it honest and specific.

---

## Cumulative takeaways (through Week 2)

Worth saying in an interview, not just a changelog:

1. **Goodput is the product.** The toy MLP is disposable. The deliverable is measurable useful-compute fraction under failure — the same loop Meta/Google care about at cluster scale (MTBF collapses as GPU count grows; checkpoint frequency is an explicit trade-off).
2. **Synchronized training changes the failure model.** One dead worker stalls everyone waiting on all-reduce. That’s why kill → restore → resume is infra, not “retry the job from zero.”
3. **Honesty about fidelity.** We use a **toy barrier + shared-tensor all-reduce**, not full PyTorch DDP yet. Saying that out loud is better than pretending `torchrun` when CI runs on CPU spawn.
4. **Provider interfaces are a CI strategy.** Mocks for checkpoint/fault/metrics mean PRs never need GPUs, SIGKILL, or multi-GB downloads — same pattern as swappable OCR providers in a product repo, adapted for ML.
5. **Scope discipline.** No DB, no frontend, no FastAPI for MVP. Artifacts are JSON/`.pt` files. That kept Phase 0–1 about the metric, not a SaaS shell.
6. **Resume correctness ≠ weight restore.** Restoring model + optimizer + step is necessary but not sufficient; the **data iterator position** must match an uninterrupted run or you silently train on the wrong batches.
7. **Process hygiene is part of the portfolio.** Numbered tickets, Done-when criteria, protected `main`, CI on every PR, weekly learned log — the story is “I can run an infra project,” not only “I wrote training code.”
8. **Automation helps and conflicts.** Bugbot caught a real medium bug (batch stream on resume) and also raced a local fix (divergent commits / rejected push). Learn to verify remote, then rebase/reset — don’t force-push blindly.

---

## Week 1 — Phase 0 Foundation

**Shipped:**

- Private GitHub repo (`cletusabumah/goodput`), docs scaffold (vision, master plan, MVP, ML strategy, architecture, testing, privacy, tickets)
- Stack locked in writing: PyTorch CPU-first, pydantic-settings, Docker Compose later, **no DB / no UI / no serving API** for MVP
- Importable package, `.env.example`, `gdone` weekly todos, GitHub Actions CI (ruff + pytest)
- Branch protection via ruleset (PR required, `lint-and-test` required, no force-push/delete on default branch)
- Synthetic-only data protocol (no PII, no real datasets in git)

**Hard problem:**

- GitHub **rulesets** vs classic branch protection: first ruleset had empty target (`include: []`) → “does not target any resources”; second ruleset included an **`update`** rule that blocked *all* updates to `main` including PR merges, with no bypass actors. Fix: target `~DEFAULT_BRANCH` / `main`, keep PR + status checks, remove restrict-updates (or add admin bypass).

**Metrics / evidence:**

- Fresh clone → venv (Python ≥3.11) → `pip install -r requirements.txt && pip install -e .` → `pytest -q` in &lt;30 minutes goal
- CI green on chore smoke PR; merge path proven under protection

**Blockers:**

- Initial `gh auth login --web` timed out; private remote created later after interactive auth
- System Python 3.9 insufficient (`requires-python >=3.11`); local venv needed 3.13 via Anaconda

**Process / tooling lessons:**

- `done` is a shell reserved word → alias **`gdone`**
- `.gitignore` `data/` was too broad and ignored `src/goodput/data/` — anchor with `/data/`
- Phase 0 before features: docs + Done-when beats a giant “initial commit” of training code

**Next week focus:**

- Phase 1 tickets **1.1–1.5** (providers → naive checkpoint)

---

## Week 2

**Shipped:**

- **1.1 Providers** — `CheckpointStore` / `FaultInjector` / `MetricsSink`; mocks + `LocalFsCheckpointStore`; `build_providers()` forces mocks when `ci_mode=1`; `ProcessFaultInjector` dry-run by default (real SIGKILL deferred to 1.6)
- **1.2 Data** — deterministic synthetic Gaussian batches, `SyntheticDataLoader`, train/val split, committed `synthetic_gaussian_batch_n8.pt` (~3KB), regenerate script; constructor validation for `num_batches >= 1` (Bugbot)
- **1.3 Train** — `ToyMLP` + `train_steps` / `train_from_settings`; CLI `goodput-run --train --steps N`; loss finite and tends to decrease on fixture
- **1.4 Multiprocess** — `spawn` workers, barrier, toy all-reduce via shared grad tensor; sharded data by rank; CLI `--workers 2`
- **1.5 Checkpoints** — naive full dump (model + optimizer + step) to `.pt`; `resume_after_crash` soft-kill path; CLI `--ckpt-dir`

**Hard problems:**

1. **Resume batch-stream bug (medium, Bugbot):** after restore, global step advanced from `start_step` but the cycled loader restarted at batch 0 → wrong data vs uninterrupted training even when weights matched. Fix: `_batch_stream(..., skip=start_step)`. Related footgun: sizing the resume loader only to `remaining_steps` can change the cycle length; pool size should match an uninterrupted run.
2. **JSON vs tensors:** early `LocalFsCheckpointStore` used JSON — fine for toy dicts, useless for real state_dicts. Switched to `torch.save` / `.pt` when checkpointing became real.
3. **Multiprocess on macOS/CI:** use `spawn` (not `fork`) for pytest-safe child processes; worker entry must be top-level picklable; parent must join/kill hung children with timeouts.
4. **Divergent autofix:** local fix commit and Cursor Agent fix commit both addressed the resume skip → push rejected. Resolution: confirm remote fix correct, `reset --hard` to remote (or rebase), don’t dual-maintain the same patch.

**Metrics / evidence:**

- Suite grew to ~38+ tests; ruff clean on touched paths
- `goodput-run --train --workers 2 --steps 5` → `ok=True` for ranks 0 and 1
- `goodput-run --train --ckpt-dir ...` then re-run → `resumed_from=<last_ckpt>`
- Bugbot risk labels: data-loader **low** (synthetic only); multiprocess **medium** (process/shared memory — expected, not a veto); concrete findings fixed before merge

**Other lessons:**

- PyTorch warns if NumPy missing (`Failed to initialize NumPy`) — add `numpy` explicitly; warning ≠ test failure
- Banner `workers=2` while running single-process path was confusing until `--workers` actually switched implementations — CLI flags should match behavior
- Soft-crash (discard memory, reload ckpt) is the right Done-when for 1.5; **real SIGKILL** is a separate ticket (1.6) so CI stays deterministic
- Provider/factory pattern paid off immediately: CI never needs process kills or GPU
- Ticket-sized PRs (1.1 → 1.5) stayed reviewable; Bugbot comments were actionable because diffs were small

**Blockers:**

- Push rejection from divergent resume-fix commits (resolved by adopting remote)
- None that stopped the Week 2 exit criteria (N workers + naive ckpt + resume step match)

**Next week focus:**

- **1.6** SIGKILL fault injector on a schedule  
- **1.7** Goodput + ckpt timing metrics JSON  
- **1.8** Experiment YAML + `goodput-run --config`  
- **1.9** CI training smoke (≤10 steps, fail on NaN)  
- Then Phase 2: fast/incremental ckpt A/B and goodput-vs-failure-rate chart

---

## Week 3

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 

---

## Week 4

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 

---

## Week 5

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 

---

## Week 6

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 

---

## Week 7

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 

---

## Week 8

**Shipped:**

- 

**Hard problem:**

- 

**Metrics / evidence:**

- 

**Blockers:**

- 

**Next week focus:**

- 
