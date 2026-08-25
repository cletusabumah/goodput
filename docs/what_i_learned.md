# What I learned

Weekly engineering log. Fill at end of each week. Keep it honest and specific.

---

## Cumulative takeaways (through Week 8)

Worth stating clearly, not just as a changelog:

1. **Goodput is the product.** The toy MLP is disposable. The deliverable is measurable useful-compute fraction under failure — the same loop Meta/Google care about at cluster scale (MTBF collapses as GPU count grows; checkpoint frequency is an explicit trade-off).
2. **Synchronized training changes the failure model.** One dead worker stalls everyone waiting on all-reduce. That’s why kill → restore → resume is infra, not “retry the job from zero.”
3. **Honesty about fidelity.** We use a **toy barrier + shared-tensor all-reduce**, not full PyTorch DDP yet. Saying that out loud is better than pretending `torchrun` when CI runs on CPU spawn.
4. **Provider interfaces are a CI strategy.** Mocks for checkpoint/fault/metrics mean PRs never need GPUs, SIGKILL, or multi-GB downloads — same pattern as swappable OCR providers in a product repo, adapted for ML.
5. **Scope discipline.** No DB, no frontend, no FastAPI for MVP. Artifacts are JSON/`.pt` files. That kept Phase 0–1 about the metric, not a SaaS shell.
6. **Resume correctness ≠ weight restore.** Restoring model + optimizer + step is necessary but not sufficient; the **data iterator position** must match an uninterrupted run or you silently train on the wrong batches. Same class of bug: resume **loader pool size** must match uninterrupted `min(steps, 16)`, not grow with `ckpt + remaining`.
7. **Process hygiene is part of the portfolio.** Numbered tickets, Done-when criteria, protected `main`, CI on every PR, weekly learned log — the story is “I can run an infra project,” not only “I wrote training code.”
8. **Automation helps and conflicts.** Bugbot caught real bugs and also raced local fixes (divergent commits / rejected push). Default recovery: `fetch` + `reset --hard origin/<branch>`, then re-apply only what remote missed (e.g. a regression test) — don’t force-push blindly.
9. **Define the metric clocks.** Goodput is only as honest as **wall vs useful**. Warm-up (device/model/loader rebuild) should be excluded from both (ml-strategy); restore/save overhead is wall but not useful. Inconsistent timers between `train_from_settings` and `resume_after_crash` silently understate goodput on the kill path.
10. **Path filters are part of the test plan.** A dedicated ML smoke job that asserts `report.json` must trigger on every package it depends on (`providers/`, `checkpointing/`), not only `training/` — otherwise sink/ckpt regressions skip the job that was meant to catch them.
11. **CLI wiring is product surface.** `use_checkpoint_store` that only applies when `num_workers <= 1` means a committed 2-worker baseline never checkpoints and `ckpt_save_s` stays zero. Feature flags must reach every code path the docs claim to exercise.
12. **Incremental checkpointing is an optimizer-state bet.** The fast path skips Adam (or any heavy optimizer) between full bases; a tiny SGD toy can make naive look as cheap, and goodput need not be higher. Pick the fixture that matches the claim.
13. **Resume is contamination until you isolate the cell.** A sweep that shares `ckpt_dir` will load leftover `latest.pt` and turn `failure_rate=0` into a resume of the previous job. Wipe per cell.
14. **Optional extras keep default CI honest.** Matplotlib lives in `[viz]`; plot tests `importorskip`. Default pytest stays green without chart deps — same idea as mock providers.
15. **Assert the mechanism, not noisy wall-clock.** End-to-end `torch.save` means on GitHub runners inverted a 2.1 “incremental is faster” check (~13× the wrong way). Interleaved in-memory capture medians test clone/serialize; disk I/O is not a Done-when clock.
16. **Compose is demo topology, not DDP.** Two containers are two **single-process** ranks on a shared volume — not cross-container all-reduce. Portfolio realism is named nodes + shared ckpt + host SIGKILL, with the fidelity gap stated in docs.
17. **Shared checkpoint volume needs one writer.** When ranks share `latest.pt`, only rank 0 persists (`GOODPUT_RANK` / `settings.rank`). Otherwise concurrent writers race and clobber the restore pointer — same lesson as multiprocess rank-0-only dumps.
18. **Readiness signals must match the story.** Killing on the first `step_*.pt` landed at step 10 of 400 and skipped worker-1’s report mid-run. Waiting for **post-train reports** aligns the Compose kill demo with “training finished, then failure.”
19. **Latency table ≠ SIGKILL path.** The 2.5 table times restore via the 1.5 soft-resume path per cell (no SIGKILL in the matrix). On a tiny naive dump, save/restore stay roughly flat; train **wall** grows with spawn + barriers as N increases — that’s the honest scale story for MVP.
20. **Hang detection is a liveness check, not an exit code.** The worker is still alive; you need a heartbeat / progress timeout. Kill is “process gone”; hang is “process stuck.” Same split as cluster health checks vs crash handlers.
21. **Silent corruption is a different product than crash.** A sign-bit flip keeps L2 norm, so a magnitude-only detector misses it. Cross-rank agreement (or checksums) is the real hook. Training **continues** — goodput can look fine while the run is poisoned.
22. **Inject races are Done-when failures.** Parent and workers share hang/bitflip slots. Arm hang before rank 0 can publish the next step; pre-set bitflip `(rank, step)` at spawn. If progress advances “during detection,” you did not hang.
23. **Dollar models must stay labeled.** Measured toy Δgoodput × simulated cluster × public list price is back-of-envelope arithmetic, **not a quote**. Millisecond sweep walls will print million-dollar swings of the wrong sign — the artifact is the formula + disclaimer, not the dollar figure.
24. **Repro is three IDs, not one field.** `git_sha` = code, `config_hash` = training knobs, `package_versions` = deps. Same-seed **loss** must match within tolerance; wall-clock is allowed to drift. `git_dirty: true` means “SHA + local edits.”
25. **Hash what changes the experiment, not the laptop.** `config_hash` excludes `artifacts_dir`, `ckpt_dir`, and provider backends so committed YAML hashes the same in CI `tmp_path` and on your machine.
26. **The demo script is the deliverable.** `./scripts/portfolio-demo.sh` regenerates smoke + sweep + latency + dollar from committed YAML; matplotlib stays optional (`[viz]`) like mocks — skip plot, not fail the walkthrough.
27. **N as scale is a failure-rate model, not spawn count.** Ticket 4.1 maps cluster interruption rate to `N × per_gpu_rate` (MTBF ~ 1/N) on the same 1-process crash/resume path as the Phase 2 sweep. Confusing that with the 2.5 latency table (real worker spawn) mixes two different stories.
28. **A Colab demo is documented knobs plus CPU fallback.** The notebook must call `train_from_settings`, not copy a trainer. YAML ↔ notebook lockstep belongs in CI; CI must never start Colab or require a GPU.
29. **Optional SDKs need an offline fallback.** `GOODPUT_TRACKER=mlflow` still has to log a run when `mlflow` is not installed — JSON under `artifacts/mlflow/` — or CI cannot prove the Done-when. Cloud accounts and extras stay optional.
30. **A library comparison is a note, not a rewrite.** Timing naive/incremental `torch.save` against `torch.distributed.checkpoint` on 1-rank CPU is regenerable evidence of the API gap. DCP slower here does not mean DCP loses at 16k GPUs; `available: no` on some torch builds is a documented skip, not a CI failure.

---

## Demo walkthrough (~5 minutes)

Regenerate everything:

```bash
source .venv/bin/activate
./scripts/portfolio-demo.sh
```

### 60-second pitch

At cluster scale, GPUs fail often enough that **restart-from-scratch wastes most of the wall clock**. **Goodput** is the fraction of GPU-hours that actually advance training. This repo is a **scaled-down simulator**: toy multi-process training, periodic checkpoints, inject kill / hang / silent grad corruption, measure goodput and recovery — then scale a measured delta to a **labeled** dollar story. It is infra evidence, not a model zoo.

### Three charts (Phase 3 exit criteria)

| # | Say this | Command | Open this |
|---|----------|---------|-----------|
| 1 | “Incremental checkpointing vs naive under injected failure rate” | `goodput-run --sweep experiments/sweep.yaml --plot` | `artifacts/plots/goodput_vs_failure_rate.png` |
| 2 | “Save/restore latency vs worker count on the soft-resume path” | `goodput-run --latency experiments/latency.yaml` | `artifacts/sweeps/latency-table/table.md` |
| 3 | “Back-of-envelope $ if this Δgoodput happened on a Llama-scale cluster” | `goodput-run --dollar experiments/dollar.yaml` | `artifacts/sweeps/dollar-impact/table.md` |

Every `report.json` also logs **`git_sha`**, **`config_hash`**, and **`package_versions`** so a number is traceable.

### Failure modes (optional 2 minutes)

| Mode | One line | Demo |
|------|----------|------|
| **Kill** | Process gone → restore from ckpt | `goodput-run --config experiments/fault-hang.yaml` is hang; kill via `--fault-kill` or Compose |
| **Hang** | Process alive, progress stalled → timeout detect | `experiments/fault-hang.yaml` |
| **Bit-flip** | Silent grad corruption, training continues | `experiments/fault-bitflip.yaml` |

### Honest caveats (always say one)

- Toy **barrier all-reduce**, not production DDP.
- Compose is **two single-process nodes** on a shared volume, not cross-container sync.
- Sweep matrix uses **soft crash + resume**, not SIGKILL in every cell.
- Dollar table uses **public list $/GPU-hr** — not a quote; toy millisecond walls make the $ magnitude illustrative only.

### Deeper narrative

Weekly build log and cumulative takeaways **1–26** below — pick 2–3 that match the question at hand (sync training, provider mocks for CI, resume iterator bugs, inject races, sign-bit detection, etc.).

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

- **1.6 SIGKILL** — Multiprocess train → wait for durable `step_{k}.pt` → pin `latest.pt` → real `ProcessFaultInjector(dry_run=False)` on one rank → reap barrier-stuck peers → `resume_after_crash`; CLI `--fault-kill`
- **1.7 Metrics report** — Time useful/wall + ckpt save/restore; `build_run_report` / `emit_run_report` with required MVP fields; emit via `MetricsSink` from `--train` / `--fault-kill`
- **1.8 Experiment YAML** — `load_experiment_yaml` → `Settings` + `mode`; `goodput-run --config experiments/baseline.yaml` produces `artifacts/reports/<name>/report.json`; CLI flags override YAML
- **1.9 ML CI smoke** — `experiments/ci-smoke.yaml` (≤10 steps), pytest NaN guard + report checks, real `ml-ci.yml` job (not a placeholder import)

**Hard problems:**

1. **Kill-at-checkpoint races:** Rank 0 can keep training and advance `latest.pt` past the intended kill step. Fix: wait for the *specific* `step_{kill_at}.pt`, then pin `latest` before SIGKILL. Peers hanging on the next barrier are expected — parent must `_reap_all` (try/finally) or the job leaks.
2. **Resume pool-size regression:** `resume_after_crash` briefly sized the batch pool with `max(steps, ckpt+remaining)`, changing cycle length vs uninterrupted `min(steps, 16)` → wrong batches again (same family as Week 2 skip bug). Fix + regression test. Bugbot/local races → prefer reset-to-remote, re-apply only the unique test.
3. **Metric clock consistency:** `train_from_settings` excluded warm-up from wall; `resume_after_crash` counted model/loader rebuild → kill-path reports looked worse than equivalent resume-via-train. Align both to restore + train loop only.
4. **Multiprocess ignored checkpoint flag:** YAML baseline uses `num_workers: 2`, but `use_checkpoint_store` only wired the single-process branch → no `.pt` files, `ckpt_save_s=0`. Fix: pass `ckpt_dir` through `launch_workers` and return rank-0 save timings.
5. **CI path-filter gap:** ML smoke asserts report/ckpt health but initially omitted `providers/**` and `checkpointing/**` from `paths:` — sink/ckpt-only PRs would skip the job. Filters must match the dependency graph of the smoke.

**Metrics / evidence:**

- `goodput-run --fault-kill --workers 2 …` recovers with `resumed_from=<kill_at>`
- `goodput-run --config experiments/baseline.yaml` → report with `goodput`, `ckpt_save_s`, `ckpt_restore_s`, `wasted_gpu_hours` (after MP ckpt fix, save latency &gt; 0)
- `goodput-run --config experiments/ci-smoke.yaml` → `steps_completed=8`, finite `final_loss`, `goodput ∈ [0,1]`
- Phase 1 exit story closed: kill → restore → **number** in JSON, plus one-command YAML + CI smoke

**Other lessons:**

- Required report fields are a contract: tests that only check “file exists” are weaker than asserting keys + ranges
- Warm-up exclusion is a product decision — document it (ml-strategy) and enforce it in every entrypoint that builds a report
- `gdone` todo marks can lag merges if they never land on `main`; treat todo updates as part of the ticket PR when possible
- Small follow-up commits (wall policy, MP ckpt, path filters) after Bugbot/review are normal; keep them focused

**Blockers:**

- Repeated push rejects from Bugbot autofix vs local commits on 1.6 (resolved with reset-to-remote pattern)
- None that blocked Phase 1 Done-when once the MP checkpoint + wall-clock fixes landed

**Next week focus:**

- Phase 2: fast/incremental checkpoint path vs naive  
- Goodput vs injected failure-rate chart (with/without fast ckpt)  
- Keep reports reproducible from committed `experiments/*.yaml`

---

## Week 4 — Phase 2 start (fast ckpt + goodput curve)

**Shipped:**

- **2.1 Incremental ckpt** — Full model+optimizer base every `ckpt_full_every`; model-only deltas in between (`incremental_v1` / `incremental_full_v1`). Restore loads the base via `store.load_at_step` and reattaches optimizer state. Default capture path stays naive.
- **2.2 Sweep runner** — `goodput-run --sweep experiments/sweep.yaml`: matrix of `ckpt_mode` × `failure_rate` → `artifacts/sweeps/<name>/comparison.json` + `.csv`. `failure_rate=0` is uninterrupted; `>0` is one **soft crash** at `kill_at ≈ 1/rate` snapped to `ckpt_interval` (1.5 resume path — no SIGKILL in the matrix).
- **2.3 Goodput plot** — `goodput-run --plot` (or `--sweep … --plot`) writes `artifacts/plots/goodput_vs_failure_rate.png` (gitignored). Recipe in `docs/testing.md`. Matplotlib is `pip install -e ".[viz]"`.

**Hard problems:**

1. **What “faster ckpt” actually is:** Deltas omit optimizer clone/serialize. That is a real win on **Adam**; the committed sweep uses the default **SGD** trainer, so incremental is not guaranteed to beat naive on goodput. Don’t sell the Phase 2 chart as “fast ckpt always wins” on a toy that barely has optimizer state.
2. **Sweep cell isolation:** Rerunning a cell reused `latest.pt` from the previous job → `failure_rate=0` silently resumed. Fix: `_reset_ckpt_dir` (wipe) before every cell + a regression test.
3. **Failure rate is a schedule, not a Poisson process:** One crash per cell, snapped to a durable step with resume room. Honest for a tiny `steps=8` matrix; not a cluster MTBF simulator. Document the mapping (`kill_at_for_rate`) so the x-axis of the plot is interpretable.
4. **CI timing flake vs 2.3:** After `gdone goodput-plot`, lint-and-test failed on the **2.1** disk-save mean check (incremental ~13× *slower*). Unrelated todo JSON; naive-only warmup + mean-of-all-saves (including full dumps) + GHA disk noise. Fix: interleaved in-memory `capture()` medians of **deltas vs naive**, not `torch.save` wall time.
5. **`gdone` re-escaped Unicode:** Default `json.dumps` wrote `\u2014` / `\u00d7` back into `week-04.json`. `save_todos(..., ensure_ascii=False)` + a test so the next mark-done doesn’t undo week-file hygiene.

**Metrics / evidence:**

- Incremental **delta capture median** &lt; naive full-dump median on the Adam fixture (`hidden=256`)
- `experiments/sweep.yaml`: naive + incremental × rates `{0, 0.25, 0.5}` → comparison table with `goodput`, `ckpt_save_s`, `kill_at`, `wasted_gpu_hours`
- `goodput-run --sweep experiments/sweep.yaml --plot` → PNG under `artifacts/plots/`
- Plot tests skip without matplotlib (`ss` on default CI); series grouping always runs

**Other lessons:**

- Soft-crash in the sweep is the right CI trade-off; real SIGKILL stays on the 1.6 / Compose path (2.4)
- A green plot PR can still go red on a later `gdone` push if an unrelated timing test is flaky — read the failing node, not the commit title
- `gdone` is part of the docs pipeline: it rewrites the whole week JSON, not just `done: true`

**Blockers:**

- Transient `Could not resolve host: github.com` on push (retry; HTTPS remote)
- False-alarm CI on the 2.1 timing test after the plot+gdone push — not a 2.3 regression
- None that blocked Phase 2 tickets **2.1–2.3**

**Next week focus:**

- **2.4** Docker Compose multi-node (2–4 services) + documented kill script
- **2.5** Checkpoint/restore latency vs worker count table
- Phase 2 exit: keep the goodput-vs-rate chart regenerable from committed YAML

---

## Week 5 — Phase 2 close (Compose + ckpt latency table)

**Shipped:**

- **2.4 Compose cluster** — `Dockerfile` (CPU torch index), `docker/compose.yaml` (`worker-0` / `worker-1` default; `--profile four` for 2–3), `docker/kill-worker.sh` (`--dry-run`, waits for reports), `experiments/compose.yaml`. Containers share `./artifacts`; each node is **one single-process rank** (`num_workers: 1`), not cross-container DDP. Workers `sleep infinity` after train so the host can SIGKILL a live container.
- **Rank-gated checkpointing** — `GOODPUT_RANK` / `settings.rank`: only rank 0 writes when nodes share a volume; rank ≥ 1 trains but `ckpt_save_s=0` in its report. CLI `_run_train` applies the store only when `num_workers > 1 or rank == 0`.
- **Kill readiness fix** — First version waited on first `step_*.pt` (step 10 of 400) → worker-1 killed mid-run, no report. Fix: `kill-worker.sh` waits for **both** `artifacts/reports/compose-worker-{0,1}/report.json` before SIGKILL.
- **2.5 Latency table** — `goodput-run --latency experiments/latency.yaml`: sweep `worker_counts` `{1, 2, 4}` → `artifacts/sweeps/latency-table/latency.json` + `.csv` + `table.md`. Each cell trains with periodic ckpt, then times restore via `resume_after_crash` (1.5 path — no SIGKILL in the matrix). Same per-cell `_reset_ckpt_dir` isolation as the 2.2 sweep.
- **Docs / tests** — `docker/README.md`, recipes in `README.md` and `docs/testing.md`; `tests/test_compose.py` (YAML + kill script dry-run, no daemon); `tests/test_latency.py` (N=1–2 in CI, committed YAML includes N=4 for manual runs).

**Hard problems:**

1. **Compose fidelity vs narrative:** Two containers do not share the in-process barrier/all-reduce from 1.4/1.6 — they are parallel single-process trainers on a shared volume. The portfolio story is “named nodes + shared ckpt + documented kill,” not “I replicated DDP in Docker.” Say that explicitly.
2. **Shared `latest.pt` races:** Without rank-gated writes, every container would overwrite the same path. Rank 0 alone persists; others must not clobber restore pointers.
3. **Wrong readiness signal (real bug):** Checkpoint file appearance ≠ “job done.” At `ckpt_interval=10` and `steps=400`, the first `step_*.pt` fires at step 10 while worker-1 still has 390 steps left. Reports are the correct gate for the post-train kill demo.
4. **What the latency table measures:** Restore is probed with one soft-resume step per cell, not the 1.6 SIGKILL path. On a tiny naive dump, **save/restore seconds stay roughly flat** across N; **train wall** grows with spawn + barriers — that is the expected MVP shape, not “ckpt I/O scales with worker count” for unsharded dumps.
5. **Local Docker friction:** Mac needed an engine (Docker Desktop or Colima); first `up --build` is slow; recipe docs must wipe compose ckpt/report dirs; zsh `*` globs can fail with `no matches found` — use explicit paths in copy-paste recipes.

**Metrics / evidence:**

- `docker compose -f docker/compose.yaml up --build` → rank 0 writes `artifacts/checkpoints/compose/step_*.pt`; both ranks emit `artifacts/reports/compose-worker-*/report.json`
- `./docker/kill-worker.sh --dry-run` → prints SIGKILL command without daemon; live run → worker-1 status **137** after both reports exist
- `goodput-run --latency experiments/latency.yaml` → markdown table with `num_workers`, `ckpt_save_s`, `ckpt_restore_s`, `wall_seconds` for N ∈ {1, 2, 4}
- Phase 2 exit (master-plan): goodput-vs-rate chart (2.3) **and** ckpt latency vs worker count table (2.5) both regenerable from committed YAML

**Other lessons:**

- Default pytest still does not start Compose — same CI boundary as real SIGKILL in 1.6
- Latency runner mirrors sweep structure (`load_*_yaml`, per-cell ckpt wipe, comparison JSON/CSV) — third evaluation axis reuses the harness pattern
- Phase 2 is closed on paper; Phase 3 adds failure modes (hang, bitflip) and the dollar narrative

**Blockers:**

- No Docker on PATH until Desktop/Colima installed locally (documented in `docker/README.md`; not a CI blocker)
- None that blocked tickets **2.4–2.5** once kill readiness and rank-gated ckpt landed

**Next week focus:**

- Phase 3: **3.1** hang injector + health-check timeout
- **3.2** bit-flip stub; **3.3** dollar-impact script from measured goodput deltas
- Keep the three-story portfolio honest: goodput curve + latency table done; dollar model next

---

## Week 6 — Phase 3 start (hang, bitflip, dollar model)

**Shipped:**

- **3.1 Hang injector** — One worker blocks on `hang_rank` after a durable checkpoint; parent detects via rank-0 **progress stall** (`wait_for_hang_detection`, `health_check_timeout_s`), not process exit; reap; `resume_after_crash`. CLI `--fault-hang`, `mode: fault_hang`, `experiments/fault-hang.yaml`.
- **3.2 Bit-flip injector** — Target rank XORs one float32 **sign bit** on a local gradient before the toy all-reduce; training **continues** (no ckpt recovery). Optional rank-0 detector. CLI `--fault-bitflip`, `experiments/fault-bitflip.yaml`. Done-when: loss trajectory diverges from a clean baseline after the flip step.
- **3.3 Dollar-impact script** — `goodput-run --dollar experiments/dollar.yaml`: pair naive vs incremental goodput from the 2.2 comparison table, then `cluster_size × public $/GPU-hr × hours × Δgoodput`. Default: 16,384 GPUs × 1,296 h × Lambda Cloud 8× H100 SXM on-demand list **$3.99/GPU-hr** (public page, labeled not a quote). Writes `dollar.json` / `.csv` / `table.md`.
- **Three-story portfolio (partial exit):** goodput-vs-rate chart (2.3) + latency table (2.5) + dollar model (3.3) are all regenerable from committed YAML.

**Hard problems:**

1. **Hang inject race:** First version waited for the checkpoint, then set `hang_rank` — rank 0 could publish the next step before the target blocked, so detection saw progress advance. Fix: arm hang as soon as `progress >= fault_at`, then pin `latest.pt`; `wait_for_hang_detection` **fails** if progress moves or every process exits. Done-when is “alive + stalled,” not SIGKILL with extra sleep.
2. **Sign-bit flips are invisible to L2-norm detectors:** XOR `1 << 31` negates a value; magnitude stays the same, so “one rank’s norm ≫ median” never fires on the real injector. Fix: bitflip demos share a data seed so local grads match until corruption; detector flags **cross-rank desync** when norms still match, and keeps the norm-outlier rule for synthetic spikes. Sharded batches would hide a sign flip.
3. **Bitflip is not kill/hang:** No restore path. Useful-compute can count poisoned steps as “useful but wrong” (ml-strategy **corrupted steps**). Detector is optional; the loss-divergence test is the contract, not a crash.
4. **Dollar scaling vs toy noise:** The 2.2 sweep walls are milliseconds; incremental vs naive goodput at `failure_rate=0` can flip sign from spawn/I/O jitter. Scaling that Δ to 16k GPUs prints tens of millions either way. The honest artifact is **labeled arithmetic** (public list price, simulated cluster, measured delta) — not “incremental saved $40M.”

**Metrics / evidence:**

- `goodput-run --config experiments/fault-hang.yaml` → `ok=True`, `detection_s≈1.0` (matches `health_check_timeout_s: 1.0`), `resumed_from=4`
- `goodput-run --config experiments/fault-bitflip.yaml` → `ok=True`, `applied=True`, `detected=True` at `fault_at`; `test_bitflip_changes_loss_trajectory` matches clean losses through the flip step, then diverges
- `goodput-run --dollar experiments/dollar.yaml` → `artifacts/sweeps/dollar-impact/table.md` with disclaimer + public price source; missing comparison JSON runs `experiments/sweep.yaml` first
- Tests: `tests/test_hang.py`, `tests/test_bitflip.py`, `tests/test_dollar.py` (CI does not need Compose or a prior sweep artifact)

**Other lessons:**

- Hang time is wall, not useful (ml-strategy). Detection latency is its own report field — don’t fold it into `ckpt_restore_s`.
- Pre-set bitflip `(rank, step)` in shared memory at spawn (unlike hang’s parent `maybe_inject`) so the worker cannot miss the schedule.
- Evaluation runners keep cloning the same harness: YAML → cells/rows → JSON/CSV/markdown under `artifacts/sweeps/<name>/`. Dollar is a **post-processor** of measured goodput, not a fourth trainer.
- `gdone` on 3.1–3.3 belonged in those ticket PRs so Week 6 status on `main` does not lag the merges.

**Blockers:**

- None that blocked tickets **3.1–3.3**. Hang detection flake was a real race, fixed before merge; dollar “wrong-sign” millions are expected toy noise, not a product bug.

**Next week focus:**

- **3.4** Reproducibility pack: git SHA, versions, config hash in reports; same-seed runs match within tolerance
- **3.5** Portfolio polish: architecture diagram, README demo script, learned log walkthrough
- Phase 3 exit: keep the three charts honest in one clone-and-run story (curve + latency + labeled $) 

---

## Week 7 — Phase 3 close (reproducibility + portfolio polish)

**Shipped:**

- **3.4 Reproducibility pack** — Every `build_run_report` payload includes `git_sha`, `git_dirty`, `config_hash`, `package_versions`, `versions_hash` (schema **1.1**). `config_hash` hashes training knobs only (paths/providers excluded). CLI prints `repro git=… config=…`. Done-when: two same-seed trains match loss within tolerance (`tests/test_reproducibility.py`).
- **3.5 Portfolio polish** — `./scripts/portfolio-demo.sh` (pytest → ci-smoke → sweep → optional plot → latency → dollar); expanded `docs/architecture.md` (CLI flow, fault sequence, three-story table, fidelity notes); **§ Demo walkthrough** + README Phase 3 status.
- **Phase 3 exit:** goodput-vs-rate chart (2.3) + latency table (2.5) + dollar model (3.3) regenerable from one clone-and-run story; repro fields on every per-run JSON.

**Hard problems:**

1. **What belongs in `config_hash`:** Including `artifacts_dir` / `run_name` would make identical experiment YAML hash differently per machine or pytest `tmp_path`. Fix: explicit `CONFIG_HASH_FIELDS` list — training + fault knobs only; paths stay out.
2. **Repro Done-when vs wall clock:** Same seed can produce identical losses but different `wall_seconds` / `goodput` if the OS schedules differently. Tests assert **loss trajectory + config_hash**, not timing equality — honest for a portfolio metric story.
3. **Schema bump ripples:** Adding `git_sha`, `config_hash`, and `package_versions` to required report fields touches every report test and CI smoke. Bumping `schema_version` to 1.1 documents the contract change.
4. **Portfolio demo pacing:** Full `./scripts/portfolio-demo.sh` runs pytest + six sweep cells + N=1,2,4 latency + dollar (~1–3 min). Matplotlib is optional — script skips plot with a hint rather than failing; `--dry-run` / `--skip-tests` keep CI and quick checks fast.

**Metrics / evidence:**

- `goodput-run --config experiments/ci-smoke.yaml` → report with `git_sha`, `config_hash`, `package_versions`; CLI `repro git=… dirty config=…`
- `./scripts/portfolio-demo.sh` → pytest green; sweep six cells; plot PNG when matplotlib installed; `latency-table/table.md`; `dollar-impact/table.md`
- `./scripts/portfolio-demo.sh --dry-run` covered by `tests/test_portfolio_demo.py`
- Phase 3 master-plan exit: three-story portfolio + `./scripts/portfolio-demo.sh` walkthrough in `docs/what_i_learned.md`

**Other lessons:**

- Pin core deps in `package_versions`, not full `pip freeze` — editable install paths make freeze hashes unstable.
- The demo walkthrough lives **above** the weekly log so a first-time reader does not scroll through six weeks of changelog.
- Week 7 closes the 6–8 week MVP arc on paper; Phase 4 is explicitly stretch / optional defer.

**Blockers:**

- None that blocked **3.4–3.5**. Dollar millions on a full demo run are expected toy noise at Llama-scale scaling — say the disclaimer, don’t debug the sign.

**Next week focus:**

- Phase 4 stretch (pick one or defer with reason): **4.1** goodput vs worker count, **4.2** Colab GPU notebook, **4.3** tracker stub, **4.4** torch.distributed.checkpoint comparison note
- Or: stop at Phase 3 — exit criteria already met 

---

## Week 8 — Phase 4 stretch (scale, Colab, tracker, DCP note)

**Shipped:**

- **4.1 Goodput vs worker count** — `goodput-run --scale experiments/scale.yaml`: `cluster_failure_rate = N × per_gpu_failure_rate` on the Phase 2 soft-crash path. N is **simulated cluster size** (crash schedule + wasted GPU-hours), not multiprocess spawn. JSON/CSV/`table.md` + optional PNG. Write-up: `docs/scale.md`.
- **4.2 Colab GPU demo** — `notebooks/colab_gpu_demo.ipynb` + `experiments/colab.yaml` (12 steps, seed 42, naive ckpt). Runtime → T4; `train_from_settings` falls back to CPU. CI checks YAML ↔ notebook knobs; does not start Colab.
- **4.3 Tracker provider** — `Tracker.log_run` on `Providers`; `none` / `mlflow` / `wandb`. `GOODPUT_TRACKER=mlflow goodput-run --config experiments/tracker.yaml` logs a run. Without the SDK, JSON under `artifacts/mlflow/`; optional `pip install -e ".[tracker]"` for a local `file:` store.
- **4.4 DCP latency note** — `goodput-run --dcp-compare experiments/dcp-compare.yaml`: naive dump vs incremental model-only save vs `torch.distributed.checkpoint` on 1-rank `ToyMLP`+Adam. `docs/dcp-compare.md` + regenerable `table.md`. Not a DCP rewrite of training.
- **Phase 4 exit:** all four stretch tickets shipped (not deferred). The 6–8 week plan is complete including optional stretch.

**Hard problems:**

1. **Two meanings of “workers”:** 4.1’s N scales interruption rate on a 1-process crash cell; 2.5 actually spawns N processes for save/restore latency. Mixing them in one chart would be a lying axis. Docs have to say which N is which.
2. **Colab vs CI:** A notebook that `pip install`s from GitHub can drift from `main`. Done-when is documented settings + CPU-safe tests, not “CI opened Colab.” GPU quota exhaustion is why fallback is part of the contract.
3. **Tracker Done-when vs extras:** Requiring real MLflow in default CI would add a heavy optional dep. JSON fallback makes `GOODPUT_TRACKER=mlflow` true in CI; the extra is for people who want `mlruns/`.
4. **DCP on 1-rank CPU:** Torch warns that distributed is uninitialized and assumes single-process save/load. Incremental can be faster than naive here (skip optimizer pickle); DCP can be slower. That is expected overhead of a sharded protocol on a tiny unsharded blob — not a verdict against DCP at cluster scale. Some builds skip DCP without a process group; that is `available: no`, not a red X.

**Metrics / evidence:**

- `--scale` → earlier `kill_at` at larger N; `artifacts/sweeps/goodput-vs-workers/table.md`
- Colab notebook + `goodput-run --config experiments/colab.yaml` on CPU; `tests/test_colab_demo.py`
- `GOODPUT_TRACKER=mlflow` → `tracker=mlflow run=…` (JSON fallback or `mlruns/`)
- `--dcp-compare` → three rows; on this laptop naive ~0.9 ms save, incremental ~0.3 ms, DCP ~2.9 ms `available=yes` (torch single-process warnings)

**Other lessons:**

- Phase 4 is labeled stretch so shipping all four is extra, not a promise from Week 1. Stopping after 4.1 would still have met “at least one.”
- `report.json` and tracker JSON are different files. Seeing `report=artifacts/reports/tracker-demo/report.json` means the train sink worked; the tracker line is `tracker=mlflow run=…`.
- Torch DCP warnings on macOS (`Redirects are currently not supported`) are library noise, not a failed compare.

**Blockers:**

- None that blocked **4.1–4.4**. Colab GPU quota and optional MLflow extra are expected limits, handled by fallbacks.

**Next week focus:**

- No Week 9 ticket list. The numbered plan through Phase 4 is done.
- Optional follow-ups (not tickets): real multi-rank DCP, a hosted MLflow server, or a goodput-vs-N chart that *also* spawns N processes — only if a new question needs that fidelity. 
