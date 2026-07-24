# Phase 1 Tickets

Owner for all tickets: **Cletus**. Map each PR to a step id (`feat/1.1-providers`).

## Suggested build order

```text
1.1 providers ─┬─► 1.2 data loader ─► 1.3 single-process train
               │                              │
               ├─► 1.7 metrics writer ◄────────┤
               │                              ▼
               └─► 1.4 multi-process ─► 1.5 naive ckpt ─► 1.6 SIGKILL
                                                      │
                                                      ▼
                                               1.8 experiment YAML
                                                      │
                                                      ▼
                                               1.9 CI smoke
```

## Ticket table

| ID | Title | Branch | Done when |
|----|-------|--------|-----------|
| 1.1 | Provider ABCs + mocks (Checkpoint, Fault, Metrics) | `feat/1.1-providers` | Unit tests pass; CI uses mocks |
| 1.2 | Synthetic data loader + fixtures | `feat/1.2-data-loader` | `test_data_loaders.py` green |
| 1.3 | Toy model + single-process train smoke | `feat/1.3-toy-train` | Loss finite; steps complete |
| 1.4 | Multi-process worker launcher | `feat/1.4-multiprocess` | N=2 short run succeeds |
| 1.5 | Naive checkpoint save/restore | `feat/1.5-naive-checkpoint` | Resume step equals last ckpt |
| 1.6 | SIGKILL fault injector | `feat/1.6-sigkill` | Mid-run kill triggers recovery path |
| 1.7 | Goodput metrics JSON writer | `feat/1.7-metrics` | Required fields present in report |
| 1.8 | Baseline experiment YAML + CLI | `feat/1.8-baseline-config` | `goodput-run --config experiments/baseline.yaml` works |
| 1.9 | CI training smoke job | `feat/1.9-ml-ci` | PR CI runs ≤10-step smoke; fails on NaN |

## GitHub Issue template (copy-paste)

```markdown
## Goal
Ticket **1.X** — <one sentence>

## Tasks
- [ ] …
- [ ] …

## Definition of done
- [ ] Works locally (`pytest` / documented manual steps)
- [ ] Security/data handling considered (synthetic only)
- [ ] Another engineer could run without a walkthrough
- [ ] Merged via reviewed PR (self-review checklist OK for solo)

## Test plan
- [ ] `pytest -q` passes
- [ ] Fresh clone steps still work
- [ ] Metrics on fixture (if applicable): …
```

## PR title convention

`feat(1.5): naive checkpoint save/restore`

## Out of scope for Phase 1

Incremental checkpoint, Docker Compose cluster, hang/bitflip, dollar model, plots (those are Phase 2–3).
