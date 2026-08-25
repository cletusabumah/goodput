# Documentation index

Read in this order for Phase 0 orientation:

1. [`vision.md`](vision.md) — why this exists
2. [`mvp-spec.md`](mvp-spec.md) — must-haves vs stretch
3. [`master-plan.md`](master-plan.md) — numbered steps + Done when
4. [`ml-strategy.md`](ml-strategy.md) — metrics, baselines, ablations
5. [`architecture.md`](architecture.md) — system diagram
6. [`phase-1-tickets.md`](phase-1-tickets.md) — build order + issue template
7. [`team-workflow.md`](team-workflow.md) — git, PRs, Definition of Done
8. [`todos.md`](todos.md) — weekly JSON + `gdone` CLI
9. [`testing.md`](testing.md) — test layers
10. [`privacy-security.md`](privacy-security.md) + [`test-data-protocol.md`](test-data-protocol.md)
11. [`github-access.md`](github-access.md) — 2FA + branch protection
12. [`roadmap.md`](roadmap.md) — high-level phases
13. [`what_i_learned.md`](what_i_learned.md) — weekly engineering log + **§ Demo walkthrough**
14. [`scale.md`](scale.md) — goodput vs worker count (ticket 4.1 MTBF illustration)
15. [`../notebooks/README.md`](../notebooks/README.md) — Colab GPU demo (ticket 4.2)

`GOODPUT_TRACKER=mlflow` logs a run (ticket 4.3; JSON fallback if the SDK is missing).

## Product / persona notes

[`product-features.md`](product-features.md) maps features to personas (operator, metrics reader, collaborator). This is **not** a consumer SaaS product — discard Pigeon-style end-user UX framing where it does not apply.
