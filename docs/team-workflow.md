# Team workflow (solo)

Adapted from the two-founder Pigeon workflow. There is **one owner**: Cletus. “Review” means self-review against the checklist below; invite an external reviewer when useful.

## Git

- `main` is protected and always deployable (docs + green CI).
- Branches: `feat/…`, `fix/…`, `docs/…`
- Prefer one ticket per PR. Title includes step id: `feat(1.2): synthetic data loader`

## PR body format

```markdown
## Summary
- Bullet points (closes ticket X.Y)

## Test plan
- [ ] pytest passes
- [ ] Fresh clone setup verified (if setup changed)
- [ ] Metrics on fixture: …
```

## Definition of Done (every ticket)

- Works locally
- Tests or documented manual steps
- Security/data handling considered (no secrets, synthetic fixtures only)
- Someone else could run it without a walkthrough
- Merged via PR (not direct push to `main`)

## Weekly rhythm

| Day | Activity |
|-----|----------|
| Start of week | Set `todos/week-NN.json` + point `todos/current.json`; write week goal |
| During week | `gdone <task-id>` as tasks finish; small PRs |
| End of week | Update `what_i_learned.md`; open next week file |

## Commit hygiene

- No secrets, no `artifacts/`, no large binaries
- No “initial commit” kitchen-sink after Phase 0
- Refactors need a ticket or an explicit `chore/` note tied to a step

## Releases / tags

Optional for MVP. Tag `v0.1.0` when Phase 1 exit criteria are met.
