# Privacy & security

## Data handling

- **No PII.** All training data is synthetic (random tensors / generated fixtures).
- **No human subjects, consent forms, or scraped personal data.**
- Do not copy production cluster logs that may contain hostnames, IPs, or user identifiers into the repo without scrubbing.

## Secrets

- All secrets via environment variables (see `.env.example`).
- Never commit `.env`, API tokens, cloud keys, or WandB/MLflow credentials.
- CI uses defaults / mocks — no cloud credentials required for green builds.

## Model safety / bias checklist (portfolio narrative)

This project does not ship a user-facing model. Still apply infra safety habits:

- [ ] Reports do not embed secrets or machine-identifying paths unnecessarily
- [ ] Fault injection only targets processes you own (local/Compose), never third-party systems
- [ ] Document that bit-flip experiments are **simulated** corruptions, not attacks on real clusters
- [ ] Public dollar estimates use **public** GPU pricing and clearly labeled back-of-envelope math

## Threat model (MVP)

| Threat | Mitigation |
|--------|------------|
| Accidental secret commit | `.gitignore`, `.env.example` only, pre-commit optional later |
| Destructive kill scripts | Default to mock injector in CI; real SIGKILL gated by env flag |
| Dependency compromise | Pin ranges in requirements; review upgrades |
| Artifact bloat / leak | `artifacts/` gitignored |

## Future (if cloud used)

- Least-privilege IAM
- No long-lived keys in repo
- Separate staging project/account for experiments
