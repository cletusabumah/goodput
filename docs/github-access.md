# GitHub access checklist

Solo owner: **Cletus Abumah** (`cletusabumah`). Complete before treating `main` as production-ready.

## Account

- [ ] GitHub account owns private repo `goodput`
- [ ] **2FA enabled** ([settings](https://github.com/settings/security))
- [ ] SSH or HTTPS auth works locally (`git fetch`, `git push`)

## Branch protection on `main`

- [ ] Require a pull request before merging
- [ ] Require status checks to pass (`CI` workflow)
- [ ] Do not allow force pushes
- [ ] Do not allow deletions
- [ ] Optional: require linear history

## Verify locally

```bash
git remote -v
git fetch origin
git push origin HEAD   # on a feature branch, not main
```

When branch protection is on, direct pushes to `main` should fail and PRs should wait for green CI.

## Collaborators

None required for MVP. If adding a reviewer later, grant least privilege and re-check branch protection (reviews required).
