# Weekly todos

Track weeks in JSON so you can mark tasks done from the terminal and see progress.

## Setup (once)

```bash
chmod +x scripts/done scripts/setup-shell.sh
./scripts/setup-shell.sh    # adds `gdone` to ~/.zshrc
exec zsh
gdone who cletus
```

### Why `gdone` and not `done`?

In zsh/bash, `done` is a **reserved word**. The alias is **`gdone`** (goodput done).

## Mark complete

```bash
gdone git-init
gdone docs-scaffold
gdone ci
```

Partial names work if unique:

```bash
gdone git
gdone smoke
```

## Status

```bash
gdone status
gdone list
```

Without the alias:

```bash
./scripts/done status
./scripts/done git-init
```

## File format

`todos/week-01.json`:

```json
{
  "week": 1,
  "phase": "Phase 0 — Foundation",
  "goal": "One sentence weekly goal",
  "label": "Week 1",
  "tasks": [
    {
      "id": "git-init",
      "title": "Initialize private git repo + GitHub remote (ticket 0.1)",
      "owner": "cletus",
      "done": false,
      "completed_at": null,
      "completed_by": null
    }
  ]
}
```

`todos/current.json` points at the active week file:

```json
{ "file": "week-01.json" }
```

## Solo note

Owner is always `cletus`. The CLI still uses an identity file (`.goodput-user`) so the same tool stays compatible if a collaborator joins later.
