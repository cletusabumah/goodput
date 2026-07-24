#!/usr/bin/env bash
# Installs `gdone` alias into ~/.zshrc (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARKER="# goodput-gdone"
LINE="alias gdone='$ROOT/scripts/done'"
RC="${HOME}/.zshrc"

touch "$RC"
if grep -Fq "$MARKER" "$RC"; then
  echo "gdone already configured in $RC"
else
  {
    echo ""
    echo "$MARKER"
    echo "$LINE"
  } >> "$RC"
  echo "Added gdone alias to $RC"
fi
echo "Run: exec zsh && gdone who cletus"
