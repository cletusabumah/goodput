#!/usr/bin/env bash
# Portfolio demo — regenerate the three-story chart set from committed YAML (ticket 3.5).
#
# Story 1: goodput vs failure rate (sweep + optional plot)
# Story 2: checkpoint/restore latency vs worker count
# Story 3: labeled dollar impact from measured goodput deltas
#
# Usage:
#   ./scripts/portfolio-demo.sh              # full demo (~1–3 min on a laptop)
#   ./scripts/portfolio-demo.sh --dry-run    # print commands only (CI / smoke)
#   ./scripts/portfolio-demo.sh --skip-tests  # skip pytest preamble
#   ./scripts/portfolio-demo.sh --skip-plot   # skip matplotlib chart
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
SKIP_TESTS=0
SKIP_PLOT=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-plot) SKIP_PLOT=1 ;;
    -h | --help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run: $*"
  else
    echo ""
    echo ">>> $*"
    "$@"
  fi
}

section() {
  echo ""
  echo "=== $* ==="
}

if [[ "$DRY_RUN" -eq 0 ]] && [[ ! -d .venv ]]; then
  echo "hint: create .venv and pip install -e . first (see README Fresh clone)" >&2
fi

section "0. Smoke + reproducibility"
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  run pytest -q
fi
run goodput-run --config experiments/ci-smoke.yaml

section "1. Goodput vs failure rate (chart)"
run goodput-run --sweep experiments/sweep.yaml
if [[ "$SKIP_PLOT" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    run goodput-run --plot artifacts/sweeps/phase2-sweep/comparison.json
  elif python -c "import matplotlib" 2>/dev/null; then
    run goodput-run --plot artifacts/sweeps/phase2-sweep/comparison.json
  else
    echo "skip plot: matplotlib not installed (pip install -e '.[viz]')"
  fi
fi

section "2. Checkpoint latency vs worker count (table)"
run goodput-run --latency experiments/latency.yaml

section "3. Dollar impact from measured deltas (labeled estimate)"
run goodput-run --dollar experiments/dollar.yaml

section "Artifacts (gitignored)"
cat <<EOF
  artifacts/reports/ci-smoke/report.json          — repro fields + goodput
  artifacts/sweeps/phase2-sweep/comparison.json   — sweep table (story 1)
  artifacts/plots/goodput_vs_failure_rate.png     — chart (if --plot ran)
  artifacts/sweeps/latency-table/table.md         — latency table (story 2)
  artifacts/sweeps/dollar-impact/table.md         — dollar estimate (story 3)
Walkthrough: docs/what_i_learned.md § Interview walkthrough
EOF
