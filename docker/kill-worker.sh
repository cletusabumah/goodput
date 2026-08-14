#!/usr/bin/env bash
# SIGKILL a Compose worker after both ranks finish training (ticket 2.4).
#
# Workers train to completion, write reports, then sleep infinity. This script
# waits for those reports — not the first step_*.pt — so a mid-run kill cannot
# skip compose-worker-1's report.
#
# Usage (from repo root, after `docker compose -f docker/compose.yaml up --build`):
#   ./docker/kill-worker.sh
#   ./docker/kill-worker.sh --dry-run
#   ./docker/kill-worker.sh worker-1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker/compose.yaml"
REPORT_0="$ROOT/artifacts/reports/compose-worker-0/report.json"
REPORT_1="$ROOT/artifacts/reports/compose-worker-1/report.json"

DRY_RUN=0
SERVICE="worker-1"
WAIT_READY=1

usage() {
  cat <<'EOF'
Usage: docker/kill-worker.sh [--dry-run] [--no-wait] [SERVICE]

SIGKILL a Compose worker (default: worker-1) after both ranks have finished
training and written artifacts/reports/compose-worker-*/report.json.

  --dry-run   Print the docker compose kill command; do not wait or signal
  --no-wait   Skip waiting for reports (still requires a running service)
  SERVICE     Compose service name (default: worker-1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-wait)
      WAIT_READY=0
      shift
      ;;
    -*)
      echo "unknown flag: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SERVICE="$1"
      shift
      ;;
  esac
done

cd "$ROOT"
COMPOSE=(docker compose -f "$COMPOSE_FILE")
KILL_CMD=("${COMPOSE[@]}" kill -s SIGKILL "$SERVICE")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'dry-run:'
  printf ' %q' "${KILL_CMD[@]}"
  printf '\n'
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not on PATH; install Docker Desktop / Engine, or pass --dry-run" >&2
  exit 2
fi

if [[ "$WAIT_READY" -eq 1 ]]; then
  echo "waiting for post-train reports under artifacts/reports/compose-worker-*/ ..."
  found=0
  for _ in $(seq 1 120); do
    if [[ -f "$REPORT_0" && -f "$REPORT_1" ]]; then
      found=1
      break
    fi
    sleep 1
  done
  if [[ "$found" -eq 0 ]]; then
    echo "reports missing after 120s; is compose up and still training?" >&2
    echo "  docker compose -f docker/compose.yaml up --build" >&2
    echo "expected: $REPORT_0" >&2
    echo "          $REPORT_1" >&2
    exit 1
  fi
  echo "both reports present; sending SIGKILL to $SERVICE"
fi

if ! "${COMPOSE[@]}" ps --status running --services 2>/dev/null | grep -qx "$SERVICE"; then
  echo "$SERVICE is not running. Start the cluster first:" >&2
  echo "  docker compose -f docker/compose.yaml up --build" >&2
  exit 1
fi

"${KILL_CMD[@]}"
echo "killed $SERVICE"
"${COMPOSE[@]}" ps -a
