#!/usr/bin/env bash
# SIGKILL a Compose worker after a durable checkpoint exists (ticket 2.4).
#
# Usage (from repo root, after `docker compose -f docker/compose.yaml up --build`):
#   ./docker/kill-worker.sh
#   ./docker/kill-worker.sh --dry-run
#   ./docker/kill-worker.sh worker-1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker/compose.yaml"
CKPT_DIR="$ROOT/artifacts/checkpoints/compose"

DRY_RUN=0
SERVICE="worker-1"
WAIT_CKPT=1

usage() {
  cat <<'EOF'
Usage: docker/kill-worker.sh [--dry-run] [--no-wait] [SERVICE]

SIGKILL a Compose worker (default: worker-1) after rank 0 has written
artifacts/checkpoints/compose/step_*.pt on the shared volume.

  --dry-run   Print the docker compose kill command; do not wait or signal
  --no-wait   Skip waiting for a checkpoint (still requires docker compose)
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
      WAIT_CKPT=0
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

if [[ "$WAIT_CKPT" -eq 1 ]]; then
  echo "waiting for checkpoint under $CKPT_DIR ..."
  found=0
  for _ in $(seq 1 90); do
    if compgen -G "$CKPT_DIR"/step_*.pt >/dev/null 2>&1; then
      found=1
      break
    fi
    sleep 1
  done
  if [[ "$found" -eq 0 ]]; then
    echo "no step_*.pt after 90s; is compose up?  docker compose -f docker/compose.yaml up --build" >&2
    exit 1
  fi
  echo "checkpoint present; sending SIGKILL to $SERVICE"
fi

if ! "${COMPOSE[@]}" ps --status running --services 2>/dev/null | grep -qx "$SERVICE"; then
  echo "$SERVICE is not running. Start the cluster first:" >&2
  echo "  docker compose -f docker/compose.yaml up --build" >&2
  exit 1
fi

"${KILL_CMD[@]}"
echo "killed $SERVICE"
"${COMPOSE[@]}" ps -a
