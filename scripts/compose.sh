#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  exec docker compose "$@"
fi

if command -v podman-compose >/dev/null 2>&1; then
  exec podman-compose "$@"
fi

echo "No usable compose command found. Install docker compose v2 or podman-compose." >&2
exit 1
