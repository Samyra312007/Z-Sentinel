#!/usr/bin/env bash
# Phase 1 fraud-detection demo launcher for IBM LinuxONE (s390x / Linux on Z).
#
# Requires only python3 >= 3.10 (Ubuntu 24.04 ships 3.12). No pip packages:
# the pipeline is pure standard library, so nothing needs s390x wheels.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Install with:  sudo apt update && sudo apt install -y python3" >&2
  exit 1
fi
# a real interpreter must run and be >= 3.10 (catches Windows Store stubs etc.)
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo "'$PY' is not a working Python >= 3.10. On Ubuntu:  sudo apt install -y python3" >&2
  exit 1
fi

echo "host        : $(uname -sr) ($(uname -m))"
echo "interpreter : $("$PY" --version 2>&1) at $(command -v "$PY")"
exec "$PY" -m fraud_detection "$@"