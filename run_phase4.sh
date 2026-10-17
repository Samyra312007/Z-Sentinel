#!/usr/bin/env bash
# Phase 4 launcher for IBM LinuxONE (s390x / Linux on Z).
#
# 1. trains the fraud model if the artifact is missing (or --retrain is given)
# 2. streams transactions into the review queue
# 3. serves the investigator dashboard and monitoring API
#
# Requires only python3 >= 3.10 (Ubuntu 24.04 ships 3.12). No pip packages.
# Stop the dashboard with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
MODEL="models/fraud_model.json"
DB="${PHASE4_DB:-logs/phase4-reviews.sqlite}"
AUDIT="${PHASE4_AUDIT:-logs/phase4-decisions.jsonl}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Install with:  sudo apt update && sudo apt install -y python3" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo "'$PY' is not a working Python >= 3.10. On Ubuntu:  sudo apt install -y python3" >&2
  exit 1
fi

echo "host        : $(uname -sr) ($(uname -m))"
echo "interpreter : $("$PY" --version 2>&1) at $(command -v "$PY")"

if [ "${1:-}" = "--retrain" ]; then
  shift
  echo "retraining model ..."
  "$PY" -m fraud_detection.train --events 15000 --warmup 2000 --users 200 \
      --fraud-rate 0.03 --trees 60 --depth 3 --out "$MODEL"
elif [ ! -f "$MODEL" ]; then
  echo "no model artifact found — training ..."
  "$PY" -m fraud_detection.train --events 15000 --warmup 2000 --users 200 \
      --fraud-rate 0.03 --trees 60 --depth 3 --out "$MODEL"
else
  echo "using cached model: $MODEL"
fi

echo
mkdir -p "$(dirname "$DB")" "$(dirname "$AUDIT")"
exec "$PY" -m fraud_detection.phase4 --model "$MODEL" \
     --transactions 2000 --users 200 --fraud-rate 0.03 --warmup 2000 \
     --db "$DB" --audit-log "$AUDIT" "$@"
