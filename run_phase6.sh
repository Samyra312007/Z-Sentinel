#!/usr/bin/env bash
set -o pipefail

# Phase 6: Testing & Validation end-to-end
#
# Phase 6 is validation and operational readiness rather than a new runtime
# feature, so the launcher exercises the suites that already cover the plan's
# validation cells: the unit suite, the Phase 4 smoke run, the Phase 5 smoke
# run, and the new Phase 6 validation suite.
#
# Runbook-style commands used by the demo:
#   ./run_phase6.sh                 # full validation pass
#   ./run_phase6.sh --phase4        # Phase 4 smoke only
#   ./run_phase6.sh --phase5        # Phase 5 smoke only
#   ./run_phase6.sh --unit          # unit suites only (Phase 2/3/4/5/6)
#   ./run_phase6.sh --unit tests/test_phase6.py  # Phase 6 suite only

USAGE="Usage: $0 [--unit [--only FILE]] [--phase4] [--phase5] [--fresh]"

ONLY=""
DO_PHASE4=0
DO_PHASE5=0
DO_UNIT=0
FRESH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --unit)
      DO_UNIT=1
      shift
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --phase4)
      DO_PHASE4=1
      shift
      ;;
    --phase5)
      DO_PHASE5=1
      shift
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    *)
      echo "$USAGE" >&2
      exit 2
      ;;
  esac
done

# Default: do everything Phase 6 cares about.
if [ "$DO_PHASE4" = "0" ] && [ "$DO_PHASE5" = "0" ] && [ "$DO_UNIT" = "0" ]; then
  DO_PHASE4=1
  DO_PHASE5=1
  DO_UNIT=1
fi

FAIL=0
PASS=0

count_result() {
  local label="$1"
  local rc="$2"
  if [ "$rc" -eq 0 ]; then
    echo "[PASS] $label"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $label (exit $rc)"
    FAIL=$((FAIL + 1))
  fi
}

if [ "$DO_UNIT" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    echo "=== Phase 6: unit suite ($ONLY) ==="
    set -o pipefail
    python3 -m unittest discover -s tests -v -p "$(basename "$ONLY")" 2>&1 | tail -n 20
    count_result "unit suite ($ONLY)" $?
  else
    echo "=== Phase 6: unit suite (Phase 2/3/4/5/6) ==="
    set -o pipefail
    python3 -m unittest discover -s tests -v 2>&1 | tail -n 20
    count_result "unit suite" $?
  fi
fi

if [ "$DO_PHASE4" -eq 1 ]; then
  echo "=== Phase 6: Phase 4 validation smoke ==="
  set -o pipefail
  ./run_phase4.sh \
    --model models/fraud_model.json \
    --transactions 300 \
    --warmup 200 \
    --fresh \
    --check \
    --port 0 \
    > /tmp/phase6-phase4.log 2>&1
  count_result "Phase 4 smoke" $?
fi

if [ "$DO_PHASE5" -eq 1 ]; then
  echo "=== Phase 6: Phase 5 validation smoke ==="
  set -o pipefail
  ./run_phase5.sh \
    --model models/fraud_model.json \
    --transactions 200 \
    --demo-reviews 2 \
    --fresh \
    --check \
    --port 0 \
    > /tmp/phase6-phase5.log 2>&1
  count_result "Phase 5 smoke" $?
fi

echo
echo "Phase 6 summary: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
