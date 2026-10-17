# Phase 3 — Real-Time Scoring Integration

This phase puts the Phase 2 model inside the payment transaction path:

```
SDK session aggregates
        │  (salted subject hash, consent, timing only)
        ▼
Behavioral ingestion ──► per-user baseline ──┐
                                             │
Transaction ──► Feature store snapshot ──────┼──► timeout-bounded scoring
                (7d/30d aggregates,          │    (calibrated GBDT + behavior
                 velocity, devices, geo)     │     + ring boost)
                                             ▼
                                    Decision rules (30/60/80/95)
                                             │
                         COMMITTED / PENDING_STEP_UP / HELD / DECLINED
                                             │
                                    Append-only JSONL audit
```

Everything runs in-process on the LinuxONE / z/OS application side with no
Python packages outside the standard library. The same interfaces map onto a
CICS transaction, an IMS application, or a z/OS Connect service.

## Components

| Component | Module | Purpose |
|---|---|---|
| Online feature store | `fraud_detection/feature_store.py` | Precomputed per-user aggregates, behavioral baseline; O(1)-ish reads |
| Behavioral ingestion | `fraud_detection/collection.py` | Validate SDK payloads, enforce consent, pseudonymise identity |
| Browser/mobile SDK | `sdk/behavioral-biometrics.js` | Collect keystroke/mouse/touch aggregates, never content |
| Transaction service | `fraud_detection/service.py` | Synchronous timeout-bounded scoring, policies, audit |
| Demo CLI | `fraud_detection/realtime.py` | Streams labeled traffic through the full path and verifies it |
| Launcher | `run_phase3.sh` | Trains if needed, then runs the Phase 3 demo |

## Feature store (plan 3.3)

`FeatureStore` maintains, per customer reference:

- average / deviation of the last 200 payments (the Phase 2 model baseline),
- true 7-day and 30-day rolling amount sums/counts,
- transaction velocity over 1h and 24h, and repeat-recipient counts,
- device, recipient and last-location history,
- account age and the confirmed-fraud flag (set via `mark_fraud`, never from
  evaluation labels),
- the behavioral-biometrics baseline (`fraud_detection.biometrics`).

Reads are snapshots; writes prune expired entries. The demo measures feature
store read p99 and checks it against the sub-millisecond target.

```python
from fraud_detection.feature_store import FeatureStore
from fraud_detection.features import FeatureEngine

store = FeatureStore()
engine = FeatureEngine(store)      # or FeatureEngine() to get its own store

features = engine.compute(txn, behavior_score, channels)
```

## Behavioral biometrics SDK (plan 3.2)

```js
import { hashSubjectId, createCollector } from "./sdk/behavioral-biometrics.js";

const subjectRef = await hashSubjectId(customerId, serverIssuedSalt);
const collector = createCollector({ subjectRef, consent: userConsent });
collector.attach();

// at payment time
const payload = collector.snapshot();     // session aggregates only
await collector.flush("/api/fraud/behavior");
```

Collected values: key hold distribution, inter-key interval, pointer velocity
and direction entropy, normalised click positions, and touch
pressure/speed/angle. The SDK never records text, key codes, element values,
DOM paths or raw pointer paths. `hashSubjectId` requires HTTPS/localhost (Web
Crypto) and a server-issued salt.

Server-side `parse_payload` enforces the same rules: unknown fields, raw
content, event arrays and out-of-range metrics are rejected. A rejected or
opted-out payload is **not** an error for the payment — the transaction is
scored with transaction-only features and the audit record says
`behavior_status: "rejected"` or `"opted_out"`.

Payload schema (`behavioral.session.v1`):

```json
{
  "schema": "behavioral.session.v1",
  "sdk_version": "1.0.0",
  "subject_ref": "sha256:...",
  "session_id": "S000001",
  "collected_at": "2026-09-01T12:00:00+00:00",
  "consent": {"behavioral_biometrics": true},
  "keystroke": {"key_hold_mean_ms": 108.2, "key_hold_std_ms": 13.1, "inter_key_ms": 176.4, "samples": 84},
  "mouse": {"velocity_px_s": 655.0, "entropy": 3.2, "clicks": 3, "click_x_mean_norm": 0.47, "click_y_mean_norm": 0.58, "samples": 412},
  "touch": {"pressure_mean": 0.0, "speed_px_s": 0.0, "swipe_angle_deg": 0.0, "samples": 0}
}
```

## Scoring service (plan 3.1)

```python
from fraud_detection.service import AuditLog, ScoringPolicy, ScoringService

with ScoringService(
    model_path="models/fraud_model.json",
    policy=ScoringPolicy(timeout_ms=50, failure_mode="fail_open"),
    audit=AuditLog("logs/phase3-decisions.jsonl"),
) as service:
    service.warm(historical_events)       # builds profiles, not audited
    result = service.score(txn, payload)  # synchronous, returns ScoreResult
```

`ScoreResult` carries the risk score, action, settlement outcome, segment,
model version, top-3 signals, per-stage latency, behavior status and
degradation details.

### Timeout and failure policy

Inference runs on a worker thread while feature computation and state updates
stay on the caller thread. The caller waits at most `timeout_ms`; a timed-out
inference is abandoned without losing transaction history.

| `failure_mode` | Degraded action | Use when |
|---|---|---|
| `fail_open` (default) | `ALLOW` (risk 0) | Payment availability matters more than a missed hold |
| `fail_closed` | `BLOCK` (risk 100) | Policy requires no unvetted payment to settle |

Timeouts, model exceptions and policy fallbacks are all written to the audit
trail with `degraded: true` and `fallback_reason`.

### Decision and settlement mapping

| Risk score | Action | Settlement outcome |
|---|---|---|
| `< 30` | `ALLOW` | `COMMITTED` |
| `30–60` | `ALLOW_WITH_LOGGING` | `COMMITTED` |
| `60–80` + behavior anomaly | `STEP_UP_2FA` | `PENDING_STEP_UP` |
| `60–80` otherwise | `ALLOW_WITH_LOGGING` | `COMMITTED` |
| `80–95` | `HOLD_FOR_REVIEW` | `HELD_FOR_REVIEW` |
| `>= 95` | `BLOCK` | `DECLINED` |

Segment thresholds (`new_user`, `high_value`, `standard`) remain configurable
through `DecisionConfig`.

## Audit trail

One JSON object per decision, appended and flushed immediately:

```json
{
  "action": "HOLD_FOR_REVIEW",
  "amount": 1240.0,
  "behavior_status": "scored",
  "channel": "transfer",
  "correlation_id": "CORR-TXN0001234",
  "currency": "USD",
  "decided_at": "2026-09-01T12:31:02.123+00:00",
  "degraded": false,
  "fallback_reason": null,
  "feature_latency_ms": 0.071,
  "latency_ms": 0.912,
  "model_version": "phase2-1",
  "outcome": "HELD_FOR_REVIEW",
  "risk_score": 88.4,
  "segment": "standard",
  "top_signals": ["unrecognized device", "new recipient", "impossible travel"],
  "txn_id": "TXN0001234",
  "user_ref": "sha256:31e803a67c4f5..."
}
```

`user_ref` is always the salted pseudonym — raw customer IDs are never
written.

## Runbook

```bash
./run_phase3.sh                                  # full 2000-transaction demo
./run_phase3.sh --verbose
./run_phase3.sh --retrain                        # retrain, then run

# fail-open and fail-closed policy checks (injected slow inference)
python3 -m fraud_detection.realtime --model models/fraud_model.json \
    --transactions 100 --fault-rate 0.2 --fault-delay-ms 250 --fail-mode open
python3 -m fraud_detection.realtime --model models/fraud_model.json \
    --transactions 100 --fault-rate 0.2 --fault-delay-ms 250 --fail-mode closed

# biometric opt-out path
python3 -m fraud_detection.realtime --model models/fraud_model.json \
    --transactions 200 --opt-out-rate 0.25
```

The demo exits non-zero if audit records are missing, required fields or
explanations are absent, identities are not pseudonymous, degraded decisions
violate the policy, or the p99 scoring target is missed when no fault is
injected.

### Monitoring expectations

- Scoring path p99 under 50 ms; feature-store read p99 under 1 ms.
- `degraded` rate should stay near zero; a sustained rise means model/runtime
  instability and the policy fallback is absorbing it.
- Compare the action mix and score distribution between runs for drift.
- `behavior_status` counts show SDK coverage; a rising `rejected` count means
  a client/schema regression.

### Configuration

| Setting | Default | Notes |
|---|---|---|
| `FRAUD_SUBJECT_SALT` | demo salt | Must be a secret in production; changing it orphans baselines and history |
| `ScoringPolicy.timeout_ms` | 50 | Transaction-path budget |
| `ScoringPolicy.failure_mode` | `fail_open` | `fail_closed` for strict policy |
| `--audit-log` | `logs/phase3-decisions.jsonl` | Append-only JSONL |

## Tests

```bash
python -m unittest discover -s tests -v
```

Phase 3 coverage: feature-store aggregates/windows/latency/baseline, SDK
payload round-trip and rejection, opt-out fallback, timeout fail-open and
fail-closed, model-error fallback, audit JSONL completeness, and a trained
model end-to-end run.

## Limitations

- The feature store is in-process memory. A multi-instance deployment needs a
  shared low-latency store (IMS/DB2 cache or attached Redis) behind the same
  interface.
- The timeout is caller-side; timings are measured in CPython, not on z/OS
  hardware. Benchmark on the target LPAR before quoting production numbers.
- The JS SDK is a reference implementation and is not bundled/minified.
- Behavioral data is sampled synthetic traffic; no production biometric data
  was used.
