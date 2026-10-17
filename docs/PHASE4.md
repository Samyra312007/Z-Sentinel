# Phase 4 — Decision Engine, Investigator Dashboard & Monitoring

Phase 4 makes every escalated payment actionable and observable:

```
scoring decision (HOLD / STEP_UP)
        │
        ▼
SQLite review queue ──► investigator dashboard ──► confirmed / cleared
        │                        │                         │
        │                        └── evidence: features,    │
        │                            signals, biometrics,   │
        │                            graph context          │
        ▼                                                   ▼
monitoring snapshot ◄──────────────────────────── feedback labels
(latency, drift, capture, FPR, alerts)                 (retraining export)
```

Everything is standard library only: `sqlite3`, `http.server`, and vanilla
browser JavaScript.

## 4.1 Configurable decision policy

`DecisionConfig` now loads and validates a JSON file. Partial files are merged
over safe defaults, so a deployment can override only the segments it cares
about.

```json
{
  "thresholds": {
    "standard": {"log": 30, "step_up": 60, "hold": 80, "block": 95},
    "new_user": {"log": 25, "step_up": 55, "hold": 75, "block": 92},
    "high_value": {"log": 30, "step_up": 55, "hold": 75, "block": 92}
  },
  "graph_ring_boost": 15
}
```

Validation enforces `0 <= log < step_up < hold < block <= 100`, numeric
weights, known segments/keys and a bounded ring boost. See
`config/decision-thresholds.example.json`.

At the hold band, an account with 2FA enrolled is challenged (`STEP_UP_2FA`)
instead of being held; unenrolled accounts go to `HOLD_FOR_REVIEW`.

```python
from fraud_detection.decision import DecisionConfig
from fraud_detection.service import ScoringService

config = DecisionConfig.load("config/decision-thresholds.example.json")
service = ScoringService(model_path="models/fraud_model.json", config=config)
```

Pass `--config` to `fraud_detection.phase4` to run the demo with a custom file.

## 4.2 Investigator dashboard

`fraud_detection/dashboard.py` serves the API and the static UI in
`dashboard/`.

### Web UI

The console shows:

- open queue sorted by risk, with filters for status and sort order;
- transaction context, feature evidence, top contributing signals and their
  relative weights;
- behavioral biometric z-scores per channel, graph context and action history;
- one-click **confirm fraud**, **clear false positive**, **request more info**
  and **reopen** actions with investigator ID and note;
- score histogram, latency cards, capture/FPR cards and live alerts.

Run it with:

```bash
./run_phase4.sh
# or
python3 -m fraud_detection.phase4 --model models/fraud_model.json \
    --transactions 2000 --demo-reviews 12
```

The default URL is `http://127.0.0.1:8080`. `--check` starts the server,
verifies the API and exits, which is useful in CI. `--serve-seconds N` serves
for a bounded period.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness, review count and model version |
| `GET` | `/api/cases?status=OPEN&sort=risk&limit=50` | Risk-sorted queue |
| `GET` | `/api/cases/{id}` | Full case evidence plus action history |
| `GET` | `/api/cases/{id}/actions` | Investigator audit trail |
| `POST` | `/api/cases/{id}/actions` | `{"action":"confirm_fraud", "investigator":"analyst-1", "note":"..."}` |
| `GET` | `/api/metrics` | Monitoring snapshot, review stats, alerts, config |
| `GET` | `/api/config` | Active segment thresholds |
| `GET` | `/api/feedback` | Labeled cases for retraining |

Actions: `confirm_fraud` (label 1), `clear_fp` (label 0), `request_info`,
`reopen`. The queue and action history are persisted in SQLite
(`logs/phase4-reviews.sqlite` by default) and survive restarts.

> **Security:** the dashboard has no built-in authentication. It binds to
> `127.0.0.1` and is intended for a demo or for deployment behind z/OS
> Connect / RACF (or an authenticated reverse proxy). Do not expose it
> directly.

## 4.3 Monitoring and alerting

`fraud_detection/monitoring.py` consumes `ScoreResult` objects and exposes a
rolling snapshot:

- action/outcome mix and 10-bin risk histogram;
- scoring latency p50/p95/p99 versus the configured budget;
- degraded/timeout rate;
- detection (elevated-risk) rate versus a baseline window, plus mean score
  shift, to catch model drift in either direction;
- capture rate and false positive rate once review labels exist;
- active alerts, emitted once per code to an optional alert sink.

| Alert | Trigger |
|---|---|
| `latency_p99` | p99 above `latency_target_ms` |
| `degraded_rate` | fallback rate above `max_degraded_rate` |
| `detection_rate_drift` | elevated-rate shift above `drift_threshold` |
| `score_distribution_drift` | mean score shift above `score_shift_threshold` |
| `capture_rate` | labeled fraud capture below `min_capture` |
| `false_positive_rate` | labeled legit elevated rate above `max_fpr` |

```python
from fraud_detection.monitoring import MetricsMonitor

monitor = MetricsMonitor(alert_sink=lambda alert: print(alert.to_dict()))
monitor.observe(result)
monitor.record_feedback(result.txn_id, 1)  # after investigator review
```

The demo passes stream labels to `observe` for evaluation only. In production,
labels arrive through `record_feedback` when an investigator acts or a
chargeback is confirmed.

## Feedback for retraining

`ReviewStore.feedback()` and `GET /api/feedback` return rows with the
transaction reference, label, feature snapshot, risk score and model version.
`run_phase4.sh --demo-reviews N` demonstrates the loop by auto-reviewing N
cases using demo ground truth.

## Runbook

```bash
./run_phase4.sh                         # generate traffic + dashboard
./run_phase4.sh --demo-reviews 12       # review a sample automatically
./run_phase4.sh --config config/decision-thresholds.example.json
./run_phase4.sh --fresh                 # clear the demo review database

# CI-style checks
python3 -m fraud_detection.phase4 --model models/fraud_model.json \
    --transactions 300 --warmup 300 --db /tmp/p4.sqlite \
    --audit-log /tmp/p4.jsonl --fresh --check

# all tests
python3 -m unittest discover -s tests -v
```

If the dashboard cannot bind its port, pass `--port 0` to choose an ephemeral
port (the printed URL is authoritative). If the review database is corrupted,
stop the process and move the `logs/phase4-reviews.sqlite*` files aside; the
store recreates the schema on the next start.

## Plan mapping

| Plan item | Implementation |
|---|---|
| Configurable per-segment thresholds | `DecisionConfig.load/from_dict/save`, `--config` |
| Rules layer on ML score | `DecisionEngine.decide`, 2FA hold-band policy |
| Investigator queue | `ReviewStore`, `list_cases`, risk sort/filter |
| Side-by-side evidence | `/api/cases/{id}`, dashboard detail panel |
| One-click actions + feedback | `POST /api/cases/{id}/actions`, `ReviewStore.act` |
| Investigator audit trail | `case_actions` table, `/actions` endpoint |
| Dashboard web UI | `dashboard/index.html`, `dashboard.js`, `dashboard.css` |
| Latency / drift / FPR monitoring | `MetricsMonitor`, `alerts`, `/api/metrics` |
| Real-time alerting | `alert_sink`, active alerts in dashboard |
| Feedback → retraining | `ReviewStore.feedback()`, `/api/feedback` |

## Limitations

- No authentication or role separation; put the dashboard behind an
  authenticated proxy before production.
- SQLite is suitable for a single demo node. A multi-node deployment needs a
  shared database service and the same store interface.
- The monitoring baseline is the first `baseline_size` observations; retrain
  or restart the monitor when the traffic mix intentionally changes.
- Alerts are computed in-process. Forward `alert_sink` to a real paging or
  webhook service for on-call delivery.
- `--demo-reviews` uses demo ground truth to populate the feedback loop; it is
  not a production labeling mechanism.
- Capture and false-positive rates use the reviewed subset, so they are
  selection-biased until every held case receives a disposition or labels
  arrive from chargebacks/complaints.
