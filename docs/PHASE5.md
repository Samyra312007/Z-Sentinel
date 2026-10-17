# Phase 5 — Security, Compliance & Responsible AI

Phase 5 hardens the operations layer and makes every model decision
explainable, governed and reviewable:

```
transaction ──► scoring ──► explanation + decision ──► tamper-evident audit
                    │                    │
                    │                    ├── HOLD / STEP_UP ──► review queue
                    │                    └── block confirmation by segment
                    ▼
          model registry (hash, approve, activate, rollback)
                    │
                    └── governance events ──► same audit chain

monitoring ──► per-group capture / FPR / gaps ──► fairness alerts
closed cases ──► retention policy ──► purge
```

Everything remains standard-library only. The demo uses a local TLS-capable
HTTP server, SQLite and a file-backed registry; the same interfaces map to
z/OS Connect, RACF, DB2 and an enterprise keyring in production.

## 5.1 IBM Z security integration

### Encryption in transit

`fraud_detection.security.build_server_ssl_context(certfile, keyfile)` builds a
TLS 1.2+ server context. Pass it to `DashboardServer(tls_context=...)`; the
server then serves HTTPS and the printed URL uses `https://`.

```python
from fraud_detection.dashboard import DashboardServer
from fraud_detection.security import build_server_ssl_context

server = DashboardServer(
    store, monitor,
    access=policy,
    tls_context=build_server_ssl_context("/run/secrets/dashboard.crt",
                                         "/run/secrets/dashboard.key"),
)
```

Certificate and key material belong in the z/OS keyring or a mounted secret,
never in the repository.

### Encryption at rest

The application writes pseudonymous references, aggregate features and review
evidence; it does not store PANs or raw behavioral input. At-rest encryption is
delegated to the platform: zFS encryption for the audit/registry files and DB2
or IMS encryption for the review tables, with CPACF-backed key material and
RACF keyrings. This is a deployment requirement, not application code.

### Access control

`fraud_detection/access.py` implements bearer-token authentication with
constant-time digest comparison and three roles:

| Role | Scopes | Can do |
|---|---|---|
| `viewer` | `read` | queue, cases, metrics, config, model catalogue |
| `investigator` | `read`, `review` | viewer plus case dispositions |
| `admin` | `read`, `review`, `admin` | investigator plus feedback export and model governance |

The dashboard asks for a token in the browser and stores it in `localStorage`.
`AccessPolicy.save` writes the demo token file with mode `0600`; in production,
issue short-lived tokens from the enterprise identity provider and map RACF
profiles / groups to these roles. Without a policy the server keeps the Phase 4
localhost-only behaviour.

```python
from fraud_detection.access import ADMIN, VIEWER, AccessPolicy

policy = AccessPolicy()
policy.add_token(token, "analyst-7", ADMIN)
server = DashboardServer(store, monitor, access=policy)
```

### Immutable audit trail

`SecureAuditLog` appends one JSONL record per decision **and** per governance
event. Each record contains:

```json
{"seq": 42, "prev_hash": "…", "hash": "…", "mac": "…", "entry": {"…": "…"}}
```

- SHA-256 covers `seq`, `prev_hash` and the canonical entry;
- HMAC-SHA256 authenticates the hash with a key from `FRAUD_AUDIT_KEY` or a
  file created with mode `0600`;
- editing, deleting or reordering a record breaks `verify()`, and the log
  refuses to open when verification fails.

```python
from fraud_detection.security import SecureAuditLog, ensure_key_file

key = ensure_key_file("logs/phase5-audit.key")
with SecureAuditLog("logs/phase5-decisions.jsonl", key) as audit:
    audit.write(result)                  # scoring decision
    audit.write({                        # non-decision event
        "event_type": "threshold_change",
        "actor": "release-1",
        "segment": "high_value",
    })
    assert audit.verify().ok
```

The decision entry includes the model version, threshold segment, action,
outcome, latency, degradation reason, top signals and the full explanation
list. Governance events include register, approve, reject, activate and
rollback with the actor and previous active version.

### Secure execution and artifact integrity

Model artifacts are plain auditable JSON. The registry stores a SHA-256 digest
at registration and re-hashes the file before every approval and activation,
so a swapped artifact cannot inherit an approved version. In production the
artifact directory is read-only under RACF, and inference runs in the
ML-for-z/OS secure execution boundary; the application-side fallback is the
same in-process scorer used by the demo.

## 5.2 Responsible AI controls

### Explainability

Every `ScoreResult`, audit record and review case carries a signed contribution
list and the attribution method:

```json
"top_signals": ["new recipient", "amount vs. user's 30-day history"],
"explanations": [
  {"feature": "new_recipient_flag", "label": "new recipient",
   "contribution": 0.42, "direction": "increases_risk"},
  {"feature": "amount_dev_30d_z", "label": "amount vs. user's 30-day history",
   "contribution": 0.31, "direction": "increases_risk"}
],
"explanation_method": "gbdt_saabas_path_attribution"
```

The model path uses Saabas path attribution over the gradient-boosted trees
(`GBDT.contributions`) and adds behavioral and ring contributions. The
rules-only fallback emits the deterministic rule points it applied. The
dashboard renders these values as the “Why it was flagged” bars, and
`GET /api/cases/{id}` is the right-to-explanation payload an analyst can share
with a customer or regulator.

### Bias and fairness monitoring

`MetricsMonitor.observe(result, label=..., group=...)` accepts an optional
group. `snapshot()["fairness"]` reports, per group:

- volume and elevated-risk rate;
- capture rate on labeled fraud;
- false-positive rate on labeled legitimate payments;
- the max capture gap and max FPR gap across comparable groups.

Groups smaller than `min_group_labeled` are excluded from gap calculations.
Alerts `fairness_capture_gap` and `fairness_fpr_gap` fire when a gap exceeds
`max_capture_gap` / `max_fpr_gap`. The demo groups by channel to exercise the
mechanism; production groups must be defined with legal and privacy review,
and only where collecting the attribute is lawful. Do not infer protected
attributes from proxies.

### Human in the loop

`DecisionConfig.block_confirmation_segments` lists segments whose `BLOCK`
decisions must be confirmed by an analyst instead of declining immediately:

```json
{
  "block_confirmation_segments": ["high_value"]
}
```

Those decisions return `HOLD_FOR_REVIEW`, set
`block_confirmation_required: true` and `proposed_action: "BLOCK"`, and are
queued with the evidence. `confirm_fraud` records the analyst's confirmation
and the label for retraining; `clear_fp` releases the payment. Unlisted
segments keep the normal block path, and the default is an empty list so
existing behaviour is unchanged.

### Model governance

`fraud_detection/governance.py` provides a file-backed registry:

```
CANDIDATE --approve--> APPROVED --activate--> ACTIVE
     |                     |                     |
     +------reject-------->+                     +--rollback--> RETIRED
```

- `register(path, version, actor, metrics)` hashes the artifact and records the
  trainer and metrics.
- `approve` is the validation gate; only approved artifacts can be activated.
- `activate` re-hashes the file, retires the previous active model and records
  `previous_active`.
- `rollback` restores the previous active (or an explicit) approved version,
  again after a hash check.
- Every transition can write a `model_governance` event to the secure audit.

```python
from fraud_detection.governance import ModelRegistry

registry = ModelRegistry("logs/phase5-model-registry.json", audit=audit)
registry.register("models/fraud_model.json", version="phase2-1", actor="trainer")
registry.approve("phase2-1", actor="validator", note="holdout approved")
registry.activate("phase2-1", actor="release")
registry.rollback(actor="release")          # back to the previous active model
```

The version, digest, approval actors, metrics and event history are available
through `registry.summary()`, `GET /api/models` (viewer) and the admin
promotion endpoints. Model cards remain the human-readable validation record.

### Privacy

- The SDK sends session aggregates only; `parse_payload` rejects raw text,
  key codes, event arrays and pointer paths.
- Customer identity is replaced by a salted SHA-256 reference before scoring;
  raw customer IDs never reach the audit, review store or dashboard.
- Review cases store aggregate features and evidence, not raw behavioral input.
- `RetentionPolicy` deletes closed cases and their action history after the
  configured window. Open and `NEED_INFO` cases are retained, because deleting
  them would erase pending work.
- The decision audit is immutable and is not rewritten. Ship immutable audit
  segments to the enterprise archive under the records-retention schedule.

```python
from fraud_detection.privacy import RetentionPolicy

result = RetentionPolicy(review_days=365).apply(store)
# {"enabled": True, "cutoff": "...", "cases": 4, "actions": 6}
```

## 5.3 Compliance notes

- **PCI DSS** — no PAN, CVV or track data is accepted or stored. Tokens and
  salted subject references are used throughout; the card-data boundary is the
  issuer/payment processor, not this service. Confirm scope with a QSA.
- **GDPR / CCPA** — the per-case explanation and audit entry provide the
  “right to explanation”; pseudonymous references support access/erasure
  workflows, and retention removes closed review data. The immutable audit
  retention period must be reconciled with the erasure policy by legal.
- **SR 11-7-style model risk management** — the model card covers development
  and validation; the registry records approval, activation and rollback; the
  monitoring snapshot covers ongoing performance, drift and fairness. The
  `docs/` files are the model-risk evidence pack.
- **Audit retention** — the HMAC chain is tamper-evident, not a substitute for
  WORM storage. Archive segments to immutable storage and protect the audit key
  in RACF/z/OS keyrings.

## Dashboard API additions

| Method | Path | Scope | Purpose |
|---|---|---|---|
| `GET` | `/api/models` | `read` | Registry summary, active model, events |
| `POST` | `/api/models/{version}/approve` | `admin` | Approve a candidate |
| `POST` | `/api/models/{version}/reject` | `admin` | Reject a candidate |
| `POST` | `/api/models/{version}/activate` | `admin` | Serve an approved model |
| `POST` | `/api/models/rollback` | `admin` | Restore the previous active model |

```bash
TOKEN=$(python3 -c 'import json;print(json.load(open("logs/phase5-tokens.json"))["tokens"][2]["token"])')
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8085/api/models | python3 -m json.tool
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"actor":"release-1","note":"approved"}' \
  http://127.0.0.1:8085/api/models/phase2-1/approve
```

Use the third token in the demo file for `admin`; the first is `viewer`, the
second `investigator`. The file is created with mode `0600`.

## Runbook

```bash
./run_phase5.sh                                  # full demo + dashboard
./run_phase5.sh --check                          # CI-style verification and exit
./run_phase5.sh --fresh --transactions 500 --warmup 500 --demo-reviews 4
./run_phase5.sh --config config/decision-thresholds.example.json
./run_phase5.sh --no-serve                       # data run only

python3 -m fraud_detection.phase5 --model models/fraud_model.json \
    --transactions 500 --warmup 500 --demo-reviews 4 --demo-rollback --check \
    --db /tmp/p5.sqlite --audit-log /tmp/p5.jsonl --audit-key /tmp/p5.key \
    --registry /tmp/p5-registry.json --tokens /tmp/p5-tokens.json --port 0
```

The demo checks audit integrity/completeness, an active governed model, at
least one fairness group, the retention policy, the rollback drill (when
enabled), bearer-token enforcement, governance API, case explanations,
fairness API and dashboard API.

## Verified controls

`python3 -m unittest discover -s tests` covers:

- HMAC chain verification, tamper/reorder/wrong-key rejection, owner-only key
  files and token files;
- RBAC scopes, 401/403 boundaries, HTTPS round trip;
- signed explanations from the model, rules path and service audit;
- block-confirmation segments and validation;
- registry approval gate, hash-tamper rejection and rollback;
- retention purge of closed cases while preserving pending work;
- per-group fairness rates and gap alerts.

## Limitations

- The demo token file contains plaintext bearer tokens; issue short-lived
  tokens from an identity provider in production and keep the file `0600`.
- The registry is a single-node JSON file. Multi-instance deployment needs a
  shared transactional table (DB2) and a locking/compare-and-set strategy.
- HMAC integrity detects tampering but does not prevent deletion of the whole
  file or of a trailing segment; ship segments to WORM storage.
- At-rest encryption and secure enclave execution are platform controls, not
  implemented by this application.
- Fairness groups are only as meaningful as the legal basis and data quality
  behind them; channel is a demo grouping, not a protected attribute.
- Retention purges closed review cases only. Audit retention follows the
  enterprise archive schedule.
