# Phase 6 — Testing & Validation

Phase 6 validates the fraud-detection path against the plan's testing and
operational-readiness cells rather than introducing a new runtime feature. The
codebase already implements most of what Phase 6 asks for; where it does not,
the gap is an explicit deployment dependency rather than a missing feature.

```text
unit + integration regression
        │
        ▼
validation suite (feature parity, decision coverage, latency/degraded mode,
                    adversarial/missing data, artifact integrity, ops framing)
        │
        ▼
Phase 4 smoke -> Phase 5 smoke -> Phase 6 suite
        │                         │
        ▼                         ▼
scoring + dashboard +        security + governance +
reviews + monitoring          fairness + retention + RBAC
```

Everything here is standard-library only, matching the rest of the project.

## 6.1 What Phase 6 validates

| Plan cell | How it is validated | Status |
|-----------|---------------------|--------|
| Feature computation parity | `TestFeatureComputationParity` verifies the real-time feature path is identical to the offline feature definitions (`fraud_detection/features.py`, `fraud_detection/feature_store.py`) | done in test suite |
| Scoring integration | Phase 4 end-to-end demo (`fraud_detection/phase4.py`, `run_phase4.sh`) streams transactions through scoring, review, dashboard and monitoring | done in smoke run |
| Decision engine coverage | `TestDecisionCoverage` exercises allow / allow-with-logging / step-up / hold / block bands and the segment and block-confirmation overrides | done in test suite |
| Latency budget / timeout | `TestScoringLatencyAndDegradedMode` validates timeout-bounded inference and the fail-open default vs. fail-closed policy | done in test suite |
| Model validation | `models/MODEL_CARD.md` records the trained model; `fraud_detection/model.py` is the loadable artifact; synthetic fraud strategies in `fraud_detection/simulator.py` exercise the fraud path | done in smoke run + tests |
| Adversarial testing | `TestAdversarialAndMissingDataBehavior` validates malformed biometrics, cross-user telemetry, missing scorer and missing behavior | done in test suite |
| Stress test on missing/partial data | same as above, plus degraded-mode fallback when the scorer is unavailable | done in test suite |
| Operational readiness | `docs/PHASE6.md`, `fraud_detection/monitoring.py` alerts, `fraud_detection/phase5.py` rollback drill, and `TestOperationalReadinessFraming` | done in docs + tests |

## 6.2 Out of scope here (deployment dependencies)

Phase 6 asks for a few things that are deployment concerns rather than
application code. They are documented as outstanding so the checklist is honest:

- **Load test at target IBM Z transaction volume.** The Python path already
  meets the configured latency budget (`ScoringPolicy.timeout_ms` /
  `MonitorConfig.latency_target_ms`), and Phase 4/5 smoke runs plus the Phase 6
  suite exercise the scoring path. A true plateau load test at target volume on
  IBM Z is a deployment dependency, not something this deliverable runs.
- **ML for z/OS deployment and shared DB2/IMS store.** The application is
  standard-library-only and is shaped to map onto z/OS Connect / CICS / IMS /
  DB2, but the actual z/OS transport, the shared review database, and the
  ML-for-z/OS inference boundary are deployment steps. The `sdk/behavioral-biometrics.js`
  browser SDK is a Phase 7 packaging item too.
- **Capacity sizing.** Sizing for peak transaction volume depends on the actual
  IBM Z LPAR and traffic profile; the current runbook documents the knobs
  (`--transactions`, `--warmup`, `--port`, `--timeout-ms`) so operations can
  size it in the real environment.
- **On-call paging/webhook sink.** Alerts are emitted through `alert_sink`; the
  demo leaves it null. Production wiring to the paging system is a deployment
  step.
- **Real historical data and real fraud labels.** The project's stated fallback
  strategy is synthetic data plus the transaction simulator when real data is
  unavailable; that strategy is reflected in `fraud_detection/simulator.py` and
  the Phase 1–6 docs. A real dataset changes the model and the validation story.

## 6.3 Validation runbook

```bash
# Full validation pass
./run_phase6.sh

# Unit suites only (Phase 2/3/4/5/6)
./run_phase6.sh --unit

# A single suite
./run_phase6.sh --unit --only test_phase6.py

# Phase 4 smoke
./run_phase6.sh --phase4

# Phase 5 smoke
./run_phase6.sh --phase5
```

The launchers inside `run_phase4.sh` and `run_phase5.sh` are themselves part of
the validation story:

```bash
./run_phase4.sh --model models/fraud_model.json --transactions 2000 --fresh --check
./run_phase5.sh --model models/fraud_model.json --transactions 200 --demo-reviews 2 --fresh --check
python3 -m unittest discover -s tests -v
```

## 6.4 Degraded-mode policy (fail-open / fail-closed)

The transaction path is timeout-bounded. If inference exceeds the policy budget
or raises, the policy chooses the degraded action and the decision is flagged in
the audit trail.

- **Fail-open (default)** — the payment is allowed when scoring is unavailable.
  A fraud system should not stop all payments when the model is down.
- **Fail-closed** — configurable per policy; the payment is blocked when scoring
  is unavailable. Use it only where the loss from a missed fraud exceeds the
  loss from blocked legitimate payments.

Either mode writes `degraded: true` and a `fallback_reason` into the audit
entry, so degraded traffic is visible in the dashboard metrics and the audit.

## 6.5 Limitations

- The latency target in the plan is **scoring p99 under 50 ms**. The configured
  budget and the Phase 4/5/6 runs show the Python path is well within that
  budget; the production IBM Z number is the deployment boundary.
- The model registry and audit log are single-node file stores. A multi-instance
  deployment needs a transactional shared store (DB2/IMS) — see Phase 5 docs.
- Demo bearer tokens are plaintext in a `0600` file; production should use
  short-lived identity-provider tokens behind z/OS Connect / RACF.
- Fairness groups require an explicit legal basis; `txn.channel` is a demo
  grouping.
- HMAC integrity detects tampering and reordering but cannot prevent deletion of
  the entire audit file or its trailing segment; archive segments to WORM store.
- Behavioral biometrics and the browser SDK are ready in the backend; the
  browser SDK is a Phase 7 packaging item.
- No real historical transaction data or real fraud labels: the project uses the
  synthetic simulator as its demo/fallback strategy.
