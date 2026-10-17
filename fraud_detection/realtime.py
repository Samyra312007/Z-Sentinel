"""Phase 3 demo: real-time scoring integration end to end.

Streams simulated payments through the production-shaped path:

    SDK behavioral payload -> feature store -> timeout-bounded model call
    -> decision rules -> commit / step-up / hold / decline -> audit entry

Usage:
    python -m fraud_detection.realtime --model models/fraud_model.json
    python -m fraud_detection.realtime --model models/fraud_model.json \
        --audit-log logs/phase3-decisions.jsonl --verbose

Reports capture rate / false positive rate, action and settlement outcomes,
timeout degradation counts, feature-store read latency, total scoring latency
against the plan's p99 < 50ms target, and verifies that every decision has a
complete audit record.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, field

from fraud_detection.collection import payload_from_sample
from fraud_detection.decision import BLOCK, HOLD, STEP_UP
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.model import FraudModel
from fraud_detection.service import (
    FAIL_CLOSED,
    FAIL_OPEN,
    AuditLog,
    ScoringPolicy,
    ScoringService,
)
from fraud_detection.simulator import TransactionSimulator

FLAGGED_ACTIONS = (STEP_UP, HOLD, BLOCK)
ELEVATED_RISK = 60.0
AUDIT_REQUIRED = {
    "decided_at", "correlation_id", "txn_id", "user_ref", "risk_score",
    "action", "outcome", "segment", "model_version", "top_signals",
    "latency_ms", "behavior_status", "degraded",
}


class FaultInjector:
    """Wraps the ensemble to simulate slow inference and exercise timeouts."""

    def __init__(self, inner, rate: float, delay_ms: float, seed: int = 42) -> None:
        self.inner = inner
        self.rate = rate
        self.delay_ms = delay_ms
        self.rng = random.Random(seed)

    def score(self, features: dict, behavior_score: float, graph_features: dict) -> dict:
        if self.rng.random() < self.rate:
            time.sleep(self.delay_ms / 1000.0)
        return self.inner.score(features, behavior_score, graph_features)


@dataclass
class Summary:
    total: int = 0
    fraud: int = 0
    fraud_elevated: int = 0
    fraud_escalated: int = 0
    legit_elevated: int = 0
    escalated: int = 0
    behavior_payloads: int = 0
    opt_outs: int = 0
    behavior_rejected: int = 0
    degraded: int = 0
    timeouts: int = 0
    errors: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    outcome_counts: dict[str, int] = field(default_factory=dict)
    strategy_hits: dict[str, int] = field(default_factory=dict)
    degraded_actions: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    feature_latencies_ms: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.fraud_elevated / self.fraud if self.fraud else 0.0

    @property
    def fp_rate(self) -> float:
        legit = self.total - self.fraud
        return self.legit_elevated / legit if legit else 0.0


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _record(summary: Summary, txn, result) -> None:
    summary.total += 1
    if result.behavior_status == "scored":
        summary.behavior_payloads += 1
    elif result.behavior_status == "opted_out":
        summary.opt_outs += 1
    elif result.behavior_status == "rejected":
        summary.behavior_rejected += 1
    summary.action_counts[result.action] = summary.action_counts.get(result.action, 0) + 1
    summary.outcome_counts[result.outcome] = summary.outcome_counts.get(result.outcome, 0) + 1
    summary.latencies_ms.append(result.latency_ms)
    summary.feature_latencies_ms.append(result.feature_latency_ms)

    if result.degraded:
        summary.degraded += 1
        reason = (result.fallback_reason or "").split(":")[0]
        if reason == "timeout":
            summary.timeouts += 1
        else:
            summary.errors += 1
        summary.degraded_actions[result.action] = summary.degraded_actions.get(result.action, 0) + 1

    elevated = result.risk_score >= ELEVATED_RISK
    if elevated and not txn.is_fraud:
        summary.legit_elevated += 1
    if result.action in FLAGGED_ACTIONS:
        summary.escalated += 1
    if txn.is_fraud:
        summary.fraud += 1
        strategy = txn.fraud_strategy or "unknown"
        summary.strategy_hits[strategy] = summary.strategy_hits.get(strategy, 0) + 1
        if elevated:
            summary.fraud_elevated += 1
        if result.action in FLAGGED_ACTIONS:
            summary.fraud_escalated += 1


def run(
    n_transactions: int,
    n_users: int,
    fraud_rate: float,
    seed: int,
    warmup: int,
    model_path: str | None,
    timeout_ms: float,
    failure_mode: str,
    audit_path: str,
    fault_rate: float,
    fault_delay_ms: float,
    opt_out_rate: float,
    verbose: bool,
    out=sys.stdout,
) -> tuple[Summary, ScoringService, dict]:
    model = FraudModel.load(model_path) if model_path else None
    scorer = EnsembleScorer(model) if model is not None else None
    if scorer is not None and fault_rate > 0:
        scorer = FaultInjector(scorer, fault_rate, fault_delay_ms, seed=seed)

    audit = AuditLog(audit_path)
    existing = 0
    if audit_path and os.path.exists(audit_path):
        existing = len(AuditLog.read(audit_path))
    service = ScoringService(
        model=model,
        scorer=scorer,
        policy=ScoringPolicy(timeout_ms=timeout_ms, failure_mode=failure_mode),
        audit=audit,
        subject_salt=None,
    )
    sim = TransactionSimulator(n_users=n_users, fraud_rate=fraud_rate, seed=seed)
    summary = Summary()
    rng = random.Random(seed + 7)

    warmed = service.warm(sim.events(max(0, warmup)))
    print(f"warm-up                : {warmed} historical events "
          f"(feature store users={service.store.users})", file=out)

    stream_started = time.perf_counter()
    for txn in sim.events(n_transactions):
        payload = None
        if txn.behavior is not None:
            opted_out = rng.random() < opt_out_rate
            payload = payload_from_sample(txn.behavior, consent=not opted_out)
        result = service.score(
            txn, payload, correlation_id=f"CORR-{txn.txn_id}"
        )
        _record(summary, txn, result)

        if verbose:
            signals = ", ".join(result.top_signals)
            extra = " DEGRADED" if result.degraded else ""
            print(
                f"{txn.txn_id} {txn.timestamp:%Y-%m-%d %H:%M} "
                f"score={result.risk_score:6.2f} {result.action:18s} "
                f"outcome={result.outcome:15s} {result.latency_ms:6.2f}ms "
                f"signals=[{signals}]{extra}"
                + (f" truth=FRAUD/{txn.fraud_strategy}" if txn.is_fraud else ""),
                file=out,
            )
    elapsed = time.perf_counter() - stream_started
    service.close()

    records = AuditLog.read(audit_path) if audit_path else list(audit.records)
    new_records = records[existing:]
    checks = _verify(summary, new_records, service, elapsed, fault_rate, failure_mode)
    return summary, service, checks


def _verify(
    summary: Summary,
    records: list[dict],
    service: ScoringService,
    elapsed: float,
    fault_rate: float,
    failure_mode: str,
) -> dict:
    missing = [r.get("txn_id") for r in records if not AUDIT_REQUIRED.issubset(r)]
    no_explanation = [r.get("txn_id") for r in records if not r.get("top_signals")]
    no_version = [r.get("txn_id") for r in records if not r.get("model_version")]
    degraded_records = [r for r in records if r.get("degraded")]
    expected_fallback = "BLOCK" if failure_mode == FAIL_CLOSED else "ALLOW"

    return {
        "measured": summary.total,
        "audit_records": len(records),
        "audit_complete": len(records) == summary.total,
        "audit_missing_fields": missing,
        "audit_missing_explanation": no_explanation,
        "audit_missing_version": no_version,
        "audit_user_refs_hashed": all(
            str(r.get("user_ref", "")).startswith("sha256:") for r in records
        ),
        "degraded_records": len(degraded_records),
        "degraded_records_match": len(degraded_records) == summary.degraded,
        "degraded_policy_ok": all(
            r.get("action") == expected_fallback for r in degraded_records
        ),
        "feature_users": service.store.users,
        "feature_transactions": service.store.stats()["transactions"],
        "elapsed_s": elapsed,
        "throughput": summary.total / elapsed if elapsed > 0 else 0.0,
        "latency_target_applicable": fault_rate == 0.0,
        "scoring_p99_ms": _p(summary.latencies_ms, 0.99),
        "feature_p99_ms": _p(summary.feature_latencies_ms, 0.99),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 3 real-time scoring integration demo"
    )
    p.add_argument("--transactions", type=int, default=2000)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--model", default=None, help="trained model JSON (default: rules fallback)")
    p.add_argument("--timeout-ms", type=float, default=50.0, help="scoring budget (default 50)")
    p.add_argument("--fail-mode", choices=("open", "closed"), default="open",
                   help="degraded-mode policy (default open => allow)")
    p.add_argument("--audit-log", default="logs/phase3-decisions.jsonl",
                   help="append-only decision audit trail")
    p.add_argument("--fault-rate", type=float, default=0.0,
                   help="fraction of scoring calls delayed past the timeout (default 0)")
    p.add_argument("--fault-delay-ms", type=float, default=250.0)
    p.add_argument("--opt-out-rate", type=float, default=0.0,
                   help="fraction of sessions that revoke biometric consent (default 0)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    if args.model and not os.path.exists(args.model):
        print(f"Model not found: {args.model}\n"
              f"Train it first:  python -m fraud_detection.train --out {args.model}",
              file=sys.stderr)
        return 1
    if not 0.0 <= args.fault_rate <= 1.0 or not 0.0 <= args.opt_out_rate <= 1.0:
        print("--fault-rate and --opt-out-rate must be in [0, 1]", file=sys.stderr)
        return 1

    failure_mode = FAIL_OPEN if args.fail_mode == "open" else FAIL_CLOSED
    s, service, checks = run(
        args.transactions, args.users, args.fraud_rate, args.seed, args.warmup,
        args.model, args.timeout_ms, failure_mode, args.audit_log,
        args.fault_rate, args.fault_delay_ms, args.opt_out_rate, args.verbose,
    )
    if s.total == 0:
        print("No transactions generated (check --transactions).", file=sys.stderr)
        return 1

    mode = "Phase 2 model ensemble" if args.model else "rules-only fallback"
    print("\n=== Phase 3: real-time scoring integration ===")
    print(f"scoring path           : {mode} (model {service.model_version})")
    fallback_action = "ALLOW" if failure_mode == FAIL_OPEN else "BLOCK"
    print(f"policy                 : timeout={args.timeout_ms:g}ms, "
          f"fail-{args.fail_mode} -> {fallback_action}")
    print(f"transactions processed : {s.total} (after {args.warmup} warm-up events)")
    print(f"throughput             : {checks['throughput']:.0f} txns/s")
    print(f"fraud in stream        : {s.fraud} ({100 * s.fraud / s.total:.1f}%)")
    print(f"action mix             : " + ", ".join(
        f"{k}={v}" for k, v in sorted(s.action_counts.items())))
    print(f"settlement outcomes    : " + ", ".join(
        f"{k}={v}" for k, v in sorted(s.outcome_counts.items())))
    print(f"fraud score >= 60      : {s.fraud_elevated}/{s.fraud} -> capture rate {100 * s.recall:.1f}%")
    print(f"fraud escalated        : {s.fraud_escalated}/{s.fraud} (step-up/hold/block)")
    print(f"legit score >= 60      : {s.legit_elevated} -> false positive rate {100 * s.fp_rate:.1f}%")
    print(f"biometric payloads     : {s.behavior_payloads} scored, {s.opt_outs} opted out, "
          f"{s.behavior_rejected} rejected")
    print(f"degraded decisions     : {s.degraded} "
          f"(timeouts={s.timeouts}, errors={s.errors}, "
          f"fallbacks={s.degraded_actions or '{}'})")
    print(f"feature store          : {checks['feature_users']} users, "
          f"{checks['feature_transactions']} applied transactions")

    print("\n--- latency ---")
    print(f"feature store read     : p99={checks['feature_p99_ms']:.3f}ms "
          f"(target sub-ms -> {'PASS' if checks['feature_p99_ms'] < 1.0 else 'FAIL'})")
    print(f"full scoring path      : p50={_p(s.latencies_ms, 0.50):.3f}ms "
          f"p95={_p(s.latencies_ms, 0.95):.3f}ms p99={checks['scoring_p99_ms']:.3f}ms")
    if checks["latency_target_applicable"]:
        print(f"latency target         : p99 < {args.timeout_ms:g}ms -> "
              f"{'PASS' if checks['scoring_p99_ms'] < args.timeout_ms else 'FAIL'}")
    else:
        print("latency target         : not applicable (fault injection enabled)")

    print("\n--- audit trail ---")
    print(f"audit log              : {args.audit_log} (+{checks['audit_records']} entries)")
    print(f"one record per decision: {checks['audit_complete']} "
          f"({checks['audit_records']}/{checks['measured']})")
    print(f"required fields        : {'PASS' if not checks['audit_missing_fields'] else 'FAIL'}")
    print(f"explanation present    : {'PASS' if not checks['audit_missing_explanation'] else 'FAIL'}")
    print(f"model version present  : {'PASS' if not checks['audit_missing_version'] else 'FAIL'}")
    print(f"identities pseudonymous: {'PASS' if checks['audit_user_refs_hashed'] else 'FAIL'}")
    print(f"degraded policy honored: "
          f"{'PASS' if checks['degraded_policy_ok'] and checks['degraded_records_match'] else 'FAIL'}")

    ok = (
        checks["audit_complete"]
        and not checks["audit_missing_fields"]
        and not checks["audit_missing_explanation"]
        and not checks["audit_missing_version"]
        and checks["audit_user_refs_hashed"]
        and checks["degraded_policy_ok"]
        and checks["degraded_records_match"]
        and (not checks["latency_target_applicable"]
             or checks["scoring_p99_ms"] < args.timeout_ms)
    )
    print(f"\nPhase 3 integration    : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
