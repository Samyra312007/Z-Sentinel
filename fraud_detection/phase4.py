"""Phase 4 demo: decision policy, review queue, dashboard and monitoring.

Usage:
    python -m fraud_detection.phase4 --model models/fraud_model.json
    python -m fraud_detection.phase4 --model models/fraud_model.json --check
    python -m fraud_detection.phase4 --model models/fraud_model.json \
        --transactions 2000 --demo-reviews 12 --port 8080

The command streams simulated payments through the Phase 3 scoring path,
persists hold/step-up cases, optionally records a few demo investigator
decisions, and serves the dashboard until Ctrl-C (or for `--serve-seconds`).
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import urlopen

from fraud_detection.collection import payload_from_sample
from fraud_detection.dashboard import DashboardServer
from fraud_detection.decision import (
    HOLD,
    STEP_UP,
    DecisionConfig,
)
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.model import FraudModel
from fraud_detection.monitoring import MetricsMonitor, MonitorConfig
from fraud_detection.reviews import ReviewStore
from fraud_detection.service import AuditLog, ScoringPolicy, ScoringService
from fraud_detection.simulator import TransactionSimulator


@dataclass
class Phase4Summary:
    total: int = 0
    fraud: int = 0
    hold_cases: int = 0
    stepup_cases: int = 0
    queued: int = 0
    queue_failures: int = 0
    demo_reviews: int = 0
    audit_records: int = 0
    cases: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def expected_cases(self) -> int:
        return self.hold_cases + self.stepup_cases


def run(
    *,
    transactions: int,
    users: int,
    fraud_rate: float,
    seed: int,
    warmup: int,
    model_path: str | None,
    config: DecisionConfig | None,
    db_path: str,
    audit_path: str,
    fresh: bool = False,
    demo_reviews: int = 0,
    monitor_config: MonitorConfig | None = None,
    scorer: Any | None = None,
) -> tuple[Phase4Summary, ReviewStore, MetricsMonitor, str]:
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            path = db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    model = FraudModel.load(model_path) if model_path else None
    if scorer is None and model is not None:
        scorer = EnsembleScorer(model)
    audit = AuditLog(audit_path)
    existing_audit = len(AuditLog.read(audit_path)) if audit_path and os.path.exists(audit_path) else 0
    store = ReviewStore(db_path)
    monitor = MetricsMonitor(monitor_config or MonitorConfig())
    service = ScoringService(
        model=model,
        scorer=scorer,
        config=config or DecisionConfig(),
        audit=audit,
        review_sink=store.add_case,
    )
    sim = TransactionSimulator(n_users=users, fraud_rate=fraud_rate, seed=seed)
    summary = Phase4Summary()
    truth: dict[str, bool] = {}

    service.warm(sim.events(max(0, warmup)))
    for txn in sim.events(transactions):
        payload = payload_from_sample(txn.behavior) if txn.behavior else None
        result = service.score(txn, payload, correlation_id=f"CORR-{txn.txn_id}")
        monitor.observe(result, label=txn.is_fraud)  # demo labels only
        truth[txn.txn_id] = txn.is_fraud
        summary.total += 1
        if txn.is_fraud:
            summary.fraud += 1
        if result.action == HOLD:
            summary.hold_cases += 1
        elif result.action == STEP_UP:
            summary.stepup_cases += 1
        if result.review_queued:
            summary.queued += 1
        if result.review_error:
            summary.queue_failures += 1
    service.close()

    if demo_reviews > 0:
        for case in store.list_cases(status="OPEN", limit=min(demo_reviews, 200)):
            labeled = truth.get(case["txn_id"])
            if labeled is None:
                continue
            action = "confirm_fraud" if labeled else "clear_fp"
            store.act(case["case_id"], action, "demo-analyst", "demo review")
            monitor.record_feedback(case["txn_id"], 1 if labeled else 0)
            summary.demo_reviews += 1

    summary.audit_records = (
        len(AuditLog.read(audit_path)) - existing_audit if audit_path else 0
    )
    summary.cases = store.stats()
    summary.metrics = monitor.snapshot()
    version = (
        model.version if model is not None
        else getattr(scorer, "version", "rules-only")
    )
    return summary, store, monitor, version


def _dashboard_check(url: str) -> dict:
    with urlopen(url + "/api/health", timeout=5) as response:
        health = json.loads(response.read().decode("utf-8"))
    with urlopen(url + "/api/cases?status=all&limit=1", timeout=5) as response:
        cases = json.loads(response.read().decode("utf-8"))
    with urlopen(url + "/api/metrics", timeout=5) as response:
        metrics = json.loads(response.read().decode("utf-8"))
    return {"health": health, "cases": cases, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4 fraud operations demo")
    p.add_argument("--transactions", type=int, default=2000)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--model", default=None)
    p.add_argument("--config", default=None, help="decision threshold JSON")
    p.add_argument("--db", default="logs/phase4-reviews.sqlite")
    p.add_argument("--audit-log", default="logs/phase4-decisions.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fresh", action="store_true", help="clear the demo review database")
    p.add_argument("--demo-reviews", type=int, default=0,
                   help="auto-review N queued cases using demo truth")
    p.add_argument("--serve-seconds", type=int, default=0,
                   help="serve this many seconds (0 = until Ctrl-C)")
    p.add_argument("--no-serve", action="store_true")
    p.add_argument("--check", action="store_true", help="verify dashboard API and exit")
    args = p.parse_args(argv)

    if args.model and not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    config = DecisionConfig.load(args.config) if args.config else DecisionConfig()
    if not 0.0 < args.fraud_rate < 1.0:
        print("--fraud-rate must be in (0, 1)", file=sys.stderr)
        return 1
    if args.demo_reviews < 0 or args.serve_seconds < 0:
        print("--demo-reviews and --serve-seconds must be non-negative", file=sys.stderr)
        return 1

    summary, store, monitor, model_version = run(
        transactions=args.transactions,
        users=args.users,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        warmup=args.warmup,
        model_path=args.model,
        config=config,
        db_path=args.db,
        audit_path=args.audit_log,
        fresh=args.fresh,
        demo_reviews=args.demo_reviews,
    )

    metrics = summary.metrics
    print("\n=== Phase 4: decision engine, review queue and monitoring ===")
    print(f"scoring path           : {model_version}")
    print(f"transactions processed : {summary.total}")
    print(f"fraud in stream        : {summary.fraud}")
    print(f"cases queued           : {summary.queued} "
          f"(hold={summary.hold_cases}, step-up={summary.stepup_cases}, "
          f"failures={summary.queue_failures})")
    print(f"review queue           : open={summary.cases.get('open', 0)}, "
          f"confirmed={summary.cases.get('confirmed_fraud', 0)}, "
          f"cleared={summary.cases.get('cleared', 0)}, "
          f"need-info={summary.cases.get('need_info', 0)}")
    if summary.demo_reviews:
        print(f"demo investigator      : {summary.demo_reviews} cases reviewed")
    latency = metrics.get("latency_ms", {})
    labeling = metrics.get("labeling", {})
    print(f"scoring latency        : p50={latency.get('p50', 0):.2f}ms "
          f"p95={latency.get('p95', 0):.2f}ms p99={latency.get('p99', 0):.2f}ms")
    if labeling.get("capture_rate") is not None:
        print(f"capture / false pos    : {100 * labeling['capture_rate']:.1f}% / "
              f"{100 * (labeling.get('false_positive_rate') or 0):.1f}%")
    alerts = metrics.get("alerts", [])
    print(f"active alerts          : {len(alerts)}"
          + (f" ({', '.join(a['code'] for a in alerts)})" if alerts else ""))

    checks = {
        "cases_match_actions": summary.queued == summary.expected_cases,
        "queue_failures_zero": summary.queue_failures == 0,
        "audit_complete": summary.audit_records == summary.total,
        "queue_has_evidence": _queue_has_evidence(store, summary),
        "metrics_have_window": metrics.get("window_size", 0) > 0,
    }
    print("\n--- checks ---")
    for name, ok in checks.items():
        print(f"{name:24s}: {'PASS' if ok else 'FAIL'}")

    server = DashboardServer(
        store, monitor,
        config=config,
        model_version=model_version,
        host=args.host,
        port=args.port,
    )
    dashboard_ok = True
    try:
        if args.check or not args.no_serve:
            url = server.start()
            print(f"\ndashboard              : {url}")
            if args.check:
                try:
                    probe = _dashboard_check(url)
                    dashboard_ok = (
                        probe["health"].get("status") == "ok"
                        and probe["metrics"].get("window_size", 0) > 0
                    )
                    print(f"dashboard API          : {'PASS' if dashboard_ok else 'FAIL'}")
                except Exception as exc:
                    dashboard_ok = False
                    print(f"dashboard API          : FAIL ({type(exc).__name__}: {exc})")
            elif args.serve_seconds > 0:
                print(f"serving for {args.serve_seconds}s (Ctrl-C to stop)")
                time.sleep(args.serve_seconds)
            else:
                print("serving until Ctrl-C")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        else:
            print("\ndashboard              : not started (--no-serve)")
    finally:
        server.stop()
        store.close()

    ok = all(checks.values()) and dashboard_ok
    print(f"\nPhase 4 integration    : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _queue_has_evidence(store: ReviewStore, summary: Phase4Summary) -> bool:
    if summary.expected_cases == 0:
        return True
    rows = store.list_cases(status=None, limit=200)
    for row in rows:
        if not row.get("top_signals") or row.get("behavior_status") is None:
            return False
        detail = store.get(row["case_id"])
        if not detail or not detail.get("features"):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
