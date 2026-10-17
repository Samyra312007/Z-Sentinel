"""Phase 5 demo: security, governance, explainability and privacy.

Usage:
    python -m fraud_detection.phase5 --model models/fraud_model.json --check
    python -m fraud_detection.phase5 --model models/fraud_model.json \
        --transactions 1500 --demo-reviews 8 --check
    ./run_phase5.sh

The run streams payments through the Phase 4 path with a tamper-evident audit
log, RBAC/TLS-ready dashboard, model registry and retention policy. It then
verifies every control and (optionally) serves the dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fraud_detection.access import load_or_create_tokens
from fraud_detection.dashboard import DashboardServer
from fraud_detection.decision import DecisionConfig
from fraud_detection.governance import CANDIDATE, ModelRegistry
from fraud_detection.model import FraudModel
from fraud_detection.monitoring import MonitorConfig
from fraud_detection.phase4 import Phase4Summary, run as run_phase4
from fraud_detection.privacy import RetentionPolicy
from fraud_detection.security import SecureAuditLog, ensure_key_file


@dataclass
class Phase5Summary:
    phase4: Phase4Summary
    audit_ok: bool
    audit_entries: int
    registry_active: str | None
    registry_events: int
    fairness_groups: int
    retention: dict[str, Any]
    demo_rollback_ok: bool = False
    checks: dict[str, bool] = field(default_factory=dict)


def _remove_fresh(paths: list[str]) -> None:
    for path in paths:
        for suffix in ("", "-wal", "-shm"):
            target = path + suffix
            if os.path.exists(target):
                os.remove(target)


def _prepare_registry(
    registry: ModelRegistry,
    model_path: str,
    model: FraudModel,
    *,
    demo_rollback: bool,
) -> tuple[str, bool]:
    """Register, approve and activate the model; optionally exercise rollback."""
    version = model.version
    record = registry.get(version)
    if record is None:
        record = registry.register(
            model_path,
            version=version,
            actor="phase5-demo",
            notes="phase 5 model governance demo",
            metrics=model.metrics,
        )
    if record["status"] == CANDIDATE:
        registry.approve(version, actor="phase5-validator", note="demo approval")
    if registry.active_version != version:
        registry.activate(version, actor="phase5-release", note="demo activation")

    rollback_ok = True
    if demo_rollback and registry.active_version == version:
        candidate_path = os.path.join(
            os.path.dirname(os.path.abspath(registry.path)),
            "phase5-candidate-model.json",
        )
        candidate_version = f"{version}-candidate"
        if registry.get(candidate_version) is None:
            if os.path.exists(candidate_path):
                os.remove(candidate_path)
            candidate = FraudModel.load(model_path)
            candidate.version = candidate_version
            candidate.created_at = model.created_at
            candidate.save(candidate_path)
            registry.register(
                candidate_path,
                version=candidate_version,
                actor="phase5-demo",
                notes="rollback drill candidate",
            )
        if registry.get(candidate_version)["status"] == CANDIDATE:
            registry.approve(candidate_version, actor="phase5-validator")
        registry.activate(candidate_version, actor="phase5-release")
        registry.rollback(actor="phase5-release", to_version=version)
        rollback_ok = registry.active_version == version
    return version, rollback_ok


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
    key_path: str,
    registry_path: str,
    token_path: str,
    retention_days: int,
    confirm_blocks: tuple[str, ...],
    demo_reviews: int,
    fresh: bool = False,
    demo_rollback: bool = False,
) -> tuple[Phase5Summary, Any, Any, str]:
    if fresh:
        _remove_fresh([db_path, audit_path, registry_path, token_path, key_path])
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(registry_path)),
            "phase5-candidate-model.json",
        )
        if os.path.exists(candidate):
            os.remove(candidate)

    for directory in {
        os.path.dirname(os.path.abspath(path))
        for path in (db_path, audit_path, registry_path, token_path, key_path)
    }:
        if directory:
            os.makedirs(directory, exist_ok=True)

    key = ensure_key_file(key_path)
    audit = SecureAuditLog(audit_path, key)
    access, _created = load_or_create_tokens(token_path)

    model = FraudModel.load(model_path) if model_path else None
    registry: ModelRegistry | None = None
    active_version: str | None = None
    rollback_ok = False
    if model is not None and model_path is not None:
        registry = ModelRegistry(registry_path, audit=audit)
        active_version, rollback_ok = _prepare_registry(
            registry,
            model_path,
            model,
            demo_rollback=demo_rollback,
        )

    decision_config = config or DecisionConfig()
    if confirm_blocks:
        decision_config = replace(
            decision_config,
            block_confirmation_segments=tuple(confirm_blocks),
        )
        decision_config.validate()

    monitor_config = MonitorConfig(
        min_group_labeled=5,
        max_fpr_gap=0.05,
        max_capture_gap=0.10,
    )
    summary, store, monitor, version = run_phase4(
        transactions=transactions,
        users=users,
        fraud_rate=fraud_rate,
        seed=seed,
        warmup=warmup,
        model_path=model_path,
        config=decision_config,
        db_path=db_path,
        audit_path=audit_path,
        fresh=False,
        demo_reviews=demo_reviews,
        monitor_config=monitor_config,
        audit_log=audit,
        group_fn=lambda txn: txn.channel,
    )

    verification = audit.verify()
    retention = RetentionPolicy(review_days=retention_days).apply(store)
    metrics = summary.metrics
    fairness = metrics.get("fairness") or {}
    phase5 = Phase5Summary(
        phase4=summary,
        audit_ok=verification.ok,
        audit_entries=verification.entries,
        registry_active=registry.active_version if registry else active_version,
        registry_events=len(registry.events()) if registry else 0,
        fairness_groups=len(fairness.get("groups") or {}),
        retention=retention,
        demo_rollback_ok=rollback_ok,
    )
    return phase5, store, monitor, version


def _request_json(url: str, token: str | None = None) -> tuple[int, dict[str, Any]]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _probe(url: str, token: str) -> dict[str, Any]:
    """Verify auth is enforced and the governance/fairness APIs are live."""
    auth_required = False
    try:
        _request_json(url + "/api/health")
    except HTTPError as exc:
        auth_required = exc.code == 401
        exc.close()
    _, health = _request_json(url + "/api/health", token)
    _, models = _request_json(url + "/api/models", token)
    _, metrics = _request_json(url + "/api/metrics", token)
    return {
        "auth_required": auth_required,
        "health": health,
        "models": models,
        "metrics": metrics,
    }


def _case_explanations(url: str, token: str, limit: int = 50) -> tuple[bool, int]:
    _, payload = _request_json(url + f"/api/cases?status=all&limit={limit}", token)
    cases = payload.get("cases") or []
    detailed = 0
    ok = True
    for case in cases:
        _, detail = _request_json(
            url + "/api/cases/" + str(case["case_id"]), token
        )
        if detail.get("explanations"):
            detailed += 1
        else:
            ok = False
    return ok, detailed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5 security and governance demo")
    p.add_argument("--transactions", type=int, default=1500)
    p.add_argument("--users", type=int, default=150)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--model", default=None)
    p.add_argument("--config", default=None, help="decision threshold JSON")
    p.add_argument("--db", default="logs/phase5-reviews.sqlite")
    p.add_argument("--audit-log", default="logs/phase5-decisions.jsonl")
    p.add_argument("--audit-key", default="logs/phase5-audit.key")
    p.add_argument("--registry", default="logs/phase5-model-registry.json")
    p.add_argument("--tokens", default="logs/phase5-tokens.json")
    p.add_argument("--retention-days", type=int, default=365)
    p.add_argument(
        "--confirm-blocks",
        default="high_value",
        help="comma-separated segments requiring analyst block confirmation",
    )
    p.add_argument("--demo-reviews", type=int, default=8)
    p.add_argument("--demo-rollback", action="store_true",
                   help="exercise candidate activation and rollback")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--serve-seconds", type=int, default=0)
    p.add_argument("--no-serve", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)

    if args.model and not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    if not 0.0 < args.fraud_rate < 1.0:
        print("--fraud-rate must be in (0, 1)", file=sys.stderr)
        return 1
    if args.demo_reviews < 0 or args.serve_seconds < 0:
        print("--demo-reviews and --serve-seconds must be non-negative", file=sys.stderr)
        return 1

    config = DecisionConfig.load(args.config) if args.config else None
    confirm = tuple(
        segment.strip()
        for segment in args.confirm_blocks.split(",")
        if segment.strip()
    )
    server_config = config or DecisionConfig()
    if confirm:
        server_config = replace(
            server_config, block_confirmation_segments=tuple(confirm)
        )
        server_config.validate()

    summary, store, monitor, version = run(
        transactions=args.transactions,
        users=args.users,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        warmup=args.warmup,
        model_path=args.model,
        config=config,
        db_path=args.db,
        audit_path=args.audit_log,
        key_path=args.audit_key,
        registry_path=args.registry,
        token_path=args.tokens,
        retention_days=args.retention_days,
        confirm_blocks=confirm,
        demo_reviews=args.demo_reviews,
        fresh=args.fresh,
        demo_rollback=args.demo_rollback,
    )
    phase4 = summary.phase4
    metrics = phase4.metrics
    latency = metrics.get("latency_ms", {})
    labeling = metrics.get("labeling", {})

    print("\n=== Phase 5: security, governance and responsible AI ===")
    print(f"scoring path           : {version}")
    print(f"transactions processed : {phase4.total}")
    print(f"cases queued           : {phase4.queued} (hold={phase4.hold_cases}, "
          f"step-up={phase4.stepup_cases}, failures={phase4.queue_failures})")
    print(f"review queue           : open={phase4.cases.get('open', 0)}, "
          f"confirmed={phase4.cases.get('confirmed_fraud', 0)}, "
          f"cleared={phase4.cases.get('cleared', 0)}")
    print(f"scoring latency        : p50={latency.get('p50', 0):.2f}ms "
          f"p95={latency.get('p95', 0):.2f}ms p99={latency.get('p99', 0):.2f}ms")
    if labeling.get("capture_rate") is not None:
        print(f"capture / false pos    : {100 * labeling['capture_rate']:.1f}% / "
              f"{100 * (labeling.get('false_positive_rate') or 0):.1f}%")
    print(f"audit integrity        : {'valid' if summary.audit_ok else 'BROKEN'} "
          f"({summary.audit_entries} entries)")
    print(f"model governance       : active={summary.registry_active or 'none'}, "
          f"events={summary.registry_events}")
    if args.demo_rollback:
        print(f"rollback drill         : {'PASS' if summary.demo_rollback_ok else 'FAIL'}")
    print(f"fairness groups        : {summary.fairness_groups}")
    print(f"retention              : {summary.retention.get('cases', 0)} closed cases purged "
          f"(cutoff {summary.retention.get('cutoff')})")

    checks = {
        "audit_integrity": summary.audit_ok,
        "audit_complete": phase4.audit_records == phase4.total,
        "governance_active": summary.registry_active is not None or args.model is None,
        "fairness_groups": summary.fairness_groups > 0,
        "retention_policy": summary.retention.get("enabled", False),
        "rollback_drill": summary.demo_rollback_ok if args.demo_rollback else True,
    }

    server = DashboardServer(
        store,
        monitor,
        config=server_config,
        model_version=version,
        host=args.host,
        port=args.port,
        access=_load_access(args.tokens),
        registry=summary.registry_active and _load_registry(args.registry),
    )
    dashboard_ok = True
    try:
        if args.check or not args.no_serve:
            url = server.start()
            print(f"\ndashboard              : {url}")
            if args.check:
                try:
                    token = _first_token(args.tokens)
                    probe = _probe(url, token)
                    explanations_ok, detailed = _case_explanations(url, token)
                    checks["auth_required"] = probe["auth_required"]
                    checks["governance_api"] = (
                        args.model is None
                        or probe["models"].get("active") == summary.registry_active
                    )
                    checks["case_explanations"] = explanations_ok
                    checks["fairness_api"] = bool(
                        (probe["metrics"].get("fairness") or {}).get("groups")
                    )
                    checks["dashboard_api"] = (
                        probe["health"].get("status") == "ok"
                        and probe["metrics"].get("window_size", 0) > 0
                    )
                    print(f"dashboard API          : "
                          f"{'PASS' if checks['dashboard_api'] else 'FAIL'}")
                    print(f"auth + RBAC            : "
                          f"{'PASS' if checks['auth_required'] else 'FAIL'}")
                    print(f"case explanations      : {detailed} cases checked "
                          f"({'PASS' if explanations_ok else 'FAIL'})")
                except Exception as exc:
                    dashboard_ok = False
                    checks["dashboard_api"] = False
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

    print("\n--- checks ---")
    for name, ok in checks.items():
        print(f"{name:24s}: {'PASS' if ok else 'FAIL'}")
    ok = all(checks.values()) and dashboard_ok
    print(f"\nPhase 5 integration    : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _load_access(path: str):
    from fraud_detection.access import AccessPolicy

    return AccessPolicy.load(path) if os.path.exists(path) else None


def _load_registry(path: str) -> ModelRegistry | None:
    return ModelRegistry(path) if os.path.exists(path) else None


def _first_token(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return str(payload["tokens"][0]["token"])


if __name__ == "__main__":
    raise SystemExit(main())
