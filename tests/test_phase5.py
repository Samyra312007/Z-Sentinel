"""Phase 5 regression tests.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import ssl
import stat
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fraud_detection.access import ADMIN, INVESTIGATOR, VIEWER, AccessPolicy
from fraud_detection.dashboard import DashboardServer
from fraud_detection.decision import BLOCK, HOLD, DecisionConfig, DecisionEngine
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.governance import (
    ACTIVE,
    APPROVED,
    CANDIDATE,
    ModelRegistry,
)
from fraud_detection.model import FraudModel
from fraud_detection.monitoring import MetricsMonitor, MonitorConfig
from fraud_detection.privacy import RetentionPolicy
from fraud_detection.reviews import CONFIRMED_FRAUD, OPEN, ReviewStore
from fraud_detection.service import AuditLog, ScoringService
from fraud_detection.simulator import Transaction
from fraud_detection.security import (
    AuditIntegrityError,
    SecureAuditLog,
    build_server_ssl_context,
    ensure_key_file,
    load_key,
)


def entry(i: int) -> dict:
    return {
        "txn_id": f"T{i}",
        "risk_score": 80.0 + i,
        "action": "HOLD_FOR_REVIEW",
        "model_version": "phase2-1",
        "top_signals": ["new recipient"],
    }


class TestAuditIntegrity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fraud-audit-")
        self.key_path = os.path.join(self.dir, "audit.key")
        self.audit_path = os.path.join(self.dir, "audit.jsonl")
        self.key = ensure_key_file(self.key_path)

    def tearDown(self):
        for path in (self.key_path, self.audit_path):
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(self.dir):
            os.rmdir(self.dir)

    def test_key_file_is_owner_only_and_stable(self):
        self.assertEqual(load_key(self.key_path), self.key)
        mode = stat.S_IMODE(os.stat(self.key_path).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(ensure_key_file(self.key_path), self.key)

    def test_chain_verifies_and_survives_reopen(self):
        first = SecureAuditLog(self.audit_path, self.key)
        for i in range(3):
            first.write(entry(i))
        self.assertEqual(first.count, 3)
        self.assertTrue(first.verify().ok)
        head = first.head_hash
        first.close()

        second = SecureAuditLog(self.audit_path, self.key)
        self.assertEqual(second.seq, 3)
        self.assertEqual(second.head_hash, head)
        second.write(entry(3))
        self.assertEqual(second.verify().entries, 4)
        self.assertEqual(len(second.records), 4)
        second.close()

        self.assertEqual(
            [record["txn_id"] for record in SecureAuditLog.read_entries(self.audit_path)],
            ["T0", "T1", "T2", "T3"],
        )

    def test_tampering_breaks_verification(self):
        log = SecureAuditLog(self.audit_path, self.key)
        for i in range(3):
            log.write(entry(i))
        log.close()

        with open(self.audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        record = json.loads(lines[1])
        record["entry"]["risk_score"] = 1.0
        lines[1] = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with open(self.audit_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertFalse(verification.ok)
        self.assertEqual(verification.broken_at, 2)
        self.assertIn("hash", verification.reason or "")
        with self.assertRaises(AuditIntegrityError):
            SecureAuditLog(self.audit_path, self.key)

    def test_wrong_key_is_rejected(self):
        log = SecureAuditLog(self.audit_path, self.key)
        log.write(entry(0))
        log.close()
        verification = SecureAuditLog.verify_file(self.audit_path, b"x" * 32)
        self.assertFalse(verification.ok)
        self.assertIn("MAC", verification.reason or "")

    def test_reordering_is_rejected(self):
        log = SecureAuditLog(self.audit_path, self.key)
        for i in range(2):
            log.write(entry(i))
        log.close()
        with open(self.audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(self.audit_path, "w", encoding="utf-8") as fh:
            fh.writelines([lines[1], lines[0]])
        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertFalse(verification.ok)
        self.assertEqual(verification.broken_at, 1)

    def test_empty_audit_is_valid(self):
        open(self.audit_path, "w").close()
        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertTrue(verification.ok)
        self.assertEqual(verification.entries, 0)


def case_record(case_id: str = "T1") -> dict:
    return {
        "case_id": case_id,
        "txn_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_score": 88.0,
        "action": "HOLD_FOR_REVIEW",
        "outcome": "HELD_FOR_REVIEW",
        "user_ref": "sha256:abc",
        "amount": 10.0,
        "currency": "USD",
        "segment": "standard",
        "model_version": "phase2-1",
        "top_signals": ["new recipient"],
        "behavior_status": "scored",
        "features": {"amount": 10.0},
    }


class TestAccessPolicy(unittest.TestCase):
    def test_config_roles_and_authentication(self):
        policy = AccessPolicy.from_config(
            {
                "tokens": [
                    {"token": "read-token", "name": "viewer", "role": VIEWER},
                    {"token": "review-token", "name": "analyst", "role": INVESTIGATOR},
                    {"token": "admin-token", "name": "admin", "role": ADMIN},
                ]
            }
        )
        viewer = policy.authenticate("Bearer read-token")
        self.assertEqual(viewer.name, "viewer")
        self.assertTrue(policy.authorize(viewer, "read"))
        self.assertFalse(policy.authorize(viewer, "review"))
        self.assertTrue(policy.authorize(policy.authenticate("Bearer review-token"), "review"))
        self.assertTrue(policy.authorize(policy.authenticate("Bearer admin-token"), "admin"))
        self.assertIsNone(policy.authenticate("Bearer nope"))
        self.assertIsNone(policy.authenticate(None))
        self.assertIsNone(policy.authenticate("read-token"))

    def test_invalid_roles_and_save_load(self):
        with self.assertRaises(ValueError):
            AccessPolicy().add_token("t", "n", "root")
        path = tempfile.mktemp(suffix=".json")
        try:
            AccessPolicy.save(path, (("t", "n", VIEWER),))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            policy = AccessPolicy.load(path)
            self.assertTrue(policy.enabled)
            self.assertEqual(policy.authenticate("Bearer t").role, VIEWER)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestDashboardAccess(unittest.TestCase):
    def setUp(self):
        self.store = ReviewStore()
        self.store.add_case(case_record())
        self.monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        self.policy = AccessPolicy.from_config(
            {
                "tokens": [
                    {"token": "read-token", "name": "viewer", "role": VIEWER},
                    {"token": "review-token", "name": "analyst", "role": INVESTIGATOR},
                    {"token": "admin-token", "name": "admin", "role": ADMIN},
                ]
            }
        )
        self.server = DashboardServer(
            self.store, self.monitor, access=self.policy, port=0
        )
        self.url = self.server.start()

    def tearDown(self):
        self.server.stop()
        self.store.close()

    def _request(self, path: str, token: str | None = None, method: str = "GET", payload: dict | None = None):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=data, headers=headers, method=method)
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())

    def test_requests_require_a_bearer_token(self):
        with self.assertRaises(HTTPError) as error:
            self._request("/api/health")
        self.assertEqual(error.exception.code, 401)
        self.assertIn("Bearer", error.exception.headers.get("WWW-Authenticate", ""))
        error.exception.close()

    def test_viewer_can_read_but_not_disposition(self):
        status, health = self._request("/api/health", "read-token")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, who = self._request("/api/whoami", "read-token")
        self.assertEqual(who, {"name": "viewer", "role": VIEWER})
        with self.assertRaises(HTTPError) as denied:
            self._request(
                "/api/cases/T1/actions", "read-token", "POST",
                {"action": "confirm_fraud", "investigator": "viewer"},
            )
        self.assertEqual(denied.exception.code, 403)
        denied.exception.close()
        with self.assertRaises(HTTPError) as admin_only:
            self._request("/api/feedback", "read-token")
        self.assertEqual(admin_only.exception.code, 403)
        admin_only.exception.close()

    def test_investigator_can_disposition_and_admin_can_export(self):
        status, payload = self._request(
            "/api/cases/T1/actions", "review-token", "POST",
            {"action": "confirm_fraud", "investigator": "analyst", "note": "mule"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["case"]["status"], "CONFIRMED_FRAUD")
        status, feedback = self._request("/api/feedback", "admin-token")
        self.assertEqual(status, 200)
        self.assertEqual(feedback["count"], 1)


@unittest.skipUnless(shutil.which("openssl"), "openssl not available")
class TestDashboardTLS(unittest.TestCase):
    def test_https_round_trip(self):
        directory = tempfile.mkdtemp(prefix="fraud-tls-")
        cert = os.path.join(directory, "cert.pem")
        key = os.path.join(directory, "key.pem")
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", key, "-out", cert, "-days", "1", "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        store = ReviewStore()
        store.add_case(case_record())
        monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        server = DashboardServer(
            store, monitor, port=0,
            tls_context=build_server_ssl_context(cert, key),
        )
        try:
            url = server.start()
            self.assertTrue(url.startswith("https://"))
            host, port = url[len("https://"):].split(":")
            context = ssl._create_unverified_context()
            connection = http.client.HTTPSConnection(host, int(port), context=context, timeout=5)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            body = json.loads(response.read().decode())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(body["status"], "ok")
        finally:
            server.stop()
            store.close()
            shutil.rmtree(directory, ignore_errors=True)


def metric_result(txn_id: str, risk_score: float, *, action: str = "ALLOW") -> SimpleNamespace:
    return SimpleNamespace(
        txn_id=txn_id,
        risk_score=risk_score,
        action=action,
        outcome="COMMITTED",
        latency_ms=1.0,
        degraded=False,
    )


class TestFairnessMonitoring(unittest.TestCase):
    def _monitor(self) -> MetricsMonitor:
        return MetricsMonitor(
            MonitorConfig(
                min_volume=1,
                min_group_labeled=5,
                max_fpr_gap=0.05,
                max_capture_gap=0.10,
                evaluate_every=1000,
            )
        )

    def test_group_rates_are_reported(self):
        monitor = self._monitor()
        for i in range(5):
            monitor.observe(
                metric_result(f"A{i}", 90.0), label=True, group="card_web"
            )
        for i in range(5):
            monitor.observe(
                metric_result(f"B{i}", 10.0), label=False, group="transfer"
            )
        fairness = monitor.snapshot()["fairness"]
        self.assertEqual(fairness["groups"]["card_web"]["capture_rate"], 1.0)
        self.assertEqual(fairness["groups"]["transfer"]["false_positive_rate"], 0.0)
        self.assertEqual(fairness["capture_groups"], 1)
        self.assertEqual(fairness["fpr_groups"], 1)

    def test_false_positive_gap_raises_alert(self):
        monitor = self._monitor()
        for i in range(5):
            monitor.observe(metric_result(f"A{i}", 10.0), label=False, group="low")
        for i in range(5):
            monitor.observe(metric_result(f"B{i}", 90.0), label=False, group="high")
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["fairness"]["false_positive_rate_gap"], 1.0)
        self.assertIn(
            "fairness_fpr_gap", {alert["code"] for alert in snapshot["alerts"]}
        )

    def test_capture_gap_raises_alert(self):
        monitor = self._monitor()
        for i in range(5):
            monitor.observe(metric_result(f"A{i}", 90.0), label=True, group="low")
        for i in range(5):
            monitor.observe(metric_result(f"B{i}", 10.0), label=True, group="high")
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["fairness"]["capture_rate_gap"], 1.0)
        self.assertIn(
            "fairness_capture_gap", {alert["code"] for alert in snapshot["alerts"]}
        )

    def test_ungrouped_observations_are_ignored(self):
        monitor = self._monitor()
        monitor.observe(metric_result("T1", 90.0), label=True)
        fairness = monitor.snapshot()["fairness"]
        self.assertEqual(fairness["groups"], {})
        self.assertIsNone(fairness["false_positive_rate_gap"])


def make_txn(**overrides) -> Transaction:
    base = dict(
        txn_id="T1",
        user_id="U1",
        timestamp=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        amount=50.0,
        currency="USD",
        channel="card_web",
        merchant_category="retail",
        recipient_id="R1",
        device_id="D1",
        ip_address="203.0.113.1",
        geo=(40.0, -74.0),
        account_age_days=400,
    )
    base.update(overrides)
    return Transaction(**base)


class _EvidenceScorer:
    version = "evidence-test"

    def __init__(self, risk_score: float) -> None:
        self.risk_score = risk_score

    def score(self, features, behavior_score, graph_features):
        return {
            "risk_score": self.risk_score,
            "top_signals": ["fixed test signal"],
            "signal_scores": [["fixed test signal", self.risk_score / 100.0]],
            "explanations": [
                {
                    "feature": "amount_dev_30d_z",
                    "label": "fixed test signal",
                    "contribution": self.risk_score / 100.0,
                    "direction": "increases_risk",
                }
            ],
            "explanation_method": "test_attribution",
            "model_probability": self.risk_score / 100.0,
            "ring_probability": 0.0,
            "ring_flag": False,
        }


class TestExplainability(unittest.TestCase):
    def test_rules_scores_include_signed_explanations(self):
        result = DecisionEngine().score(
            make_txn(amount=5000.0, channel="card_web"),
            {
                "amount": 5000.0,
                "amount_dev_30d_z": 6.0,
                "new_recipient_flag": True,
                "device_change_flag": True,
                "txn_count_1h": 5,
                "same_recipient_1h": 0,
                "behavior_anomaly_score": 20.0,
                "is_odd_hours_flag": False,
            },
        )
        self.assertTrue(result["explanations"])
        self.assertEqual(result["explanation_method"], "deterministic_rules")
        for item in result["explanations"]:
            self.assertTrue(item["feature"])
            self.assertEqual(item["direction"], "increases_risk")
            self.assertGreater(item["contribution"], 0)

    def test_ensemble_exposes_model_and_ensemble_contributions(self):
        path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.json")
        scorer = EnsembleScorer(FraudModel.load(path))
        result = scorer.score(
            {"amount": 5000.0, "amount_dev_30d_z": 8.0},
            80.0,
            {"recipient_distinct_payers": 5.0, "recipient_payers_1h": 4.0},
        )
        self.assertEqual(result["explanation_method"], "gbdt_saabas_path_attribution")
        features = {item["feature"] for item in result["explanations"]}
        self.assertIn("behavior_anomaly_score", features)
        self.assertEqual(result["top_signals"], [item["label"] for item in result["explanations"]])

    def test_service_audits_explanations_and_case_evidence(self):
        cases: list[dict] = []
        audit = AuditLog()
        with ScoringService(
            scorer=_EvidenceScorer(85.0), audit=audit, review_sink=cases.append
        ) as service:
            result = service.score(make_txn())
        self.assertEqual(result.explanations[0]["feature"], "amount_dev_30d_z")
        self.assertEqual(result.explanation_method, "test_attribution")
        self.assertEqual(audit.records[-1]["explanations"][0]["contribution"], 0.85)
        self.assertEqual(cases[0]["explanations"][0]["feature"], "amount_dev_30d_z")


class TestHumanInTheLoop(unittest.TestCase):
    def test_blocks_are_held_when_segment_requires_confirmation(self):
        config = DecisionConfig(block_confirmation_segments=("standard",))
        cases: list[dict] = []
        audit = AuditLog()
        with ScoringService(
            scorer=_EvidenceScorer(99.0),
            config=config,
            audit=audit,
            review_sink=cases.append,
        ) as service:
            result = service.score(make_txn())
        self.assertEqual(result.action, HOLD)
        self.assertEqual(result.proposed_action, BLOCK)
        self.assertTrue(result.block_confirmation_required)
        self.assertTrue(result.review_queued)
        self.assertTrue(cases[0]["block_confirmation_required"])
        self.assertTrue(audit.records[-1]["block_confirmation_required"])

    def test_default_policy_still_blocks_without_review(self):
        result = DecisionEngine().decide(make_txn(), 99.0)
        self.assertEqual(result["action"], BLOCK)
        self.assertFalse(result["block_confirmation_required"])
        self.assertEqual(result["proposed_action"], BLOCK)

    def test_confirmation_segments_are_validated(self):
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict(
                {"block_confirmation_segments": ["not-a-segment"]}
            )
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict(
                {"block_confirmation_segments": ["standard", "standard"]}
            )


class TestModelGovernance(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fraud-governance-")
        base = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.json")
        self.base = os.path.abspath(base)
        self.first = os.path.join(self.dir, "first.json")
        self.second = os.path.join(self.dir, "second.json")
        shutil.copy(self.base, self.first)
        shutil.copy(self.base, self.second)
        self.registry_path = os.path.join(self.dir, "registry.json")
        self.audit = AuditLog()
        self.registry = ModelRegistry(self.registry_path, audit=self.audit)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_approval_is_required_before_activation(self):
        record = self.registry.register(
            self.first, version="v1", actor="trainer", metrics={"auc": 0.91}
        )
        self.assertEqual(record["status"], CANDIDATE)
        with self.assertRaises(ValueError):
            self.registry.activate("v1", actor="release")
        approved = self.registry.approve("v1", actor="validator", note="holdout ok")
        self.assertEqual(approved["status"], APPROVED)
        active = self.registry.activate("v1", actor="release")
        self.assertEqual(active["status"], ACTIVE)
        self.assertEqual(self.registry.active_version, "v1")
        self.assertTrue(self.registry.verify("v1"))
        self.assertEqual(self.audit.records[-1]["event_type"], "model_governance")

    def test_tampered_artifact_cannot_be_approved(self):
        self.registry.register(self.first, version="v1", actor="trainer")
        with open(self.first, "a", encoding="utf-8") as fh:
            fh.write(" ")
        with self.assertRaises(ValueError):
            self.registry.approve("v1", actor="validator")
        self.assertFalse(self.registry.verify("v1"))

    def test_prepare_registry_refuses_artifact_changed_after_registration(self):
        from fraud_detection.phase5 import _prepare_registry

        model = FraudModel.load(self.first)
        model.version = "v1"
        self.registry.register(self.first, version="v1", actor="trainer")
        self.registry.approve("v1", actor="validator")
        self.registry.activate("v1", actor="release")
        with open(self.first, "a", encoding="utf-8") as fh:
            fh.write(" ")
        # without a pre-flight this used to pass silently (record already
        # active) and only blow up inside the rollback drill
        with self.assertRaisesRegex(ValueError, "registered hash"):
            _prepare_registry(self.registry, self.first, model, demo_rollback=False)
        self.assertFalse(self.registry.verify("v1"))

    def test_rollback_restores_previous_active_model(self):
        self.registry.register(self.first, version="v1", actor="trainer")
        self.registry.register(self.second, version="v2", actor="trainer")
        self.registry.approve("v1", actor="validator")
        self.registry.approve("v2", actor="validator")
        self.registry.activate("v1", actor="release")
        self.registry.activate("v2", actor="release")
        rolled_back = self.registry.rollback(actor="release")
        self.assertEqual(rolled_back["version"], "v1")
        self.assertEqual(self.registry.active_version, "v1")
        self.assertEqual(self.registry.get("v2")["status"], "RETIRED")

    def test_duplicate_versions_and_active_rejection_are_refused(self):
        self.registry.register(self.first, version="v1", actor="trainer")
        with self.assertRaises(ValueError):
            self.registry.register(self.second, version="v1", actor="trainer")
        self.registry.approve("v1", actor="validator")
        self.registry.activate("v1", actor="release")
        with self.assertRaises(ValueError):
            self.registry.reject("v1", actor="validator")
        with self.assertRaises(KeyError):
            self.registry.approve("missing", actor="validator")


class TestRetentionPolicy(unittest.TestCase):
    def test_closed_cases_are_purged_and_pending_work_is_kept(self):
        store = ReviewStore()
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=500)).isoformat()
        closed = case_record("CLOSED")
        closed["created_at"] = old
        store.add_case(closed)
        store.act("CLOSED", "confirm_fraud", "analyst")

        open_case = case_record("OPEN")
        open_case["created_at"] = old
        store.add_case(open_case)

        pending = case_record("PENDING")
        pending["created_at"] = old
        store.add_case(pending)
        store.act("PENDING", "request_info", "analyst")

        result = RetentionPolicy(review_days=365).apply(store, now=now)
        self.assertEqual(result["cases"], 1)
        self.assertEqual(result["actions"], 1)
        self.assertIsNone(store.get("CLOSED"))
        self.assertIsNotNone(store.get("OPEN"))
        self.assertIsNotNone(store.get("PENDING"))
        self.assertEqual(store.stats()["by_status"][OPEN], 1)
        store.close()

    def test_disabled_policy_is_a_noop_and_days_are_validated(self):
        store = ReviewStore()
        result = RetentionPolicy(enabled=False).apply(store)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["cases"], 0)
        with self.assertRaises(ValueError):
            RetentionPolicy(review_days=0)
        store.close()


class TestDashboardGovernance(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fraud-gov-api-")
        self.model_path = os.path.join(self.dir, "model.json")
        base = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.json")
        shutil.copy(os.path.abspath(base), self.model_path)
        self.registry = ModelRegistry(os.path.join(self.dir, "registry.json"))
        self.registry.register(self.model_path, version="v1", actor="trainer")
        self.store = ReviewStore()
        self.monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        self.policy = AccessPolicy.from_config(
            {
                "tokens": [
                    {"token": "read-token", "name": "viewer", "role": VIEWER},
                    {"token": "admin-token", "name": "admin", "role": ADMIN},
                ]
            }
        )
        self.server = DashboardServer(
            self.store,
            self.monitor,
            access=self.policy,
            registry=self.registry,
            port=0,
        )
        self.url = self.server.start()

    def tearDown(self):
        self.server.stop()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _request(self, path, token, method="GET", payload=None):
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=data, headers=headers, method=method)
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())

    def test_viewer_reads_registry_but_cannot_promote(self):
        status, summary = self._request("/api/models", "read-token")
        self.assertEqual(status, 200)
        self.assertEqual(summary["count"], 1)
        with self.assertRaises(HTTPError) as denied:
            self._request(
                "/api/models/v1/approve",
                "read-token",
                "POST",
                {"actor": "viewer"},
            )
        self.assertEqual(denied.exception.code, 403)
        denied.exception.close()

    def test_admin_can_approve_and_activate(self):
        status, payload = self._request(
            "/api/models/v1/approve",
            "admin-token",
            "POST",
            {"actor": "validator", "note": "holdout ok"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"]["status"], APPROVED)
        status, payload = self._request(
            "/api/models/v1/activate",
            "admin-token",
            "POST",
            {"note": "release"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["active"], "v1")
        self.assertEqual(payload["models"]["by_status"][ACTIVE], 1)


if __name__ == "__main__":
    unittest.main()
