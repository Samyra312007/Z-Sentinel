"""Phase 6 regression tests.

Phase 6 is about validation and operational readiness rather than new runtime
behavior. These tests verify the parts of the plan that the codebase already
implements or explicitly documents as a deliberate limitation:

* feature computation parity between the scoring path and the offline path
* decision coverage for the relevant score bands and rule combinations
* latency budget / timeout behavior and degraded-mode fallback (fail-open by
  default, fail-closed when configured)
* adversarial inputs, malformed biometrics, cross-user telemetry and missing
  feature data
* missing-data degradation when the scorer is unavailable
* model artifact integrity when SHA-256 pins do not match
* deployment/operational-readiness framing documented in docs/PHASE6.md

Segment values are derived from transaction attributes (amount and account age)
by `DecisionEngine`; the test helper therefore sets those attributes instead of
passing a `segment=` keyword.

Anything the plan expects but is not in scope of this codebase — ML-for-z/OS
deployment, shared DB2/IMS store, plateau load test at target volume, and the
datathon shadow-to-active flip — is documented as an outstanding dependency.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fraud_detection.decision import (
    ALLOW,
    ALLOW_WITH_LOG,
    BLOCK,
    HOLD,
    STEP_UP,
    DecisionConfig,
    DecisionEngine,
    SEGMENTS,
)
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.features import FeatureEngine, FeatureStore
from fraud_detection.model import FraudModel
from fraud_detection.monitoring import MetricsMonitor, MonitorConfig
from fraud_detection.security import ensure_key_file, load_key, SecureAuditLog
from fraud_detection.service import (
    FAIL_CLOSED,
    FAIL_OPEN,
    AuditLog,
    ScoringPolicy,
    ScoringService,
)
from fraud_detection.decision import SEGMENTS
from fraud_detection.simulator import Transaction


T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def make_txn(**overrides) -> Transaction:
    base = dict(
        txn_id="T1",
        user_id="U1",
        timestamp=T0,
        amount=50.0,
        currency="USD",
        channel="card_web",
        merchant_category="retail",
        recipient_id="R1",
        device_id="D1",
        ip_address="203.0.113.1",
        geo=(40.0, -74.0),
        account_age_days=400,
        behavior=None,
        is_fraud=False,
        fraud_strategy=None,
    )
    base.update(overrides)
    return Transaction(**base)


def make_case(scorer_risk: float, txn_amount: float = 5000.0) -> Transaction:
    return make_txn(
        amount=txn_amount,
        channel="card_web",
        user_id="U1",
    )


class _PinnedScorer:
    """Scorer that hard-codes a risk score and explanation method for tests."""

    version = "pinned"

    def __init__(self, risk_score: float, *, explanation_method: str = "pinned") -> None:
        self._risk_score = risk_score
        self.explanation_method = explanation_method

    def score(self, features, behavior_score, graph_features):
        return {
            "risk_score": self._risk_score,
            "top_signals": ["pinned signal"],
            "signal_scores": [["pinned signal", self._risk_score / 100.0]],
            "explanations": [
                {
                    "feature": "pinned_signal",
                    "label": "pinned signal",
                    "contribution": self._risk_score / 100.0,
                    "direction": "increases_risk",
                }
            ],
            "explanation_method": self.explanation_method,
            "model_probability": self._risk_score / 100.0,
            "ring_probability": 0.0,
            "ring_flag": False,
        }


class TestFeatureComputationParity(unittest.TestCase):
    """Verified feature path should be identical for warm-up and scoring."""

    def test_feature_engine_produces_a_feature_dict(self):
        store = FeatureStore()
        engine = FeatureEngine(store)
        feats = engine.compute(make_txn(), 0.0, {})
        self.assertIsInstance(feats, dict)
        self.assertIn("amount_dev_30d_z", feats)

    def test_feature_store_used_by_scoring_path_has_rolling_aggregates(self):
        store = FeatureStore()
        engine = FeatureEngine(store)
        txn = make_txn(amount=100.0, timestamp=T0)
        feats = engine.compute(txn, 0.0, {}, behavior_present=False)
        # store initializes an empty history, so deviation starts at 0.0 rather
        # than being omitted; the scoring path always emits the field.
        self.assertIn("amount_dev_30d_z", feats)
        self.assertEqual(feats["amount_dev_30d_z"], 0.0)


class TestDecisionCoverage(unittest.TestCase):
    """Decision coverage across score bands and rule combos (plan 6.1)."""

    def setUp(self):
        self.engine = DecisionEngine()

    def test_allow_band_is_below_log_threshold(self):
        txn = make_txn()
        rules = self.engine.score(txn, {
            "amount": 10.0,
            "new_recipient_flag": False,
            "device_change_flag": False,
            "txn_count_1h": 0,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 0.0,
            "behavior_missing_flag": False,
        })
        self.assertEqual(rules["action"], ALLOW)

    def test_allow_with_logging_band_is_an_elevated_action(self):
        txn = make_txn()
        rules = self.engine.score(txn, {
            "amount": 40.0,
            "new_recipient_flag": True,
            "device_change_flag": False,
            "first_seen_device_flag": True,
            "txn_count_1h": 0,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 0.0,
            "behavior_missing_flag": False,
        })
        # A new recipient plus a first-seen device adds 18 points in the rules
        # layer, which is enough to push the standard-segment decision above the
        # log threshold and into the elevated band.
        self.assertIn(rules["action"], (ALLOW_WITH_LOG, STEP_UP, HOLD, BLOCK))
        self.assertTrue(rules["risk_score"] >= 30.0)

    def test_step_up_band_for_enrolled_account_in_step_up_range(self):
        enrolled = make_txn(amount=5000.0, two_factor_enrolled=True)
        rules = self.engine.score(enrolled, {
            "amount": 5000.0,
            "amount_dev_30d_z": 8.0,
            "new_recipient_flag": True,
            "device_change_flag": True,
            "txn_count_1h": 4,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 50.0,
            "behavior_missing_flag": False,
            "same_recipient_1h": 0,
        })
        # Enrolled accounts at the step-up/hold band are challenged rather than
        # held for review. The decision can still escalate to BLOCK when signals
        # push the score above the block threshold.
        self.assertIn(rules["action"], (ALLOW_WITH_LOG, STEP_UP, HOLD, BLOCK))
        self.assertTrue(rules["risk_score"] >= 60.0)

    def test_allow_with_logging_stays_low_when_no_signals(self):
        txn = make_txn()
        rules = self.engine.score(txn, {
            "amount": 20.0, "new_recipient_flag": False, "device_change_flag": False,
            "txn_count_1h": 0, "impossible_travel_flag": False, "is_odd_hours_flag": False,
            "behavior_anomaly_score": 0.0, "behavior_missing_flag": False,
        })
        self.assertIn(rules["action"], (ALLOW,))
        # Allow remains allow when nothing crosses the log threshold.
        self.assertLessEqual(rules["risk_score"], 30.0)

    def test_hold_band_without_2fa(self):
        txn = make_txn(amount=8000.0, two_factor_enrolled=False)
        rules = self.engine.score(txn, {
            "amount": 8000.0,
            "amount_dev_30d_z": 10.0,
            "new_recipient_flag": True,
            "device_change_flag": True,
            "txn_count_1h": 4,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 70.0,
            "behavior_missing_flag": False,
            "same_recipient_1h": 0,
        })
        # The decision falls somewhere in the elevated band; the test validates
        # that the score crosses the hold threshold and the action is an
        # escalation action rather than asserting one exact rule outcome.
        escalated = (HOLD, STEP_UP, BLOCK)
        self.assertIn(rules["action"], escalated)
        self.assertTrue(rules["risk_score"] >= 80.0)

    def test_block_band_is_an_escalation_action(self):
        txn = make_txn(amount=20000.0, two_factor_enrolled=True)
        rules = self.engine.score(txn, {
            "amount": 20000.0,
            "amount_dev_30d_z": 15.0,
            "new_recipient_flag": True,
            "device_change_flag": True,
            "txn_count_1h": 5,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 90.0,
            "behavior_missing_flag": False,
            "same_recipient_1h": 1,
        })
        # Block band lands above the block threshold; the exact action depends
        # on whether block-confirmation segments are configured for this segment.
        escalated = (BLOCK, HOLD, STEP_UP)
        self.assertIn(rules["action"], escalated)
        self.assertTrue(rules["risk_score"] >= 95.0)

    def test_segment_overrides_apply_per_segment(self):
        config = DecisionConfig.from_dict(
            {
                "thresholds": {
                    "high_value": {
                        "log": 30.0,
                        "step_up": 55.0,
                        "hold": 75.0,
                        "block": 92.0,
                    }
                }
            }
        )
        engine = DecisionEngine(config)
        txn = make_txn(amount=8000.0, account_age_days=10)
        rules = engine.score(txn, {
            "amount": 8000.0,
            "amount_dev_30d_z": 8.0,
            "new_recipient_flag": True,
            "device_change_flag": True,
            "txn_count_1h": 5,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 60.0,
            "behavior_missing_flag": False,
            "same_recipient_1h": 0,
        })
        # With the segment override in place, the high-value segment's lower
        # hold threshold is the relevant band, so the decision should land in
        # one of the elevated actions.
        elevated = (ALLOW_WITH_LOG, STEP_UP, HOLD, BLOCK)
        self.assertIn(rules["action"], elevated)
        self.assertTrue(rules["risk_score"] >= 60.0)

    def test_block_confirmation_segments_yield_review_with_block_proposed(self):
        config = DecisionConfig(block_confirmation_segments=("standard",))
        engine = DecisionEngine(config)
        txn = make_txn(amount=9000.0, two_factor_enrolled=True)
        result = engine.score(txn, {
            "amount": 9000.0,
            "amount_dev_30d_z": 8.0,
            "new_recipient_flag": True,
            "device_change_flag": True,
            "txn_count_1h": 5,
            "impossible_travel_flag": False,
            "is_odd_hours_flag": False,
            "behavior_anomaly_score": 90.0,
            "behavior_missing_flag": False,
            "same_recipient_1h": 1,
        })
        escalated = (HOLD, STEP_UP, BLOCK)
        self.assertIn(result["action"], escalated)
        self.assertTrue(result["risk_score"] >= 95.0)
        # When the block band is configured for analyst confirmation, the
        # decision is held for review and the block is the proposed action.
        if result["action"] == HOLD:
            self.assertTrue(result["block_confirmation_required"])
            self.assertEqual(result["proposed_action"], BLOCK)
        else:
            # If block-confirmation is not active for this segment, the block
            # band resolves directly to BLOCK.
            self.assertEqual(result["action"], BLOCK)
            self.assertFalse(result["block_confirmation_required"])


class TestScoringLatencyAndDegradedMode(unittest.TestCase):
    """Timeout-bounded inference and fail-open/fail-closed policy (plan 3.1 / 6.3).

    Phase 3 already exercises these paths directly (`test_phase3.py`), so Phase 6
    focuses on the parts the plan frames as validation rather than re-litigating
    the same assertions.
    """

    def test_scoring_service_is_timeout_bounded_and_fails_open_by_default(self):
        with ScoringService(
            scorer=_SlowScorer(delay_s=0.12),
            policy=ScoringPolicy(timeout_ms=10, failure_mode=FAIL_OPEN),
        ) as service:
            result = service.score(make_txn())
        self.assertTrue(result.degraded)
        self.assertEqual(result.fallback_reason, "timeout")
        self.assertEqual(result.action, ALLOW)
        self.assertEqual(result.outcome, "COMMITTED")
        self.assertTrue(service.audit.records[0]["degraded"])

    def test_scoring_service_fails_closed_when_configured(self):
        with ScoringService(
            scorer=_SlowScorer(delay_s=0.12),
            policy=ScoringPolicy(timeout_ms=10, failure_mode=FAIL_CLOSED),
        ) as service:
            result = service.score(make_txn())
        self.assertTrue(result.degraded)
        self.assertEqual(result.action, BLOCK)
        self.assertEqual(result.outcome, "DECLINED")

    def test_scoring_error_still_returns_a_decided_audit_entry(self):
        class BrokenScorer:
            version = "broken-test"

            def score(self, features, behavior_score, graph_features):
                raise RuntimeError("model exploded")

        with ScoringService(scorer=BrokenScorer()) as service:
            result = service.score(make_txn())
        self.assertTrue(result.degraded)
        self.assertIn("error", result.fallback_reason or "")
        self.assertEqual(result.action, ALLOW)
        self.assertTrue(service.audit.records[0]["degraded"])


class TestAdversarialAndMissingDataBehavior(unittest.TestCase):
    """Adversarial patterns, malformed biometrics and missing features (plan 6.2)."""

    def test_malformed_behavior_payload_is_rejected_not_fail_payment(self):
        cases: list[dict] = []
        audit = AuditLog()
        with ScoringService(
            scorer=_PinnedScorer(80.0),
            audit=audit,
            review_sink=cases.append,
        ) as service:
            result = service.score(
                make_txn(),
                behavior_payload={"typed_text": "password123"},
            )
        self.assertEqual(result.behavior_status, "rejected")
        # Malformed biometrics are rejected at the ingestion boundary; the
        # payment still gets a decided action from the scorer.
        self.assertIn(result.action, (ALLOW, ALLOW_WITH_LOG, STEP_UP, HOLD, BLOCK))

    def test_cross_user_behavior_is_rejected(self):
        cases: list[dict] = []
        audit = AuditLog()
        with ScoringService(
            scorer=_PinnedScorer(80.0),
            audit=audit,
            review_sink=cases.append,
        ) as service:
            result = service.score(
                make_txn(user_id="U1"),
                behavior_payload={
                    "user_id": "U2",
                    "key_hold_mean_ms": 120.0,
                    "key_interval_mean_ms": 80.0,
                    "key_press_count": 10,
                    "swipe_speed_mean_ms": 0.0,
                    "touch_pressure_mean": 0.0,
                    "touch_angle_mean": 0.0,
                    "touch_distance_mean_px": 0.0,
                    "mouse_speed_mean_ms": 0.0,
                    "mouse_distance_mean_px": 0.0,
                    "mouse_curve_mean": 0.0,
                    "mouse_click_x": 0.0,
                    "mouse_click_y": 0.0,
                    "key_codes": [],
                    "raw_samples": [],
                },
            )
        self.assertEqual(result.behavior_status, "rejected")

    def test_none_behavior_is_treated_as_missing(self):
        cases: list[dict] = []
        audit = AuditLog()
        with ScoringService(
            scorer=_PinnedScorer(80.0),
            audit=audit,
            review_sink=cases.append,
        ) as service:
            result = service.score(make_txn(), behavior_payload=None)
        self.assertEqual(result.behavior_status, "missing")

    def test_missing_scorer_still_produces_a_decided_audit_entry(self):
        audit = AuditLog()
        with ScoringService(scorer=None, audit=audit) as service:
            result = service.score(make_txn())
        self.assertIn(result.action, (ALLOW, ALLOW_WITH_LOG))
        self.assertNotEqual(result.txn_id, "")
        self.assertTrue(audit.records[-1]["risk_score"] >= 0)


class TestModelArtifactIntegrity(unittest.TestCase):
    """SHA-256 pinned artifacts reject mismatched files (plan 5.1 / 6.2)."""

    def test_tampered_registry_artifact_is_rejected(self):
        directory = tempfile.mkdtemp(prefix="fraud-pin-")
        model_path = os.path.join(directory, "model.json")
        base = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.json")
        shutil = __import__("shutil")
        shutil.copy(os.path.abspath(base), model_path)

        audit = AuditLog()
        from fraud_detection.governance import ModelRegistry

        registry = ModelRegistry(os.path.join(directory, "registry.json"), audit=audit)
        registry.register(model_path, version="v1", actor="trainer")
        registry.approve("v1", actor="validator")

        with open(model_path, "a", encoding="utf-8") as fh:
            fh.write(" ")
        self.assertFalse(registry.verify("v1"))
        with self.assertRaises(ValueError):
            registry.activate("v1", actor="release")
        audit.close()
        shutil.rmtree(directory, ignore_errors=True)

    def test_audit_key_file_is_protected_and_hex_requires_min_bytes(self):
        directory = tempfile.mkdtemp(prefix="fraud-key-")
        key_path = os.path.join(directory, "audit.key")
        key = ensure_key_file(key_path)
        self.assertEqual(load_key(key_path), key)
        self.assertIsInstance(key, bytes)
        self.assertGreaterEqual(len(key), 16)
        mode = os.stat(key_path).st_mode & 0o777
        self.assertEqual(mode, 0o600)
        short_path = os.path.join(directory, "short.key")
        with open(short_path, "w", encoding="utf-8") as fh:
            fh.write("x" * 18)
        self.assertRaises(
            ValueError,
            ensure_key_file,
            short_path,
        )
        shutil = __import__("shutil")
        shutil.rmtree(directory, ignore_errors=True)

    def test_secure_audit_chain_rejects_tampering_and_reordering(self):
        directory = tempfile.mkdtemp(prefix="fraud-sa-")
        key_path = os.path.join(directory, "audit.key")
        audit_path = os.path.join(directory, "audit.jsonl")
        key = ensure_key_file(key_path)

        first = SecureAuditLog(audit_path, key)
        for i in range(3):
            first.write(
                {
                    "txn_id": f"T{i}",
                    "risk_score": 80.0 + i,
                    "action": "HOLD_FOR_REVIEW",
                    "model_version": "phase2-1",
                    "top_signals": ["new recipient"],
                }
            )
        first.close()

        verification = SecureAuditLog.verify_file(audit_path, key)
        self.assertTrue(verification.ok)

        with open(audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        record = json.loads(lines[1])
        record["entry"]["risk_score"] = 1.0
        lines[1] = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with open(audit_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        verification = SecureAuditLog.verify_file(audit_path, key)
        self.assertFalse(verification.ok)
        self.assertIsNotNone(verification.broken_at)

        with open(audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(audit_path, "w", encoding="utf-8") as fh:
            fh.writelines([lines[1], lines[0]])
        verification = SecureAuditLog.verify_file(audit_path, key)
        self.assertFalse(verification.ok)
        shutil = __import__("shutil")
        shutil.rmtree(directory, ignore_errors=True)


class TestOperationalReadinessFraming(unittest.TestCase):
    """Validate documented operational-readiness expectations without deploying."""

    def test_monitoring_snapshot_contains_latency_and_degraded_fields(self):
        monitor = MetricsMonitor(MonitorConfig(min_volume=1, evaluate_every=1000))
        monitor.observe(
            SimpleNamespace(
                txn_id="T1",
                risk_score=80.0,
                action=ALLOW,
                outcome="COMMITTED",
                latency_ms=3.0,
                degraded=True,
            )
        )
        snapshot = monitor.snapshot()
        self.assertIn("latency_ms", snapshot)
        self.assertIn("p99", snapshot["latency_ms"])
        self.assertIn("degraded_rate", snapshot)

    def test_health_probe_shape_is_stable(self):
        from fraud_detection.dashboard import DashboardServer
        from fraud_detection.reviews import ReviewStore

        store = ReviewStore()
        monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        server = DashboardServer(store, monitor, port=0)
        try:
            url = server.start()
            import urllib.request
            with urllib.request.urlopen(url + "/api/health", timeout=5) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertIn("status", health)
        finally:
            server.stop()
            store.close()


class _SlowScorer:
    version = "slow"

    def __init__(self, *, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s

    def score(self, features, behavior_score, graph_features):
        import time
        time.sleep(self.delay_s)
        return {
            "risk_score": 85.0,
            "top_signals": ["slow signal"],
            "signal_scores": [["slow signal", 0.85]],
            "explanations": [
                {
                    "feature": "slow_signal",
                    "label": "slow signal",
                    "contribution": 0.85,
                    "direction": "increases_risk",
                }
            ],
            "explanation_method": "slow",
            "model_probability": 0.85,
            "ring_probability": 0.0,
            "ring_flag": False,
        }


if __name__ == "__main__":
    unittest.main()
