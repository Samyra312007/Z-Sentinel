"""Phase 3 regression tests (stdlib unittest, no dependencies).

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from fraud_detection.collection import (
    hash_subject_id,
    parse_payload,
    payload_from_sample,
)
from fraud_detection.features import FeatureEngine
from fraud_detection.feature_store import FeatureStore
from fraud_detection.service import (
    ALLOW,
    BLOCK,
    COMMITTED,
    DECLINED,
    FAIL_CLOSED,
    FAIL_OPEN,
    AuditLog,
    ScoringPolicy,
    ScoringService,
)
from fraud_detection.simulator import BehavioralSample, Transaction
from fraud_detection.training import train

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


def make_sample(user_id: str = "U1", **overrides) -> BehavioralSample:
    base = dict(
        session_id="S1",
        user_id=user_id,
        timestamp=T0,
        key_hold_mean_ms=110.0,
        key_hold_std_ms=14.0,
        inter_key_ms=180.0,
        mouse_velocity_px_s=640.0,
        mouse_entropy=3.4,
    )
    base.update(overrides)
    return BehavioralSample(**base)


class TestFeatureStore(unittest.TestCase):
    def test_snapshot_aggregates_and_windows(self):
        store = FeatureStore()
        for i in range(3):
            store.apply(
                make_txn(
                    txn_id=f"T{i}",
                    timestamp=T0 + timedelta(minutes=10 * i),
                    amount=10.0 + i,
                    recipient_id=f"R{i}",
                    device_id=f"D{i}",
                )
            )
        snap = store.snapshot(
            make_txn(
                timestamp=T0 + timedelta(minutes=30),
                recipient_id="R-new",
                device_id="D-new",
            )
        )
        self.assertEqual(snap.transaction_count, 3)
        self.assertAlmostEqual(snap.avg_amount_30d, 11.0, places=2)
        self.assertAlmostEqual(snap.avg_amount_7d, 11.0, places=2)
        self.assertEqual(snap.txn_count_1h, 3)
        self.assertEqual(snap.txn_count_24h, 3)
        self.assertEqual(snap.distinct_devices, 3)
        self.assertEqual(snap.distinct_recipients, 3)
        self.assertEqual(snap.distinct_locations, 1)
        self.assertEqual(snap.home_geo_distance_km, 0.0)
        self.assertTrue(snap.new_recipient_flag)
        self.assertTrue(snap.first_seen_device_flag)

    def test_old_payments_leave_rolling_windows(self):
        store = FeatureStore()
        store.apply(make_txn(timestamp=T0, amount=100.0))
        later = T0 + timedelta(days=31)
        store.apply(make_txn(txn_id="T2", timestamp=later, amount=20.0))
        snap = store.snapshot(make_txn(timestamp=later + timedelta(minutes=1)))
        # Both payments still drive the 30d average immediately after the
        # second apply; the 7d/30d buffers are pruned to the new event time.
        self.assertEqual(store.get("U1").count_7d, 1)
        self.assertEqual(store.get("U1").count_30d, 1)
        self.assertAlmostEqual(snap.avg_amount_7d, 20.0, places=2)

    def test_historical_fraud_flag_and_stats(self):
        store = FeatureStore()
        store.mark_fraud("U1")
        snap = store.snapshot(make_txn())
        self.assertTrue(snap.historical_fraud_flag)
        store.apply(make_txn())
        self.assertEqual(store.stats(), {"users": 1, "transactions": 1})

    def test_reads_stay_sub_millisecond(self):
        store = FeatureStore()
        for i in range(100):
            store.apply(
                make_txn(
                    txn_id=f"T{i}",
                    timestamp=T0 + timedelta(minutes=i),
                    recipient_id=f"R{i % 10}",
                )
            )
        txn = make_txn(timestamp=T0 + timedelta(hours=2))
        start = time.perf_counter()
        for _ in range(3000):
            store.snapshot(txn)
        average_ms = (time.perf_counter() - start) * 1000.0 / 3000
        self.assertLess(average_ms, 1.0)

    def test_feature_engine_backed_by_store(self):
        store = FeatureStore()
        engine = FeatureEngine(store)
        first = engine.compute(make_txn(txn_id="T1"))
        second = engine.compute(
            make_txn(
                txn_id="T2",
                timestamp=T0 + timedelta(minutes=3),
                amount=70.0,
                recipient_id="R2",
                device_id="D2",
            )
        )
        self.assertEqual(first["txn_count_1h"], 0)
        self.assertEqual(second["txn_count_1h"], 1)
        self.assertTrue(second["new_recipient_flag"])
        self.assertTrue(second["device_change_flag"])
        self.assertEqual(store.stats()["transactions"], 2)
        self.assertIn("avg_amount_7d", second)

    def test_behavior_baseline_lives_in_store(self):
        store = FeatureStore()
        for i in range(6):
            store.score_behavior(make_sample(session_id=f"S{i}"))
        score, channels = store.score_behavior(
            make_sample(
                session_id="S-anom",
                key_hold_mean_ms=60.0,
                key_hold_std_ms=5.0,
                inter_key_ms=90.0,
                mouse_velocity_px_s=1800.0,
                mouse_entropy=1.0,
            )
        )
        self.assertGreater(score, 40.0)
        self.assertIn("key_hold_mean_ms", channels)


class TestCollection(unittest.TestCase):
    def test_payload_roundtrip_and_pseudonymity(self):
        sample = make_sample(
            touch_pressure_mean=0.4,
            touch_speed_px_s=900.0,
            touch_swipe_angle_deg=12.5,
            click_count=7,
            click_x_mean_norm=0.42,
            click_y_mean_norm=0.61,
        )
        payload = payload_from_sample(sample)
        self.assertNotIn("U1", json.dumps(payload))
        self.assertTrue(payload["subject_ref"].startswith("sha256:"))
        parsed = parse_payload(payload)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.user_id, hash_subject_id("U1"))
        self.assertEqual(parsed.key_hold_mean_ms, sample.key_hold_mean_ms)
        self.assertEqual(parsed.mouse_entropy, sample.mouse_entropy)
        self.assertEqual(parsed.click_count, 7)
        self.assertEqual(parsed.click_x_mean_norm, 0.42)
        self.assertEqual(parsed.touch_pressure_mean, 0.4)

    def test_opt_out_returns_none(self):
        payload = payload_from_sample(make_sample(), consent=False)
        self.assertFalse(payload["consent"]["behavioral_biometrics"])
        self.assertIsNone(parse_payload(payload))

    def test_rejects_raw_content_and_event_arrays(self):
        payload = payload_from_sample(make_sample())
        payload["keystroke"]["text"] = "hunter2"
        with self.assertRaises(ValueError):
            parse_payload(payload)

        payload = payload_from_sample(make_sample())
        payload["mouse"]["points"] = [[0, 0], [1, 1]]
        with self.assertRaises(ValueError):
            parse_payload(payload)

    def test_rejects_bad_schema_and_ranges(self):
        payload = payload_from_sample(make_sample())
        payload["schema"] = "behavioral.session.v0"
        with self.assertRaises(ValueError):
            parse_payload(payload)

        payload = payload_from_sample(make_sample())
        payload["keystroke"]["key_hold_mean_ms"] = -1
        with self.assertRaises(ValueError):
            parse_payload(payload)

    def test_requires_keystroke_and_mouse(self):
        payload = payload_from_sample(make_sample())
        del payload["mouse"]
        with self.assertRaises(ValueError):
            parse_payload(payload)


class _SlowScorer:
    version = "slow-test"

    def __init__(self, delay_s: float, risk_score: float = 99.0) -> None:
        self.delay_s = delay_s
        self.risk_score = risk_score

    def score(self, features, behavior_score, graph_features):
        time.sleep(self.delay_s)
        return {
            "risk_score": self.risk_score,
            "top_signals": ["slow model"],
            "signal_scores": [],
            "model_probability": 0.99,
            "ring_probability": 0.0,
            "ring_flag": False,
        }


class TestScoringService(unittest.TestCase):
    def test_rules_fallback_commits_and_audits(self):
        with ScoringService() as service:
            result = service.score(make_txn())
            self.assertEqual(result.model_version, "rules-only")
            self.assertEqual(result.outcome, COMMITTED)
            self.assertTrue(service.audit.records)
            record = service.audit.records[0]
            self.assertEqual(record["txn_id"], "T1")
            for key in ("decided_at", "risk_score", "action", "model_version", "top_signals"):
                self.assertIn(key, record)
            self.assertTrue(record["user_ref"].startswith("sha256:"))
            self.assertNotIn("U1", json.dumps(record))

    def test_timeout_fails_open(self):
        scorer = _SlowScorer(delay_s=0.12)
        with ScoringService(
            scorer=scorer,
            policy=ScoringPolicy(timeout_ms=10, failure_mode=FAIL_OPEN),
        ) as service:
            result = service.score(make_txn())
            self.assertTrue(result.degraded)
            self.assertEqual(result.fallback_reason, "timeout")
            self.assertEqual(result.action, ALLOW)
            self.assertEqual(result.outcome, COMMITTED)
            self.assertEqual(service.store.stats()["transactions"], 1)
            record = service.audit.records[0]
            self.assertTrue(record["degraded"])
            self.assertEqual(record["action"], ALLOW)
            self.assertIn("scoring_timeout", record["top_signals"])

    def test_timeout_fails_closed(self):
        scorer = _SlowScorer(delay_s=0.12)
        with ScoringService(
            scorer=scorer,
            policy=ScoringPolicy(timeout_ms=10, failure_mode=FAIL_CLOSED),
        ) as service:
            result = service.score(make_txn())
            self.assertTrue(result.degraded)
            self.assertEqual(result.action, BLOCK)
            self.assertEqual(result.outcome, DECLINED)

    def test_scoring_error_falls_back(self):
        class BrokenScorer:
            version = "broken-test"

            def score(self, features, behavior_score, graph_features):
                raise RuntimeError("model exploded")

        with ScoringService(scorer=BrokenScorer()) as service:
            result = service.score(make_txn())
            self.assertTrue(result.degraded)
            self.assertIn("error", result.fallback_reason or "")
            self.assertEqual(result.action, ALLOW)

    def test_mismatched_behavior_is_rejected_not_scored(self):
        payload = payload_from_sample(make_sample(user_id="U2"))
        with ScoringService() as service:
            result = service.score(make_txn(), payload)
            self.assertEqual(result.behavior_status, "rejected")
            self.assertEqual(result.behavior_score, 0.0)
            self.assertFalse(result.degraded)
            self.assertEqual(service.audit.records[0]["behavior_status"], "rejected")

    def test_malformed_behavior_falls_back_to_transaction_only(self):
        payload = payload_from_sample(make_sample())
        payload["schema"] = "behavioral.session.v0"
        with ScoringService() as service:
            result = service.score(make_txn(), payload)
            self.assertEqual(result.behavior_status, "rejected")
            self.assertEqual(result.outcome, COMMITTED)

    def test_opt_out_disables_behavior(self):
        payload = payload_from_sample(make_sample(), consent=False)
        with ScoringService() as service:
            result = service.score(make_txn(), payload)
            self.assertEqual(result.behavior_score, 0.0)
            self.assertEqual(result.behavior_status, "opted_out")
            self.assertFalse(result.degraded)

    def test_audit_file_is_jsonl_and_complete(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with ScoringService(audit=AuditLog(path)) as service:
                for i in range(5):
                    service.score(
                        make_txn(
                            txn_id=f"T{i}",
                            timestamp=T0 + timedelta(minutes=i),
                            recipient_id=f"R{i}",
                        )
                    )
            records = AuditLog.read(path)
            self.assertEqual(len(records), 5)
            required = {
                "decided_at", "correlation_id", "txn_id", "user_ref", "risk_score",
                "action", "outcome", "segment", "model_version", "top_signals",
                "latency_ms", "feature_latency_ms", "behavior_status",
                "degraded", "fallback_reason",
            }
            for record in records:
                self.assertTrue(required.issubset(record))
                self.assertTrue(record["top_signals"])
                self.assertTrue(record["decided_at"])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_trained_model_end_to_end(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            model = train(
                events=400, warmup=80, users=40, fraud_rate=0.1, seed=9,
                trees=4, out=path, card=None, quiet=True,
            )
            with ScoringService(model_path=path) as service:
                from fraud_detection.simulator import TransactionSimulator

                sim = TransactionSimulator(n_users=30, fraud_rate=0.1, seed=9)
                service.warm(sim.events(80))
                for txn in sim.events(40):
                    payload = payload_from_sample(txn.behavior) if txn.behavior else None
                    result = service.score(txn, payload)
                    self.assertEqual(result.model_version, model.version)
                    self.assertIsNotNone(result.model_probability)
                    self.assertGreaterEqual(result.risk_score, 0.0)
                    self.assertLessEqual(result.risk_score, 100.0)
            self.assertEqual(service.audit.count, 40)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
