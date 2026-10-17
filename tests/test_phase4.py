"""Phase 4 regression tests.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from fraud_detection.decision import (
    HOLD,
    STEP_UP,
    DecisionConfig,
    DecisionEngine,
)
from fraud_detection.reviews import (
    CLEARED,
    CONFIRMED_FRAUD,
    NEED_INFO,
    OPEN,
    ReviewStore,
)
from fraud_detection.simulator import Transaction

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def make_case(case_id: str = "T1", risk_score: float = 85.0, **overrides) -> dict:
    base = dict(
        case_id=case_id,
        txn_id=case_id,
        created_at=T0.isoformat(),
        risk_score=risk_score,
        action=HOLD,
        outcome="HELD_FOR_REVIEW",
        user_ref="sha256:abc123",
        amount=123.45,
        currency="USD",
        segment="standard",
        model_version="phase2-1",
        top_signals=["unrecognized device", "new recipient"],
        behavior_score=55.0,
        behavior_status="scored",
        behavior_channels={"key_hold_mean_ms": -2.1},
        features={"amount": 123.45, "new_recipient_flag": True},
        graph_features={"recipient_distinct_payers": 4.0},
        ring_flag=False,
    )
    base.update(overrides)
    return base


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


class TestDecisionConfig(unittest.TestCase):
    def test_defaults_validate_and_roundtrip(self):
        config = DecisionConfig()
        config.validate()
        restored = DecisionConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_partial_override_merges_with_defaults(self):
        config = DecisionConfig.from_dict(
            {"thresholds": {"standard": {"hold": 85.0}}}
        )
        self.assertEqual(config.thresholds["standard"]["hold"], 85.0)
        self.assertEqual(config.thresholds["standard"]["step_up"], 60.0)
        self.assertEqual(config.thresholds["new_user"]["hold"], 75.0)

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict(
                {"thresholds": {"standard": {"hold": 40.0}}}
            )

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict({"thresholds": {"gold": {}}})
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict({"nope": 1})
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict(
                {"thresholds": {"standard": {"hold": 85.0, "magic": 1}}}
            )

    def test_non_numeric_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            DecisionConfig.from_dict(
                {"thresholds": {"standard": {"log": True}}}
            )

    def test_save_and_load(self):
        config = DecisionConfig.from_dict(
            {"thresholds": {"high_value": {"block": 90.0}}}
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            config.save(path)
            loaded = DecisionConfig.load(path)
            self.assertEqual(loaded.thresholds["high_value"]["block"], 90.0)
            with open(path, "r", encoding="utf-8") as fh:
                self.assertIn("thresholds", json.load(fh))
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestDecisionPolicy(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()
        self.txn = make_txn()

    def test_hold_band_challenges_enrolled_users(self):
        self.assertEqual(self.engine.decide(self.txn, 85.0)["action"], HOLD)
        self.assertEqual(
            self.engine.decide(self.txn, 85.0, two_factor_enrolled=True)["action"],
            STEP_UP,
        )

    def test_custom_thresholds_change_action(self):
        config = DecisionConfig.from_dict(
            {"thresholds": {"standard": {"log": 20.0, "step_up": 40.0,
                                          "hold": 50.0, "block": 90.0}}}
        )
        engine = DecisionEngine(config)
        self.assertEqual(engine.decide(self.txn, 55.0)["action"], HOLD)

    def test_transaction_carries_2fa_capability(self):
        enrolled = make_txn(two_factor_enrolled=True)
        self.assertEqual(self.engine.decide(enrolled, 85.0)["action"], STEP_UP)


class TestReviewStore(unittest.TestCase):
    def setUp(self):
        self.store = ReviewStore()

    def tearDown(self):
        self.store.close()

    def test_queue_sorted_and_filtered(self):
        for case_id, risk in (("T70", 70.0), ("T90", 90.0), ("T80", 80.0)):
            self.store.add_case(make_case(case_id, risk))
        rows = self.store.list_cases()
        self.assertEqual([r["case_id"] for r in rows], ["T90", "T80", "T70"])
        self.assertEqual(rows[0]["top_signals"][0], "unrecognized device")
        self.assertEqual(len(self.store.list_cases(limit=2)), 2)
        self.assertEqual(self.store.list_cases(status=CONFIRMED_FRAUD), [])

    def test_detail_contains_evidence(self):
        self.store.add_case(make_case())
        case = self.store.get("T1")
        self.assertIsNotNone(case)
        assert case is not None
        self.assertEqual(case["features"]["amount"], 123.45)
        self.assertEqual(case["behavior_channels"]["key_hold_mean_ms"], -2.1)
        self.assertEqual(case["status"], OPEN)
        self.assertEqual(self.store.actions("T1"), [])

    def test_duplicate_transaction_is_idempotent(self):
        first = self.store.add_case(make_case("T1", 80.0))
        second = self.store.add_case(make_case("T1", 99.0))
        self.assertEqual(first["case_id"], second["case_id"])
        self.assertEqual(second["risk_score"], 80.0)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_confirm_fraud_captures_label_and_audit(self):
        self.store.add_case(make_case())
        updated = self.store.act("T1", "confirm_fraud", "analyst-1", "verified mule")
        self.assertEqual(updated["status"], CONFIRMED_FRAUD)
        self.assertEqual(updated["feedback_label"], 1)
        actions = self.store.actions("T1")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["previous_status"], OPEN)
        self.assertEqual(actions[0]["new_status"], CONFIRMED_FRAUD)
        self.assertEqual(actions[0]["investigator"], "analyst-1")
        stats = self.store.stats()
        self.assertEqual(stats["reviewed"], 1)
        self.assertEqual(stats["confirmed_rate"], 1.0)
        feedback = self.store.feedback()
        self.assertEqual(feedback[0]["label"], 1)
        self.assertEqual(feedback[0]["features"]["amount"], 123.45)

    def test_clear_and_request_info_states(self):
        self.store.add_case(make_case())
        self.store.act("T1", "request_info", "analyst-2", "need proof")
        self.assertEqual(self.store.get("T1")["status"], NEED_INFO)  # type: ignore[index]
        self.store.act("T1", "reopen", "analyst-2")
        self.assertEqual(self.store.get("T1")["status"], OPEN)  # type: ignore[index]
        cleared = self.store.act("T1", "clear_fp", "analyst-2")
        self.assertEqual(cleared["status"], CLEARED)
        self.assertEqual(cleared["feedback_label"], 0)
        self.assertEqual(len(self.store.feedback()), 1)

    def test_invalid_actions_and_missing_cases(self):
        self.store.add_case(make_case())
        with self.assertRaises(ValueError):
            self.store.act("T1", "delete", "analyst")
        with self.assertRaises(ValueError):
            self.store.act("T1", "clear_fp", "")
        with self.assertRaises(KeyError):
            self.store.act("missing", "clear_fp", "analyst")

    def test_sqlite_persistence_across_restart(self):
        path = tempfile.mktemp(suffix=".sqlite")
        try:
            first = ReviewStore(path)
            first.add_case(make_case())
            first.act("T1", "confirm_fraud", "analyst-1", "confirmed")
            first.close()

            second = ReviewStore(path)
            self.assertEqual(second.get("T1")["status"], CONFIRMED_FRAUD)  # type: ignore[index]
            self.assertEqual(len(second.actions("T1")), 1)
            second.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)


if __name__ == "__main__":
    unittest.main()
