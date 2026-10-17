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


if __name__ == "__main__":
    unittest.main()
