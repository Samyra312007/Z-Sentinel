"""Phase 1 + Phase 2 regression tests (stdlib unittest, no dependencies).

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from fraud_detection.biometrics import BehavioralAnomalyDetector
from fraud_detection.decision import (
    ALLOW,
    ALLOW_WITH_LOG,
    BLOCK,
    HOLD,
    STEP_UP,
    DecisionEngine,
)
from fraud_detection.features import FeatureEngine, as_row, vectorize
from fraud_detection.gbdt import GBDT
from fraud_detection.graph import GraphFeatureExtractor, RingModel
from fraud_detection.model import FraudModel, PlattCalibrator
from fraud_detection.simulator import BehavioralSample, Transaction
from fraud_detection.training import build_dataset, roc_auc, train


def make_txn(**overrides) -> Transaction:
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    base = dict(
        txn_id="T1",
        user_id="U1",
        timestamp=now,
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


class TestDecisionRules(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()
        self.txn = make_txn()

    def test_thresholds_standard_segment(self):
        self.assertEqual(self.engine.decide(self.txn, 10)["action"], ALLOW)
        self.assertEqual(self.engine.decide(self.txn, 45)["action"], ALLOW_WITH_LOG)
        # 60-80 without behavior anomaly -> log; with anomaly -> step-up
        self.assertEqual(self.engine.decide(self.txn, 70, behavior_score=0)["action"], ALLOW_WITH_LOG)
        self.assertEqual(self.engine.decide(self.txn, 70, behavior_score=60)["action"], STEP_UP)
        self.assertEqual(self.engine.decide(self.txn, 85, behavior_score=0)["action"], HOLD)
        self.assertEqual(self.engine.decide(self.txn, 99, behavior_score=0)["action"], BLOCK)

    def test_new_user_segment_is_stricter(self):
        new_user = make_txn(account_age_days=10)
        self.assertEqual(self.engine.decide(new_user, 77)["action"], HOLD)  # vs log for standard
        self.assertEqual(self.engine.decide(new_user, 93)["action"], BLOCK)


class TestFeatureEngine(unittest.TestCase):
    def test_deterministic_and_flags(self):
        engine = FeatureEngine()
        t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        first = make_txn(timestamp=t0, amount=50.0, recipient_id="R1")
        f1 = engine.compute(first)
        self.assertTrue(f1["new_recipient_flag"])
        self.assertTrue(f1["first_seen_device_flag"])

        second = make_txn(
            txn_id="T2",
            timestamp=t0 + timedelta(minutes=5),
            amount=70.0,
            recipient_id="R2",
            device_id="D2",
            geo=(41.0, -87.0),  # ~1000km away -> impossible travel
        )
        f2 = engine.compute(second)
        # R2 has never been seen before, so it is flagged as a new recipient
        self.assertTrue(f2["new_recipient_flag"])
        self.assertTrue(f2["device_change_flag"])
        self.assertTrue(f2["impossible_travel_flag"])
        self.assertEqual(f2["txn_count_1h"], 1)

    def test_vectorize_stable(self):
        engine = FeatureEngine()
        feats = engine.compute(make_txn())
        v1 = vectorize(feats)
        v2 = vectorize(feats)
        self.assertEqual(list(v1), list(v2))
        self.assertEqual(as_row(feats, list(v1)), [v1[k] for k in v1])
        self.assertEqual(v1["channel_card_web"], 1.0)


class TestGBDT(unittest.TestCase):
    def test_learns_and_roundtrips(self):
        # label = a AND b (with some noise-free structure)
        X, y = [], []
        for a in (0, 1):
            for b in (0, 1):
                for _ in range(40):
                    X.append([float(a), float(b), 0.5])
                    y.append(a & b)
        model = GBDT(n_estimators=20, learning_rate=0.3, max_depth=2,
                     min_samples_leaf=5, pos_weight=4.0, seed=1)
        model.fit(X, y, ["a", "b", "noise"])
        preds = model.predict_proba([[0, 0, 0.5], [0, 1, 0.5], [1, 1, 0.5]])
        self.assertLess(preds[0], 0.1)
        self.assertLess(preds[1], 0.1)
        self.assertGreater(preds[2], 0.9)

        tmp = tempfile.mktemp(suffix=".json")
        try:
            with open(tmp, "w") as fh:
                json.dump(model.to_dict(), fh)
            with open(tmp) as fh:
                restored = GBDT.from_dict(json.load(fh))
            again = restored.predict_proba([[0, 0, 0.5], [0, 1, 0.5], [1, 1, 0.5]])
            for a, b in zip(preds, again):
                self.assertAlmostEqual(a, b, places=5)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_contributions_sum_direction(self):
        X = [[float(i % 2), float((i // 2) % 2)] for i in range(200)]
        y = [int(x[0] == 1) for x in X]
        model = GBDT(n_estimators=8, max_depth=2, min_samples_leaf=10).fit(X, y, ["a", "b"])
        contrib = model.contributions([1.0, 0.0])
        self.assertIn(0, contrib)  # feature 'a' must appear for a model driven by 'a'


class TestCalibration(unittest.TestCase):
    def test_monotonic(self):
        cal = PlattCalibrator().fit([-2.0, -0.5, 0.5, 2.0], [0, 0, 1, 1])
        self.assertLess(cal.predict(-2.0), cal.predict(0.5))
        self.assertLess(cal.predict(0.5), cal.predict(2.0))


class TestGraph(unittest.TestCase):
    def test_shared_recipient_grows_payers(self):
        g = GraphFeatureExtractor()
        t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        for i in range(3):
            txn = make_txn(
                txn_id=f"T{i}", user_id=f"U{i}", recipient_id="MULE",
                timestamp=t0 + timedelta(minutes=i),
                device_id=f"D{i}", ip_address=f"203.0.113.{i}",
            )
            feats = g.extract(txn)
            self.assertEqual(feats["recipient_distinct_payers"], float(i))
            self.assertEqual(feats["payer_new_to_recipient"], 1.0)
            g.update(txn)
        last = g.extract(make_txn(user_id="U3", recipient_id="MULE",
                                  timestamp=t0 + timedelta(minutes=5)))
        self.assertEqual(last["recipient_distinct_payers"], 3.0)
        self.assertEqual(last["recipient_txn_count_1h"], 3.0)

    def test_ring_model_separates(self):
        X = [[0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 0, 100],
             [0, 0, 0, 0, 0, 1, 10], [4, 3, 5, 4, 4, 0, 900],
             [1, 0, 1, 1, 1, 1, 50], [5, 4, 6, 5, 5, 0, 1500]]
        y = [0, 1, 0, 1, 0, 1]
        ring = RingModel().fit(X, y)
        probs = [ring.predict_proba(dict(zip(RingModel().feature_names, row))) for row in X]
        for label, p in zip(y, probs):
            if label:
                self.assertGreater(p, 0.5)
            else:
                self.assertLess(p, 0.5)


class TestTrainingPipeline(unittest.TestCase):
    def test_small_training_run(self):
        ds = build_dataset(events=600, warmup=100, users=30, fraud_rate=0.12, seed=3)
        self.assertGreater(len(ds.samples), 100)
        model = train(events=600, warmup=100, users=30, fraud_rate=0.12,
                      seed=3, trees=8, out="/tmp/test_model.json", card=None, quiet=True)
        # test slice must contain frauds for AUC to be defined
        self.assertGreater(model.metrics["pipeline"]["test_fraud"], 0)
        self.assertGreater(model.metrics["model1"]["roc_auc"], 0.6)
        self.assertEqual(model.gbdt.train_info["trees"], 8)
        self.assertTrue(0.0 <= model.metrics["pipeline"]["capture_rate"] <= 1.0)

    def test_model_predictions_are_probabilities(self):
        model = train(events=300, warmup=80, users=25, fraud_rate=0.1,
                      seed=4, trees=5, out="/tmp/test_model2.json", card=None, quiet=True)
        feats = build_dataset(events=5, warmup=50, users=10, fraud_rate=0.1, seed=4).samples[0]
        p = model.predict_proba(feats.features)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)
        contrib = model.contributions(feats.features)
        self.assertIsInstance(contrib, dict)

    def test_auc_known_values(self):
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)


class TestModelArtifact(unittest.TestCase):
    def test_save_load_parity(self):
        model = train(events=300, warmup=60, users=20, fraud_rate=0.1,
                      seed=5, trees=5, out="/tmp/artifact_test.json", card=None, quiet=True)
        loaded = FraudModel.load("/tmp/artifact_test.json")
        sample = build_dataset(events=10, warmup=80, users=15, fraud_rate=0.1, seed=5).samples[0]
        a = model.predict_proba(sample.features)
        b = loaded.predict_proba(sample.features)
        # artifact stores 6-decimal values, so allow 1e-6 drift
        self.assertAlmostEqual(a, b, places=5)
        self.assertAlmostEqual(model.ring_probability(sample.graph),
                               loaded.ring_probability(sample.graph), places=5)


if __name__ == "__main__":
    unittest.main()