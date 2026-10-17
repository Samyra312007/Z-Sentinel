"""Phase 2 training pipeline (plan 2.3).

Replays the same simulator through the SAME feature engine / biometrics /
graph extractors used in production scoring, so offline and online features
are identical by construction (plan 6.1 parity). Then:

  1. generate a labeled dataset with leakage-free per-event features
  2. split chronologically (70/15/15) — never randomly (plan 2.3)
  3. train the gradient-boosting transaction model (class-weighted)
  4. calibrate probabilities on the validation split (Platt scaling)
  5. train the graph fraud-ring classifier
  6. evaluate on the held-out time period and write a model card

Run:
    python -m fraud_detection.train --events 15000 --out models/fraud_model.json
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from fraud_detection.biometrics import BehavioralAnomalyDetector
from fraud_detection.decision import BLOCK, HOLD, STEP_UP, DecisionEngine
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.features import FeatureEngine, as_row, vectorize
from fraud_detection.gbdt import GBDT
from fraud_detection.graph import GraphFeatureExtractor, RingModel
from fraud_detection.model import FraudModel, PlattCalibrator
from fraud_detection.simulator import TransactionSimulator

FLAGGED_ACTIONS = (STEP_UP, HOLD, BLOCK)
ELEVATED_RISK = 60.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    features: dict
    graph: dict
    behavior: float
    label: int
    strategy: str | None
    amount: float


@dataclass
class Dataset:
    samples: list[Sample] = field(default_factory=list)

    def subsets(self) -> tuple[list[Sample], list[Sample], list[Sample]]:
        n = len(self.samples)
        if n < 30:
            raise ValueError(f"dataset too small to split ({n} events)")
        n_train = int(n * 0.70)
        n_val = int(n * 0.85)
        return self.samples[:n_train], self.samples[n_train:n_val], self.samples[n_val:]


def build_dataset(
    events: int,
    warmup: int,
    users: int,
    fraud_rate: float,
    seed: int,
) -> Dataset:
    sim = TransactionSimulator(n_users=users, fraud_rate=fraud_rate, seed=seed)
    features_engine = FeatureEngine()
    bio = BehavioralAnomalyDetector()
    graph = GraphFeatureExtractor()
    ds = Dataset()

    t0 = time.perf_counter()
    # ---- warm-up: build profiles/baselines exactly as production would ----
    for txn in sim.events(max(0, warmup)):
        if txn.behavior is not None:
            bio.score(txn.behavior)
        features_engine.compute(txn)
        graph.extract(txn)
        graph.update(txn)

    # ---- measured events: capture features + labels ----
    for txn in sim.events(events):
        if txn.behavior is not None:
            b_score, b_channels = bio.score(txn.behavior)
        else:
            b_score, b_channels = 0.0, {}
        feats = features_engine.compute(txn, b_score, b_channels)
        gfeats = graph.extract(txn)  # read BEFORE update: leakage-free
        graph.update(txn)
        ds.samples.append(
            Sample(
                features=feats,
                graph=gfeats,
                behavior=b_score,
                label=1 if txn.is_fraud else 0,
                strategy=txn.fraud_strategy,
                amount=txn.amount,
            )
        )
    print(f"  built {len(ds.samples)} labeled events in {time.perf_counter() - t0:.1f}s")
    return ds


# ---------------------------------------------------------------------------
# Metrics (pure python)
# ---------------------------------------------------------------------------

def roc_auc(y: list[int], s: list[float]) -> float:
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranked = sorted(zip(s, y), key=lambda t: t[0])
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, yi) in zip(ranks, ranked) if yi)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: list[int], s: list[float]) -> float:
    ranked = sorted(zip(s, y), key=lambda t: -t[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    n_pos = sum(y) or 1
    for _, yi in ranked:
        if yi:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        if recall > prev_recall:
            ap += (recall - prev_recall) * (tp / (tp + fp))
            prev_recall = recall
    return ap


def precision_at_recall(y: list[int], s: list[float], target: float) -> float:
    ranked = sorted(zip(s, y), key=lambda t: -t[0])
    tp = fp = 0
    n_pos = sum(y) or 1
    for _, yi in ranked:
        if yi:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target:
            return tp / (tp + fp)
    return tp / (tp + fp) if tp + fp else 0.0


def expected_calibration_error(y: list[int], p: list[float], bins: int = 10) -> float:
    total = len(y)
    ece = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, v in enumerate(p) if (lo <= v < hi) or (k == bins - 1 and v == 1.0)]
        if not idx:
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(conf - acc)
    return ece


def evaluate_pipeline(model: FraudModel, test: list[Sample]) -> dict:
    scorer = EnsembleScorer(model)
    decision = DecisionEngine()
    elevated = escalated = fraud = 0
    legit = 0
    legit_elevated = 0
    by_action: dict[str, int] = {}
    latencies: list[float] = []

    for sample in test:
        t0 = time.perf_counter()
        result = scorer.score(sample.features, sample.behavior, sample.graph)
        action = decision.decide(
            _TxnView(sample.amount), result["risk_score"], sample.behavior,
            result["top_signals"],
        )["action"]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        by_action[action] = by_action.get(action, 0) + 1
        if result["risk_score"] >= ELEVATED_RISK:
            if sample.label:
                elevated += 1
            else:
                legit_elevated += 1
        if action in FLAGGED_ACTIONS:
            escalated += 1
        if sample.label:
            fraud += 1
        else:
            legit += 1

    latencies.sort()
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    return {
        "test_events": len(test),
        "test_fraud": fraud,
        "capture_rate": round(elevated / fraud, 4) if fraud else 0.0,
        "false_positive_rate": round(legit_elevated / legit, 4) if legit else 0.0,
        "escalated": escalated,
        "action_mix": dict(sorted(by_action.items())),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
        "latency_p99_ms": round(p99, 3),
    }


class _TxnView:
    """Minimal transaction interface the decision engine needs."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.account_age_days = 365


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train(
    events: int = 15000,
    warmup: int = 2000,
    users: int = 200,
    fraud_rate: float = 0.03,
    seed: int = 42,
    trees: int = 60,
    depth: int = 3,
    learning_rate: float = 0.12,
    out: str = "models/fraud_model.json",
    card: str | None = "models/MODEL_CARD.md",
    quiet: bool = False,
) -> FraudModel:
    print("Phase 2 training")
    print(f"  generating dataset: events={events} warmup={warmup} users={users} "
          f"fraud_rate={fraud_rate} seed={seed}")
    ds = build_dataset(events, warmup, users, fraud_rate, seed)
    train_s, val_s, test_s = ds.subsets()

    order = list(vectorize(train_s[0].features).keys())
    X_train = [as_row(s.features, order) for s in train_s]
    y_train = [s.label for s in train_s]
    X_val = [as_row(s.features, order) for s in val_s]
    y_val = [s.label for s in val_s]
    X_test = [as_row(s.features, order) for s in test_s]
    y_test = [s.label for s in test_s]

    pos = sum(y_train)
    neg = len(y_train) - pos
    pos_weight = min(25.0, max(2.0, neg / max(1, pos)))
    print(f"  train={len(train_s)} val={len(val_s)} test={len(test_s)} "
          f"positives(train)={pos} pos_weight={pos_weight:.1f}")

    t0 = time.perf_counter()
    gbdt = GBDT(
        n_estimators=trees,
        learning_rate=learning_rate,
        max_depth=depth,
        pos_weight=pos_weight,
        seed=seed,
    ).fit(X_train, y_train, order)
    print(f"  trained {trees} trees in {time.perf_counter() - t0:.1f}s")

    calibrator = PlattCalibrator().fit(gbdt.predict_logit(X_val), y_val)

    ring = RingModel().fit(
        [[s.graph[name] for name in RingModel().feature_names] for s in train_s],
        [1 if s.strategy == "fraud_ring" else 0 for s in train_s],
    )
    ring_probs_val = [ring.predict_proba(s.graph) for s in val_s]
    ring.choose_threshold(ring_probs_val, [1 if s.strategy == "fraud_ring" else 0 for s in val_s])

    model = FraudModel(version="phase2-1", gbdt=gbdt, calibrator=calibrator, ring=ring)
    model.dataset = {
        "events": len(ds.samples),
        "warmup_events": warmup,
        "users": users,
        "fraud_rate": fraud_rate,
        "seed": seed,
        "train": len(train_s),
        "validation": len(val_s),
        "test": len(test_s),
        "train_positives": pos,
    }

    # ---- metrics ----
    raw_test = gbdt.predict_proba(X_test)
    cal_test = [calibrator.predict(v) for v in gbdt.predict_logit(X_test)]
    model.metrics = {
        "model1": {
            "roc_auc": round(roc_auc(y_test, raw_test), 4),
            "pr_auc": round(average_precision(y_test, raw_test), 4),
            "precision_at_recall_0.8": round(precision_at_recall(y_test, cal_test, 0.8), 4),
            "precision_at_recall_0.9": round(precision_at_recall(y_test, cal_test, 0.9), 4),
            "ece": round(expected_calibration_error(y_test, cal_test), 4),
        },
        "ring_model": {
            "threshold": ring.threshold,
            "test_auc": round(
                roc_auc(
                    [1 if s.strategy == "fraud_ring" else 0 for s in test_s],
                    [ring.predict_proba(s.graph) for s in test_s],
                ),
                4,
            ),
        },
    }
    model.metrics["pipeline"] = evaluate_pipeline(model, test_s)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    print(f"  saved model -> {out} (version {model.version})")

    if card:
        os.makedirs(os.path.dirname(card) or ".", exist_ok=True)
        with open(card, "w", encoding="utf-8") as fh:
            fh.write(model_card(model))
        print(f"  wrote model card -> {card}")

    if not quiet:
        print_report(model)
    return model


def print_report(model: FraudModel) -> None:
    m1 = model.metrics["model1"]
    pipe = model.metrics["pipeline"]
    print("\n=== Phase 2 evaluation (held-out time period) ===")
    print(f"  model ROC-AUC          : {m1['roc_auc']:.3f}")
    print(f"  model PR-AUC           : {m1['pr_auc']:.3f}")
    print(f"  precision@recall 0.80  : {m1['precision_at_recall_0.8']:.3f}")
    print(f"  precision@recall 0.90  : {m1['precision_at_recall_0.9']:.3f}")
    print(f"  calibration ECE        : {m1['ece']:.3f}")
    print(f"  pipeline capture >=60  : {100 * pipe['capture_rate']:.1f}% "
          f"({pipe['test_fraud']} fraud events)")
    print(f"  pipeline FPR >=60      : {100 * pipe['false_positive_rate']:.2f}%")
    print(f"  actions                : " + ", ".join(f"{k}={v}" for k, v in pipe["action_mix"].items()))
    print(f"  scoring latency        : p50={pipe['latency_p50_ms']:.3f}ms "
          f"p99={pipe['latency_p99_ms']:.3f}ms (target < 50ms)")


def model_card(model: FraudModel) -> str:
    d = model.dataset
    m1 = model.metrics.get("model1", {})
    ring = model.metrics.get("ring_model", {})
    pipe = model.metrics.get("pipeline", {})
    params = model.gbdt.train_info
    return f"""# Fraud Detection Model Card — {model.version}

Generated: {model.created_at or datetime.utcnow().isoformat(timespec="seconds")}
Training framework: pure-Python GBDT (`fraud_detection/gbdt.py`), artifact `models/fraud_model.json`

## Intended use
Real-time risk scoring of payment transactions inside the transaction path.
Outputs a 0-100 risk index, an action (allow / log / step-up / hold / block),
and the top contributing signals for each decision.

## Data
- Synthetic transaction + behavioral stream from `fraud_detection/simulator.py`
  (the plan's documented fallback when no real transaction history exists).
- {d.get("events", "?")} measured events after {d.get("warmup_events", "?")} warm-up events,
  {d.get("users", "?")} users, fraud rate {d.get("fraud_rate", "?")}.
- Chronological split: train {d.get("train", "?")} / validation {d.get("validation", "?")} /
  test {d.get("test", "?")} (no random shuffling — avoids look-ahead leakage).
- Features are computed by the same `FeatureEngine` used in production scoring,
  so offline/online parity holds by construction.

## Model
- **Model 1** — gradient-boosted trees on transaction context:
  {params.get("trees", "?")} trees, depth {model.gbdt.max_depth},
  {params.get("features", "?")} features, class-weighted for fraud prevalence.
- **Model 2** — online behavioral-biometrics anomaly detector (per-user baseline
  over keystroke/mouse aggregates). Unsupervised; no labels used.
- **Model 3** — logistic classifier over sender→recipient graph features
  (distinct payers, hourly growth, shared devices/IPs); ring flag adds +15 risk.
- Calibration: Platt scaling fitted on the validation split.

## Held-out performance
| Metric | Value |
|---|---|
| ROC-AUC | {m1.get("roc_auc", "n/a")} |
| PR-AUC | {m1.get("pr_auc", "n/a")} |
| Precision @ 80% recall | {m1.get("precision_at_recall_0.8", "n/a")} |
| Precision @ 90% recall | {m1.get("precision_at_recall_0.9", "n/a")} |
| Calibration ECE | {m1.get("ece", "n/a")} |
| Pipeline capture rate (risk >= 60) | {pipe.get("capture_rate", "n/a")} |
| Pipeline false positive rate | {pipe.get("false_positive_rate", "n/a")} |
| Ring model threshold / AUC | {ring.get("threshold", "n/a")} / {ring.get("test_auc", "n/a")} |
| Scoring latency p50 / p99 | {pipe.get("latency_p50_ms", "n/a")} / {pipe.get("latency_p99_ms", "n/a")} ms |

## Limitations
- Trained on synthetic data; absolute performance will not transfer to real
  portfolios without retraining on historical labeled transactions.
- The graph model uses hand-built structural features, not Node2Vec/GraphSAGE
  embeddings — an intentional dependency-free simplification.
- Behavioral baselines need several sessions per user; new accounts fall back
  to transaction-only scoring (`behavior_missing_flag`).
- Model 1 is a reference GBDT implementation, not LightGBM/XGBoost; retrain with
  LightGBM offline when the training environment allows it.
- No ONNX export yet; the JSON artifact is the deployment format for the demo.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the Phase 2 fraud model")
    p.add_argument("--events", type=int, default=15000, help="measured events (default 15000)")
    p.add_argument("--warmup", type=int, default=2000, help="profile warm-up events (default 2000)")
    p.add_argument("--users", type=int, default=200, help="simulated users (default 200)")
    p.add_argument("--fraud-rate", type=float, default=0.03, help="fraud rate (default 0.03)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=60, help="boosting rounds (default 60)")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.12)
    p.add_argument("--out", default="models/fraud_model.json", help="model artifact path")
    p.add_argument("--card", default="models/MODEL_CARD.md", help="model card path")
    p.add_argument("--no-card", action="store_true", help="skip model card")
    args = p.parse_args(argv)

    train(
        events=args.events,
        warmup=args.warmup,
        users=args.users,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        trees=args.trees,
        depth=args.depth,
        learning_rate=args.learning_rate,
        out=args.out,
        card=None if args.no_card else args.card,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())