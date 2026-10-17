"""End-to-end demo CLI: stream -> biometrics -> features -> model -> decision.

Usage:
    python -m fraud_detection                                  # Phase 1 rules
    python -m fraud_detection --model models/fraud_model.json  # Phase 2 model
    python -m fraud_detection --model models/fraud_model.json --verbose

Without `--model` the rules-only fallback scores transactions. With a model
artifact the full Phase 2 ensemble runs: calibrated GBDT probability +
behavioral-biometrics deviation + graph ring boost, then the plan's
approval thresholds produce an action and top-3 explanations.

Reports capture rate / false positive rate (labels are used only in the
summary, never inside scoring) and the measured p99 latency of the scoring
path against the plan's <50ms target.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

from fraud_detection.biometrics import BehavioralAnomalyDetector
from fraud_detection.decision import BLOCK, HOLD, STEP_UP, DecisionEngine
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.features import FeatureEngine
from fraud_detection.graph import GraphFeatureExtractor
from fraud_detection.model import FraudModel
from fraud_detection.simulator import TransactionSimulator

FLAGGED_ACTIONS = (STEP_UP, HOLD, BLOCK)


@dataclass
class Summary:
    total: int = 0
    fraud: int = 0
    fraud_elevated: int = 0  # fraud with score >= 60 (plan capture rate)
    fraud_escalated: int = 0  # fraud stepped-up / held / blocked
    leg_elevated: int = 0  # legitimate txns with score >= 60
    escalated: int = 0  # all txns stepped-up / held / blocked
    action_counts: dict[str, int] = field(default_factory=dict)
    strategy_hits: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    max_latency_txn: tuple[str, float] | None = None

    @property
    def recall(self) -> float:
        return self.fraud_elevated / self.fraud if self.fraud else 0.0

    @property
    def fp_rate(self) -> float:
        legit = self.total - self.fraud
        return self.leg_elevated / legit if legit else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(len(s) - 1, int(len(s) * 0.99))]


def run(
    n_transactions: int,
    n_users: int,
    fraud_rate: float,
    seed: int,
    warmup: int = 1000,
    verbose: bool = False,
    model_path: str | None = None,
    out=sys.stdout,
) -> Summary:
    """Stream `warmup` historical events (profiles/baselines pre-exist in a
    real deployment), then score `n_transactions` measured events."""
    sim = TransactionSimulator(n_users=n_users, fraud_rate=fraud_rate, seed=seed)
    features_engine = FeatureEngine()
    bio = BehavioralAnomalyDetector()
    engine = DecisionEngine()
    graph = GraphFeatureExtractor()
    model = FraudModel.load(model_path) if model_path else None
    scorer = EnsembleScorer(model) if model is not None else None
    summary = Summary()

    # -- warm history: same code path, no scoring, no accounting -----------
    for txn in sim.events(max(0, warmup)):
        if txn.behavior is not None:
            bio.score(txn.behavior)
        features_engine.compute(txn)
        engine.observe_recipient(txn)
        graph.extract(txn)
        graph.update(txn)

    for txn in sim.events(n_transactions):
        # 1. behavioral biometrics score (session aggregates arrive with txn)
        if txn.behavior is not None:
            b_score, b_channels = bio.score(txn.behavior)
        else:
            b_score, b_channels = 0.0, {}

        t0 = time.perf_counter()
        # 2. real-time feature computation
        feats = features_engine.compute(txn, b_score, b_channels)

        if scorer is not None:
            # 3a. Phase 2: calibrated model + behavior + graph ensemble
            gfeats = graph.extract(txn)  # read before update: leakage-free
            graph.update(txn)
            result = scorer.score(feats, b_score, gfeats)
            decision = engine.decide(
                txn, result["risk_score"], b_score, result["top_signals"]
            )
            decision["model_probability"] = result["model_probability"]
            decision["ring_probability"] = result["ring_probability"]
        else:
            # 3b. Phase 1 fallback: rules-only scorer + ring flag
            ring = engine.observe_recipient(txn)
            graph.extract(txn)
            graph.update(txn)
            decision = engine.score(txn, feats, ring_flag=ring)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary.latencies_ms.append(elapsed_ms)
        if summary.max_latency_txn is None or elapsed_ms > summary.max_latency_txn[1]:
            summary.max_latency_txn = (txn.txn_id, elapsed_ms)

        # bookkeeping (evaluation only — uses labels after scoring is done)
        summary.total += 1
        action = decision["action"]
        summary.action_counts[action] = summary.action_counts.get(action, 0) + 1
        elevated = decision["risk_score"] >= 60.0
        if elevated:
            summary.leg_elevated += 1 if not txn.is_fraud else 0
        if action in FLAGGED_ACTIONS:
            summary.escalated += 1
        if txn.is_fraud:
            summary.fraud += 1
            if txn.fraud_strategy:
                summary.strategy_hits[txn.fraud_strategy] = summary.strategy_hits.get(txn.fraud_strategy, 0) + 1
            if elevated:
                summary.fraud_elevated += 1
                summary.strategy_hits.setdefault(f"{txn.fraud_strategy}!elevated", 0)
                summary.strategy_hits[f"{txn.fraud_strategy}!elevated"] += 1
            if action in FLAGGED_ACTIONS:
                summary.fraud_escalated += 1

        if verbose:
            signals = ", ".join(decision["top_signals"])
            extra = ""
            if scorer is not None:
                extra = (f"p={decision['model_probability']:.3f} "
                         f"ring={decision['ring_probability']:.2f}")
            print(
                f"{txn.txn_id} {txn.timestamp:%Y-%m-%d %H:%M} "
                f"score={decision['risk_score']:6.2f} {action:17s} "
                f"seg={decision['segment']:10s} {extra} signals=[{signals}]"
                + (f" truth=FRAUD/{txn.fraud_strategy}" if txn.is_fraud else ""),
                file=out,
            )

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fraud detection demo (Phase 1 rules / Phase 2 model)")
    p.add_argument("--transactions", type=int, default=500, help="events to stream (default 500)")
    p.add_argument("--users", type=int, default=80, help="simulated user pool (default 80)")
    p.add_argument("--fraud-rate", type=float, default=0.02, help="fraud injection rate (default 0.02)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p.add_argument("--warmup", type=int, default=1000, help="historical events used to build profiles (default 1000)")
    p.add_argument("--model", default=None, help="trained model JSON (default: rules-only fallback)")
    p.add_argument("--verbose", "-v", action="store_true", help="print each scored event")
    args = p.parse_args(argv)

    if args.model and not os.path.exists(args.model):
        print(f"Model not found: {args.model}\n"
              f"Train it first:  python -m fraud_detection.train --out {args.model}",
              file=sys.stderr)
        return 1

    s = run(args.transactions, args.users, args.fraud_rate, args.seed,
            args.warmup, args.verbose, args.model)
    if s.total == 0:
        print("No transactions generated (check --transactions).", file=sys.stderr)
        return 1

    mode = "Phase 2 (trained model)" if args.model else "Phase 1 (rules fallback)"
    print(f"\n=== {mode}: real-time scoring pipeline ===")
    if args.model:
        print(f"model artifact         : {args.model}")
    print(f"transactions processed : {s.total} (after profile/baseline warm-up)")
    print(f"fraud in stream        : {s.fraud} ({100 * s.fraud / s.total:.1f}%)")
    print(f"action mix             : " + ", ".join(f"{k}={v}" for k, v in sorted(s.action_counts.items())))
    print(f"fraud score >= 60      : {s.fraud_elevated}/{s.fraud} -> capture rate {100 * s.recall:.1f}%")
    print(f"fraud escalated        : {s.fraud_escalated}/{s.fraud} (step-up/hold/block)")
    print(f"legit score >= 60      : {s.leg_elevated} -> false positive rate {100 * s.fp_rate:.1f}%")
    tallies = sorted((k, v) for k, v in s.strategy_hits.items() if "!" not in k)
    print("fraud strategies       : " + (", ".join(
        f"{b}: {s.strategy_hits.get(f'{b}!elevated', 0)}/{s.strategy_hits.get(b, 0)} elevated" for b, _ in tallies
    ) if tallies else "none"))
    lat = sorted(s.latencies_ms)
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    path = "features+model+decision" if args.model else "features+rules"
    print(f"latency ({path}) : p50={p50:.3f}ms p95={p95:.3f}ms p99={s.p99_ms:.3f}ms "
          f"(target p99 < 50ms -> {'PASS' if s.p99_ms < 50 else 'FAIL'})")
    if s.max_latency_txn:
        print(f"max single txn path    : {s.max_latency_txn[0]} at {s.max_latency_txn[1]:.3f}ms")
    if not args.model:
        print("\nTip: train the Phase 2 model and re-run with --model models/fraud_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())