"""Transactional AI integration (plan 3.1).

    transaction arrives
      -> behavioral payload ingested (opt-out aware)
      -> feature-store snapshot
      -> synchronous, timeout-bounded model scoring
      -> decision rules
      -> committed / pending step-up / held / declined
      -> immutable decision audit entry

The scoring call runs on a worker thread with a hard caller-side timeout. If
inference exceeds the policy budget (or raises), the policy decides whether
to fail open (allow, the default — a fraud system should not stop all
payments when the model is unavailable) or fail closed (block). Either way
the degraded decision is flagged in the audit trail.

Feature computation and state updates stay on the caller thread, so a
timed-out inference can be abandoned without losing transaction history.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable

from fraud_detection.collection import hash_subject_id, parse_payload
from fraud_detection.decision import (
    ALLOW,
    ALLOW_WITH_LOG,
    BLOCK,
    HOLD,
    STEP_UP,
    DecisionConfig,
    DecisionEngine,
)
from fraud_detection.ensemble import EnsembleScorer
from fraud_detection.features import FeatureEngine
from fraud_detection.feature_store import FeatureStore
from fraud_detection.graph import GraphFeatureExtractor
from fraud_detection.model import FraudModel
from fraud_detection.simulator import BehavioralSample, Transaction

FAIL_OPEN = "fail_open"
FAIL_CLOSED = "fail_closed"
FAILURE_MODES = (FAIL_OPEN, FAIL_CLOSED)

COMMITTED = "COMMITTED"
PENDING_STEP_UP = "PENDING_STEP_UP"
HELD_FOR_REVIEW = "HELD_FOR_REVIEW"
DECLINED = "DECLINED"

_OUTCOMES = {
    ALLOW: COMMITTED,
    ALLOW_WITH_LOG: COMMITTED,
    STEP_UP: PENDING_STEP_UP,
    HOLD: HELD_FOR_REVIEW,
    BLOCK: DECLINED,
}

_UNSET = object()


@dataclass(frozen=True)
class ScoringPolicy:
    """Latency budget and degraded-mode behaviour for the scoring call."""

    timeout_ms: float = 50.0
    failure_mode: str = FAIL_OPEN

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.failure_mode not in FAILURE_MODES:
            raise ValueError(f"failure_mode must be one of {FAILURE_MODES}")

    @property
    def fail_closed(self) -> bool:
        return self.failure_mode == FAIL_CLOSED

    @property
    def fallback_action(self) -> str:
        return BLOCK if self.fail_closed else ALLOW


@dataclass(frozen=True)
class AuditEntry:
    """One immutable decision record (plan 3.1 required fields, plus context)."""

    decided_at: str
    correlation_id: str
    txn_id: str
    user_ref: str
    amount: float
    currency: str
    channel: str
    risk_score: float
    action: str
    outcome: str
    segment: str
    model_version: str
    top_signals: tuple[str, ...]
    latency_ms: float
    feature_latency_ms: float
    behavior_status: str = "missing"
    degraded: bool = False
    fallback_reason: str | None = None
    explanations: tuple[dict[str, Any], ...] = ()
    explanation_method: str = "none"
    block_confirmation_required: bool = False
    proposed_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decided_at": self.decided_at,
            "correlation_id": self.correlation_id,
            "txn_id": self.txn_id,
            "user_ref": self.user_ref,
            "amount": self.amount,
            "currency": self.currency,
            "channel": self.channel,
            "risk_score": self.risk_score,
            "action": self.action,
            "outcome": self.outcome,
            "segment": self.segment,
            "model_version": self.model_version,
            "top_signals": list(self.top_signals),
            "latency_ms": self.latency_ms,
            "feature_latency_ms": self.feature_latency_ms,
            "behavior_status": self.behavior_status,
            "degraded": self.degraded,
            "fallback_reason": self.fallback_reason,
            "explanations": [dict(item) for item in self.explanations],
            "explanation_method": self.explanation_method,
            "block_confirmation_required": self.block_confirmation_required,
            "proposed_action": self.proposed_action,
        }


class AuditLog:
    """Append-only JSONL decision audit trail.

    When a path is supplied every entry is appended and flushed immediately;
    entries are also retained in memory for tests and in-process verification.
    No raw customer identifier is ever written — `user_ref` is the salted hash.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fh = None
        if path:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def write(self, entry: AuditEntry | dict[str, Any]) -> dict[str, Any]:
        record = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self.records.append(record)
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
        return record

    @property
    def count(self) -> int:
        return len(self.records)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def read(path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class ScoreResult:
    txn_id: str
    user_ref: str
    correlation_id: str
    risk_score: float
    action: str
    outcome: str
    segment: str
    model_version: str
    top_signals: list[str]
    latency_ms: float
    feature_latency_ms: float
    behavior_score: float = 0.0
    behavior_status: str = "missing"
    features: dict[str, Any] = field(default_factory=dict)
    graph_features: dict[str, Any] = field(default_factory=dict)
    behavior_channels: dict[str, Any] = field(default_factory=dict)
    signal_scores: list[Any] = field(default_factory=list)
    case_id: str | None = None
    review_queued: bool = False
    review_error: str | None = None
    model_probability: float | None = None
    ring_probability: float | None = None
    ring_flag: bool = False
    degraded: bool = False
    fallback_reason: str | None = None
    decided_at: str = ""
    explanations: list[dict[str, Any]] = field(default_factory=list)
    explanation_method: str = "none"
    block_confirmation_required: bool = False
    proposed_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ScoringService:
    """The synchronous scoring call that sits inside the transaction path."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: FraudModel | None = None,
        scorer: Any | None = None,
        policy: ScoringPolicy | None = None,
        config: DecisionConfig | None = None,
        audit: AuditLog | None = None,
        store: FeatureStore | None = None,
        subject_salt: str | None = None,
        workers: int = 4,
        review_sink: Callable[[dict[str, Any]], Any] | None = None,
        review_actions: tuple[str, ...] = (HOLD, STEP_UP),
    ) -> None:
        if model is None and model_path is not None:
            model = FraudModel.load(model_path)
        self.model = model
        self.scorer = scorer if scorer is not None else (
            EnsembleScorer(model) if model is not None else None
        )
        self.policy = policy or ScoringPolicy()
        self.store = store or FeatureStore()
        self.features = FeatureEngine(self.store)
        self.decision = DecisionEngine(config)
        self.graph = GraphFeatureExtractor()
        self.audit = audit or AuditLog()
        self.subject_salt = subject_salt
        self.review_sink = review_sink
        self.review_actions = tuple(review_actions)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="fraud-score"
        )
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def model_version(self) -> str:
        return getattr(self.model, "version", None) or getattr(
            self.scorer, "version", "rules-only"
        )

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True)
            self.audit.close()
            self._closed = True

    def __enter__(self) -> "ScoringService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- warm history -------------------------------------------------------

    def warm(self, events) -> int:
        """Replay profile history through the same feature/graph path.

        Warm-up events are not scored and not audited; they build the feature
        store, behavioral baselines and sender→recipient graph exactly as a
        long-running production service would already have them.
        """
        count = 0
        for txn in events:
            scoring_txn = self._prepare(txn)
            sample = self._sample_from_transaction(scoring_txn)
            if sample is not None:
                self.store.score_behavior(sample)
            self.features.compute(scoring_txn, 0.0, {}, behavior_present=sample is not None)
            self.decision.observe_recipient(scoring_txn)
            self.graph.extract(scoring_txn)
            self.graph.update(scoring_txn)
            count += 1
        return count

    # -- scoring ------------------------------------------------------------

    def score(
        self,
        txn: Transaction,
        behavior_payload: dict | None | object = _UNSET,
        *,
        correlation_id: str | None = None,
    ) -> ScoreResult:
        """Score one transaction and write the audit entry.

        `behavior_payload` may be:
          * omitted — use the sample already attached to the transaction;
          * the SDK payload dict — validated/ingested here;
          * None — explicitly no behavior (e.g. the SDK failed to load).
        """
        started = time.perf_counter()
        correlation = correlation_id or f"CORR-{uuid.uuid4().hex[:16]}"
        scoring_txn = self._prepare(txn)

        sample, behavior_status = self._resolve_behavior(scoring_txn, behavior_payload)
        if sample is not None:
            behavior_score, behavior_channels = self.store.score_behavior(sample)
        else:
            behavior_score, behavior_channels = 0.0, {}

        feature_started = time.perf_counter()
        feats = self.features.compute(
            scoring_txn,
            behavior_score,
            behavior_channels,
            behavior_present=sample is not None,
        )
        feature_latency_ms = (time.perf_counter() - feature_started) * 1000.0

        graph_features = self.graph.extract(scoring_txn)  # read before update
        self.graph.update(scoring_txn)
        ring_flag = self.decision.observe_recipient(scoring_txn)

        scored: dict[str, Any] | None = None
        degraded = False
        fallback_reason: str | None = None
        try:
            future = self._executor.submit(
                self._infer, scoring_txn, feats, behavior_score, graph_features, ring_flag
            )
            scored = future.result(timeout=self.policy.timeout_ms / 1000.0)
        except FutureTimeout:
            degraded, fallback_reason = True, "timeout"
        except Exception as exc:  # model/runtime failure inside the call
            degraded, fallback_reason = True, f"error:{type(exc).__name__}"

        if scored is None:
            scored = self._fallback(scoring_txn, fallback_reason or "unavailable")

        latency_ms = (time.perf_counter() - started) * 1000.0
        decided_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        action = str(scored["action"])
        outcome = _OUTCOMES.get(action, COMMITTED)

        entry = AuditEntry(
            decided_at=decided_at,
            correlation_id=correlation,
            txn_id=txn.txn_id,
            user_ref=scoring_txn.user_id,
            amount=round(txn.amount, 2),
            currency=txn.currency,
            channel=txn.channel,
            risk_score=round(float(scored["risk_score"]), 2),
            action=action,
            outcome=outcome,
            segment=str(scored.get("segment", "standard")),
            model_version=self.model_version,
            top_signals=tuple(scored.get("top_signals") or []),
            latency_ms=round(latency_ms, 3),
            feature_latency_ms=round(feature_latency_ms, 3),
            behavior_status=behavior_status,
            degraded=degraded,
            fallback_reason=fallback_reason,
            explanations=tuple(scored.get("explanations") or ()),
            explanation_method=str(scored.get("explanation_method", "none")),
            block_confirmation_required=bool(
                scored.get("block_confirmation_required", False)
            ),
            proposed_action=scored.get("proposed_action"),
        )
        self.audit.write(entry)

        result = ScoreResult(
            txn_id=txn.txn_id,
            user_ref=scoring_txn.user_id,
            correlation_id=correlation,
            risk_score=entry.risk_score,
            action=action,
            outcome=outcome,
            segment=entry.segment,
            model_version=entry.model_version,
            top_signals=list(entry.top_signals),
            latency_ms=entry.latency_ms,
            feature_latency_ms=entry.feature_latency_ms,
            behavior_score=round(behavior_score, 2),
            behavior_status=behavior_status,
            features=feats,
            graph_features=graph_features,
            behavior_channels=behavior_channels,
            signal_scores=list(scored.get("signal_scores") or []),
            model_probability=scored.get("model_probability"),
            ring_probability=scored.get("ring_probability"),
            ring_flag=bool(scored.get("ring_flag", False)),
            degraded=degraded,
            fallback_reason=fallback_reason,
            decided_at=decided_at,
            explanations=list(entry.explanations),
            explanation_method=entry.explanation_method,
            block_confirmation_required=entry.block_confirmation_required,
            proposed_action=entry.proposed_action,
        )
        self._enqueue_review(result, txn, scoring_txn)
        return result

    # -- review queue -------------------------------------------------------

    def _enqueue_review(
        self,
        result: ScoreResult,
        txn: Transaction,
        scoring_txn: Transaction,
    ) -> None:
        if self.review_sink is None or result.action not in self.review_actions:
            return
        case = self.build_case(result, txn, scoring_txn)
        try:
            created = self.review_sink(case)
        except Exception as exc:  # keep the payment decision authoritative
            result.review_error = f"{type(exc).__name__}: {exc}"
            return
        result.review_queued = True
        if isinstance(created, dict) and created.get("case_id"):
            result.case_id = str(created["case_id"])
        else:
            result.case_id = str(case["case_id"])

    @staticmethod
    def build_case(
        result: ScoreResult,
        txn: Transaction,
        scoring_txn: Transaction,
    ) -> dict[str, Any]:
        """Build the evidence package investigator dashboards consume."""
        return {
            "case_id": txn.txn_id,
            "txn_id": txn.txn_id,
            "correlation_id": result.correlation_id,
            "created_at": result.decided_at,
            "user_ref": scoring_txn.user_id,
            "amount": txn.amount,
            "currency": txn.currency,
            "channel": txn.channel,
            "merchant_category": txn.merchant_category,
            "recipient_id": txn.recipient_id,
            "device_id": txn.device_id,
            "ip_address": txn.ip_address,
            "geo": list(txn.geo),
            "account_age_days": txn.account_age_days,
            "two_factor_enrolled": bool(getattr(txn, "two_factor_enrolled", False)),
            "risk_score": result.risk_score,
            "action": result.action,
            "outcome": result.outcome,
            "segment": result.segment,
            "model_version": result.model_version,
            "top_signals": list(result.top_signals),
            "explanations": list(result.explanations),
            "explanation_method": result.explanation_method,
            "signal_scores": result.signal_scores,
            "model_probability": result.model_probability,
            "ring_probability": result.ring_probability,
            "ring_flag": result.ring_flag,
            "behavior_status": result.behavior_status,
            "behavior_score": result.behavior_score,
            "behavior_channels": result.behavior_channels,
            "features": result.features,
            "graph_features": result.graph_features,
            "degraded": result.degraded,
            "fallback_reason": result.fallback_reason,
            "block_confirmation_required": result.block_confirmation_required,
            "proposed_action": result.proposed_action,
            "latency_ms": result.latency_ms,
        }

    # -- internals ----------------------------------------------------------

    def _prepare(self, txn: Transaction) -> Transaction:
        """Replace the customer id with its salted reference."""
        if txn.user_id.startswith("sha256:"):
            return txn
        return replace(txn, user_id=self.subject_ref(txn.user_id))

    def subject_ref(self, user_id: str) -> str:
        return hash_subject_id(user_id, self.subject_salt)

    def _sample_from_transaction(self, txn: Transaction) -> BehavioralSample | None:
        sample = txn.behavior
        if sample is None:
            return None
        if sample.user_id != txn.user_id:
            sample = replace(sample, user_id=txn.user_id)
        return sample

    def _resolve_behavior(
        self,
        scoring_txn: Transaction,
        behavior_payload: dict | None | object,
    ) -> tuple[BehavioralSample | None, str]:
        """Return (sample, status); bad telemetry never fails the payment."""
        if behavior_payload is _UNSET:
            sample = self._sample_from_transaction(scoring_txn)
            return sample, "scored" if sample is not None else "missing"
        if behavior_payload is None:
            return None, "missing"
        try:
            sample = parse_payload(behavior_payload)  # type: ignore[arg-type]
        except ValueError:
            # Malformed or content-bearing payload: score transaction-only and
            # mark the rejection in the audit trail instead of failing payment.
            return None, "rejected"
        if sample is None:
            return None, "opted_out"
        if sample.user_id != scoring_txn.user_id:
            return None, "rejected"
        return sample, "scored"

    def _infer(
        self,
        txn: Transaction,
        features: dict,
        behavior_score: float,
        graph_features: dict,
        ring_flag: bool,
    ) -> dict[str, Any]:
        if self.scorer is not None:
            result = self.scorer.score(features, behavior_score, graph_features)
            decision = self.decision.decide(
                txn, result["risk_score"], behavior_score, result["top_signals"],
                two_factor_enrolled=bool(getattr(txn, "two_factor_enrolled", False)),
            )
            return {
                **result,
                "action": decision["action"],
                "segment": decision["segment"],
                "proposed_action": decision["proposed_action"],
                "block_confirmation_required": decision[
                    "block_confirmation_required"
                ],
            }
        result = self.decision.score(txn, features, ring_flag=ring_flag)
        return {
            "risk_score": result["risk_score"],
            "top_signals": result["top_signals"],
            "segment": result["segment"],
            "action": result["action"],
            "proposed_action": result["proposed_action"],
            "block_confirmation_required": result["block_confirmation_required"],
            "explanations": result["explanations"],
            "explanation_method": result["explanation_method"],
            "signal_scores": [],
            "behavior_score": round(behavior_score, 2),
            "model_probability": None,
            "ring_probability": None,
            "ring_flag": ring_flag,
        }

    def _fallback(self, txn: Transaction, reason: str) -> dict[str, Any]:
        risk = 100.0 if self.policy.fail_closed else 0.0
        signal = f"scoring_{reason.split(':')[0]}"
        decision = self.decision.decide(txn, risk, 0.0, [signal])
        return {
            "risk_score": decision["risk_score"],
            "top_signals": decision["top_signals"],
            "segment": decision["segment"],
            "action": decision["action"],
            "proposed_action": decision["proposed_action"],
            "block_confirmation_required": decision[
                "block_confirmation_required"
            ],
            "explanations": [
                {
                    "feature": signal,
                    "label": signal,
                    "contribution": risk / 100.0,
                    "direction": "increases_risk",
                }
            ] if risk else [],
            "explanation_method": "degraded_fallback",
            "signal_scores": [],
            "behavior_score": 0.0,
            "model_probability": None,
            "ring_probability": None,
            "ring_flag": False,
        }
