"""Placeholder decision engine (plan 4.1).

Rules-only scorer used until the Phase 2 gradient-boosting model is wired
in — the plan's own failover ("fallback to rules-only" / risk table row 2).
It produces the same contract the model ensemble will: a risk score 0-100,
the top contributing signals, and an Allow/Log/Step-up/Hold/Block action
using the plan's thresholds:

    < 30            ALLOW
    30-60           ALLOW_WITH_LOGGING
    60-80           STEP_UP (?behavioral anomaly -> step-up)
    80-95           HOLD_FOR_REVIEW
    > 95            BLOCK

Thresholds and segment overrides live in `DecisionConfig` so Phase 4 can
make them externally configurable without touching call sites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fraud_detection.simulator import Transaction

ALLOW = "ALLOW"
ALLOW_WITH_LOG = "ALLOW_WITH_LOGGING"
STEP_UP = "STEP_UP_2FA"
HOLD = "HOLD_FOR_REVIEW"
BLOCK = "BLOCK"

SEGMENTS = ("standard", "new_user", "high_value")
THRESHOLD_KEYS = ("log", "step_up", "hold", "block")


@dataclass(frozen=True)
class DecisionConfig:
    # segment: "standard" | "new_user" (account_age < 90d) | "high_value"
    thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "standard": {"log": 30.0, "step_up": 60.0, "hold": 80.0, "block": 95.0},
            "new_user": {"log": 25.0, "step_up": 55.0, "hold": 75.0, "block": 92.0},
            "high_value": {"log": 30.0, "step_up": 55.0, "hold": 75.0, "block": 92.0},
        }
    )
    # additive risk points per signal, summed then clipped to 0-100:
    signal_weights: dict[str, float] = field(
        default_factory=lambda: {
            "behavior_anomaly_score": 0.45,
            "amount_dev_30d_z_4+": 20.0,
            "amount_dev_30d_z_2-4": 10.0,
            "new_recipient_flag": 10.0,
            "device_change_flag": 12.0,
            "first_seen_device_flag": 8.0,
            "impossible_travel_flag": 25.0,
            "txn_count_1h_5+": 30.0,
            "txn_count_1h_3-4": 25.0,
            "new_recipient_and_new_device": 10.0,
            "odd_hours_high_amount": 8.0,
            "behavior_missing_flag": 4.0,
            "mule_recipient_3plus_users": 25.0,
            "card_testing_retry_pattern": 70.0,  # sub-$10 retry to same recipient within 1h
        }
    )
    graph_ring_boost: float = 15.0  # plan: ring flag boosts score by 15
    # Segments whose block decisions must be confirmed by an analyst before
    # the payment is declined (plan 5.2 human-in-the-loop). Empty by default.
    block_confirmation_segments: tuple[str, ...] = ()

    # -- validation / external configuration (plan 4.1) --------------------

    def validate(self) -> None:
        """Raise ValueError unless every segment has sane, ordered thresholds."""
        if set(self.thresholds) != set(SEGMENTS):
            raise ValueError(f"thresholds must define exactly {SEGMENTS}")
        for segment, values in self.thresholds.items():
            if set(values) != set(THRESHOLD_KEYS):
                raise ValueError(
                    f"segment {segment!r} must define {THRESHOLD_KEYS}"
                )
            ordered = [values[key] for key in THRESHOLD_KEYS]
            for key, value in zip(THRESHOLD_KEYS, ordered):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{segment}.{key} must be numeric")
            if not (0.0 <= ordered[0] < ordered[1] < ordered[2] < ordered[3] <= 100.0):
                raise ValueError(
                    f"segment {segment!r} thresholds must satisfy "
                    "0 <= log < step_up < hold < block <= 100"
                )
        for name, weight in self.signal_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"signal weight {name!r} must be a non-negative number")
        if (
            isinstance(self.graph_ring_boost, bool)
            or not isinstance(self.graph_ring_boost, (int, float))
            or not 0.0 <= self.graph_ring_boost <= 100.0
        ):
            raise ValueError("graph_ring_boost must be between 0 and 100")
        if not isinstance(self.block_confirmation_segments, tuple):
            raise ValueError("block_confirmation_segments must be a tuple")
        unknown_segments = set(self.block_confirmation_segments) - set(SEGMENTS)
        if unknown_segments:
            raise ValueError(
                f"unknown block confirmation segments: {sorted(unknown_segments)}"
            )
        if len(set(self.block_confirmation_segments)) != len(
            self.block_confirmation_segments
        ):
            raise ValueError("block_confirmation_segments must not contain duplicates")

    def to_dict(self) -> dict:
        return {
            "thresholds": {k: dict(v) for k, v in self.thresholds.items()},
            "signal_weights": dict(self.signal_weights),
            "graph_ring_boost": self.graph_ring_boost,
            "block_confirmation_segments": list(self.block_confirmation_segments),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionConfig":
        """Merge a partial JSON config over the safe defaults, then validate."""
        if not isinstance(data, dict):
            raise ValueError("decision config must be an object")
        unknown = set(data) - {
            "thresholds", "signal_weights", "graph_ring_boost",
            "block_confirmation_segments",
        }
        if unknown:
            raise ValueError(f"unknown decision config fields: {sorted(unknown)}")

        base = cls()
        thresholds = {k: dict(v) for k, v in base.thresholds.items()}
        raw_thresholds = data.get("thresholds", {})
        if not isinstance(raw_thresholds, dict):
            raise ValueError("thresholds must be an object")
        for segment, values in raw_thresholds.items():
            if segment not in thresholds:
                raise ValueError(f"unknown segment {segment!r}; expected {SEGMENTS}")
            if not isinstance(values, dict):
                raise ValueError(f"thresholds.{segment} must be an object")
            merged = {**thresholds[segment], **values}
            unknown_keys = set(values) - set(THRESHOLD_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"unknown thresholds for {segment!r}: {sorted(unknown_keys)}"
                )
            thresholds[segment] = merged

        signal_weights = dict(base.signal_weights)
        raw_weights = data.get("signal_weights", {})
        if not isinstance(raw_weights, dict):
            raise ValueError("signal_weights must be an object")
        signal_weights.update(raw_weights)

        raw_segments = data.get(
            "block_confirmation_segments", list(base.block_confirmation_segments)
        )
        if not isinstance(raw_segments, (list, tuple)):
            raise ValueError("block_confirmation_segments must be a list")
        segments = tuple(str(segment) for segment in raw_segments)

        config = cls(
            thresholds=thresholds,
            signal_weights=signal_weights,
            graph_ring_boost=data.get("graph_ring_boost", base.graph_ring_boost),
            block_confirmation_segments=segments,
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str) -> "DecisionConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")


def resolve_segment(txn_txn_amount: float, account_age_days: int) -> str:
    if account_age_days < 90:
        return "new_user"
    if txn_txn_amount >= 4000.0:
        return "high_value"
    return "standard"


class DecisionEngine:
    """Scores a feature vector and applies the business-rules layer."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        # recipient_id -> distinct payers, for the fraud-ring lookup
        # (stand-in for Model 3's graph embedding flag)
        self._recipient_payers: dict[str, set[str]] = {}

    # -- ring signal (running over observed stream, as a daily graph job would)

    def observe_recipient(self, txn: Transaction) -> bool:
        """Record the payer; return True if this recipient now looks like a
        mule (>= 3 distinct payers within the observed window)."""
        payers = self._recipient_payers.setdefault(txn.recipient_id, set())
        payers.add(txn.user_id)
        return len(payers) >= 3

    # -- scoring -------------------------------------------------------------

    def score(
        self,
        txn: Transaction,
        features: dict[str, float | int | bool],
        ring_flag: bool = False,
    ) -> dict:
        w = self.config.signal_weights
        signals: list[tuple[str, float]] = []
        score = 0.0

        def add(name: str, points: float) -> None:
            nonlocal score
            if points:
                score += points
                signals.append((name, round(points, 1)))

        dev_z = float(features.get("amount_dev_30d_z", 0.0))
        if dev_z >= 4.0:
            add("amount_dev_z>=4", w["amount_dev_30d_z_4+"])
        elif dev_z >= 2.0:
            add("amount_dev_z>=2", w["amount_dev_30d_z_2-4"])

        new_rcpt = bool(features.get("new_recipient_flag"))
        new_dev = bool(features.get("device_change_flag") or features.get("first_seen_device_flag"))
        if new_rcpt:
            add("new_recipient", w["new_recipient_flag"])
        if new_dev:
            add("new_device", w["device_change_flag"])
        if new_rcpt and new_dev:
            add("new_recipient+x_new_device", w["new_recipient_and_new_device"])
        if features.get("impossible_travel_flag"):
            add("impossible_travel", w["impossible_travel_flag"])

        count_1h = int(features.get("txn_count_1h", 0))
        amount = float(features.get("amount", txn.amount))
        if count_1h >= 5:
            add("velocity_1h>=5", w["txn_count_1h_5+"])
        elif count_1h >= 3:
            add("velocity_1h>=3", w["txn_count_1h_3-4"])
        if (
            count_1h >= 2
            and amount <= 10.0
            and int(features.get("same_recipient_1h", 0)) >= 1
        ):
            add("card_testing_retry", w["card_testing_retry_pattern"])

        if features.get("is_odd_hours_flag") and dev_z >= 4.0:
            add("odd_hours+high_amount", w["odd_hours_high_amount"])
        if features.get("behavior_missing_flag"):
            add("behavior_missing", w["behavior_missing_flag"])

        behavior_score = float(features.get("behavior_anomaly_score", 0.0))
        score += behavior_score * w["behavior_anomaly_score"]
        if behavior_score > 0:
            signals.append((f"behavior_anomaly={behavior_score:.0f}", round(
                behavior_score * w["behavior_anomaly_score"], 1)))

        ring_boost = 0.0
        if ring_flag:
            ring_boost = self.config.graph_ring_boost
            add("graph_ring_flag", ring_boost)

        score = round(max(0.0, min(100.0, score)), 2)
        signals.sort(key=lambda kv: -abs(kv[1]))
        top = [name for name, _ in signals[:3]] or ["no_risk_signals"]

        result = self.decide(
            txn, score, behavior_score, top,
            two_factor_enrolled=bool(getattr(txn, "two_factor_enrolled", False)),
        )
        result["ring_boost"] = ring_boost
        result["signal_scores"] = signals
        result["explanations"] = [
            {
                "feature": name,
                "label": name,
                "contribution": points,
                "direction": "increases_risk",
            }
            for name, points in signals[:3]
        ]
        result["explanation_method"] = "deterministic_rules"
        return result

    # -- business rules on top of any 0-100 risk score ----------------------

    def decide(
        self,
        txn: Transaction,
        risk_score: float,
        behavior_score: float = 0.0,
        top_signals: list[str] | None = None,
        *,
        two_factor_enrolled: bool | None = None,
    ) -> dict:
        """Apply the plan's threshold policy to a model or rules risk score.

        Used by both paths: rules-only fallback and the Phase 2 ensemble.
        At the hold band, a 2FA-enrolled user is challenged (step-up) instead
        of being held for manual review; unenrolled users are held.
        """
        score = round(float(risk_score), 2)
        segment = resolve_segment(txn.amount, txn.account_age_days)
        if two_factor_enrolled is None:
            two_factor_enrolled = bool(getattr(txn, "two_factor_enrolled", False))
        t = self.config.thresholds[segment]
        confirmation_required = False
        if score >= t["block"]:
            confirmation_required = segment in self.config.block_confirmation_segments
            action = HOLD if confirmation_required else BLOCK
        elif score >= t["hold"]:
            action = STEP_UP if two_factor_enrolled else HOLD
        elif score >= t["step_up"]:
            # plan rules layer: >60 + behavioral anomaly -> 2FA, else log.
            # Note: score 60-80 WITHOUT a behavior anomaly stays ALLOW_WITH_LOG
            # per the plan; callers should treat score>=60 as "elevated risk"
            # for capture-rate accounting, not only escalated actions.
            action = STEP_UP if behavior_score > 40.0 else ALLOW_WITH_LOG
        elif score >= t["log"]:
            action = ALLOW_WITH_LOG
        else:
            action = ALLOW

        return {
            "risk_score": score,
            "action": action,
            "proposed_action": BLOCK if confirmation_required else action,
            "block_confirmation_required": confirmation_required,
            "top_signals": top_signals or [],
            "segment": segment,
        }