"""Graph-based fraud-ring detection (plan Phase 2, Model 3).

A lightweight, dependency-free stand-in for Node2Vec/GraphSAGE: the
sender→recipient graph is maintained incrementally in memory, and each
transaction is described by structural features around its recipient
(distinct payers, growth rate, shared devices/IPs). A logistic classifier
trained on those features emits a ring probability, which the ensemble turns
into the plan's "+15 ring boost" when it crosses the flag threshold.

All reads happen BEFORE the current transaction is added, so features are
leakage-free both offline (training replay) and online (real-time path).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from fraud_detection.simulator import Transaction

GRAPH_FEATURES = (
    "recipient_distinct_payers",
    "recipient_payers_1h",
    "recipient_txn_count_1h",
    "recipient_distinct_devices",
    "recipient_distinct_ips",
    "payer_new_to_recipient",
    "recipient_avg_amount",
)

GRAPH_FEATURE_LABELS = {
    "recipient_distinct_payers": "recipient shared by many payers",
    "recipient_payers_1h": "many payers in the last hour",
    "recipient_txn_count_1h": "recipient active in the last hour",
    "recipient_distinct_devices": "recipient seen from many devices",
    "recipient_distinct_ips": "recipient seen from many IPs",
    "payer_new_to_recipient": "first payment to this recipient",
    "recipient_avg_amount": "unusual recipient amount profile",
}

_WINDOW = timedelta(hours=1)
_KEEP = timedelta(hours=24)


@dataclass
class _RecipientState:
    payers: set[str] = field(default_factory=set)
    pairs: set[tuple[str, str]] = field(default_factory=set)  # (payer, recipient)
    events: deque[tuple[datetime, str]] = field(default_factory=deque)  # (time, payer)
    devices: set[str] = field(default_factory=set)
    ips: set[str] = field(default_factory=set)
    amount_sum: float = 0.0
    amount_count: int = 0


class GraphFeatureExtractor:
    """Incremental sender→recipient graph with per-recipient features."""

    def __init__(self) -> None:
        self._recipients: dict[str, _RecipientState] = defaultdict(_RecipientState)

    def extract(self, txn: Transaction) -> dict[str, float]:
        st = self._recipients.get(txn.recipient_id)
        if st is None:
            return {
                "recipient_distinct_payers": 0.0,
                "recipient_payers_1h": 0.0,
                "recipient_txn_count_1h": 0.0,
                "recipient_distinct_devices": 0.0,
                "recipient_distinct_ips": 0.0,
                "payer_new_to_recipient": 1.0,
                "recipient_avg_amount": 0.0,
            }
        recent = [p for t, p in st.events if txn.timestamp - t <= _WINDOW]
        return {
            "recipient_distinct_payers": float(len(st.payers)),
            "recipient_payers_1h": float(len(set(recent))),
            "recipient_txn_count_1h": float(len(recent)),
            "recipient_distinct_devices": float(len(st.devices)),
            "recipient_distinct_ips": float(len(st.ips)),
            "payer_new_to_recipient": 0.0 if (txn.user_id, txn.recipient_id) in st.pairs else 1.0,
            "recipient_avg_amount": round(st.amount_sum / st.amount_count, 2)
            if st.amount_count else 0.0,
        }

    def update(self, txn: Transaction) -> None:
        st = self._recipients[txn.recipient_id]
        st.payers.add(txn.user_id)
        st.pairs.add((txn.user_id, txn.recipient_id))
        st.devices.add(txn.device_id)
        st.ips.add(txn.ip_address)
        st.amount_sum += txn.amount
        st.amount_count += 1
        st.events.append((txn.timestamp, txn.user_id))
        cutoff = txn.timestamp - _KEEP
        while st.events and st.events[0][0] < cutoff:
            st.events.popleft()


@dataclass
class RingModel:
    """Standardised logistic regression over graph features."""

    feature_names: list[str] = field(default_factory=lambda: list(GRAPH_FEATURES))
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    threshold: float = 0.5
    epochs: int = 900
    lr: float = 0.25
    l2: float = 0.01
    pos_weight: float = 8.0

    def fit(self, X: list[list[float]], y: list[int]) -> "RingModel":
        n = len(X)
        nf = len(self.feature_names)
        self.mean = [sum(row[j] for row in X) / n for j in range(nf)]
        self.std = [
            (sum((row[j] - self.mean[j]) ** 2 for row in X) / n) ** 0.5 or 1.0
            for j in range(nf)
        ]
        Z = [[(row[j] - self.mean[j]) / self.std[j] for j in range(nf)] for row in X]
        self.weights = [0.0] * nf
        self.bias = 0.0
        w_pos = max(1.0, self.pos_weight)
        weights_per_row = [w_pos if yi else 1.0 for yi in y]
        for _ in range(self.epochs):
            gw = [0.0] * nf
            gb = 0.0
            for i in range(n):
                z = self.bias + sum(self.weights[j] * Z[i][j] for j in range(nf))
                p = _sigmoid(z)
                err = weights_per_row[i] * (p - y[i])
                for j in range(nf):
                    gw[j] += err * Z[i][j]
                gb += err
            scale = self.lr / n
            for j in range(nf):
                self.weights[j] -= scale * (gw[j] + self.l2 * self.weights[j])
            self.bias -= scale * gb
        return self

    def predict_proba(self, row: dict[str, float]) -> float:
        z = self.bias
        for j, name in enumerate(self.feature_names):
            z += self.weights[j] * ((row.get(name, 0.0) - self.mean[j]) / self.std[j])
        return _sigmoid(z)

    def choose_threshold(self, probs: list[float], y: list[int]) -> float:
        best_t, best_f1 = 0.5, -1.0
        for k in range(1, 10):
            t = k / 10.0
            tp = sum(1 for p, yi in zip(probs, y) if p >= t and yi)
            fp = sum(1 for p, yi in zip(probs, y) if p >= t and not yi)
            fn = sum(1 for p, yi in zip(probs, y) if p < t and yi)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = round(best_t, 4)
        return self.threshold

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "mean": [round(v, 6) for v in self.mean],
            "std": [round(v, 6) for v in self.std],
            "weights": [round(v, 6) for v in self.weights],
            "bias": round(self.bias, 6),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RingModel":
        return cls(
            feature_names=list(d["feature_names"]),
            mean=list(d["mean"]),
            std=list(d["std"]),
            weights=list(d["weights"]),
            bias=float(d["bias"]),
            threshold=float(d["threshold"]),
        )


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
