"""Versioned model artifact: calibrated transaction model + ring model.

The saved JSON is the plan's "export model artifact" deliverable (2.3/2.4):
plain, auditable, and loadable on IBM Z without any ML runtime. Converting
the same structure to ONNX for ML for z/OS is a later deployment step.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field

from fraud_detection.features import as_row
from fraud_detection.gbdt import GBDT, sigmoid
from fraud_detection.graph import GRAPH_FEATURES, RingModel


@dataclass
class PlattCalibrator:
    """Platt scaling: p = sigmoid(a * (logit - mu)/sd + b)."""

    a: float = 1.0
    b: float = 0.0
    mu: float = 0.0
    sd: float = 1.0
    epochs: int = 2500
    lr: float = 0.05

    def fit(self, logits: list[float], y: list[int]) -> "PlattCalibrator":
        if len(logits) < 2 or sum(y) in (0, len(y)):
            return self
        self.mu = sum(logits) / len(logits)
        var = sum((v - self.mu) ** 2 for v in logits) / len(logits)
        self.sd = math.sqrt(var) or 1.0
        z = [(v - self.mu) / self.sd for v in logits]
        for _ in range(self.epochs):
            grad_a = 0.0
            grad_b = 0.0
            for zi, yi in zip(z, y):
                p = sigmoid(self.a * zi + self.b)
                grad_a += (p - yi) * zi
                grad_b += p - yi
            n = len(logits)
            self.a -= self.lr * grad_a / n
            self.b -= self.lr * grad_b / n
        return self

    def predict(self, logit: float) -> float:
        return sigmoid(self.a * ((logit - self.mu) / self.sd) + self.b)

    def to_dict(self) -> dict:
        return {
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "mu": round(self.mu, 6),
            "sd": round(self.sd, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        return cls(a=float(d["a"]), b=float(d["b"]), mu=float(d["mu"]), sd=float(d["sd"]))


@dataclass
class FraudModel:
    """Everything needed to score a transaction, loaded from one JSON file."""

    version: str = "phase2-1"
    created_at: str = ""
    gbdt: GBDT = field(default_factory=GBDT)
    calibrator: PlattCalibrator = field(default_factory=PlattCalibrator)
    ring: RingModel = field(default_factory=RingModel)
    metrics: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)

    # -- scoring ------------------------------------------------------------

    def logit(self, features: dict) -> float:
        return self.gbdt.predict_logit([as_row(features, self.gbdt.feature_names)])[0]

    def predict_proba(self, features: dict) -> float:
        return self.calibrator.predict(self.logit(features))

    def contributions(self, features: dict) -> dict[str, float]:
        """Per-feature log-odds contribution (Saabas approximation)."""
        idx = self.gbdt.contributions(as_row(features, self.gbdt.feature_names))
        return {
            self.gbdt.feature_names[i]: round(v, 4)
            for i, v in sorted(idx.items(), key=lambda kv: -abs(kv[1]))
        }

    def ring_probability(self, graph_features: dict) -> float:
        return self.ring.predict_proba(graph_features)

    def ring_flag(self, graph_features: dict) -> bool:
        return self.ring_probability(graph_features) >= self.ring.threshold

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "framework": "fraud_detection.gbdt (pure python)",
            "gbdt": self.gbdt.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "ring": self.ring.to_dict(),
            "graph_features": list(GRAPH_FEATURES),
            "metrics": self.metrics,
            "dataset": self.dataset,
        }

    def save(self, path: str) -> None:
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))

    @classmethod
    def load(cls, path: str) -> "FraudModel":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            version=str(d.get("version", "unknown")),
            created_at=str(d.get("created_at", "")),
            gbdt=GBDT.from_dict(d["gbdt"]),
            calibrator=PlattCalibrator.from_dict(d["calibrator"]),
            ring=RingModel.from_dict(d["ring"]),
            metrics=dict(d.get("metrics", {})),
            dataset=dict(d.get("dataset", {})),
        )
