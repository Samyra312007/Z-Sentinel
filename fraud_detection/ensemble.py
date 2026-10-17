"""Ensemble scoring (plan 2.2): model probability + behavior + ring boost.

Final risk index is 0-100 so the decision engine's thresholds (30/60/80/95)
keep their meaning:

    risk = 100 * P(fraud | transaction features)
           + behavior_weight * behavioral_anomaly_score   (0-100)
           + ring_boost (15) when the graph model flags a ring

The transaction model is the primary scorer; behavioral deviation is an
additive boost (an account takeover with a strong model score still escalates
even when the model saw a legitimate-looking payment), and the graph model
supplies the plan's fixed "+15 ring risk" bump.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraud_detection.features import label_for
from fraud_detection.graph import GRAPH_FEATURE_LABELS
from fraud_detection.model import FraudModel


@dataclass
class EnsembleScorer:
    model: FraudModel
    behavior_weight: float = 0.30
    ring_boost: float = 15.0
    top_signals: int = 3

    def score(
        self,
        features: dict,
        behavior_score: float,
        graph_features: dict,
    ) -> dict:
        prob = self.model.predict_proba(features)
        ring_prob = self.model.ring_probability(graph_features)
        ring_flagged = ring_prob >= self.model.ring.threshold

        risk = 100.0 * prob
        risk += self.behavior_weight * float(behavior_score)
        if ring_flagged:
            risk += self.ring_boost
        risk = round(min(100.0, max(0.0, risk)), 2)

        # Signed, per-feature contributions (Saabas path attribution). The
        # dashboard and review payloads expose these so every score is
        # explainable, not just labelled with a coarse signal name.
        contributions: list[dict] = []
        for name, value in self.model.contributions(features).items():
            if value > 0.1:
                contributions.append(
                    {
                        "feature": name,
                        "label": label_for(name),
                        "contribution": round(float(value), 4),
                        "direction": "increases_risk",
                    }
                )
        if behavior_score >= 15.0:
            contributions.append(
                {
                    "feature": "behavior_anomaly_score",
                    "label": "behavioral biometrics deviation",
                    "contribution": round(behavior_score / 100.0, 4),
                    "direction": "increases_risk",
                }
            )
        if ring_flagged:
            contributions.append(
                {
                    "feature": "ring_flag",
                    "label": "fraud-ring pattern (shared recipient)",
                    "contribution": round(ring_prob, 4),
                    "direction": "increases_risk",
                }
            )
        for name, value in graph_features.items():
            if name in ("recipient_distinct_payers", "recipient_payers_1h") and value >= 3:
                contributions.append(
                    {
                        "feature": name,
                        "label": GRAPH_FEATURE_LABELS[name],
                        "contribution": round(value / 10.0, 4),
                        "direction": "increases_risk",
                    }
                )

        contributions.sort(key=lambda item: -abs(item["contribution"]))
        top_contributions = contributions[: self.top_signals]
        top = [item["label"] for item in top_contributions] or ["no_risk_signals"]

        return {
            "risk_score": risk,
            "model_probability": round(prob, 4),
            "behavior_score": round(float(behavior_score), 2),
            "ring_probability": round(ring_prob, 4),
            "ring_flag": ring_flagged,
            "top_signals": top,
            "signal_scores": [
                (item["label"], item["contribution"]) for item in top_contributions
            ],
            "explanations": top_contributions,
            "explanation_method": "gbdt_saabas_path_attribution",
        }
