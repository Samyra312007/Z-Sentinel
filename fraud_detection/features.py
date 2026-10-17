"""Real-time feature computation (plan 1.3 / Phase 3.3).

`FeatureEngine` sits in the transaction path and computes a feature dict for
each transaction from the precomputed per-user aggregates in `FeatureStore`.
The same feature definitions are used by the offline training pipeline, so
real-time and batch features match by construction (plan 6.1).

Latency notes: the store snapshot and this function perform no I/O; measured
p99 is reported by the Phase 3 demo.
"""

from __future__ import annotations

from fraud_detection.feature_store import FeatureSnapshot, FeatureStore, haversine_km
from fraud_detection.simulator import Transaction

__all__ = [
    "FeatureEngine",
    "FeatureSnapshot",
    "FeatureStore",
    "haversine_km",
    "vectorize",
    "as_row",
    "label_for",
]


class FeatureEngine:
    """Computes the model feature vector from a feature-store snapshot."""

    def __init__(self, store: FeatureStore | None = None) -> None:
        self.store = store or FeatureStore()

    # -- public -------------------------------------------------------------

    def compute(
        self,
        txn: Transaction,
        behavior_score: float = 0.0,
        behavior_channels: dict[str, float] | None = None,
        *,
        behavior_present: bool | None = None,
    ) -> dict[str, float | int | bool]:
        snap = self.store.snapshot(txn)
        now = txn.timestamp

        # --- session / time features ---
        hour = now.hour + now.minute / 60.0
        behavior_missing = (
            txn.behavior is None if behavior_present is None else not behavior_present
        )

        features: dict[str, float | int | bool] = {
            "amount": round(txn.amount, 2),
            "amount_dev_30d_z": round(snap.amount_dev_30d_z, 3),
            "avg_amount_30d": round(snap.avg_amount_30d, 2),
            "avg_amount_7d": round(snap.avg_amount_7d, 2),
            "time_since_last_txn_s": round(snap.time_since_last_txn_s, 1),
            "txn_count_1h": snap.txn_count_1h,
            "txn_count_24h": snap.txn_count_24h,
            "same_recipient_1h": snap.same_recipient_1h,
            "loc_distance_last_km": round(snap.loc_distance_last_km, 2),
            "device_change_flag": snap.device_change_flag,
            "first_seen_device_flag": snap.first_seen_device_flag,
            "new_recipient_flag": snap.new_recipient_flag,
            "impossible_travel_flag": snap.impossible_travel_flag,
            "account_age_days": snap.account_age_days,
            "historical_fraud_flag": snap.historical_fraud_flag,
            "distinct_devices": snap.distinct_devices,
            "distinct_recipients": snap.distinct_recipients,
            "distinct_locations": snap.distinct_locations,
            "home_geo_distance_km": round(snap.home_geo_distance_km, 2),
            "hour_of_day": round(hour, 2),
            "is_odd_hours_flag": hour < 6.0 or hour >= 23.0,
            "channel": txn.channel,
            "merchant_category": txn.merchant_category,
            "behavior_anomaly_score": float(behavior_score),
            "behavior_missing_flag": behavior_missing,
        }
        channels = behavior_channels or {}
        features["behavior_max_channel_z"] = round(
            max((abs(v) for v in channels.values()), default=0.0), 3
        )

        self.store.apply(txn)
        return features


# ---------------------------------------------------------------------------
# Model vectorisation (shared by the offline trainer and the scoring path, so
# there is exactly one feature definition — plan 6.1 parity requirement)
# ---------------------------------------------------------------------------

CHANNEL_VALUES = ("card_pos", "card_web", "mobile", "transfer")
MERCHANT_VALUES = (
    "grocery", "restaurant", "fuel", "retail", "travel",
    "electronics", "wire_transfer", "gambling", "crypto", "subscription",
)
BOOLEAN_FEATURES = (
    "device_change_flag",
    "first_seen_device_flag",
    "new_recipient_flag",
    "impossible_travel_flag",
    "is_odd_hours_flag",
    "behavior_missing_flag",
)
NUMERIC_FEATURES = (
    "amount",
    "amount_dev_30d_z",
    "avg_amount_30d",
    "time_since_last_txn_s",
    "txn_count_1h",
    "txn_count_24h",
    "same_recipient_1h",
    "loc_distance_last_km",
    "account_age_days",
    "hour_of_day",
    "behavior_anomaly_score",
    "behavior_max_channel_z",
)


def vectorize(features: dict) -> dict[str, float]:
    """Flatten the feature dict to a stable numeric vector (dict form)."""
    out: dict[str, float] = {}
    for name in NUMERIC_FEATURES:
        out[name] = float(features.get(name, 0.0) or 0.0)
    for name in BOOLEAN_FEATURES:
        out[name] = 1.0 if features.get(name) else 0.0
    channel = str(features.get("channel", ""))
    for c in CHANNEL_VALUES:
        out[f"channel_{c}"] = 1.0 if channel == c else 0.0
    mcc = str(features.get("merchant_category", ""))
    for m in MERCHANT_VALUES:
        out[f"mcc_{m}"] = 1.0 if mcc == m else 0.0
    return out


def as_row(features: dict, order: list[str]) -> list[float]:
    """Vectorise and order by the trained model's feature order."""
    vec = vectorize(features)
    return [vec.get(name, 0.0) for name in order]


# Human-readable labels for the top-signal explanations (plan 5.2)
SIGNAL_LABELS: dict[str, str] = {
    "amount": "payment amount",
    "amount_dev_30d_z": "amount vs. user's 30-day history",
    "avg_amount_30d": "user's average payment",
    "time_since_last_txn_s": "time since previous payment",
    "txn_count_1h": "payments in the last hour",
    "txn_count_24h": "payments in the last 24h",
    "same_recipient_1h": "repeat payments to same recipient",
    "loc_distance_last_km": "distance from previous payment",
    "account_age_days": "account age",
    "hour_of_day": "time of day",
    "behavior_anomaly_score": "typing / mouse behavior anomaly",
    "behavior_max_channel_z": "strongest biometric deviation",
    "device_change_flag": "device changed",
    "first_seen_device_flag": "unrecognized device",
    "new_recipient_flag": "new recipient",
    "impossible_travel_flag": "impossible travel",
    "is_odd_hours_flag": "unusual hour",
    "behavior_missing_flag": "no behavioral sample",
    "channel_card_pos": "channel: card present",
    "channel_card_web": "channel: card online",
    "channel_mobile": "channel: mobile",
    "channel_transfer": "channel: transfer",
}


def label_for(feature_name: str) -> str:
    if feature_name in SIGNAL_LABELS:
        return SIGNAL_LABELS[feature_name]
    if feature_name.startswith("mcc_"):
        return f"merchant category: {feature_name[4:]}"
    return feature_name.replace("_", " ")
