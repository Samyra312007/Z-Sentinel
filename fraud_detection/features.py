"""Real-time feature computation (plan 1.3 / Phase 3.3).

`FeatureEngine` sits in the transaction path and computes a feature dict for
each transaction from per-user rolling state — the same feature definitions
the offline training pipeline will need, so real-time and batch features
match by construction (plan 6.1).

Latency notes: all state is in-memory per-user, reads are O(velocity
window); measured p99 is reported by the demo.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fraud_detection.simulator import Transaction


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = (dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * (dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class _UserState:
    txn_times: deque[tuple[datetime, str]] = field(default_factory=deque)  # (time, recipient)
    amounts: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    last_txn_at: datetime | None = None
    last_geo: tuple[float, float] | None = None
    last_device: str | None = None
    devices_seen: set[str] = field(default_factory=set)
    recipients_seen: set[str] = field(default_factory=set)
    account_age_days: int = 0


_VELOCITY_WINDOW_1H = timedelta(hours=1)
_VELOCITY_WINDOW_24H = timedelta(hours=24)


class FeatureEngine:
    """Stateful, in-line feature computation for the scoring path."""

    def __init__(self) -> None:
        self._users: dict[str, _UserState] = defaultdict(_UserState)

    # -- public -------------------------------------------------------------

    def compute(
        self,
        txn: Transaction,
        behavior_score: float = 0.0,
        behavior_channels: dict[str, float] | None = None,
    ) -> dict[str, float | int | bool]:
        st = self._users[txn.user_id]
        now = txn.timestamp

        # --- transaction amount features ---
        amounts = list(st.amounts)
        if amounts:
            mean = sum(amounts) / len(amounts)
            sd = math.sqrt(sum((a - mean) ** 2 for a in amounts) / len(amounts)) or 1e-6
            amount_dev_z = (txn.amount - mean) / sd
            avg_30d = mean  # amounts deque ~ 30d rolling history in sim time
        else:
            amount_dev_z = 0.0
            avg_30d = txn.amount

        # --- velocity features ---
        recent_1h = 0
        recent_24h = 0
        same_recipient_1h = 0
        for t, rcpt in st.txn_times:
            if now - t <= _VELOCITY_WINDOW_1H:
                recent_1h += 1
                if rcpt == txn.recipient_id:
                    same_recipient_1h += 1
            if now - t <= _VELOCITY_WINDOW_24H:
                recent_24h += 1
        time_since_last_s = (
            (now - st.last_txn_at).total_seconds() if st.last_txn_at else -1.0
        )

        # --- device / geo / recipient ---
        device_change = st.last_device is not None and txn.device_id != st.last_device
        first_seen_device = txn.device_id not in st.devices_seen
        if st.last_geo is not None:
            geo_dist_km = haversine_km(st.last_geo, txn.geo)
            impossible_travel = (
                geo_dist_km > 800.0 and time_since_last_s >= 0.0
                and time_since_last_s < 4 * 3600.0
            )
        else:
            geo_dist_km = 0.0
            impossible_travel = False
        new_recipient = txn.recipient_id not in st.recipients_seen

        # --- session / time features ---
        hour = now.hour + now.minute / 60.0
        behavior_missing = txn.behavior is None

        features: dict[str, float | int | bool] = {
            "amount": round(txn.amount, 2),
            "amount_dev_30d_z": round(amount_dev_z, 3),
            "avg_amount_30d": round(avg_30d, 2),
            "time_since_last_txn_s": round(time_since_last_s, 1),
            "txn_count_1h": recent_1h,
            "txn_count_24h": recent_24h,
            "same_recipient_1h": same_recipient_1h,
            "loc_distance_last_km": round(geo_dist_km, 2),
            "device_change_flag": device_change,
            "first_seen_device_flag": first_seen_device,
            "new_recipient_flag": new_recipient,
            "impossible_travel_flag": impossible_travel,
            "account_age_days": st.account_age_days,
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

        self._update_state(st, txn)
        return features

    # -- state maintenance (runs only after features are read) ---------------

    def _update_state(self, st: _UserState, txn: Transaction) -> None:
        st.txn_times.append((txn.timestamp, txn.recipient_id))
        st.amounts.append(txn.amount)
        st.last_txn_at = txn.timestamp
        st.last_geo = txn.geo
        st.last_device = txn.device_id
        st.devices_seen.add(txn.device_id)
        st.recipients_seen.add(txn.recipient_id)
        st.account_age_days = txn.account_age_days
