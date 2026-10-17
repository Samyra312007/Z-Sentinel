"""Real-time feature store (plan 3.3).

Per-user aggregates are maintained incrementally as transactions stream past:
the scoring path reads a precomputed snapshot instead of recomputing rolling
windows, so serving latency stays flat as history grows.

Contents of a `UserState`:

* 7-day and 30-day amount aggregates (sums/counts maintained on update)
* the last 200 payment amounts (the Phase 1/2 model's deviation baseline)
* payment velocity windows (1h / 24h)
* device, recipient and location history
* account age and the account-level fraud flag (set by confirmed-fraud
  feedback, never by stream labels — those are evaluation-only)
* the behavioral-biometrics baseline for the user

Snapshot reads are O(payments in the 24h window) and never mutate state; in
practice that is a handful of entries, measured in microseconds. Updates only
prune expired entries, so each write is O(expired).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fraud_detection.biometrics import BehavioralAnomalyDetector
from fraud_detection.simulator import BehavioralSample, Transaction

VELOCITY_WINDOW_1H = timedelta(hours=1)
VELOCITY_WINDOW_24H = timedelta(hours=24)
AMOUNT_WINDOW_7D = timedelta(days=7)
AMOUNT_WINDOW_30D = timedelta(days=30)

# Kept for parity with the Phase 1/2 model: `amount_dev_30d_z` and
# `avg_amount_30d` are defined over the most recent 200 payments. The true
# time-window aggregates (avg_amount_7d) are additional store features.
_AMOUNT_HISTORY = 200


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = (dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * (dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class UserState:
    """Mutable per-user aggregate record owned by the store."""

    txn_times: deque[tuple[datetime, str]] = field(default_factory=deque)
    amounts: deque[float] = field(default_factory=lambda: deque(maxlen=_AMOUNT_HISTORY))
    amount_7d: deque[tuple[datetime, float]] = field(default_factory=deque)
    amount_30d: deque[tuple[datetime, float]] = field(default_factory=deque)
    sum_7d: float = 0.0
    count_7d: int = 0
    sum_30d: float = 0.0
    count_30d: int = 0
    last_txn_at: datetime | None = None
    last_geo: tuple[float, float] | None = None
    last_device: str | None = None
    devices_seen: set[str] = field(default_factory=set)
    recipients_seen: set[str] = field(default_factory=set)
    # coarse location histogram (0.1 degree grid) for the user's typical area
    geo_counts: dict[tuple[float, float], int] = field(default_factory=dict)
    account_age_days: int = 0
    historical_fraud_flag: bool = False
    transactions: int = 0


@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable view of everything the store knows before a transaction."""

    user_id: str
    transaction_count: int
    avg_amount_30d: float
    amount_std_30d: float
    amount_dev_30d_z: float
    avg_amount_7d: float
    txn_count_1h: int
    txn_count_24h: int
    same_recipient_1h: int
    time_since_last_txn_s: float
    loc_distance_last_km: float
    impossible_travel_flag: bool
    device_change_flag: bool
    first_seen_device_flag: bool
    new_recipient_flag: bool
    account_age_days: int
    historical_fraud_flag: bool
    distinct_devices: int
    distinct_recipients: int
    distinct_locations: int
    home_geo: tuple[float, float] | None
    home_geo_distance_km: float
    last_geo: tuple[float, float] | None
    last_device: str | None


class FeatureStore:
    """In-memory online feature store for the transaction path.

    The store is intentionally single-process and dependency-free: on IBM Z
    the same interface maps to an IMS/DB2 cached table or an attached Redis.
    """

    def __init__(self) -> None:
        self._users: dict[str, UserState] = defaultdict(UserState)
        self.behavior = BehavioralAnomalyDetector()

    # -- reads --------------------------------------------------------------

    def snapshot(self, txn: Transaction) -> FeatureSnapshot:
        """Precomputed features for `txn`, read before it is applied."""
        st = self._users[txn.user_id]
        now = txn.timestamp

        amounts = list(st.amounts)
        if amounts:
            mean = sum(amounts) / len(amounts)
            sd = math.sqrt(sum((a - mean) ** 2 for a in amounts) / len(amounts)) or 1e-6
            amount_dev_z = (txn.amount - mean) / sd
        else:
            mean = txn.amount
            sd = 0.0
            amount_dev_z = 0.0

        recent_1h = 0
        recent_24h = 0
        same_recipient_1h = 0
        for t, rcpt in st.txn_times:
            if now - t <= VELOCITY_WINDOW_24H:
                recent_24h += 1
                if now - t <= VELOCITY_WINDOW_1H:
                    recent_1h += 1
                    if rcpt == txn.recipient_id:
                        same_recipient_1h += 1

        if st.last_txn_at is not None:
            time_since_last_s = (now - st.last_txn_at).total_seconds()
        else:
            time_since_last_s = -1.0

        if st.last_geo is not None:
            geo_dist_km = haversine_km(st.last_geo, txn.geo)
            impossible_travel = (
                geo_dist_km > 800.0
                and 0.0 <= time_since_last_s < 4 * 3600.0
            )
        else:
            geo_dist_km = 0.0
            impossible_travel = False

        avg_7d = st.sum_7d / st.count_7d if st.count_7d else mean
        home_geo = (
            max(st.geo_counts, key=lambda geo: st.geo_counts[geo])
            if st.geo_counts
            else None
        )
        home_geo_distance_km = (
            haversine_km(home_geo, txn.geo) if home_geo is not None else 0.0
        )

        return FeatureSnapshot(
            user_id=txn.user_id,
            transaction_count=st.transactions,
            avg_amount_30d=round(mean, 2),
            amount_std_30d=sd,
            amount_dev_30d_z=amount_dev_z,
            avg_amount_7d=round(avg_7d, 2),
            txn_count_1h=recent_1h,
            txn_count_24h=recent_24h,
            same_recipient_1h=same_recipient_1h,
            time_since_last_txn_s=time_since_last_s,
            loc_distance_last_km=geo_dist_km,
            impossible_travel_flag=impossible_travel,
            device_change_flag=st.last_device is not None and txn.device_id != st.last_device,
            first_seen_device_flag=txn.device_id not in st.devices_seen,
            new_recipient_flag=txn.recipient_id not in st.recipients_seen,
            account_age_days=st.account_age_days,
            historical_fraud_flag=st.historical_fraud_flag,
            distinct_devices=len(st.devices_seen),
            distinct_recipients=len(st.recipients_seen),
            distinct_locations=len(st.geo_counts),
            home_geo=home_geo,
            home_geo_distance_km=home_geo_distance_km,
            last_geo=st.last_geo,
            last_device=st.last_device,
        )

    # -- writes -------------------------------------------------------------

    def apply(self, txn: Transaction) -> None:
        """Advance the store with a transaction that has already been scored."""
        st = self._users[txn.user_id]
        now = txn.timestamp
        st.transactions += 1

        st.txn_times.append((now, txn.recipient_id))
        cutoff = now - VELOCITY_WINDOW_24H
        while st.txn_times and st.txn_times[0][0] < cutoff:
            st.txn_times.popleft()

        st.amounts.append(txn.amount)

        st.amount_7d.append((now, txn.amount))
        st.sum_7d += txn.amount
        st.count_7d += 1
        cutoff_7d = now - AMOUNT_WINDOW_7D
        while st.amount_7d and st.amount_7d[0][0] < cutoff_7d:
            _, expired = st.amount_7d.popleft()
            st.sum_7d -= expired
            st.count_7d -= 1

        st.amount_30d.append((now, txn.amount))
        st.sum_30d += txn.amount
        st.count_30d += 1
        cutoff_30d = now - AMOUNT_WINDOW_30D
        while st.amount_30d and st.amount_30d[0][0] < cutoff_30d:
            _, expired = st.amount_30d.popleft()
            st.sum_30d -= expired
            st.count_30d -= 1

        st.last_txn_at = now
        st.last_geo = txn.geo
        st.last_device = txn.device_id
        st.devices_seen.add(txn.device_id)
        st.recipients_seen.add(txn.recipient_id)
        geo_key = (round(txn.geo[0], 1), round(txn.geo[1], 1))
        st.geo_counts[geo_key] = st.geo_counts.get(geo_key, 0) + 1
        st.account_age_days = txn.account_age_days

    def score_behavior(self, sample: BehavioralSample) -> tuple[float, dict[str, float]]:
        """Update/read the per-user behavioral baseline (plan 3.3)."""
        return self.behavior.score(sample)

    def mark_fraud(self, user_id: str, flagged: bool = True) -> None:
        """Set the account-level historical fraud flag from confirmed feedback."""
        self._users[user_id].historical_fraud_flag = flagged

    # -- introspection ------------------------------------------------------

    @property
    def users(self) -> int:
        return len(self._users)

    def stats(self) -> dict[str, int]:
        return {
            "users": len(self._users),
            "transactions": sum(st.transactions for st in self._users.values()),
        }

    def get(self, user_id: str) -> UserState:
        return self._users[user_id]
