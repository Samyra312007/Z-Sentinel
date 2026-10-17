"""Synthetic transaction + behavioral event generator.

Mirrors the plan's "Data Mocking Strategy": no real transaction data is
assumed. The simulator produces a realistic stream of payment transactions
with embedded fraud patterns (card testing, account takeover, fraud ring
mule payouts), each transaction optionally carrying a behavioral-biometrics
sample collected during the login/payment session.

Every generated event carries its ground-truth label so the demo and any
offline evaluation can measure capture rate and false positive rate. Labels
never reach the feature engine or decision engine.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Event schemas (the "wire format" a CICS/IMS transaction stream would carry)
# ---------------------------------------------------------------------------

FRAUD_STRATEGIES = ("card_testing", "account_takeover", "fraud_ring")


@dataclass(frozen=True)
class BehavioralSample:
    """Session-level aggregates of keystroke/mouse dynamics.

    Privacy note (plan 3.2): only timing/shape statistics are present, never
    typed content or raw pointer paths.
    """

    session_id: str
    user_id: str
    timestamp: datetime
    key_hold_mean_ms: float  # mean key press duration
    key_hold_std_ms: float  # variability of press duration
    inter_key_ms: float  # mean interval between consecutive keys
    mouse_velocity_px_s: float  # mean pointer velocity
    mouse_entropy: float  # movement entropy (0 = robotic, high = human)
    # Click shape: count plus mean normalised viewport position (0..1).
    click_count: int = 0
    click_x_mean_norm: float = 0.0
    click_y_mean_norm: float = 0.0
    # Touch dynamics (mobile sessions); 0.0 when the session had no touch input.
    touch_pressure_mean: float = 0.0
    touch_speed_px_s: float = 0.0
    touch_swipe_angle_deg: float = 0.0


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    user_id: str
    timestamp: datetime
    amount: float
    currency: str
    channel: str  # "card_pos" | "card_web" | "mobile" | "transfer"
    merchant_category: str
    recipient_id: str
    device_id: str
    ip_address: str
    geo: tuple[float, float]  # (latitude, longitude)
    account_age_days: int
    behavior: BehavioralSample | None = None
    # Ground truth — evaluation only, never given to scoring code:
    is_fraud: bool = False
    fraud_strategy: str | None = None


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    user_id: str
    home_geo: tuple[float, float]
    amount_mean: float
    amount_sigma: float  # log-space sigma
    devices: list[str]
    recipients: list[str]  # established counterparties
    account_age_days: int
    # "True" biometric baseline (what enrollment captured):
    bio_key_hold_mean: float
    bio_key_hold_std: float
    bio_inter_key: float
    bio_mouse_velocity: float
    bio_mouse_entropy: float


@dataclass
class World:
    users: list[UserProfile]
    mule_recipients: list[str]  # shared drop accounts for fraud_ring
    fraud_recipients: list[str]  # fresh throwaway recipients for takeovers


GEO_ZONES: dict[str, tuple[float, float]] = {
    "new_york": (40.7128, -74.0060),
    "boston": (42.3601, -71.0589),
    "chicago": (41.8781, -87.6298),
    "austin": (30.2672, -97.7431),
    "los_angeles": (34.0522, -118.2437),
    "miami": (25.7617, -80.1918),
    "london": (51.5074, -0.1278),
    "toronto": (43.6532, -79.3832),
}

MERCHANT_CATEGORIES = (
    "grocery", "restaurant", "fuel", "retail", "travel",
    "electronics", "wire_transfer", "gambling", "crypto", "subscription",
)
CHANNELS = ("card_pos", "card_web", "mobile", "transfer")


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(float, (a[0], a[1], b[0], b[1]))
    r = 6371.0
    p1, p2 = lat1 * 0.017453292519943295, lat2 * 0.017453292519943295
    dp = (lat2 - lat1) * 0.017453292519943295
    dl = (lon2 - lon1) * 0.017453292519943295
    h = (dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * (dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, h ** 0.5))


class TransactionSimulator:
    """Yields labeled Transaction events in chronological order.

    Fraud injection (plan section 5 — realistic patterns):
      * card_testing   — burst of tiny payments to fresh recipients
      * account_takeover — legitimate user suddenly pays a stranger from a
        new device/geo with visibly stressed biometrics
      * fraud_ring     — several distinct users pay the same mule recipient
    """

    def __init__(
        self,
        n_users: int = 80,
        fraud_rate: float = 0.02,
        seed: int = 42,
        start: datetime | None = None,
    ) -> None:
        if not 0.0 < fraud_rate < 1.0:
            raise ValueError("fraud_rate must be in (0, 1)")
        self.rng = random.Random(seed)
        self.fraud_rate = fraud_rate
        self.clock = start or datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        self._ids = itertools.count(1)
        self._session_ids = itertools.count(1)
        self.world = self._build_world(n_users)
        self._fraud_budget = 0  # drip-feed fraud across the stream
        self._card_testing_left = 0
        self._card_testing_recipient = ""
        self._card_testing_user: UserProfile | None = None

    # -- world construction ------------------------------------------------

    def _build_world(self, n_users: int) -> World:
        rng = self.rng
        zones = list(GEO_ZONES)
        users = []
        for i in range(n_users):
            uid = f"U{i:04d}"
            bio_mu = rng.gauss(115.0, 15.0)  # key hold ms
            bio_ik = rng.gauss(180.0, 30.0)  # inter-key ms
            users.append(
                UserProfile(
                    user_id=uid,
                    home_geo=GEO_ZONES[rng.choice(zones)],
                    amount_mean=rng.uniform(15.0, 400.0),
                    amount_sigma=rng.uniform(0.35, 0.75),
                    devices=[f"dev-{uid}-a"] + ([f"dev-{uid}-b"] if rng.random() < 0.4 else []),
                    recipients=[f"rcpt-{uid}-{k}" for k in range(rng.randint(3, 12))],
                    account_age_days=int(rng.triangular(30, 4000, 900)),
                    bio_key_hold_mean=bio_mu,
                    bio_key_hold_std=abs(rng.gauss(14.0, 4.0)),
                    bio_inter_key=bio_ik,
                    bio_mouse_velocity=abs(rng.gauss(650.0, 120.0)),
                    bio_mouse_entropy=rng.uniform(2.5, 4.5),
                )
            )
        return World(
            users=users,
            mule_recipients=[f"mule-{k:03d}" for k in range(6)],
            fraud_recipients=[f"throwaway-{k:04d}" for k in range(200)],
        )

    # -- behavioral sampling ----------------------------------------------

    def _sample_behavior(self, user: UserProfile, stressed: bool) -> BehavioralSample:
        rng = self.rng
        if stressed:  # takeover: fast, robotic, erratic typing/mouse
            hold = user.bio_key_hold_mean * rng.uniform(0.6, 0.8)
            std = user.bio_key_hold_std * rng.uniform(0.3, 0.6)
            ik = user.bio_inter_key * rng.uniform(0.5, 0.75)
            vel = user.bio_mouse_velocity * rng.uniform(1.5, 2.4)
            ent = user.bio_mouse_entropy * rng.uniform(0.35, 0.55)
        else:
            hold = rng.gauss(user.bio_key_hold_mean, max(2.0, user.bio_key_hold_std * 0.5))
            std = abs(rng.gauss(user.bio_key_hold_std, 2.0))
            ik = rng.gauss(user.bio_inter_key, user.bio_inter_key * 0.15)
            vel = abs(rng.gauss(user.bio_mouse_velocity, user.bio_mouse_velocity * 0.15))
            ent = rng.gauss(user.bio_mouse_entropy, 0.2)
        return BehavioralSample(
            session_id=f"S{next(self._session_ids):06d}",
            user_id=user.user_id,
            timestamp=self.clock,
            key_hold_mean_ms=round(max(40.0, hold), 2),
            key_hold_std_ms=round(max(3.0, std), 2),
            inter_key_ms=round(max(60.0, ik), 2),
            mouse_velocity_px_s=round(max(50.0, vel), 1),
            mouse_entropy=round(min(6.0, max(0.5, ent)), 3),
        )

    # -- generators ---------------------------------------------------------

    def _normal_event(self) -> Transaction:
        rng = self.rng
        user = rng.choice(self.world.users)
        amount = min(9000.0, rng.lognormvariate(math.log(user.amount_mean), user.amount_sigma))
        if rng.random() < 0.05:  # occasional large legitimate purchase
            amount = min(9000.0, amount * rng.uniform(3.0, 7.0))

        # legitimate novelty: new recipients / devices / travel all happen
        if rng.random() < 0.08:
            recipient = f"rcpt-{user.user_id}-new-{rng.randint(1000, 9999)}"
            user.recipients.append(recipient)  # established after first payment
        else:
            recipient = rng.choice(user.recipients)
        if rng.random() < 0.03:
            device = f"dev-{user.user_id}-new-{rng.randint(1, 999)}"
            user.devices.append(device)
        else:
            device = rng.choice(user.devices)
        if rng.random() < 0.03:  # the user is travelling this week
            zone = rng.choice(list(GEO_ZONES.values()))
            geo = (zone[0] + rng.gauss(0, 0.2), zone[1] + rng.gauss(0, 0.2))
        else:
            geo = (
                user.home_geo[0] + rng.gauss(0, 0.15),
                user.home_geo[1] + rng.gauss(0, 0.15),
            )
        return Transaction(
            txn_id=f"TXN{next(self._ids):07d}",
            user_id=user.user_id,
            timestamp=self.clock,
            amount=round(amount, 2),
            currency="USD",
            channel=rng.choice(CHANNELS),
            merchant_category=rng.choice(MERCHANT_CATEGORIES[:-3]),
            recipient_id=recipient,
            device_id=device,
            ip_address=f"203.0.113.{rng.randint(1, 254)}",
            geo=(round(geo[0], 4), round(geo[1], 4)),
            account_age_days=user.account_age_days,
            behavior=self._sample_behavior(user, stressed=False),
            is_fraud=False,
        )

    def _fraud_event(self, strategy: str) -> Transaction:
        rng = self.rng
        if strategy == "card_testing":
            # whole burst hits the SAME victim card within minutes
            if self._card_testing_left <= 0 or self._card_testing_user is None:
                raise RuntimeError("card_testing burst not initialized")
            user = self._card_testing_user
            recipient = self._card_testing_recipient
            self._card_testing_left -= 1
            amount = round(rng.uniform(0.5, 15.0), 2)
            device = rng.choice(user.devices)  # stolen card, device may look usual
            behavior = self._sample_behavior(user, stressed=False)
            geo = (user.home_geo[0] + 0.02, user.home_geo[1] + 0.02)
            channel = "card_web"
        elif strategy == "account_takeover":
            user = rng.choice(self.world.users)
            recipient = rng.choice(self.world.fraud_recipients)
            amount = round(rng.lognormvariate(math.log(user.amount_mean * 2.2), 0.7), 2)
            if rng.random() < 0.3:  # high-value takeovers overlap normal spend
                amount = round(amount * rng.uniform(2.0, 4.0), 2)
            device = f"dev-{user.user_id}-attacker-{rng.randint(1, 9)}"  # new device
            behavior = self._sample_behavior(user, stressed=True)
            far = rng.choice([z for z in GEO_ZONES.values() if _haversine_km(user.home_geo, z) > 800])
            geo = (far[0] + rng.gauss(0, 0.1), far[1] + rng.gauss(0, 0.1))
            channel = rng.choice(("card_web", "transfer"))
        else:  # fraud_ring: distinct users -> shared mule recipient
            user = rng.choice(self.world.users)
            recipient = rng.choice(self.world.mule_recipients)
            amount = round(rng.uniform(150.0, 2000.0), 2)
            device = rng.choice(user.devices)
            behavior = self._sample_behavior(user, stressed=False)
            geo = (user.home_geo[0] + rng.gauss(0, 0.1), user.home_geo[1] + rng.gauss(0, 0.1))
            channel = "transfer"
        return Transaction(
            txn_id=f"TXN{next(self._ids):07d}",
            user_id=user.user_id,
            timestamp=self.clock,
            amount=amount,
            currency="USD",
            channel=channel,
            merchant_category="wire_transfer" if strategy == "fraud_ring" else rng.choice(MERCHANT_CATEGORIES),
            recipient_id=recipient,
            device_id=device,
            ip_address=f"198.51.100.{rng.randint(1, 254)}",
            geo=(round(geo[0], 4), round(geo[1], 4)),
            account_age_days=user.account_age_days,
            behavior=behavior,
            is_fraud=True,
            fraud_strategy=strategy,
        )

    # -- stream -------------------------------------------------------------

    def events(self, n: int):
        """Generate n chronological transactions (one event per transaction)."""
        for _ in range(n):
            # advance simulated clock 30s .. 10min between events; a running
            # card-testing burst fires seconds apart, as it does in reality
            if self._card_testing_left > 0:
                self.clock += timedelta(seconds=self.rng.randint(10, 45))
            else:
                self.clock += timedelta(seconds=self.rng.randint(30, 600))
            # drip the fraud budget so fraud is spread out, not clumped at the end
            strategy: str | None = None
            if self._card_testing_left > 0:
                strategy = "card_testing"
            else:
                self._fraud_budget += self.fraud_rate
                if self._fraud_budget >= 1.0:
                    self._fraud_budget -= 1.0
                    strategy = self.rng.choice(FRAUD_STRATEGIES)
                    if strategy == "card_testing":
                        # start a burst: same victim, fresh recipient, tight clock.
                        # charge the whole burst to the budget so the requested
                        # fraud_rate stays honest (burst can push it negative).
                        length = self.rng.randint(4, 8)
                        self._card_testing_user = self.rng.choice(self.world.users)
                        self._card_testing_recipient = self.rng.choice(self.world.fraud_recipients)
                        self._card_testing_left = length
                        self._fraud_budget -= length - 1
            if strategy is None:
                yield self._normal_event()
            else:
                yield self._fraud_event(strategy)