"""Privacy and retention controls (plan 5.2).

Behavioral telemetry is reduced to session aggregates at the SDK boundary
(`fraud_detection.collection`): typed content, key codes and raw pointer paths
are rejected, and only aggregate measurements enter the feature store. This
module adds the retention half of the privacy requirement:

* `RetentionPolicy` is a validated, explicit retention window.
* `RetentionPolicy.apply` removes closed review cases (and their investigator
  action history) older than the window. Open and `NEED_INFO` cases remain,
  because deleting them would erase pending work.

The decision audit is intentionally immutable and is not purged here; a
production deployment ships immutable audit segments to the enterprise
archive under the organisation's records-retention schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fraud_detection.reviews import ReviewStore


@dataclass(frozen=True)
class RetentionPolicy:
    """How long closed review cases may remain in the operational store."""

    review_days: int = 365
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.review_days, bool) or not isinstance(self.review_days, int):
            raise ValueError("review_days must be an integer")
        if self.review_days < 1:
            raise ValueError("review_days must be positive")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

    def cutoff(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - timedelta(days=self.review_days)).isoformat(
            timespec="milliseconds"
        )

    def apply(
        self,
        store: ReviewStore,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "cases": 0, "actions": 0, "cutoff": None}
        cutoff = self.cutoff(now)
        purged = store.purge_closed_before(cutoff)
        return {"enabled": True, "cutoff": cutoff, **purged}

    def to_dict(self) -> dict[str, Any]:
        return {"review_days": self.review_days, "enabled": self.enabled}
