"""Monitoring and alerting for the real-time fraud path (plan 4.3).

`MetricsMonitor` consumes `ScoreResult` objects and produces the operational
view the investigator dashboard and on-call alerts need:

* scoring latency p50/p95/p99 against the transaction budget,
* action mix and score distribution,
* detection (elevated-risk) rate compared with a baseline window to catch
  model drift in either direction,
* degraded/timeout rate,
* capture rate and false positive rate once investigator feedback arrives.

Alerts are evaluated on a rolling window and emitted to an optional sink so a
real deployment can forward them to its paging/webhook system. Labels are
evaluation data: `observe(..., label=...)` is for the demo and backtests, while
production labels arrive through `record_feedback` after review.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

ELEVATED_RISK = 60.0
_SEVERITIES = ("INFO", "WARN", "CRITICAL")


@dataclass(frozen=True)
class MonitorConfig:
    window: int = 500
    baseline_size: int = 500
    min_volume: int = 50
    drift_threshold: float = 0.15
    score_shift_threshold: float = 12.0
    latency_target_ms: float = 50.0
    max_degraded_rate: float = 0.05
    max_fpr: float = 0.02
    min_capture: float = 0.60
    min_group_labeled: int = 10
    max_fpr_gap: float = 0.05
    max_capture_gap: float = 0.10
    evaluate_every: int = 25
    alert_history: int = 200

    def __post_init__(self) -> None:
        if self.window < 1 or self.baseline_size < 1:
            raise ValueError("window and baseline_size must be positive")
        if self.min_volume < 1:
            raise ValueError("min_volume must be positive")
        if self.evaluate_every < 1:
            raise ValueError("evaluate_every must be positive")
        if self.alert_history < 1:
            raise ValueError("alert_history must be positive")
        if self.drift_threshold < 0 or self.score_shift_threshold < 0:
            raise ValueError("drift thresholds must be non-negative")
        if not 0.0 <= self.max_fpr <= 1.0:
            raise ValueError("max_fpr must be in [0, 1]")
        if not 0.0 <= self.max_degraded_rate <= 1.0:
            raise ValueError("max_degraded_rate must be in [0, 1]")
        if not 0.0 <= self.min_capture <= 1.0:
            raise ValueError("min_capture must be in [0, 1]")
        if self.min_group_labeled < 1:
            raise ValueError("min_group_labeled must be positive")
        if not 0.0 <= self.max_fpr_gap <= 1.0:
            raise ValueError("max_fpr_gap must be in [0, 1]")
        if not 0.0 <= self.max_capture_gap <= 1.0:
            raise ValueError("max_capture_gap must be in [0, 1]")


@dataclass(frozen=True)
class Alert:
    severity: str
    code: str
    message: str
    value: float | None
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "value": self.value,
            "observed_at": self.observed_at,
        }


@dataclass
class _Observation:
    txn_id: str
    risk_score: float
    action: str
    outcome: str
    latency_ms: float
    degraded: bool
    label: bool | None = None
    group: str | None = None


class MetricsMonitor:
    """Rolling operational metrics and drift alerts for the scoring path."""

    def __init__(
        self,
        config: MonitorConfig | None = None,
        *,
        alert_sink: Callable[[Alert], Any] | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.alert_sink = alert_sink
        self._lock = threading.RLock()
        self._window: deque[_Observation] = deque(maxlen=self.config.window)
        self._baseline: list[_Observation] = []
        self._labels: dict[str, bool] = {}
        self._observed = 0
        self._actions_all: Counter[str] = Counter()
        self._outcomes_all: Counter[str] = Counter()
        self._degraded_all = 0
        self._alerts: list[Alert] = []
        self._history: deque[Alert] = deque(maxlen=self.config.alert_history)
        self._last_alert_codes: set[str] = set()

    # -- ingestion ----------------------------------------------------------

    def observe(
        self,
        result: Any,
        *,
        label: bool | None = None,
        group: str | None = None,
    ) -> None:
        """Record one score result (evaluation labels/groups are optional)."""
        txn_id = str(getattr(result, "txn_id", ""))
        observation = _Observation(
            txn_id=txn_id,
            risk_score=float(getattr(result, "risk_score", 0.0)),
            action=str(getattr(result, "action", "")),
            outcome=str(getattr(result, "outcome", "")),
            latency_ms=float(getattr(result, "latency_ms", 0.0)),
            degraded=bool(getattr(result, "degraded", False)),
            label=label,
            group=str(group) if group is not None else None,
        )
        with self._lock:
            self._window.append(observation)
            if len(self._baseline) < self.config.baseline_size:
                self._baseline.append(observation)
            self._observed += 1
            self._actions_all[observation.action] += 1
            self._outcomes_all[observation.outcome] += 1
            if observation.degraded:
                self._degraded_all += 1
            if txn_id in self._labels:
                observation.label = self._labels[txn_id]
            if self._observed % self.config.evaluate_every == 0:
                self._refresh_alerts()

    def record_feedback(self, txn_id: str, label: int | bool | None) -> None:
        """Attach (or clear) a review label on a transaction in the window."""
        key = str(txn_id)
        with self._lock:
            if label is None:
                self._labels.pop(key, None)
            else:
                self._labels[key] = bool(label)
            for observation in self._window:
                if observation.txn_id == key:
                    observation.label = None if label is None else bool(label)
            self._refresh_alerts()

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_alerts()
            observations = list(self._window)
            alerts = [alert.to_dict() for alert in self._alerts]
            baseline = self._baseline_metrics()
            observed = self._observed
            actions_all = dict(sorted(self._actions_all.items()))
            outcomes_all = dict(sorted(self._outcomes_all.items()))
            degraded_all = self._degraded_all

        n = len(observations)
        scores = [obs.risk_score for obs in observations]
        latencies = [obs.latency_ms for obs in observations]
        elevated = sum(1 for score in scores if score >= ELEVATED_RISK)
        labeled_fraud = [obs for obs in observations if obs.label is True]
        labeled_legit = [obs for obs in observations if obs.label is False]
        fraud_caught = sum(
            1 for obs in labeled_fraud if obs.risk_score >= ELEVATED_RISK
        )
        false_positives = sum(
            1 for obs in labeled_legit if obs.risk_score >= ELEVATED_RISK
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "observed_total": observed,
            "window_size": n,
            "window_limit": self.config.window,
            "action_mix": dict(sorted(Counter(obs.action for obs in observations).items())),
            "outcome_mix": dict(sorted(Counter(obs.outcome for obs in observations).items())),
            "action_mix_all_time": actions_all,
            "outcome_mix_all_time": outcomes_all,
            "score_histogram": _histogram(scores),
            "score_mean": round(sum(scores) / n, 2) if n else 0.0,
            "elevated_rate": round(elevated / n, 4) if n else 0.0,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "target": self.config.latency_target_ms,
            },
            "degraded_rate": round(
                sum(1 for obs in observations if obs.degraded) / n, 4
            ) if n else 0.0,
            "degraded_total": degraded_all,
            "labeling": {
                "labeled_fraud": len(labeled_fraud),
                "labeled_legit": len(labeled_legit),
                "capture_rate": round(fraud_caught / len(labeled_fraud), 4)
                if labeled_fraud else None,
                "false_positive_rate": round(false_positives / len(labeled_legit), 4)
                if labeled_legit else None,
            },
            "baseline": baseline,
            "fairness": _group_metrics(
                observations, self.config.min_group_labeled
            ),
            "alerts": alerts,
            "alert_count": len(alerts),
        }

    def alerts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [alert.to_dict() for alert in self._alerts]

    def alert_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [alert.to_dict() for alert in self._history]

    # -- alert evaluation ---------------------------------------------------

    def _refresh_alerts(self) -> None:
        observations = list(self._window)
        n = len(observations)
        alerts: list[Alert] = []
        if n >= self.config.min_volume:
            latencies = [obs.latency_ms for obs in observations]
            p99 = _percentile(latencies, 0.99)
            if p99 > self.config.latency_target_ms:
                alerts.append(
                    _alert(
                        "CRITICAL", "latency_p99",
                        f"scoring p99 {p99:.1f}ms exceeds "
                        f"{self.config.latency_target_ms:g}ms",
                        p99,
                    )
                )
            degraded_rate = sum(1 for obs in observations if obs.degraded) / n
            if degraded_rate > self.config.max_degraded_rate:
                alerts.append(
                    _alert(
                        "WARN", "degraded_rate",
                        f"degraded scoring rate {degraded_rate:.1%} exceeds "
                        f"{self.config.max_degraded_rate:.1%}",
                        degraded_rate,
                    )
                )

            elevated_rate = (
                sum(1 for obs in observations if obs.risk_score >= ELEVATED_RISK) / n
            )
            baseline = self._baseline_metrics()
            if (
                baseline["volume"] >= self.config.min_volume
                and baseline["elevated_rate"] is not None
            ):
                shift = elevated_rate - baseline["elevated_rate"]
                if abs(shift) >= self.config.drift_threshold:
                    direction = "spike" if shift > 0 else "drop"
                    alerts.append(
                        _alert(
                            "WARN", "detection_rate_drift",
                            f"elevated-risk rate {direction}: {elevated_rate:.1%} "
                            f"vs baseline {baseline['elevated_rate']:.1%}",
                            round(elevated_rate, 4),
                        )
                    )
                score_mean = (
                    sum(obs.risk_score for obs in observations) / n
                )
                if baseline["score_mean"] is not None:
                    score_shift = score_mean - baseline["score_mean"]
                    if abs(score_shift) >= self.config.score_shift_threshold:
                        alerts.append(
                            _alert(
                                "WARN", "score_distribution_drift",
                                f"mean risk score shifted {score_shift:+.1f} points "
                                f"from baseline",
                                round(score_mean, 2),
                            )
                        )

            labeled_fraud = [obs for obs in observations if obs.label is True]
            labeled_legit = [obs for obs in observations if obs.label is False]
            if labeled_fraud:
                capture = sum(
                    1 for obs in labeled_fraud if obs.risk_score >= ELEVATED_RISK
                ) / len(labeled_fraud)
                if capture < self.config.min_capture:
                    alerts.append(
                        _alert(
                            "WARN", "capture_rate",
                            f"fraud capture rate {capture:.1%} below "
                            f"{self.config.min_capture:.1%}",
                            round(capture, 4),
                        )
                    )
            if labeled_legit:
                fpr = sum(
                    1 for obs in labeled_legit if obs.risk_score >= ELEVATED_RISK
                ) / len(labeled_legit)
                if fpr > self.config.max_fpr:
                    alerts.append(
                        _alert(
                            "WARN", "false_positive_rate",
                            f"false positive rate {fpr:.1%} exceeds "
                            f"{self.config.max_fpr:.1%}",
                            round(fpr, 4),
                        )
                    )

            fairness = _group_metrics(observations, self.config.min_group_labeled)
            fpr_gap = fairness["false_positive_rate_gap"]
            if fpr_gap is not None and fpr_gap > self.config.max_fpr_gap:
                alerts.append(
                    _alert(
                        "WARN", "fairness_fpr_gap",
                        f"false-positive rate gap {fpr_gap:.1%} across "
                        f"{fairness['fpr_groups']} groups exceeds "
                        f"{self.config.max_fpr_gap:.1%}",
                        round(fpr_gap, 4),
                    )
                )
            capture_gap = fairness["capture_rate_gap"]
            if capture_gap is not None and capture_gap > self.config.max_capture_gap:
                alerts.append(
                    _alert(
                        "WARN", "fairness_capture_gap",
                        f"capture-rate gap {capture_gap:.1%} across "
                        f"{fairness['capture_groups']} groups exceeds "
                        f"{self.config.max_capture_gap:.1%}",
                        round(capture_gap, 4),
                    )
                )

        self._alerts = alerts
        current_codes = {alert.code for alert in alerts}
        for alert in alerts:
            if alert.code not in self._last_alert_codes:
                self._history.append(alert)
                if self.alert_sink is not None:
                    self.alert_sink(alert)
        self._last_alert_codes = current_codes

    def _baseline_metrics(self) -> dict[str, Any]:
        baseline = self._baseline
        n = len(baseline)
        if not n:
            return {"volume": 0, "elevated_rate": None, "score_mean": None}
        elevated = sum(1 for obs in baseline if obs.risk_score >= ELEVATED_RISK)
        return {
            "volume": n,
            "elevated_rate": round(elevated / n, 4),
            "score_mean": round(sum(obs.risk_score for obs in baseline) / n, 2),
        }


def _group_metrics(
    observations: list[_Observation],
    min_labeled: int,
) -> dict[str, Any]:
    """Per-group elevated, capture and false-positive rates (plan 5.2)."""
    grouped: dict[str, list[_Observation]] = {}
    for obs in observations:
        if obs.group is None:
            continue
        grouped.setdefault(obs.group, []).append(obs)

    groups: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(grouped.items()):
        labeled_fraud = [row for row in rows if row.label is True]
        labeled_legit = [row for row in rows if row.label is False]
        elevated = sum(1 for row in rows if row.risk_score >= ELEVATED_RISK)
        capture = (
            sum(1 for row in labeled_fraud if row.risk_score >= ELEVATED_RISK)
            / len(labeled_fraud)
            if labeled_fraud else None
        )
        fpr = (
            sum(1 for row in labeled_legit if row.risk_score >= ELEVATED_RISK)
            / len(labeled_legit)
            if labeled_legit else None
        )
        groups[name] = {
            "volume": len(rows),
            "elevated_rate": round(elevated / len(rows), 4) if rows else 0.0,
            "labeled_fraud": len(labeled_fraud),
            "labeled_legit": len(labeled_legit),
            "capture_rate": round(capture, 4) if capture is not None else None,
            "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        }

    fpr_comparable = [
        metrics["false_positive_rate"] for metrics in groups.values()
        if metrics["labeled_legit"] >= min_labeled
        and metrics["false_positive_rate"] is not None
    ]
    capture_comparable = [
        metrics["capture_rate"] for metrics in groups.values()
        if metrics["labeled_fraud"] >= min_labeled
        and metrics["capture_rate"] is not None
    ]
    return {
        "groups": groups,
        "min_labeled": min_labeled,
        "fpr_groups": len(fpr_comparable),
        "capture_groups": len(capture_comparable),
        "false_positive_rate_gap": round(max(fpr_comparable) - min(fpr_comparable), 4)
        if len(fpr_comparable) >= 2 else None,
        "capture_rate_gap": round(max(capture_comparable) - min(capture_comparable), 4)
        if len(capture_comparable) >= 2 else None,
    }


def _alert(severity: str, code: str, message: str, value: float | None) -> Alert:
    if severity not in _SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return Alert(
        severity=severity,
        code=code,
        message=message,
        value=round(value, 4) if value is not None else None,
        observed_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * quantile))], 3)


def _histogram(scores: list[float]) -> dict[str, int]:
    bins = {f"{low}-{low + 10}": 0 for low in range(0, 100, 10)}
    for score in scores:
        bucket = min(9, max(0, int(score // 10)))
        bins[f"{bucket * 10}-{bucket * 10 + 10}"] += 1
    return bins
