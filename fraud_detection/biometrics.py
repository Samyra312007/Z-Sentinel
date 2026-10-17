"""Behavioral biometrics feature extraction (plan 3.2 / Phase 2 Model 2).

An online anomaly scorer: each session's keystroke/mouse aggregates are
compared against the user's rolling baseline (enrollment history). Because
the baseline is built only from *earlier* sessions, there is no label
leakage — a takeover's first anomalous session can only deviate from what
came before it.

Output is a 0-100 anomaly score, i.e. the unsupervised "deviation from the
user's normal behavior" signal the plan calls for. The baseline here is a
plain running mean/std; the plan allows swapping this for an Isolation
Forest or autoencoder without changing callers.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import fmean, pstdev

from fraud_detection.simulator import BehavioralSample

# feature -> (weight in final score, higher_is_suspicious)
_CHANNELS = (
    ("key_hold_mean_ms", 0.30, True),
    ("key_hold_std_ms", 0.15, True),  # dropped variability = robotic input
    ("inter_key_ms", 0.25, True),  # faster typing than usual = scripted
    ("mouse_velocity_px_s", 0.15, True),
    ("mouse_entropy", 0.15, False),  # low entropy = inhuman curves
)

_BASELINE_WINDOW = 60  # most recent sessions used as baseline
_MIN_BASELINE = 4  # sessions required before we score a user
_Z_CAP = 6.0


class BehavioralAnomalyDetector:
    def __init__(self) -> None:
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=_BASELINE_WINDOW)
        )

    def score(self, sample: BehavioralSample) -> tuple[float, dict[str, float]]:
        """Return (anomaly_score 0-100, per-channel z-scores).

        Until the baseline has enough sessions the score is 0 with empty
        channel breakdown ("insufficient history" — stress-test case 6.2).
        """
        values = {
            "key_hold_mean_ms": sample.key_hold_mean_ms,
            "key_hold_std_ms": sample.key_hold_std_ms,
            "inter_key_ms": sample.inter_key_ms,
            "mouse_velocity_px_s": sample.mouse_velocity_px_s,
            "mouse_entropy": sample.mouse_entropy,
        }
        hist = self._history[sample.user_id]
        if len(hist) < _MIN_BASELINE:
            hist.append(values)
            return 0.0, {}

        z_channels: dict[str, float] = {}
        total = 0.0
        for name, weight, higher_suspicious in _CHANNELS:
            series = [h[name] for h in hist]
            mu = fmean(series)
            sd = pstdev(series) or _floor_for(name)
            z = (values[name] - mu) / sd
            if not higher_suspicious:
                z = -z
            z = max(-_Z_CAP, min(_Z_CAP, z))
            z_channels[name] = round(z, 3)
            total += weight * min(_Z_CAP, abs(z))
        hist.append(values)

        score = 100.0 * (total / sum(w for _, w, _ in _CHANNELS)) / _Z_CAP
        return round(min(100.0, score), 2), z_channels


def _floor_for(name: str) -> float:
    # never divide by zero: std of a flat feature is treated as a small unit
    return 1.0 if name == "mouse_entropy" else 1e-3


def combined_deviation(z_channels: dict[str, float]) -> float:
    """RMS of channel z-scores — small helper for dashboards/debug output."""
    if not z_channels:
        return 0.0
    vals = list(z_channels.values())
    return round(math.sqrt(fmean(v * v for v in vals)), 3)