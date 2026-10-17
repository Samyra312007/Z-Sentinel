"""Behavioral-biometrics session ingestion (plan 3.2).

The browser/mobile SDK (`sdk/behavioral-biometrics.js`) sends session-level
aggregates only: keystroke timing, pointer velocity/entropy and touch shape.
Typed content, key codes and raw pointer paths never leave the device.

This module is the server-side boundary. It:

* validates the SDK schema and numeric ranges,
* rejects payloads that try to smuggle raw content,
* honours the user's opt-out (returns ``None`` so scoring falls back to
  transaction-only features),
* converts the payload into the `BehavioralSample` the anomaly detector uses.

User identity is pseudonymised before it ever reaches this module. The same
salted hash is used for the transaction's customer reference, so behavioral
baselines and transaction history line up without storing a reversible ID.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from fraud_detection.simulator import BehavioralSample

SCHEMA = "behavioral.session.v1"
SDK_VERSION = "1.0.0"
DEFAULT_SUBJECT_SALT = "fraud-detection-demo"

_TOP_LEVEL = {
    "schema", "sdk_version", "subject_ref", "session_id", "collected_at",
    "consent", "keystroke", "mouse", "touch",
}
_GROUPS = {
    "keystroke": {"key_hold_mean_ms", "key_hold_std_ms", "inter_key_ms", "samples"},
    "mouse": {
        "velocity_px_s", "entropy", "clicks", "click_x_mean_norm",
        "click_y_mean_norm", "samples",
    },
    "touch": {"pressure_mean", "speed_px_s", "swipe_angle_deg", "samples"},
}
# Anything that could carry typed text or raw trajectories is refused outright.
_CONTENT_KEYS = {
    "text", "content", "value", "keys", "keystrokes", "raw", "path",
    "paths", "points", "events", "key_code", "code", "char", "character",
}


def hash_subject_id(
    subject_id: str,
    salt: str | None = None,
) -> str:
    """Return the pseudonymous subject reference used in payloads and scoring.

    In production `FRAUD_SUBJECT_SALT` must be a secret held outside the
    application config; the default exists only so the offline demo is
    reproducible.
    """
    if not subject_id:
        raise ValueError("subject_id must be non-empty")
    effective_salt = salt if salt is not None else os.environ.get(
        "FRAUD_SUBJECT_SALT", DEFAULT_SUBJECT_SALT
    )
    digest = hashlib.sha256(f"{effective_salt}:{subject_id}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def payload_from_sample(
    sample: BehavioralSample,
    *,
    salt: str | None = None,
    session_id: str | None = None,
    consent: bool = True,
) -> dict:
    """Build the wire payload a browser SDK would send for this session."""
    subject_ref = (
        sample.user_id
        if sample.user_id.startswith("sha256:")
        else hash_subject_id(sample.user_id, salt)
    )
    return {
        "schema": SCHEMA,
        "sdk_version": SDK_VERSION,
        "subject_ref": subject_ref,
        "session_id": session_id or sample.session_id,
        "collected_at": sample.timestamp.astimezone(timezone.utc).isoformat(),
        "consent": {"behavioral_biometrics": bool(consent)},
        "keystroke": {
            "key_hold_mean_ms": float(sample.key_hold_mean_ms),
            "key_hold_std_ms": float(sample.key_hold_std_ms),
            "inter_key_ms": float(sample.inter_key_ms),
            "samples": 0,
        },
        "mouse": {
            "velocity_px_s": float(sample.mouse_velocity_px_s),
            "entropy": float(sample.mouse_entropy),
            "clicks": int(sample.click_count),
            "click_x_mean_norm": float(sample.click_x_mean_norm),
            "click_y_mean_norm": float(sample.click_y_mean_norm),
            "samples": 0,
        },
        "touch": {
            "pressure_mean": float(sample.touch_pressure_mean),
            "speed_px_s": float(sample.touch_speed_px_s),
            "swipe_angle_deg": float(sample.touch_swipe_angle_deg),
            "samples": 0,
        },
    }


def parse_payload(payload: dict) -> BehavioralSample | None:
    """Validate an SDK payload and return a sample, or None if opted out."""
    if not isinstance(payload, dict):
        raise ValueError("behavioral payload must be an object")
    unknown = set(payload) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown behavioral payload fields: {sorted(unknown)}")
    _reject_content(payload)

    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported behavioral schema {payload.get('schema')!r}, expected {SCHEMA!r}"
        )

    consent = payload.get("consent")
    if not isinstance(consent, dict) or "behavioral_biometrics" not in consent:
        raise ValueError("consent.behavioral_biometrics is required")
    if not isinstance(consent["behavioral_biometrics"], bool):
        raise ValueError("consent.behavioral_biometrics must be a boolean")
    if not consent["behavioral_biometrics"]:
        return None  # opt-out: transaction-only scoring

    subject_ref = payload.get("subject_ref")
    if not isinstance(subject_ref, str) or not subject_ref.startswith("sha256:"):
        raise ValueError("subject_ref must be a salted sha256 reference")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id is required")

    collected_at = _parse_time(payload.get("collected_at"))
    keystroke = _group(payload, "keystroke", required=True)
    mouse = _group(payload, "mouse", required=True)
    touch = _group(payload, "touch", required=False)

    return BehavioralSample(
        session_id=session_id,
        user_id=subject_ref,
        timestamp=collected_at,
        key_hold_mean_ms=_num(keystroke, "key_hold_mean_ms", 0.0, 5000.0),
        key_hold_std_ms=_num(keystroke, "key_hold_std_ms", 0.0, 5000.0),
        inter_key_ms=_num(keystroke, "inter_key_ms", 0.0, 60000.0),
        mouse_velocity_px_s=_num(mouse, "velocity_px_s", 0.0, 100000.0),
        mouse_entropy=_num(mouse, "entropy", 0.0, 10.0),
        click_count=int(_num(mouse, "clicks", 0.0, 1e9, default=0.0)),
        click_x_mean_norm=_num(mouse, "click_x_mean_norm", 0.0, 1.0, default=0.0),
        click_y_mean_norm=_num(mouse, "click_y_mean_norm", 0.0, 1.0, default=0.0),
        touch_pressure_mean=_num(touch, "pressure_mean", 0.0, 10.0, default=0.0),
        touch_speed_px_s=_num(touch, "speed_px_s", 0.0, 100000.0, default=0.0),
        touch_swipe_angle_deg=_num(touch, "swipe_angle_deg", -180.0, 180.0, default=0.0),
    )


# -- validation helpers -----------------------------------------------------

def _reject_content(value, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _CONTENT_KEYS:
                raise ValueError(f"raw behavioral content is not accepted ({path}.{key})")
            _reject_content(child, f"{path}.{key}")
    elif isinstance(value, list):
        raise ValueError(f"raw event arrays are not accepted ({path})")


def _parse_time(value) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("collected_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"collected_at is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _group(payload: dict, name: str, *, required: bool) -> dict:
    group = payload.get(name)
    if group is None:
        if required:
            raise ValueError(f"{name} metrics are required")
        return {}
    if not isinstance(group, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(group) - _GROUPS[name]
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return group


def _num(
    group: dict,
    name: str,
    low: float,
    high: float,
    *,
    default: float | None = None,
) -> float:
    if name not in group:
        if default is not None:
            return float(default)
        raise ValueError(f"missing required metric {name!r}")
    value = group[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not (low <= value <= high):
        raise ValueError(f"{name}={value} outside [{low}, {high}]")
    return value
