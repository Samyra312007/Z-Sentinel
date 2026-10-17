"""Versioned model registry and approval workflow (plan 5.2).

The registry is the governance boundary around the JSON model artifacts. It
records what was registered, who approved it, what is serving, and every
transition in between. Promotion is explicit:

    CANDIDATE --approve--> APPROVED --activate--> ACTIVE
         |                     |                     |
         +------reject-------->+                     +--rollback--> RETIRED

Only an `APPROVED` artifact can be activated, and activation re-hashes the
file so a swapped artifact cannot silently inherit an approval. Rollback
re-activates the previously serving version, again after a hash check.

The registry is deliberately dependency-free and file-backed so the same
workflow works in the demo and can be mapped to a DB2 table on z/OS. When an
`audit` sink is supplied, every transition is appended to the tamper-evident
decision audit as a `model_governance` event.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

SCHEMA = "fraud.model-registry.v1"

CANDIDATE = "CANDIDATE"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
ACTIVE = "ACTIVE"
RETIRED = "RETIRED"

STATUSES = (CANDIDATE, APPROVED, REJECTED, ACTIVE, RETIRED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


class ModelRegistry:
    """File-backed registry with approval, activation and rollback."""

    def __init__(self, path: str, *, audit: Any | None = None) -> None:
        self.path = path
        self._audit = audit
        self._lock = threading.RLock()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            if not isinstance(state, dict) or state.get("schema") != SCHEMA:
                raise ValueError("unsupported model registry file")
            if not isinstance(state.get("models"), dict):
                raise ValueError("model registry 'models' must be an object")
            if not isinstance(state.get("events"), list):
                raise ValueError("model registry 'events' must be a list")
            self._state: dict[str, Any] = state
        else:
            self._state = {
                "schema": SCHEMA,
                "active": None,
                "models": {},
                "events": [],
            }
            self._save()

    # -- persistence --------------------------------------------------------

    def _save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, self.path)

    def _event(
        self,
        action: str,
        version: str | None,
        actor: str,
        note: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        event = {
            "event_type": "model_governance",
            "action": action,
            "version": version,
            "actor": actor,
            "note": note,
            "at": _now(),
            **extra,
        }
        self._state["events"].append(event)
        self._save()
        if self._audit is not None:
            self._audit.write(event)
        return deepcopy(event)

    # -- registration -------------------------------------------------------

    def register(
        self,
        model_path: str,
        *,
        version: str | None = None,
        actor: str,
        notes: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actor = (actor or "").strip()
        if not actor:
            raise ValueError("actor is required")
        if not os.path.isfile(model_path):
            raise ValueError(f"model artifact not found: {model_path}")
        digest = _digest(model_path)
        if version is None:
            try:
                with open(model_path, "r", encoding="utf-8") as fh:
                    version = str(json.load(fh).get("version") or "")
            except (OSError, ValueError, TypeError):
                version = ""
        version = (version or digest[:12]).strip()
        if not version:
            raise ValueError("version is required")

        with self._lock:
            if version in self._state["models"]:
                raise ValueError(f"model version {version!r} is already registered")
            record = {
                "version": version,
                "path": os.path.abspath(model_path),
                "sha256": digest,
                "size": os.path.getsize(model_path),
                "status": CANDIDATE,
                "registered_at": _now(),
                "registered_by": actor,
                "approved_at": None,
                "approved_by": None,
                "approvals": [],
                "notes": (notes or "").strip() or None,
                "metrics": deepcopy(metrics or {}),
            }
            self._state["models"][version] = record
            self._event(
                "register",
                version,
                actor,
                record["notes"],
                sha256=digest,
            )
            return deepcopy(record)

    # -- approval -----------------------------------------------------------

    def approve(
        self,
        version: str,
        *,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        actor = self._require_actor(actor)
        with self._lock:
            record = self._require(version)
            if record["status"] == APPROVED:
                return deepcopy(record)
            if record["status"] != CANDIDATE:
                raise ValueError(
                    f"cannot approve {version!r} in status {record['status']}"
                )
            if not self._verify_record(record):
                raise ValueError(
                    f"model artifact for {version!r} does not match its registered hash"
                )
            at = _now()
            record["status"] = APPROVED
            record["approved_at"] = at
            record["approved_by"] = actor
            record["approvals"].append({"actor": actor, "at": at, "note": note})
            self._event("approve", version, actor, note)
            return deepcopy(record)

    def reject(
        self,
        version: str,
        *,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        actor = self._require_actor(actor)
        with self._lock:
            record = self._require(version)
            if record["status"] == ACTIVE:
                raise ValueError("cannot reject the active model; roll it back first")
            if record["status"] == REJECTED:
                return deepcopy(record)
            record["status"] = REJECTED
            self._event("reject", version, actor, note)
            return deepcopy(record)

    # -- serving ------------------------------------------------------------

    def activate(
        self,
        version: str,
        *,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        return self._activate(version, actor=actor, note=note, action="activate")

    def rollback(
        self,
        *,
        actor: str,
        to_version: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Re-activate the previously serving version (or an explicit one)."""
        actor = self._require_actor(actor)
        with self._lock:
            current = self._state.get("active")
            if not current:
                raise ValueError("no active model to roll back")
            if to_version is None:
                to_version = self._previous_active(current)
            if to_version is None:
                raise ValueError("no previously active model is available")
            return self._activate(
                to_version,
                actor=actor,
                note=note or f"rollback from {current}",
                action="rollback",
                current=current,
            )

    def _activate(
        self,
        version: str,
        *,
        actor: str,
        note: str | None,
        action: str,
        current: str | None = None,
    ) -> dict[str, Any]:
        actor = self._require_actor(actor)
        with self._lock:
            record = self._require(version)
            current = current if current is not None else self._state.get("active")
            if version == current:
                return deepcopy(record)
            if record["status"] not in (APPROVED, RETIRED):
                raise ValueError(
                    f"cannot activate {version!r} in status {record['status']}"
                )
            if not self._verify_record(record):
                raise ValueError(
                    f"model artifact for {version!r} does not match its registered hash"
                )
            if current and current in self._state["models"]:
                previous = self._state["models"][current]
                if previous["status"] == ACTIVE:
                    previous["status"] = RETIRED
            record["status"] = ACTIVE
            self._state["active"] = version
            self._event(
                action,
                version,
                actor,
                note,
                previous_active=current,
            )
            return deepcopy(record)

    # -- reads --------------------------------------------------------------

    @property
    def active_version(self) -> str | None:
        return self._state.get("active")

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            version = self._state.get("active")
            if not version:
                return None
            return deepcopy(self._state["models"].get(version))

    def get(self, version: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._state["models"].get(version)
            return deepcopy(record) if record else None

    def models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(record) for record in self._state["models"].values()]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state["events"])

    def verify(self, version: str) -> bool:
        """Return True when the artifact still matches its registered digest."""
        with self._lock:
            record = self._state["models"].get(version)
            if record is None:
                return False
            return self._verify_record(record)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = {status: 0 for status in STATUSES}
            for record in self._state["models"].values():
                counts[record["status"]] += 1
            return {
                "active": self._state.get("active"),
                "count": len(self._state["models"]),
                "by_status": counts,
                "models": self.models(),
                "events": self.events(),
            }

    # -- internals ----------------------------------------------------------

    def _previous_active(self, current: str) -> str | None:
        for event in reversed(self._state["events"]):
            if event.get("action") not in ("activate", "rollback"):
                continue
            candidate = event.get("previous_active")
            if candidate and candidate != current:
                record = self._state["models"].get(candidate)
                if record and record["status"] in (APPROVED, RETIRED):
                    return str(candidate)
        return None

    def _require(self, version: str) -> dict[str, Any]:
        record = self._state["models"].get(version)
        if record is None:
            raise KeyError(version)
        return record

    @staticmethod
    def _require_actor(actor: str) -> str:
        actor = (actor or "").strip()
        if not actor:
            raise ValueError("actor is required")
        return actor

    @staticmethod
    def _verify_record(record: dict[str, Any]) -> bool:
        path = record.get("path")
        if not isinstance(path, str) or not os.path.isfile(path):
            return False
        try:
            return _digest(path) == record.get("sha256")
        except OSError:
            return False
