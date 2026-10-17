"""Security primitives for Phase 5 (plan 5.1).

The application does not invent encryption algorithms. It provides two
stdlib-only controls that fit the z/OS security boundary:

* **tamper-evident decision audit** — every JSONL record carries a sequence
  number, the previous record hash, a SHA-256 entry hash and an HMAC-SHA256
  authentication tag over the hash. Editing, deleting (except tail truncation)
  or reordering a record breaks verification.
* **protected key handling** — the HMAC key can come from a secret manager
  (`FRAUD_AUDIT_KEY`), or from a file for the offline demo; demo key files are
  created with owner-only permissions.

Data-at-rest encryption remains a platform responsibility (zFS/DB2 encryption
with CPACF-backed key material). This module deliberately does not roll its own
cipher; the audit key must live in RACF/z/OS keyrings in production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any

KEY_ENV = "FRAUD_AUDIT_KEY"
GENESIS_HASH = "0" * 64
MIN_KEY_BYTES = 16


class AuditIntegrityError(RuntimeError):
    """Raised when an existing audit chain fails verification."""


@dataclass(frozen=True)
class AuditVerification:
    ok: bool
    entries: int
    head_hash: str
    broken_at: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "entries": self.entries,
            "head_hash": self.head_hash,
            "broken_at": self.broken_at,
            "reason": self.reason,
        }


def _decode_key(raw: str) -> bytes:
    raw = raw.strip()
    if not raw:
        raise ValueError("audit key must not be empty")
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError("audit key must be hex-encoded") from exc
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"audit key must be at least {MIN_KEY_BYTES} bytes")
    return key


def load_key(path: str | None = None, *, env: str = KEY_ENV) -> bytes:
    """Load a hex audit key from a file or environment variable."""
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return _decode_key(fh.read())
    raw = os.environ.get(env)
    if not raw:
        raise ValueError(
            f"no audit key: set {env} to a hex key or pass a key file"
        )
    return _decode_key(raw)


def ensure_key_file(path: str) -> bytes:
    """Read or create an owner-only hex key file (demo/local deployments)."""
    if os.path.exists(path):
        return load_key(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    key_hex = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return load_key(path)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key_hex + "\n")
    return _decode_key(key_hex)


class SecureAuditLog:
    """Append-only, HMAC-chained decision audit compatible with AuditLog."""

    def __init__(
        self,
        path: str,
        key: bytes | None = None,
        *,
        verify_on_open: bool = True,
    ) -> None:
        self.path = path
        self._key = key or load_key()
        if len(self._key) < MIN_KEY_BYTES:
            raise ValueError(f"audit key must be at least {MIN_KEY_BYTES} bytes")
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self._seq = 0
        self._head = GENESIS_HASH
        self._fh = None

        if os.path.exists(path) and os.path.getsize(path) > 0:
            verification = self.verify_file(path, self._key)
            if verify_on_open and not verification.ok:
                raise AuditIntegrityError(
                    f"audit chain is invalid at entry {verification.broken_at}: "
                    f"{verification.reason}"
                )
            self.records = self.read_entries(path)
            if self.records:
                self._seq = len(self.records)
                self._head = verification.head_hash

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    # -- writing ------------------------------------------------------------

    def write(self, entry: Any) -> dict[str, Any]:
        entry_dict = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        with self._lock:
            seq = self._seq + 1
            payload = _entry_payload(seq, self._head, entry_dict)
            digest = hashlib.sha256(payload).hexdigest()
            mac = hmac.new(self._key, bytes.fromhex(digest), hashlib.sha256).hexdigest()
            record = {
                "seq": seq,
                "prev_hash": self._head,
                "hash": digest,
                "mac": mac,
                "entry": entry_dict,
            }
            assert self._fh is not None
            self._fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            self._fh.flush()
            self._seq = seq
            self._head = digest
            self.records.append(entry_dict)
        return record

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def head_hash(self) -> str:
        return self._head

    @property
    def seq(self) -> int:
        return self._seq

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SecureAuditLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- verification -------------------------------------------------------

    def verify(self) -> AuditVerification:
        return self.verify_file(self.path, self._key)

    @staticmethod
    def read_entries(path: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                entries.append(record["entry"])
        return entries

    @staticmethod
    def verify_file(path: str, key: bytes) -> AuditVerification:
        if not os.path.exists(path):
            return AuditVerification(False, 0, GENESIS_HASH, 0, "audit file not found")
        expected_prev = GENESIS_HASH
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    count += 1
                    record = json.loads(line)
                    if record.get("seq") != count:
                        return AuditVerification(
                            False, count - 1, expected_prev, count, "sequence gap"
                        )
                    if record.get("prev_hash") != expected_prev:
                        return AuditVerification(
                            False, count - 1, expected_prev, count, "previous hash mismatch"
                        )
                    entry = record.get("entry")
                    if not isinstance(entry, dict):
                        return AuditVerification(
                            False, count - 1, expected_prev, count, "missing entry"
                        )
                    payload = _entry_payload(count, expected_prev, entry)
                    digest = hashlib.sha256(payload).hexdigest()
                    if not hmac.compare_digest(record.get("hash", ""), digest):
                        return AuditVerification(
                            False, count - 1, expected_prev, count, "entry hash mismatch"
                        )
                    expected_mac = hmac.new(
                        key, bytes.fromhex(digest), hashlib.sha256
                    ).hexdigest()
                    if not hmac.compare_digest(record.get("mac", ""), expected_mac):
                        return AuditVerification(
                            False, count - 1, expected_prev, count, "MAC mismatch"
                        )
                    expected_prev = digest
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
            return AuditVerification(False, max(0, count - 1), expected_prev, count, str(exc))
        if count == 0:
            return AuditVerification(True, 0, GENESIS_HASH)
        return AuditVerification(True, count, expected_prev)


def _entry_payload(seq: int, prev_hash: str, entry: dict[str, Any]) -> bytes:
    return json.dumps(
        {"seq": seq, "prev_hash": prev_hash, "entry": entry},
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
