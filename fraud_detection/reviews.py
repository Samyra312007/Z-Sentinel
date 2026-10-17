"""Investigator review queue (plan 4.2).

Held and step-up cases are persisted in SQLite so the dashboard and the
scoring service can share a durable queue in one process or across restarts.
Each case stores the complete evidence package (transaction context, feature
snapshot, behavioral channels, graph features, model probabilities and top
signals) plus the investigator decision state.

Investigator actions are appended to `case_actions`; the case row keeps only
the current state. Confirming fraud (`label=1`) or clearing a false positive
(`label=0`) feeds the retraining export.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

OPEN = "OPEN"
CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
CLEARED = "CLEARED"
NEED_INFO = "NEED_INFO"

CASE_STATUSES = (OPEN, CONFIRMED_FRAUD, CLEARED, NEED_INFO)

CONFIRM_FRAUD = "confirm_fraud"
CLEAR_FP = "clear_fp"
REQUEST_INFO = "request_info"
REOPEN = "reopen"

INVESTIGATOR_ACTIONS = (CONFIRM_FRAUD, CLEAR_FP, REQUEST_INFO, REOPEN)

_ACTION_STATE = {
    CONFIRM_FRAUD: (CONFIRMED_FRAUD, 1),
    CLEAR_FP: (CLEARED, 0),
    REQUEST_INFO: (NEED_INFO, None),
    REOPEN: (OPEN, None),
}

_SORTS = {
    "risk": "risk_score DESC, created_at ASC",
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ReviewStore:
    """SQLite-backed queue of cases awaiting or carrying investigator review."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                txn_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                risk_score REAL NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT,
                user_ref TEXT NOT NULL,
                amount REAL,
                currency TEXT,
                feedback_label INTEGER,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cases_status_risk
                ON cases(status, risk_score DESC);
            CREATE TABLE IF NOT EXISTS case_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                action TEXT NOT NULL,
                investigator TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            );
            CREATE INDEX IF NOT EXISTS idx_case_actions_case
                ON case_actions(case_id, action_id);
            """
        )
        self._conn.commit()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ReviewStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- queue --------------------------------------------------------------

    def add_case(self, case: dict[str, Any], *, status: str = OPEN) -> dict[str, Any]:
        """Persist a review case; re-enqueueing the same transaction is a no-op."""
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        txn_id = str(case.get("txn_id") or "").strip()
        if not txn_id:
            raise ValueError("case.txn_id is required")
        case_id = str(case.get("case_id") or txn_id)
        if not case_id:
            raise ValueError("case.case_id is required")
        if status not in CASE_STATUSES:
            raise ValueError(f"unknown case status {status!r}")
        if "risk_score" not in case:
            raise ValueError("case.risk_score is required")
        if not case.get("user_ref"):
            raise ValueError("case.user_ref is required")

        now = _now()
        record = {
            **case,
            "case_id": case_id,
            "txn_id": txn_id,
            "status": status,
            "feedback_label": case.get("feedback_label"),
            "created_at": case.get("created_at") or now,
            "updated_at": now,
        }
        payload = json.dumps(record, separators=(",", ":"), sort_keys=True, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO cases (
                    case_id, txn_id, created_at, updated_at, status,
                    risk_score, action, outcome, user_ref, amount, currency,
                    feedback_label, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    txn_id,
                    record["created_at"],
                    now,
                    status,
                    float(record["risk_score"]),
                    str(record.get("action", "")),
                    record.get("outcome"),
                    str(record["user_ref"]),
                    float(record.get("amount") or 0.0),
                    record.get("currency"),
                    record["feedback_label"],
                    payload,
                ),
            )
            self._conn.commit()
        existing = self.get(case_id)
        if existing is None:
            # A duplicate txn_id may have been persisted under a different
            # generated case id; return the authoritative existing row.
            with self._lock:
                row = self._conn.execute(
                    "SELECT case_id FROM cases WHERE txn_id = ?", (txn_id,)
                ).fetchone()
            existing = self.get(row["case_id"]) if row is not None else None
        if existing is None:  # pragma: no cover - defensive
            raise RuntimeError(f"case {case_id} was not persisted")
        return existing

    def list_cases(
        self,
        *,
        status: str | None = OPEN,
        limit: int = 50,
        offset: int = 0,
        sort: str = "risk",
    ) -> list[dict[str, Any]]:
        if sort not in _SORTS:
            raise ValueError(f"sort must be one of {tuple(_SORTS)}")
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        sql = "SELECT * FROM cases"
        params: list[Any] = []
        if status is not None:
            if status not in CASE_STATUSES:
                raise ValueError(f"unknown case status {status!r}")
            sql += " WHERE status = ?"
            params.append(status)
        sql += f" ORDER BY {_SORTS[sort]} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._summary(row) for row in rows]

    def get(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        record = json.loads(row["payload_json"])
        record.update(
            {
                "case_id": row["case_id"],
                "txn_id": row["txn_id"],
                "status": row["status"],
                "feedback_label": row["feedback_label"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        return record

    def actions(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM case_actions WHERE case_id = ? ORDER BY action_id",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- investigator decisions --------------------------------------------

    def act(
        self,
        case_id: str,
        action: str,
        investigator: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if action not in INVESTIGATOR_ACTIONS:
            raise ValueError(f"action must be one of {INVESTIGATOR_ACTIONS}")
        investigator = (investigator or "").strip()
        if not investigator:
            raise ValueError("investigator is required")
        if note is not None:
            note = note.strip()[:2000] or None
        if self.get(case_id) is None:
            raise KeyError(case_id)

        new_status, label = _ACTION_STATE[action]
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT status, payload_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            previous_status = row["status"]
            payload = json.loads(row["payload_json"])
            payload.update(
                {
                    "status": new_status,
                    "feedback_label": label,
                    "updated_at": now,
                }
            )
            self._conn.execute(
                """
                UPDATE cases
                   SET status = ?, feedback_label = ?, updated_at = ?,
                       payload_json = ?
                 WHERE case_id = ?
                """,
                (
                    new_status,
                    label,
                    now,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
                    case_id,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO case_actions (
                    case_id, action, investigator, note, created_at,
                    previous_status, new_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (case_id, action, investigator, note, now, previous_status, new_status),
            )
            self._conn.commit()
        updated = self.get(case_id)
        if updated is None:  # pragma: no cover - defensive
            raise RuntimeError(f"case {case_id} disappeared")
        return updated

    # -- retention ----------------------------------------------------------

    def purge_closed_before(self, cutoff: str) -> dict[str, int]:
        """Delete closed cases (and their action history) older than cutoff.

        Open and `NEED_INFO` cases are retained because they still require an
        investigator decision. Behavioral raw input is never stored here; the
        payload contains only aggregate features and evidence.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT case_id FROM cases
                 WHERE status IN (?, ?) AND created_at < ?
                """,
                (CONFIRMED_FRAUD, CLEARED, cutoff),
            ).fetchall()
            case_ids = [row["case_id"] for row in rows]
            actions = 0
            for start in range(0, len(case_ids), 500):
                batch = case_ids[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                actions += self._conn.execute(
                    f"DELETE FROM case_actions WHERE case_id IN ({placeholders})",
                    batch,
                ).rowcount
                self._conn.execute(
                    f"DELETE FROM cases WHERE case_id IN ({placeholders})",
                    batch,
                )
            self._conn.commit()
        return {"cases": len(case_ids), "actions": actions}

    # -- reporting / retraining export -------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            status_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM cases GROUP BY status"
            ).fetchall()
            action_rows = self._conn.execute(
                "SELECT action, COUNT(*) AS n FROM cases GROUP BY action"
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()["n"]
        by_status = {status: 0 for status in CASE_STATUSES}
        by_status.update({row["status"]: row["n"] for row in status_rows})
        reviewed = by_status[CONFIRMED_FRAUD] + by_status[CLEARED]
        return {
            "total": total,
            "open": by_status[OPEN],
            "need_info": by_status[NEED_INFO],
            "confirmed_fraud": by_status[CONFIRMED_FRAUD],
            "cleared": by_status[CLEARED],
            "reviewed": reviewed,
            "by_status": by_status,
            "by_action": {row["action"]: row["n"] for row in action_rows},
            "confirmed_rate": round(
                by_status[CONFIRMED_FRAUD] / reviewed, 4
            ) if reviewed else 0.0,
        }

    def feedback(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """Labeled cases for retraining (confirmed fraud / cleared false positive)."""
        limit = max(1, min(int(limit), 10000))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT case_id, txn_id, feedback_label, payload_json, updated_at
                  FROM cases
                 WHERE feedback_label IS NOT NULL
                 ORDER BY updated_at DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            out.append(
                {
                    "case_id": row["case_id"],
                    "txn_id": row["txn_id"],
                    "label": int(row["feedback_label"]),
                    "user_ref": payload.get("user_ref"),
                    "features": payload.get("features", {}),
                    "risk_score": payload.get("risk_score"),
                    "model_version": payload.get("model_version"),
                    "reviewed_at": row["updated_at"],
                }
            )
        return out

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        return {
            "case_id": row["case_id"],
            "txn_id": row["txn_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "risk_score": row["risk_score"],
            "action": row["action"],
            "outcome": row["outcome"],
            "user_ref": row["user_ref"],
            "amount": row["amount"],
            "currency": row["currency"],
            "segment": payload.get("segment"),
            "model_version": payload.get("model_version"),
            "top_signals": payload.get("top_signals", []),
            "behavior_score": payload.get("behavior_score", 0.0),
            "behavior_status": payload.get("behavior_status"),
            "ring_flag": payload.get("ring_flag", False),
        }

