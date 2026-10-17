"""Phase 5 regression tests.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest

from fraud_detection.security import (
    AuditIntegrityError,
    SecureAuditLog,
    ensure_key_file,
    load_key,
)


def entry(i: int) -> dict:
    return {
        "txn_id": f"T{i}",
        "risk_score": 80.0 + i,
        "action": "HOLD_FOR_REVIEW",
        "model_version": "phase2-1",
        "top_signals": ["new recipient"],
    }


class TestAuditIntegrity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fraud-audit-")
        self.key_path = os.path.join(self.dir, "audit.key")
        self.audit_path = os.path.join(self.dir, "audit.jsonl")
        self.key = ensure_key_file(self.key_path)

    def tearDown(self):
        for path in (self.key_path, self.audit_path):
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(self.dir):
            os.rmdir(self.dir)

    def test_key_file_is_owner_only_and_stable(self):
        self.assertEqual(load_key(self.key_path), self.key)
        mode = stat.S_IMODE(os.stat(self.key_path).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(ensure_key_file(self.key_path), self.key)

    def test_chain_verifies_and_survives_reopen(self):
        first = SecureAuditLog(self.audit_path, self.key)
        for i in range(3):
            first.write(entry(i))
        self.assertEqual(first.count, 3)
        self.assertTrue(first.verify().ok)
        head = first.head_hash
        first.close()

        second = SecureAuditLog(self.audit_path, self.key)
        self.assertEqual(second.seq, 3)
        self.assertEqual(second.head_hash, head)
        second.write(entry(3))
        self.assertEqual(second.verify().entries, 4)
        self.assertEqual(len(second.records), 4)
        second.close()

        self.assertEqual(
            [record["txn_id"] for record in SecureAuditLog.read_entries(self.audit_path)],
            ["T0", "T1", "T2", "T3"],
        )

    def test_tampering_breaks_verification(self):
        log = SecureAuditLog(self.audit_path, self.key)
        for i in range(3):
            log.write(entry(i))
        log.close()

        with open(self.audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        record = json.loads(lines[1])
        record["entry"]["risk_score"] = 1.0
        lines[1] = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with open(self.audit_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertFalse(verification.ok)
        self.assertEqual(verification.broken_at, 2)
        self.assertIn("hash", verification.reason or "")
        with self.assertRaises(AuditIntegrityError):
            SecureAuditLog(self.audit_path, self.key)

    def test_wrong_key_is_rejected(self):
        log = SecureAuditLog(self.audit_path, self.key)
        log.write(entry(0))
        log.close()
        verification = SecureAuditLog.verify_file(self.audit_path, b"x" * 32)
        self.assertFalse(verification.ok)
        self.assertIn("MAC", verification.reason or "")

    def test_reordering_is_rejected(self):
        log = SecureAuditLog(self.audit_path, self.key)
        for i in range(2):
            log.write(entry(i))
        log.close()
        with open(self.audit_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(self.audit_path, "w", encoding="utf-8") as fh:
            fh.writelines([lines[1], lines[0]])
        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertFalse(verification.ok)
        self.assertEqual(verification.broken_at, 1)

    def test_empty_audit_is_valid(self):
        open(self.audit_path, "w").close()
        verification = SecureAuditLog.verify_file(self.audit_path, self.key)
        self.assertTrue(verification.ok)
        self.assertEqual(verification.entries, 0)


if __name__ == "__main__":
    unittest.main()
