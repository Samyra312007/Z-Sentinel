"""Phase 5 regression tests.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import ssl
import stat
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fraud_detection.access import ADMIN, INVESTIGATOR, VIEWER, AccessPolicy
from fraud_detection.dashboard import DashboardServer
from fraud_detection.monitoring import MetricsMonitor, MonitorConfig
from fraud_detection.reviews import ReviewStore
from fraud_detection.security import (
    AuditIntegrityError,
    SecureAuditLog,
    build_server_ssl_context,
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


def case_record(case_id: str = "T1") -> dict:
    return {
        "case_id": case_id,
        "txn_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_score": 88.0,
        "action": "HOLD_FOR_REVIEW",
        "outcome": "HELD_FOR_REVIEW",
        "user_ref": "sha256:abc",
        "amount": 10.0,
        "currency": "USD",
        "segment": "standard",
        "model_version": "phase2-1",
        "top_signals": ["new recipient"],
        "behavior_status": "scored",
        "features": {"amount": 10.0},
    }


class TestAccessPolicy(unittest.TestCase):
    def test_config_roles_and_authentication(self):
        policy = AccessPolicy.from_config(
            {
                "tokens": [
                    {"token": "read-token", "name": "viewer", "role": VIEWER},
                    {"token": "review-token", "name": "analyst", "role": INVESTIGATOR},
                    {"token": "admin-token", "name": "admin", "role": ADMIN},
                ]
            }
        )
        viewer = policy.authenticate("Bearer read-token")
        self.assertEqual(viewer.name, "viewer")
        self.assertTrue(policy.authorize(viewer, "read"))
        self.assertFalse(policy.authorize(viewer, "review"))
        self.assertTrue(policy.authorize(policy.authenticate("Bearer review-token"), "review"))
        self.assertTrue(policy.authorize(policy.authenticate("Bearer admin-token"), "admin"))
        self.assertIsNone(policy.authenticate("Bearer nope"))
        self.assertIsNone(policy.authenticate(None))
        self.assertIsNone(policy.authenticate("read-token"))

    def test_invalid_roles_and_save_load(self):
        with self.assertRaises(ValueError):
            AccessPolicy().add_token("t", "n", "root")
        path = tempfile.mktemp(suffix=".json")
        try:
            AccessPolicy.save(path, (("t", "n", VIEWER),))
            policy = AccessPolicy.load(path)
            self.assertTrue(policy.enabled)
            self.assertEqual(policy.authenticate("Bearer t").role, VIEWER)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestDashboardAccess(unittest.TestCase):
    def setUp(self):
        self.store = ReviewStore()
        self.store.add_case(case_record())
        self.monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        self.policy = AccessPolicy.from_config(
            {
                "tokens": [
                    {"token": "read-token", "name": "viewer", "role": VIEWER},
                    {"token": "review-token", "name": "analyst", "role": INVESTIGATOR},
                    {"token": "admin-token", "name": "admin", "role": ADMIN},
                ]
            }
        )
        self.server = DashboardServer(
            self.store, self.monitor, access=self.policy, port=0
        )
        self.url = self.server.start()

    def tearDown(self):
        self.server.stop()
        self.store.close()

    def _request(self, path: str, token: str | None = None, method: str = "GET", payload: dict | None = None):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=data, headers=headers, method=method)
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())

    def test_requests_require_a_bearer_token(self):
        with self.assertRaises(HTTPError) as error:
            self._request("/api/health")
        self.assertEqual(error.exception.code, 401)
        self.assertIn("Bearer", error.exception.headers.get("WWW-Authenticate", ""))
        error.exception.close()

    def test_viewer_can_read_but_not_disposition(self):
        status, health = self._request("/api/health", "read-token")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, who = self._request("/api/whoami", "read-token")
        self.assertEqual(who, {"name": "viewer", "role": VIEWER})
        with self.assertRaises(HTTPError) as denied:
            self._request(
                "/api/cases/T1/actions", "read-token", "POST",
                {"action": "confirm_fraud", "investigator": "viewer"},
            )
        self.assertEqual(denied.exception.code, 403)
        denied.exception.close()
        with self.assertRaises(HTTPError) as admin_only:
            self._request("/api/feedback", "read-token")
        self.assertEqual(admin_only.exception.code, 403)
        admin_only.exception.close()

    def test_investigator_can_disposition_and_admin_can_export(self):
        status, payload = self._request(
            "/api/cases/T1/actions", "review-token", "POST",
            {"action": "confirm_fraud", "investigator": "analyst", "note": "mule"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["case"]["status"], "CONFIRMED_FRAUD")
        status, feedback = self._request("/api/feedback", "admin-token")
        self.assertEqual(status, 200)
        self.assertEqual(feedback["count"], 1)


@unittest.skipUnless(shutil.which("openssl"), "openssl not available")
class TestDashboardTLS(unittest.TestCase):
    def test_https_round_trip(self):
        directory = tempfile.mkdtemp(prefix="fraud-tls-")
        cert = os.path.join(directory, "cert.pem")
        key = os.path.join(directory, "key.pem")
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", key, "-out", cert, "-days", "1", "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        store = ReviewStore()
        store.add_case(case_record())
        monitor = MetricsMonitor(MonitorConfig(min_volume=1000, evaluate_every=1000))
        server = DashboardServer(
            store, monitor, port=0,
            tls_context=build_server_ssl_context(cert, key),
        )
        try:
            url = server.start()
            self.assertTrue(url.startswith("https://"))
            host, port = url[len("https://"):].split(":")
            context = ssl._create_unverified_context()
            connection = http.client.HTTPSConnection(host, int(port), context=context, timeout=5)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            body = json.loads(response.read().decode())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(body["status"], "ok")
        finally:
            server.stop()
            store.close()
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
