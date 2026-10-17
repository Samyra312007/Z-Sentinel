"""Investigator dashboard server (plan 4.2).

A dependency-free HTTP API plus a small browser UI for fraud analysts:

* `GET  /api/cases`                 open/review queue, risk-sorted
* `GET  /api/cases/{id}`            transaction, feature and explanation detail
* `GET  /api/cases/{id}/actions`    investigator action audit
* `POST /api/cases/{id}/actions`    confirm fraud / clear / request info / reopen
* `GET  /api/metrics`               monitoring snapshot, review stats and alerts
* `GET  /api/config`                active segment thresholds
* `GET  /api/feedback`              labeled rows for retraining (admin)
* `GET  /api/whoami`                authenticated principal and role
* `GET  /api/health`

An `AccessPolicy` can require bearer tokens and role scopes (plan 5.1); TLS
is supplied through an `ssl.SSLContext`. Without a policy the server keeps the
Phase 4 localhost-only behaviour.
"""

from __future__ import annotations

import json
import mimetypes
import ssl
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from fraud_detection.access import AccessPolicy, Principal
from fraud_detection.decision import DecisionConfig
from fraud_detection.monitoring import MetricsMonitor
from fraud_detection.reviews import ReviewStore

_STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard"
_MAX_BODY = 1_000_000


@dataclass
class DashboardContext:
    reviews: ReviewStore
    monitor: MetricsMonitor
    config: DecisionConfig
    model_version: str = "rules-only"
    static_dir: Path = _STATIC_DIR
    access: AccessPolicy | None = None

    def metrics(self) -> dict[str, Any]:
        snapshot = self.monitor.snapshot()
        snapshot["reviews"] = self.reviews.stats()
        snapshot["config"] = self.config.to_dict()
        snapshot["model_version"] = self.model_version
        return snapshot


class DashboardServer:
    """Threaded HTTP server for the investigator UI and API."""

    def __init__(
        self,
        reviews: ReviewStore,
        monitor: MetricsMonitor,
        *,
        config: DecisionConfig | None = None,
        model_version: str = "rules-only",
        host: str = "127.0.0.1",
        port: int = 8080,
        static_dir: Path | None = None,
        access: AccessPolicy | None = None,
        tls_context: ssl.SSLContext | None = None,
    ) -> None:
        self.context = DashboardContext(
            reviews=reviews,
            monitor=monitor,
            config=config or DecisionConfig(),
            model_version=model_version,
            static_dir=static_dir or _STATIC_DIR,
            access=access,
        )
        self.host = host
        self.port = port
        self.tls_context = tls_context
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        scheme = "https" if self.tls_context is not None else "http"
        if self._httpd is None:
            return f"{scheme}://{self.host}:{self.port}"
        host, port = self._httpd.server_address[:2]
        return f"{scheme}://{host}:{port}"

    def start(self) -> str:
        handler = _make_handler(self.context)
        self._httpd = _ThreadingServer((self.host, self.port), handler)
        if self.tls_context is not None:
            self._httpd.socket = self.tls_context.wrap_socket(
                self._httpd.socket, server_side=True
            )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="fraud-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "DashboardServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class _ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(context: DashboardContext):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FraudDashboard/1.0"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.principal: Principal | None = None
            super().__init__(*args, **kwargs)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _authorize(self, scope: str) -> bool:
            if context.access is None or not context.access.enabled:
                return True
            principal = context.access.authenticate(self.headers.get("Authorization"))
            if principal is None:
                self._json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "bearer token required"},
                    extra={"WWW-Authenticate": 'Bearer realm="fraud-dashboard"'},
                )
                return False
            if not context.access.authorize(principal, scope):
                self._json(
                    HTTPStatus.FORBIDDEN,
                    {"error": f"role {principal.role!r} lacks {scope!r} scope"},
                )
                return False
            self.principal = principal
            return True

        # -- routing --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/index.html"):
                    self._static("index.html", "text/html; charset=utf-8")
                elif path.startswith("/static/"):
                    self._static(path[len("/static/"):])
                elif path == "/api/whoami":
                    if self._authorize("read"):
                        principal = self.principal
                        self._json(HTTPStatus.OK, principal.to_dict() if principal else {})
                elif path == "/api/health":
                    if self._authorize("read"):
                        self._json(
                            HTTPStatus.OK,
                            {
                                "status": "ok",
                                "reviews": context.reviews.stats(),
                                "monitoring": context.monitor.snapshot()["observed_total"],
                                "model_version": context.model_version,
                            },
                        )
                elif path == "/api/cases":
                    if self._authorize("read"):
                        self._list_cases()
                elif path == "/api/metrics":
                    if self._authorize("read"):
                        self._json(HTTPStatus.OK, context.metrics())
                elif path == "/api/config":
                    if self._authorize("read"):
                        self._json(HTTPStatus.OK, context.config.to_dict())
                elif path == "/api/feedback":
                    if self._authorize("admin"):
                        rows = context.reviews.feedback()
                        self._json(HTTPStatus.OK, {"feedback": rows, "count": len(rows)})
                elif path.startswith("/api/cases/"):
                    if self._authorize("read"):
                        self._case_route(path)
                elif path.startswith("/api/"):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "case not found"})
            except Exception as exc:  # pragma: no cover - defensive boundary
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if not path.startswith("/api/cases/") or not path.endswith("/actions"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            case_id = unquote(path[len("/api/cases/"):-len("/actions")].strip("/"))
            if not self._authorize("review"):
                return
            try:
                payload = self._read_json()
                action = str(payload.get("action", ""))
                investigator = str(payload.get("investigator", ""))
                note = payload.get("note")
                case = context.reviews.act(case_id, action, investigator, note)
                context.monitor.record_feedback(
                    case["txn_id"], case.get("feedback_label")
                )
                self._json(
                    HTTPStatus.OK,
                    {"case": case, "actions": context.reviews.actions(case_id)},
                )
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "case not found"})
            except Exception as exc:  # pragma: no cover - defensive boundary
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        # -- response helpers ----------------------------------------------

        def _json(
            self,
            status: HTTPStatus,
            payload: dict[str, Any],
            extra: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, separators=(",", ":"), default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _static(self, relative: str, content_type: str | None = None) -> None:
            relative = relative.lstrip("/")
            target = (context.static_dir / relative).resolve()
            if context.static_dir.resolve() not in target.parents and target != context.static_dir.resolve():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            body = target.read_bytes()
            ctype = content_type or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; img-src 'self' data:",
            )
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = self.headers.get("Content-Length")
            if length is None:
                raise ValueError("Content-Length is required")
            try:
                size = int(length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if size < 0 or size > _MAX_BODY:
                raise ValueError("request body is too large")
            raw = self.rfile.read(size)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _list_cases(self) -> None:
            query = parse_qs(urlsplit(self.path).query)
            status_raw = query.get("status", ["OPEN"])[0]
            status = None if status_raw in ("", "all", "ALL") else status_raw
            sort = query.get("sort", ["risk"])[0]
            try:
                limit = int(query.get("limit", ["50"])[0])
                offset = int(query.get("offset", ["0"])[0])
            except ValueError as exc:
                raise ValueError("limit and offset must be integers") from exc
            rows = context.reviews.list_cases(
                status=status, limit=limit, offset=offset, sort=sort
            )
            self._json(
                HTTPStatus.OK,
                {"cases": rows, "count": len(rows), "status": status_raw, "sort": sort},
            )

        def _case_route(self, path: str) -> None:
            remainder = path[len("/api/cases/"):].strip("/")
            if remainder.endswith("/actions"):
                case_id = unquote(remainder[:-len("/actions")].strip("/"))
                if not case_id:
                    raise ValueError("case id is required")
                actions = context.reviews.actions(case_id)
                if context.reviews.get(case_id) is None:
                    raise KeyError(case_id)
                self._json(HTTPStatus.OK, {"actions": actions, "count": len(actions)})
                return
            case_id = unquote(remainder)
            if not case_id:
                raise ValueError("case id is required")
            case = context.reviews.get(case_id)
            if case is None:
                raise KeyError(case_id)
            case["actions"] = context.reviews.actions(case_id)
            self._json(HTTPStatus.OK, case)

    return Handler
