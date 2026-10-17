"""Role-based access control for the dashboard (plan 5.1 / 5.2).

Tokens are stored hashed in memory (SHA-256) and compared in constant time.
Roles map to scopes:

* `viewer`       — read queue, cases, metrics, config
* `investigator` — viewer plus case dispositions and investigator audit
* `admin`        — investigator plus retraining feedback and audit verification

When no policy is supplied the dashboard preserves its Phase 4 localhost
behaviour. A production deployment supplies a token file (or env var) and
serves TLS behind z/OS Connect / RACF; see `docs/PHASE5.md`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

VIEWER = "viewer"
INVESTIGATOR = "investigator"
ADMIN = "admin"

ROLE_SCOPES: dict[str, frozenset[str]] = {
    VIEWER: frozenset({"read"}),
    INVESTIGATOR: frozenset({"read", "review"}),
    ADMIN: frozenset({"read", "review", "admin"}),
}


@dataclass(frozen=True)
class Principal:
    name: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "role": self.role}


class AccessPolicy:
    """Bearer-token authenticator and scope authorizer."""

    def __init__(self, tokens: dict[str, Principal] | None = None) -> None:
        self._tokens: dict[str, Principal] = dict(tokens or {})

    @property
    def enabled(self) -> bool:
        return bool(self._tokens)

    @property
    def tokens(self) -> int:
        return len(self._tokens)

    def add_token(self, token: str, name: str, role: str) -> Principal:
        token = (token or "").strip()
        name = (name or "").strip()
        if not token:
            raise ValueError("token is required")
        if not name:
            raise ValueError("token name is required")
        if role not in ROLE_SCOPES:
            raise ValueError(f"unknown role {role!r}; expected {tuple(ROLE_SCOPES)}")
        principal = Principal(name=name, role=role)
        self._tokens[_hash_token(token)] = principal
        return principal

    def authenticate(self, authorization: str | None) -> Principal | None:
        if not self._tokens:
            return None
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        candidate = _hash_token(token.strip())
        for stored, principal in self._tokens.items():
            if hmac.compare_digest(candidate, stored):
                return principal
        return None

    def authorize(self, principal: Principal | None, scope: str) -> bool:
        if principal is None:
            return False
        return scope in ROLE_SCOPES.get(principal.role, frozenset())

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "AccessPolicy":
        if not isinstance(data, dict):
            raise ValueError("access config must be an object")
        raw_tokens = data.get("tokens", [])
        if not isinstance(raw_tokens, list):
            raise ValueError("access config 'tokens' must be a list")
        policy = cls()
        for raw in raw_tokens:
            if not isinstance(raw, dict):
                raise ValueError("each token entry must be an object")
            policy.add_token(
                str(raw.get("token", "")),
                str(raw.get("name", "")),
                str(raw.get("role", "")),
            )
        return policy

    @classmethod
    def load(cls, path: str) -> "AccessPolicy":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_config(json.load(fh))

    @staticmethod
    def save(path: str, tokens: Iterable[tuple[str, str, str]]) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "tokens": [
                {"token": token, "name": name, "role": role}
                for token, name, role in tokens
            ]
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_or_create_tokens(path: str) -> tuple[AccessPolicy, dict[str, str]]:
    """Load a token file, or create a demo file with one token per role.

    Returns the policy plus the raw `role -> token` map only when the file was
    just created, so the caller can display the demo credentials once.
    """
    if os.path.exists(path):
        return AccessPolicy.load(path), {}
    import secrets

    created = {
        VIEWER: secrets.token_urlsafe(24),
        INVESTIGATOR: secrets.token_urlsafe(24),
        ADMIN: secrets.token_urlsafe(24),
    }
    AccessPolicy.save(
        path,
        (
            (created[VIEWER], "demo-viewer", VIEWER),
            (created[INVESTIGATOR], "demo-investigator", INVESTIGATOR),
            (created[ADMIN], "demo-admin", ADMIN),
        ),
    )
    policy = AccessPolicy.load(path)
    return policy, created
