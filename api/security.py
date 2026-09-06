"""Local-first API access and identity boundary.

Loopback traffic is the trusted local owner. Any non-loopback client must
present PEOPLEOS_API_TOKEN as a Bearer token. Remote role and actor identity are
configured server-side, never accepted from a client role header.
"""

import hmac
import os
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_PUBLIC_PATHS = {"/", "/api/health"}
_ALLOWED_REMOTE_ROLES = {"admin", "analyst", "viewer"}


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _set_identity(request: Request, *, actor_id: str, role: str, local: bool) -> None:
    request.state.peopleos_actor_id = actor_id
    request.state.peopleos_role = role
    request.state.peopleos_local = local


async def local_first_access_guard(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"

    if client_host in _LOOPBACK_HOSTS:
        _set_identity(request, actor_id="local-owner", role="owner", local=True)
        return await call_next(request)

    if request.url.path in _PUBLIC_PATHS:
        _set_identity(request, actor_id="public-health", role="viewer", local=False)
        return await call_next(request)

    configured_token = os.getenv("PEOPLEOS_API_TOKEN")
    if not configured_token:
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "Remote PeopleOS API access is disabled. Set PEOPLEOS_API_TOKEN "
                    "explicitly before allowing non-loopback clients."
                )
            },
        )

    supplied = _bearer_token(request.headers.get("authorization"))
    if supplied is None or not hmac.compare_digest(supplied, configured_token):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token."})

    configured_role = os.getenv("PEOPLEOS_API_ROLE", "analyst").lower()
    if configured_role not in _ALLOWED_REMOTE_ROLES:
        return JSONResponse(status_code=500, content={"detail": "PEOPLEOS_API_ROLE is invalid."})

    actor_id = os.getenv("PEOPLEOS_API_ACTOR_ID", "remote-api-user")
    _set_identity(request, actor_id=actor_id, role=configured_role, local=False)
    return await call_next(request)
