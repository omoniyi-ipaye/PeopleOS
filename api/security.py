"""Local-first API access guard.

Loopback traffic is allowed without a token. Any non-loopback client must present
PEOPLEOS_API_TOKEN as a Bearer token; if no token is configured, remote API access
is disabled even when the server was accidentally bound to a wider interface.
"""

import hmac
import os
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_PUBLIC_PATHS = {"/", "/api/health"}


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def local_first_access_guard(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    client_host = request.client.host if request.client else "unknown"
    if client_host in _LOOPBACK_HOSTS:
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

    return await call_next(request)
