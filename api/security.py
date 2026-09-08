"""Local-first identity, browser trust and request authorization boundary.

Same-origin loopback clients are the local owner. Remote access requires a
server-configured token and role; proxy forwarding never grants owner rights
when it identifies a non-loopback source. This is a single-owner desktop
boundary, not a multi-user identity provider.
"""

import hmac
import os
import re
from typing import Optional
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from api.authorization import has_permission

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_LOCAL_URL_HOSTS = _LOOPBACK_HOSTS | {"testserver"}
_PUBLIC_PATHS = {"/", "/api/health"}
_ALLOWED_REMOTE_ROLES = {"admin", "analyst", "viewer"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_DEV_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"}
# Explicit method/path mappings cover legacy writes as well as the control plane.
# New mutation routes fail closed for remote roles until assigned a permission.
_MUTATION_PERMISSIONS = (
    ('POST', r'/api/upload(?:/load-sample|/reset)?', 'dataset.write'),
    ('POST', r'/api/sentiment/upload/(?:enps|onboarding)', 'dataset.write'),
    ('POST', r'/api/scenario/(?:simulate/[^/]+|compare|sensitivity)', 'investigate'),
    ('DELETE', r'/api/scenario/[^/]+', 'investigate'),
    ('POST', r'/api/(?:intelligence/investigate|advisor/ask)', 'investigate'),
    ('POST', r'/api/search', 'dataset.read'),
    ('POST', r'/api/model-lab/optimize', 'model.train'),
    ('POST', r'/api/platform/workspaces', 'workspace.write'),
    ('POST', r'/api/platform/workspaces/[^/]+/datasets/current', 'dataset.write'),
    ('POST', r'/api/platform/workspaces/[^/]+/datasets/[^/]+/activate', 'dataset.activate'),
    ('POST', r'/api/platform/workspaces/[^/]+/models/train', 'model.train'),
    ('POST', r'/api/platform/workspaces/[^/]+/models/[^/]+/activate', 'model.activate'),
    ('POST', r'/api/platform/sessions', 'session.write'),
    ('POST', r'/api/platform/health/recover', 'health.recover'),
)


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token:
        return None
    return token


def _set_identity(request: Request, *, actor_id: str, role: str, local: bool) -> None:
    request.state.peopleos_actor_id = actor_id
    request.state.peopleos_role = role
    request.state.peopleos_local = local


def _trusted_local_transport(request: Request) -> bool:
    if not request.client or request.client.host not in _LOOPBACK_HOSTS:
        return False
    # Host validation prevents a remote hostname resolving to loopback from
    # inheriting desktop owner rights (DNS rebinding).
    if request.url.hostname not in _LOCAL_URL_HOSTS:
        return False
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded and any(host.strip() not in _LOOPBACK_HOSTS for host in forwarded.split(',')):
        return False
    # RFC Forwarded is not consumed by this local application. Do not infer a
    # local owner from a proxy chain that has not been explicitly validated.
    if request.headers.get('forwarded'):
        return False
    return True


def _trusted_browser_origin(request: Request) -> bool:
    origin = request.headers.get('origin')
    if not origin:
        return request.headers.get('sec-fetch-site') not in {'cross-site'}
    if origin in _DEV_ORIGINS:
        return True
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme in {'http', 'https'}
            and parsed.hostname in _LOCAL_URL_HOSTS
            and parsed.username is None and parsed.password is None
            and not parsed.path and not parsed.query and not parsed.fragment
            and parsed.scheme == request.url.scheme
            and parsed.hostname == request.url.hostname
            and (parsed.port or (443 if parsed.scheme == 'https' else 80))
            == (request.url.port or (443 if request.url.scheme == 'https' else 80))
        )
    except ValueError:
        return False


async def _authorized_response(request: Request, call_next):
    path = request.url.path.rstrip('/') or '/'
    # The obsolete session API accepts arbitrary filesystem paths and bypasses
    # dataset lifecycle/provenance. Current UI uses platform sessions instead.
    if path == '/api/sessions' or path.startswith('/api/sessions/'):
        return JSONResponse(status_code=410, content={'detail': (
            'Legacy file sessions are retired. Use platform investigation sessions '
            'and registered dataset version activation.'
        )})
    role = request.state.peopleos_role
    if request.method not in _SAFE_METHODS and path.startswith('/api/'):
        permission = next((p for method, pattern, p in _MUTATION_PERMISSIONS
                           if method == request.method and re.fullmatch(pattern, path)), None)
        if role != 'owner' and (permission is None or not has_permission(role, permission)):
            return JSONResponse(status_code=403, content={'detail': 'This role is not authorized for this operation.'})
    response = await call_next(request)
    if path.startswith('/api/'):
        # Workforce evidence should not persist in shared browser/proxy caches.
        response.headers['Cache-Control'] = 'no-store'
    return response


async def local_first_access_guard(request: Request, call_next):
    if _trusted_local_transport(request):
        if not _trusted_browser_origin(request):
            return JSONResponse(status_code=403, content={'detail': 'Untrusted browser origin for local PeopleOS access.'})
        _set_identity(request, actor_id='local-owner', role='owner', local=True)
        return await _authorized_response(request, call_next)

    configured_token = os.getenv('PEOPLEOS_API_TOKEN')
    supplied = _bearer_token(request.headers.get('authorization'))
    authenticated = bool(configured_token and supplied is not None and
                         hmac.compare_digest(supplied.encode('utf-8'), configured_token.encode('utf-8')))
    if request.url.path in _PUBLIC_PATHS and not authenticated:
        _set_identity(request, actor_id='public-health', role='viewer', local=False)
        if request.url.path == '/api/health':
            # Public liveness deliberately excludes workspace/data/model state.
            return JSONResponse({'status': 'running'}, headers={'Cache-Control': 'no-store'})
        return await call_next(request)
    if not configured_token:
        return JSONResponse(status_code=403, content={'detail': 'Remote PeopleOS API access is disabled. Set PEOPLEOS_API_TOKEN explicitly before allowing non-loopback clients.'})
    if not authenticated:
        return JSONResponse(status_code=401, content={'detail': 'Invalid or missing API token.'})
    configured_role = os.getenv('PEOPLEOS_API_ROLE', 'analyst').lower()
    if configured_role not in _ALLOWED_REMOTE_ROLES:
        return JSONResponse(status_code=500, content={'detail': 'PEOPLEOS_API_ROLE is invalid.'})
    actor_id = os.getenv('PEOPLEOS_API_ACTOR_ID', 'remote-api-user')
    _set_identity(request, actor_id=actor_id, role=configured_role, local=False)
    return await _authorized_response(request, call_next)
