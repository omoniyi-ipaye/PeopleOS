"""HTTP acceptance tests for local trust and remote role enforcement."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('PEOPLEOS_WORKSPACE_REGISTRY', str(tmp_path / 'registry.json'))
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'test-secret')
    from api.security import local_first_access_guard
    from api.routes import upload, sessions
    from api.dependencies import get_app_state
    changes = []
    state = SimpleNamespace(reset=lambda: changes.append('reset'))
    monkeypatch.setattr(upload, '_store', SimpleNamespace(get_workspace=lambda _: {}, _replace_workspace=lambda _: None))
    monkeypatch.setattr(upload, '_clear_active_lifecycle', lambda _: changes.append('clear'))
    app = FastAPI()
    app.middleware('http')(local_first_access_guard)
    app.dependency_overrides[get_app_state] = lambda: state
    app.include_router(upload.router)
    app.include_router(sessions.router)

    @app.get('/api/analytics/whoami')
    async def whoami(request: Request):
        return {'role': request.state.peopleos_role}

    @app.post('/api/future-mutation')
    async def mutation():
        changes.append('unknown')
        return {'ok': True}

    @app.get('/api/health')
    async def health():
        return {'status': 'healthy', 'platform': {'workspaces': [{'workspace_id': 'private-project'}]}}

    return app, changes


def client_for(app, host='198.51.100.2', base='http://127.0.0.1:8000'):
    async def with_client(scope, receive, send):
        if scope['type'] == 'http':
            scope = {**scope, 'client': (host, 45000)}
        await app(scope, receive, send)
    return TestClient(with_client, base_url=base)


@pytest.mark.parametrize('role', ['viewer', 'analyst'])
def test_remote_readers_cannot_reset_actual_dataset_route(boundary, monkeypatch, role):
    app, changes = boundary
    monkeypatch.setenv('PEOPLEOS_API_ROLE', role)
    with client_for(app) as client:
        response = client.post('/api/upload/reset', headers={'Authorization': 'Bearer test-secret'})
    assert response.status_code == 403, response.text
    assert changes == []


def test_cross_site_browser_cannot_reset_local_owner_data(boundary):
    app, changes = boundary
    with client_for(app, '127.0.0.1') as client:
        response = client.post('/api/upload/reset', headers={'Origin': 'https://untrusted.example'})
    assert response.status_code == 403, response.text
    assert changes == []


def test_legacy_session_cannot_delete_arbitrary_file(boundary, tmp_path):
    app, _ = boundary
    protected = tmp_path / 'important.txt'
    protected.write_text('must survive')
    with client_for(app, '127.0.0.1') as client:
        response = client.delete('/api/sessions', params={'filepath': str(protected)})
    assert response.status_code == 410, response.text
    assert protected.read_text() == 'must survive'


@pytest.mark.parametrize('origin', ['http://127.0.0.1:49152', 'http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001', None])
def test_desktop_random_port_and_supported_dev_origins_keep_owner_access(boundary, origin):
    app, changes = boundary
    with client_for(app, '127.0.0.1', 'http://127.0.0.1:49152') as client:
        response = client.post('/api/upload/reset', headers={'Origin': origin} if origin else {})
    assert response.status_code == 200, response.text
    assert changes == ['clear', 'reset']


def test_admin_can_reset_and_spoofed_role_does_not_elevate_viewer(boundary, monkeypatch):
    app, changes = boundary
    with client_for(app) as client:
        monkeypatch.setenv('PEOPLEOS_API_ROLE', 'viewer')
        denied = client.post('/api/upload/reset', headers={'Authorization': 'Bearer test-secret', 'X-PeopleOS-Role': 'owner'})
        assert denied.status_code == 403
        assert changes == []
        monkeypatch.setenv('PEOPLEOS_API_ROLE', 'admin')
        allowed = client.post('/api/upload/reset', headers={'Authorization': 'Bearer test-secret'})
        assert allowed.status_code == 200, allowed.text
        assert changes == ['clear', 'reset']


@pytest.mark.parametrize('role', ['viewer', 'analyst', 'admin'])
def test_unmapped_new_mutation_fails_closed_for_remote_roles(boundary, monkeypatch, role):
    app, changes = boundary
    monkeypatch.setenv('PEOPLEOS_API_ROLE', role)
    with client_for(app) as client:
        response = client.post('/api/future-mutation', headers={'Authorization': 'Bearer test-secret'})
    assert response.status_code == 403
    assert changes == []


@pytest.mark.parametrize('headers,base', [
    ({}, 'http://untrusted.example:8000'),
    ({'X-Forwarded-For': '198.51.100.3'}, 'http://127.0.0.1:8000'),
    ({'X-Forwarded-For': '198.51.100.3, 127.0.0.1'}, 'http://127.0.0.1:8000'),
    ({'Forwarded': 'for=198.51.100.3'}, 'http://127.0.0.1:8000'),
])
def test_remote_host_or_proxy_chain_does_not_become_local_owner(boundary, headers, base):
    app, changes = boundary
    with client_for(app, '127.0.0.1', base) as client:
        response = client.post('/api/upload/reset', headers=headers)
    assert response.status_code == 401, response.text
    assert changes == []


@pytest.mark.parametrize('headers', [{'Origin': 'null'}, {'Origin': 'http://127.0.0.1:49153'}, {'Sec-Fetch-Site': 'cross-site'}])
def test_opaque_foreign_and_cross_site_requests_denied(boundary, headers):
    app, changes = boundary
    with client_for(app, '127.0.0.1', 'http://127.0.0.1:49152') as client:
        response = client.post('/api/upload/reset', headers=headers)
    assert response.status_code == 403
    assert changes == []


def test_public_liveness_does_not_disclose_workspace_data(boundary):
    app, _ = boundary
    with client_for(app) as client:
        response = client.get('/api/health')
        assert response.json() == {'status': 'running'}
        assert response.headers['cache-control'] == 'no-store'
    with client_for(app, '127.0.0.1') as client:
        assert client.get('/api/health').json()['platform']['workspaces']


def test_remote_read_evidence_is_authorized_and_not_cacheable(boundary, monkeypatch):
    app, _ = boundary
    monkeypatch.setenv('PEOPLEOS_API_ROLE', 'analyst')
    with client_for(app) as client:
        assert client.get('/api/analytics/whoami').status_code == 401
        response = client.get('/api/analytics/whoami', headers={'Authorization': 'Bearer test-secret'})
    assert response.json() == {'role': 'analyst'}
    assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('role,path,expected', [
    ('viewer', '/api/sentiment/upload/enps', 403),
    ('analyst', '/api/sentiment/upload/onboarding', 403),
    ('viewer', '/api/scenario/simulate/compensation', 403),
    ('viewer', '/api/intelligence/investigate', 403),
    ('analyst', '/api/platform/workspaces/local/models/train', 403),
    ('analyst', '/api/platform/workspaces/local/datasets/d1/activate', 403),
    ('analyst', '/api/platform/health/recover', 403),
    ('analyst', '/api/intelligence/investigate', 200),
    ('viewer', '/api/search', 403),
    ('admin', '/api/sentiment/upload/enps', 200),
])
def test_permission_boundary_covers_legacy_and_control_plane_writes(boundary, monkeypatch, role, path, expected):
    app, changes = boundary
    monkeypatch.setenv('PEOPLEOS_API_ROLE', role)
    async def operation():
        changes.append(path)
        return {'ok': True}
    app.add_api_route(path, operation, methods=['POST'])
    with client_for(app) as client:
        response = client.post(path, headers={'Authorization': 'Bearer test-secret'})
    assert response.status_code == expected, response.text
    assert changes == ([path] if expected == 200 else [])
