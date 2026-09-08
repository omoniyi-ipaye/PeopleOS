"""Adversarial matrix for every API action and durable registry transition."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _remote_client(app, host='198.51.100.2'):
    async def remote(scope, receive, send):
        if scope['type'] == 'http':
            scope = {**scope, 'client': (host, 45000)}
        await app(scope, receive, send)
    return TestClient(remote, base_url='http://127.0.0.1:8000')


def _concrete_path(path: str) -> str:
    values = {
        'workspace_id': 'local', 'dataset_id': 'missing', 'model_id': 'missing',
        'scenario_id': 'missing', 'cluster_id': '1', 'employee_id': 'missing',
        'template_type': 'enps',
    }
    for name, value in values.items():
        path = path.replace('{' + name + '}', value)
    return path


def test_every_declared_api_unsafe_method_has_explicit_viewer_boundary(monkeypatch):
    """A new route cannot silently bypass the central mutation allowlist."""
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    monkeypatch.setenv('PEOPLEOS_API_ROLE', 'viewer')
    from api.main import app

    mutations = sorted({
        (method, _concrete_path(route.path))
        for route in app.routes if route.path.startswith('/api/')
        for method in (route.methods or set())
        if method not in {'GET', 'HEAD', 'OPTIONS'}
    })
    assert mutations, 'API mutation inventory unexpectedly empty'
    with _remote_client(app) as client:
        for method, path in mutations:
            response = client.request(
                method, path,
                headers={'Authorization': 'Bearer matrix-secret'},
                json={},
            )
            assert response.status_code in {403, 410}, (method, path, response.status_code, response.text)
            assert response.headers.get('cache-control') == 'no-store', (method, path, response.headers)


def test_every_declared_api_get_has_explicit_remote_viewer_outcome(monkeypatch):
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    monkeypatch.setenv('PEOPLEOS_API_ROLE', 'viewer')
    from api.main import app
    from api.security import _PROHIBITED_REMOTE_READS, _READ_PERMISSIONS, _RETIRED_ROUTES
    from api.authorization import has_permission
    import re

    reads = sorted({_concrete_path(route.path) for route in app.routes
                    if route.path.startswith('/api/') and 'GET' in (route.methods or set())})
    with _remote_client(app) as client:
        for path in reads:
            expected_denied = any(re.fullmatch(pattern, path) for pattern in _PROHIBITED_REMOTE_READS)
            expected_retired = any(method == 'GET' and re.fullmatch(pattern, path) for method, pattern in _RETIRED_ROUTES)
            permission = next((value for pattern, value in _READ_PERMISSIONS if re.fullmatch(pattern, path)), None)
            response = client.get(path, headers={'Authorization': 'Bearer matrix-secret'})
            if expected_retired:
                assert response.status_code == 410, (path, response.status_code, response.text)
            elif expected_denied or permission is None or not has_permission('viewer', permission):
                assert response.status_code == 403, (path, permission, response.status_code, response.text)
            else:
                assert response.status_code not in {401, 403}, (path, permission, response.status_code, response.text)
            assert response.headers.get('cache-control') == 'no-store', (path, response.headers)


@pytest.mark.parametrize('role,expected', [('viewer', 403), ('analyst', 400)])
def test_free_text_semantic_search_requires_sensitive_read_permission(monkeypatch, role, expected):
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    monkeypatch.setenv('PEOPLEOS_API_ROLE', role)
    from api.main import app
    with _remote_client(app) as client:
        response = client.post('/api/search?query=performance', headers={'Authorization': 'Bearer matrix-secret'})
    assert response.status_code == expected, response.text
    assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('role,expected', [('viewer', 403), ('analyst', 200)])
def test_singleton_grouped_metrics_never_reach_remote_viewer(monkeypatch, role, expected):
    """Aggregate labels do not make a one-person cell safe for a view-only role."""
    from fastapi import FastAPI
    from api.security import local_first_access_guard
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    monkeypatch.setenv('PEOPLEOS_API_ROLE', role)
    app = FastAPI()
    app.middleware('http')(local_first_access_guard)

    @app.get('/api/analytics/departments')
    async def singleton_department():
        return [{'dept': 'Executive', 'headcount': 1, 'avg_salary': 250000, 'attrition_rate': 100.0}]

    with _remote_client(app) as client:
        response = client.get('/api/analytics/departments', headers={'Authorization': 'Bearer matrix-secret'})
    assert response.status_code == expected, response.text
    if expected == 200:
        assert response.json()[0]['headcount'] == 1
    assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('headers,status', [
    ({}, 401),
    ({'Authorization': 'Bearer wrong'}, 401),
])
def test_authentication_failures_for_private_api_are_never_cacheable(monkeypatch, headers, status):
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    from api.main import app
    with _remote_client(app) as client:
        response = client.get('/api/status', headers=headers)
    assert response.status_code == status
    assert response.headers['cache-control'] == 'no-store'


def test_cross_origin_denial_is_never_cacheable(monkeypatch):
    monkeypatch.setenv('PEOPLEOS_API_TOKEN', 'matrix-secret')
    from api.main import app
    with _remote_client(app, host='127.0.0.1') as client:
        response = client.get('/api/status', headers={'Origin': 'https://attacker.example'})
    assert response.status_code == 403
    assert response.headers['cache-control'] == 'no-store'


def test_concurrent_store_instances_do_not_lose_registry_updates(tmp_path):
    from src.platform.workspace import WorkspaceStore
    registry = str(tmp_path / 'registry.json')
    stores = [WorkspaceStore(registry) for _ in range(8)]

    def register(index):
        return stores[index % len(stores)].register_dataset(
            workspace_id='local', source_name=f'synthetic-{index}.csv',
            content_hash=f'{index:064x}', row_count=1, columns=['EmployeeID'],
        ).dataset_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(register, range(64)))
    workspace = WorkspaceStore(registry).get_workspace('local')
    assert len(ids) == len(set(ids)) == 64
    assert len(workspace.datasets) == 64
    assert sorted(item.version for item in workspace.datasets) == list(range(1, 65))


def test_failed_registry_commit_removes_unregistered_dataset_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path / 'home'))
    from api.routes import upload
    from src.platform.workspace import WorkspaceStore
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    monkeypatch.setattr(upload, '_store', store)
    monkeypatch.setattr(store, 'register_active_dataset', lambda **kwargs: (_ for _ in ()).throw(OSError('registry full')))
    frame = pd.DataFrame({'EmployeeID': ['A1'], 'Dept': ['People']})
    state = SimpleNamespace(
        raw_df=frame, historical_df=frame,
        runtime_provenance={'current_fingerprint': 'verified-fingerprint'},
    )
    with pytest.raises(OSError, match='registry full'):
        upload._register_loaded_dataset(state, 'synthetic.csv', 'a' * 64)
    assert store.get_workspace('local').datasets == []
    datasets = tmp_path / 'home' / 'datasets'
    assert not datasets.exists() or list(datasets.iterdir()) == []


def test_failed_artifact_write_never_registers_ghost_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path / 'home'))
    from api.routes import upload
    from src.platform.workspace import WorkspaceStore
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    monkeypatch.setattr(upload, '_store', store)
    monkeypatch.setattr(upload, 'save_dataset_artifact', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    frame = pd.DataFrame({'EmployeeID': ['A1'], 'Dept': ['People']})
    state = SimpleNamespace(
        raw_df=frame, historical_df=frame,
        runtime_provenance={'current_fingerprint': 'verified-fingerprint'},
    )
    with pytest.raises(OSError, match='disk full'):
        upload._register_loaded_dataset(state, 'synthetic.csv', 'a' * 64)
    assert store.get_workspace('local').datasets == []


def test_session_records_established_actor_for_audit(tmp_path):
    from src.platform.workspace import WorkspaceStore
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    session = store.open_session(workspace_id='local', actor_id='remote-analyst-7')
    assert session.actor_id == 'remote-analyst-7'
    assert store.get_workspace('local').sessions[0].actor_id == 'remote-analyst-7'


def test_active_dataset_registration_is_one_consistent_registry_transition(tmp_path):
    from src.platform.workspace import DatasetState, ModelState, WorkspaceStore
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    old = store.register_dataset(
        workspace_id='local', source_name='old.csv', content_hash='a' * 64,
        row_count=1, columns=['EmployeeID'],
    )
    store.activate_dataset('local', old.dataset_id)
    model = store.create_model(workspace_id='local', dataset_id=old.dataset_id)
    store.update_model('local', model.model_id, state=ModelState.CANDIDATE)
    store.activate_model('local', model.model_id)

    new = store.register_active_dataset(
        workspace_id='local', dataset_id='ds_controlled', source_name='new.csv',
        content_hash='b' * 64, row_count=2, columns=['EmployeeID'],
    )
    workspace = store.get_workspace('local')
    assert new.state == DatasetState.ACTIVE
    assert workspace.active_dataset_id == new.dataset_id
    assert workspace.active_model_id is None
    assert next(item for item in workspace.datasets if item.dataset_id == old.dataset_id).state == DatasetState.SUPERSEDED
    assert next(item for item in workspace.models if item.model_id == model.model_id).state == ModelState.RETIRED


def test_failed_registry_replace_preserves_prior_bytes_and_removes_temp(tmp_path, monkeypatch):
    from src.platform.workspace import WorkspaceStore
    store = WorkspaceStore(str(tmp_path / 'registry.json'))
    original = store.path.read_bytes()
    monkeypatch.setattr('src.platform.workspace.os.replace', lambda *args: (_ for _ in ()).throw(OSError('replace failed')))
    with pytest.raises(OSError, match='replace failed'):
        store.ensure_workspace('secondary', 'Secondary')
    assert store.path.read_bytes() == original
    assert list(tmp_path.glob('*.tmp')) == []
