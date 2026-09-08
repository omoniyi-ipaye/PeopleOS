"""Full application recovery acceptance using real upload and analytics engines."""
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient


def test_full_api_recovery_retires_orphaned_evidence_and_preserves_sources(tmp_path, monkeypatch):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('PEOPLEOS_WORKSPACE_REGISTRY', str(tmp_path / 'registry.json'))
    from src.platform.workspace import WorkspaceStore
    from src.platform.jobs import JobStore
    from src.platform.local_dataset_store import dataset_artifact_path
    from src.platform.model_lifecycle import ModelLifecycleService
    from api import runtime_registry, main
    from api.routes import platform, upload, intelligence
    from src import database
    monkeypatch.setattr(database, '_database_instance', None)
    store = WorkspaceStore()
    registry = runtime_registry.WorkspaceRuntimeRegistry()
    monkeypatch.setattr(runtime_registry, 'runtime_registry', registry)
    for module in (platform, upload, intelligence):
        monkeypatch.setattr(module, '_store', store)
    monkeypatch.setattr(platform, '_jobs', JobStore(str(tmp_path / 'jobs.json')))
    monkeypatch.setattr(ModelLifecycleService, '_runtime_artifacts', {'orphan-model': object()})
    async def local_app(scope, receive, send):
        if scope['type'] == 'http':
            scope = {**scope, 'client': ('127.0.0.1', 45000)}
        await main.app(scope, receive, send)
    frame = pd.DataFrame({
        'EmployeeID': [f'A{i:04d}' for i in range(100)],
        'Dept': ['Engineering']*50 + ['People']*50,
        'Salary': [60000.]*100, 'Tenure': [2.]*100,
        'LastRating': [4.]*100, 'Age': [30]*100,
        'Gender': ['Female', 'Male']*50, 'JobTitle': ['Analyst']*100,
        'JobLevel': ['L03']*100, 'Location': ['Madrid']*100,
        'HireDate': ['2024-01-01']*100, 'ManagerID': ['A0000']*100,
        'Attrition': [1, 1]+[0]*98,
    })
    with TestClient(local_app, base_url='http://127.0.0.1:8000') as client:
        response = client.post('/api/upload', files={'file': ('synthetic.csv', frame.to_csv(index=False).encode(), 'text/csv')})
        assert response.status_code == 200, response.text
        original_id = response.json()['dataset_id']
        artifact = dataset_artifact_path(original_id)
        original_bytes = artifact.read_bytes()
        assert client.get('/api/analytics/summary').json()['active_count'] == 98
        prior_state = registry.get('local')
        store.path.write_bytes(b'{invalid registry with original identities')
        recovered = client.post('/api/platform/health/recover')
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()['status'] == 'metadata_reinitialized'
        assert recovered.json()['health']['status'] == 'degraded'
        assert recovered.json()['invalidated_runtimes'] == 1
        assert not prior_state.has_data()  # In-flight references also see retirement.
        assert prior_state.analytics_engine is None
        assert prior_state.scenario_cache == {}
        assert ModelLifecycleService._runtime_artifacts == {}
        assert artifact.read_bytes() == original_bytes
        unavailable = client.get('/api/analytics/summary')
        assert unavailable.status_code in (400, 409), unavailable.text
        assert 'X-PeopleOS-Dataset' not in unavailable.headers
        assert client.get('/api/status').json()['data']['loaded'] is False
        # Even an existing legacy database cannot automatically revive old data.
        fallback_calls = []
        monkeypatch.setattr(prior_state.data_loader, 'load_from_database', lambda: fallback_calls.append(True) or frame)
        assert prior_state.load_from_database() is False
        assert fallback_calls == []
        assert not prior_state.has_data()
        frame['EmployeeID'] = [f'B{i:04d}' for i in range(100)]
        frame['Attrition'] = [1]*20 + [0]*80
        uploaded = client.post('/api/upload', files={'file': ('reconciled.csv', frame.to_csv(index=False).encode(), 'text/csv')})
        assert uploaded.status_code == 200, uploaded.text
        new_id = uploaded.json()['dataset_id']
        assert new_id != original_id
        summary = client.get('/api/analytics/summary')
        assert summary.status_code == 200, summary.text
        assert summary.json()['active_count'] == 80
        assert summary.headers['X-PeopleOS-Dataset'] == new_id
        assert registry.get('local').raw_df.EmployeeID.str.startswith('B').all()
        assert artifact.read_bytes() == original_bytes
