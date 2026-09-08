"""Snapshot recovery, categorical fidelity, and synchronized commit regressions."""
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.dependencies import AppState
from src.platform.local_dataset_store import load_dataset_artifact, save_dataset_artifact
from src.platform.provenance import frame_fingerprint
from src.platform.runtime_lock import RUNTIME_MUTATION_LOCK


def frame(employee='0001'):
    return pd.DataFrame({'EmployeeID': [employee], 'ManagerID': ['0000'],
                         'Dept': ['001'], 'Location': ['002'], 'HireSource': ['003'],
                         'PerformanceText': ['0004'], 'Salary': [60000.], 'Attrition': [0]})


def state_for(data=None, generation='A'):
    state = SimpleNamespace(raw_df=data, model_provenance=None, nlp_results=None,
                            runtime_provenance=None, workspace_id='local')
    state.has_data = lambda: state.raw_df is not None and not state.raw_df.empty
    if data is not None:
        state.runtime_provenance = {
            'workspace_id': 'local', 'dataset_id': 'ds_' + generation,
            'generation': generation, 'current_fingerprint': frame_fingerprint(data),
        }
    return state


def test_categorical_lexemes_survive_artifacts_and_remain_fingerprint_significant(monkeypatch, tmp_path):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    source = frame()
    path = save_dataset_artifact('ds_A', source)
    restored = load_dataset_artifact('ds_A', expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for column in ['EmployeeID', 'ManagerID', 'Dept', 'Location', 'HireSource', 'PerformanceText']:
        assert restored[column].tolist() == source[column].tolist()
    assert restored.Salary.iloc[0] == 60000
    assert frame_fingerprint(source) == frame_fingerprint(restored)
    changed = source.assign(Dept='1')
    assert frame_fingerprint(source) != frame_fingerprint(changed)


def test_declared_measurement_and_snapshot_types_are_canonical_but_categories_are_exact():
    left = frame().assign(SnapshotDate=pd.Timestamp('2026-01-01', tz='UTC'), PromotionCount=2)
    right = left.assign(SnapshotDate='2026-01-01 00:00:00+00:00', Salary='60000.0', PromotionCount='2.0')
    assert frame_fingerprint(left) == frame_fingerprint(right)
    assert frame_fingerprint(left) != frame_fingerprint(right.assign(Location='2'))


def test_data_loader_and_artifact_share_numeric_contract(monkeypatch, tmp_path):
    from src.data_loader import DataLoader
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    source = frame().assign(Tenure='2', LastRating='4', Age='30', Gender='NA',
                            JobTitle='001', HireDate='2024-01-01', PromotionCount='02',
                            InterviewScore_Technical='04')
    upload = tmp_path / 'upload.csv'
    source.to_csv(upload, index=False)
    loader = DataLoader()
    loader.min_rows = 1
    loaded = loader.load(str(upload))
    save_dataset_artifact('ds_A', loaded)
    restored = load_dataset_artifact('ds_A')
    assert restored.Gender.iloc[0] == 'NA'
    assert restored.JobTitle.iloc[0] == '001'
    for column in ['PromotionCount', 'InterviewScore_Technical', 'Salary', 'Tenure']:
        assert pd.api.types.is_numeric_dtype(loaded[column])
        assert restored[column].tolist() == loaded[column].tolist()
    assert frame_fingerprint(loaded) == frame_fingerprint(restored)


def test_oversize_upload_is_rejected_instead_of_truncating_population(tmp_path):
    from src.data_loader import DataLoader, DataValidationError
    path = tmp_path / 'oversize.csv'
    pd.concat([frame(f'E{i}') for i in range(3)], ignore_index=True).to_csv(path, index=False)
    loader = DataLoader()
    loader.min_rows = 1
    loader.max_rows = 2
    with pytest.raises(DataValidationError, match='contains 3 rows; the supported maximum is 2'):
        loader.load(str(path))


def test_tampered_artifact_is_rejected_before_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    path = save_dataset_artifact('ds_A', frame())
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('EmployeeID,Dept\nOTHER,OTHER\n')
    with pytest.raises(ValueError, match='integrity check failed'):
        load_dataset_artifact('ds_A', expected_sha256=expected)


def test_artifact_hash_and_parser_use_one_captured_byte_sequence(monkeypatch, tmp_path):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    path = save_dataset_artifact('ds_A', frame())
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    original_read = Path.read_bytes
    reads = []

    def replace_after_read(target):
        content = original_read(target)
        if target == path:
            reads.append(content)
            target.write_text('EmployeeID,Dept\nOTHER,OTHER\n')
        return content

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    restored = load_dataset_artifact('ds_A', expected_sha256=expected)
    assert restored.EmployeeID.tolist() == ['0001']
    assert len(reads) == 1


def test_response_guard_rejects_reset_during_analysis(monkeypatch):
    from api import integrity
    state = state_for(frame())
    monkeypatch.setattr(integrity, 'get_workspace_state', lambda request: state)

    async def call_next(request):
        with RUNTIME_MUTATION_LOCK:
            state.raw_df = None
            state.runtime_provenance = None
        return JSONResponse({'old_measurement': 1})

    request = Request({'type': 'http', 'path': '/api/analytics/summary', 'headers': []})
    response = asyncio.run(integrity.evidence_snapshot_guard(request, call_next))
    assert response.status_code == 409
    assert b'old_measurement' not in response.body


def test_unchanged_response_is_stamped_with_verified_snapshot(monkeypatch):
    from api import integrity
    state = state_for(frame())
    monkeypatch.setattr(integrity, 'get_workspace_state', lambda request: state)

    async def call_next(request):
        return JSONResponse({'measurement': 1})

    request = Request({'type': 'http', 'path': '/api/analytics/summary', 'headers': []})
    response = asyncio.run(integrity.evidence_snapshot_guard(request, call_next))
    assert response.status_code == 200
    assert response.headers['X-PeopleOS-Dataset'] == 'ds_A'
    assert response.headers['X-PeopleOS-Snapshot'] == 'A'


@pytest.mark.parametrize('reset', [True, False])
def test_nlp_releases_lock_for_inference_and_rejects_stale_cache_commit(reset):
    from api.routes.nlp import get_nlp_analysis
    state = state_for(frame())
    inference_started, finish_inference = Event(), Event()
    seen = []

    def process(source):
        seen.append(source)
        inference_started.set()
        assert finish_inference.wait(3)
        return {'sentiment_summary': {}, 'topics': [], 'skills': {}, 'nlp_available': True}

    state.nlp_engine = SimpleNamespace(process_all=process)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(get_nlp_analysis, state)
        assert inference_started.wait(3)
        acquired = RUNTIME_MUTATION_LOCK.acquire(blocking=False)
        try:
            assert acquired, 'NLP inference must not hold the runtime mutation lock'
            if reset:
                state.raw_df = None
                state.runtime_provenance = None
            else:
                replacement = state_for(frame('NEW'), 'B')
                state.raw_df = replacement.raw_df
                state.runtime_provenance = replacement.runtime_provenance
        finally:
            if acquired:
                RUNTIME_MUTATION_LOCK.release()
            finish_inference.set()
        with pytest.raises(HTTPException) as error:
            future.result(timeout=3)
    assert error.value.status_code == 409
    assert state.nlp_results is None
    assert seen[0].EmployeeID.tolist() == ['0001']


def test_waiting_restore_cannot_replace_an_already_loaded_snapshot(monkeypatch):
    from src.platform import workspace
    state = state_for(frame('NEW'), 'B')
    monkeypatch.setattr(workspace, 'WorkspaceStore', lambda: pytest.fail('Already loaded state must not be restored again'))
    assert AppState.load_from_database(state) is True
    assert state.runtime_provenance['dataset_id'] == 'ds_B'


def test_nlp_commits_and_reuses_only_a_valid_same_snapshot_response():
    from api.routes.nlp import get_nlp_analysis
    state = state_for(frame())
    calls = []

    def process(source):
        calls.append(source)
        return {
            'sentiment_summary': {'avg_sentiment': 1., 'positive_count': 1,
                                  'neutral_count': 0, 'negative_count': 0,
                                  'positive_pct': 100., 'neutral_pct': 0., 'negative_pct': 0.},
            'topics': [], 'skills': {}, 'nlp_available': True,
        }

    state.nlp_engine = SimpleNamespace(process_all=process)
    first = get_nlp_analysis(state)
    second = get_nlp_analysis(state)
    assert len(calls) == 1
    assert first.model_dump() == second.model_dump() == state.nlp_results
    assert first.provenance['dataset_id'] == 'ds_A'


def test_restore_is_serialized_with_later_dataset_activation(monkeypatch, tmp_path):
    from src.platform import local_dataset_store, runtime_loader, workspace
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    path = save_dataset_artifact('ds_A', frame())
    record = SimpleNamespace(dataset_id='ds_A', quality={'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    registry = SimpleNamespace(active_dataset_id='ds_A', datasets=[record])
    monkeypatch.setattr(workspace, 'WorkspaceStore', lambda: SimpleNamespace(get_workspace=lambda key: registry))
    captured, finish_restore = Event(), Event()
    real_load = local_dataset_store.load_dataset_artifact

    def blocked_load(*args, **kwargs):
        value = real_load(*args, **kwargs)
        captured.set()
        assert finish_restore.wait(3)
        return value

    monkeypatch.setattr(local_dataset_store, 'load_dataset_artifact', blocked_load)
    state = state_for()

    def activate(target, data, **kwargs):
        target.raw_df = data
        target.runtime_provenance = {'dataset_id': kwargs['dataset_id']}

    monkeypatch.setattr(runtime_loader, 'activate_dataframe', activate)

    def activate_b():
        with RUNTIME_MUTATION_LOCK:
            registry.active_dataset_id = 'ds_B'
            state.raw_df = frame('NEW')
            state.runtime_provenance = {'dataset_id': 'ds_B'}

    with ThreadPoolExecutor(max_workers=2) as pool:
        restore = pool.submit(AppState.load_from_database, state)
        assert captured.wait(3)
        acquired = RUNTIME_MUTATION_LOCK.acquire(blocking=False)
        if acquired:
            RUNTIME_MUTATION_LOCK.release()
        try:
            assert not acquired, 'Restore must serialize its read-to-commit transaction'
            later_activation = pool.submit(activate_b)
        finally:
            finish_restore.set()
        assert restore.result(timeout=3) is True
        later_activation.result(timeout=3)
    assert state.runtime_provenance['dataset_id'] == registry.active_dataset_id == 'ds_B'
    assert state.raw_df.EmployeeID.tolist() == ['NEW']


def test_restore_rejects_tampering_without_activating_or_falling_back(monkeypatch, tmp_path):
    from src.platform import runtime_loader, workspace
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path))
    path = save_dataset_artifact('ds_A', frame())
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('EmployeeID,Dept\nOTHER,OTHER\n')
    record = SimpleNamespace(dataset_id='ds_A', quality={'artifact_sha256': expected})
    registry = SimpleNamespace(active_dataset_id='ds_A', datasets=[record])
    monkeypatch.setattr(workspace, 'WorkspaceStore', lambda: SimpleNamespace(get_workspace=lambda key: registry))
    monkeypatch.setattr(runtime_loader, 'activate_dataframe', lambda *args, **kwargs: pytest.fail('Tampered data cannot activate'))
    state = state_for()
    state.data_loader = SimpleNamespace(load_from_database=lambda: pytest.fail('Invalid canonical artifact must not fall back'))
    assert AppState.load_from_database(state) is False
    assert state.raw_df is None


def test_reset_failure_preserves_runtime_and_raises(monkeypatch):
    from src import database
    state = state_for(frame())

    def fail_clear():
        raise OSError('Controlled persistent storage failure')

    monkeypatch.setattr(database, 'get_database', lambda: SimpleNamespace(clear_all_data=fail_clear))
    with pytest.raises(RuntimeError, match='runtime data was retained'):
        AppState.reset(state)
    assert state.raw_df.EmployeeID.tolist() == ['0001']
    assert state.runtime_provenance['dataset_id'] == 'ds_A'
