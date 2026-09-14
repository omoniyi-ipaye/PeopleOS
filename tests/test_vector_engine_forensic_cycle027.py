"""Cycle 027 forensic contracts for bounded semantic retrieval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import search as search_routes
from src.platform.provenance import frame_fingerprint
from src.vector_engine import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    VectorEngine,
)


class FakeIndex:
    def __init__(self, dimension):
        self.values = np.empty((0, dimension), dtype='float32')

    @property
    def ntotal(self):
        return len(self.values)

    def add(self, values):
        self.values = values.copy()

    def search(self, query, top_k):
        distances = ((self.values - query[0]) ** 2).sum(axis=1)
        order = np.argsort(distances)[:top_k]
        return distances[order][None, :], order[None, :]


class FakeFaiss:
    IndexFlatL2 = FakeIndex


class TokenModel:
    def encode(self, texts, **_kwargs):
        return np.asarray([
            [float('python' in text.lower()), float('ventas' in text.lower()), float('team' in text.lower())]
            for text in texts
        ], dtype='float32')


class BlockingModel(TokenModel):
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def encode(self, texts, **kwargs):
        self.started.set()
        assert self.release.wait(timeout=2), 'test refresh did not release the model'
        return super().encode(texts, **kwargs)


class NonFiniteModel(TokenModel):
    def encode(self, texts, **_kwargs):
        return np.full((len(texts), 3), np.nan, dtype='float32')


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': ['E-sensitive-1', 'E-sensitive-2'],
        'PerformanceText': ['Python payroll automation', 'Spanish ventas support'],
        'Dept': ['People', 'Sales'],
    })


def provenance(source: pd.DataFrame, dataset_id: str = 'ds-current') -> dict:
    return {
        'workspace_id': 'local',
        'dataset_id': dataset_id,
        'generation': f'g-{dataset_id}',
        'current_fingerprint': frame_fingerprint(source),
    }


def engine(model=None) -> VectorEngine:
    return VectorEngine(
        model=model or TokenModel(),
        faiss_backend=FakeFaiss(),
    )


def state_for(source: pd.DataFrame, vector_engine, current_provenance: dict):
    return SimpleNamespace(
        raw_df=source,
        vector_engine=vector_engine,
        runtime_provenance=current_provenance,
        has_data=lambda: True,
        load_from_database=lambda: False,
    )


def build(engine_instance: VectorEngine, source: pd.DataFrame, index_provenance: dict):
    engine_instance.build_index(
        source['PerformanceText'].tolist(),
        source.to_dict('records'),
        provenance=index_provenance,
    )


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    def dcg(items):
        return sum(relevance.get(item, 0.0) / math.log2(rank + 2) for rank, item in enumerate(items[:k]))

    ideal = sorted(relevance, key=relevance.get, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(retrieved) / ideal_dcg if ideal_dcg else 0.0


def test_default_model_is_pinned_and_real_multilingual_scope_is_explicit():
    instance = VectorEngine.__new__(VectorEngine)
    instance.model_name = DEFAULT_EMBEDDING_MODEL
    instance.model_revision = DEFAULT_EMBEDDING_REVISION
    assert instance.model_name.endswith('MiniLM-L12-v2')
    assert len(instance.model_revision) == 40
    from scripts.validate_local_embeddings import CASES
    assert {'spanish', 'cross_language'} <= {case['scope'] for case in CASES}


def test_recall_and_ndcg_contracts_are_ranking_metrics_not_probability_claims():
    relevance = {'python': 2.0, 'payroll': 1.0}
    assert recall_at_k(['python', 'unrelated'], set(relevance), 1) == 0.5
    assert recall_at_k(['python', 'payroll'], set(relevance), 2) == 1.0
    assert ndcg_at_k(['python', 'payroll'], relevance, 2) == pytest.approx(1.0)
    assert ndcg_at_k(['payroll', 'python'], relevance, 2) < 1.0


def test_nonfinite_refresh_clears_previous_index_and_provenance():
    source = frame()
    instance = engine()
    current = provenance(source)
    build(instance, source, current)
    assert instance.is_initialized()
    with pytest.raises(ValueError, match='finite vectors'):
        broken = VectorEngine(model=NonFiniteModel(), faiss_backend=FakeFaiss())
        broken.build_index(['bad'], [{'EmployeeID': 'E-bad'}], provenance=current)
    with pytest.raises(ValueError, match='finite vectors'):
        instance.model = NonFiniteModel()
        instance.build_index(['bad'], [{'EmployeeID': 'E-bad'}], provenance=current)
    assert not instance.is_initialized()
    assert instance.metadata == []
    assert instance.index_provenance is None
    assert not instance.matches_provenance(current)


def test_index_requires_exact_active_snapshot_and_clear_removes_retained_text():
    source = frame()
    instance = engine()
    current = provenance(source)
    build(instance, source, current)
    assert instance.matches_provenance(current)
    stale = {**current, 'generation': 'g-stale'}
    assert not instance.matches_provenance(stale)
    instance.clear_index()
    assert instance.search('python') == []
    assert instance.metadata == []
    assert instance.index_provenance is None


def test_search_rejects_nonfinite_query_and_malformed_backend_results():
    source = frame()
    instance = engine()
    build(instance, source, provenance(source))
    instance.model = NonFiniteModel()
    with pytest.raises(RuntimeError, match='unavailable'):
        instance.search('python')

    instance = engine()
    build(instance, source, provenance(source))
    instance.index.search = lambda *_: (np.array([[np.inf]]), np.array([[0]]))
    assert instance.search('python') == []


def test_refresh_and_search_are_serialized_and_search_cannot_see_stale_index():
    source = frame()
    blocking = BlockingModel()
    instance = engine()
    old = provenance(source, 'ds-old')
    build(instance, source, old)
    instance.model = blocking
    replacement = source.copy(deep=True)
    replacement.loc[0, 'PerformanceText'] = 'team collaboration'
    new = provenance(replacement, 'ds-new')

    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh = pool.submit(build, instance, replacement, new)
        assert blocking.started.wait(timeout=2)
        search = pool.submit(instance.search, 'python')
        assert not search.done(), 'search ran while refresh held the index lock'
        blocking.release.set()
        refresh.result()
        results = search.result()

    assert instance.matches_provenance(new)
    assert not instance.matches_provenance(old)
    assert results and {item['PerformanceText'] for item in results} == {'team collaboration', 'Spanish ventas support'}


def test_search_api_is_fail_closed_for_missing_backend_and_stale_index():
    source = frame()
    current = provenance(source)
    app = FastAPI()

    unavailable = state_for(source, None, current)
    app.dependency_overrides[search_routes.get_app_state] = lambda: unavailable
    app.include_router(search_routes.router)
    with TestClient(app) as client:
        response = client.post('/api/search?query=python')
        assert response.status_code == 400
        assert 'unavailable' in response.json()['detail'].lower()

    instance = engine()
    build(instance, source, {**current, 'generation': 'stale'})
    stale = state_for(source, instance, current)
    app = FastAPI()
    app.dependency_overrides[search_routes.get_app_state] = lambda: stale
    app.include_router(search_routes.router)
    with TestClient(app) as client:
        response = client.post('/api/search?query=python')
        assert response.status_code == 409
        assert 'active dataset snapshot' in response.json()['detail']
        status = client.get('/api/search/status')
        assert status.status_code == 200
        assert status.json()['available'] is False


def test_search_api_strips_raw_worker_identifiers_and_returns_safe_provenance():
    source = frame()
    current = provenance(source)
    instance = engine()
    build(instance, source, current)
    state = state_for(source, instance, current)
    app = FastAPI()
    app.dependency_overrides[search_routes.get_app_state] = lambda: state
    app.include_router(search_routes.router)
    with TestClient(app) as client:
        response = client.post('/api/search?query=python')
    assert response.status_code == 200, response.text
    body = response.json()
    assert 'employee_id' not in body['results'][0]
    assert 'E-sensitive-' not in response.text
    assert body['provenance']['dataset_id'] == 'ds-current'


def test_search_api_preserves_no_store_boundary_for_sensitive_results():
    source = frame()
    current = provenance(source)
    instance = engine()
    build(instance, source, current)
    state = state_for(source, instance, current)
    app = FastAPI()

    @app.middleware('http')
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    app.dependency_overrides[search_routes.get_app_state] = lambda: state
    app.include_router(search_routes.router)
    with TestClient(app) as client:
        response = client.post('/api/search?query=python')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'


def test_search_prepare_binds_current_snapshot_and_keeps_worker_ids_out_of_index(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import search as search_routes

    source = frame()
    current = provenance(source)
    captured = {}

    class FakeVectorEngine:
        dimension = 3

        def __init__(self):
            self.metadata = []
            self.index_provenance = None

        def build_index(self, texts, metadata, *, provenance):
            captured['texts'] = texts
            captured['metadata'] = metadata
            self.metadata = list(metadata)
            self.index_provenance = dict(provenance)

        def is_initialized(self):
            return bool(self.metadata)

        def matches_provenance(self, value):
            return self.index_provenance == value

    state = state_for(source, None, current)
    app = FastAPI()
    app.dependency_overrides[search_routes.get_app_state] = lambda: state
    app.include_router(search_routes.router)
    # The isolated route test does not install the production identity
    # middleware; the full enterprise controls matrix covers that boundary.
    monkeypatch.setattr(search_routes, 'require_permission', lambda request, permission: None)
    monkeypatch.setattr(search_routes, '_new_vector_engine', FakeVectorEngine)
    with TestClient(app) as client:
        response = client.post('/api/search/prepare')
        status = client.get('/api/search/status')
    assert response.status_code == 200, response.text
    assert status.status_code == 200, status.text
    assert response.json()['available'] is True
    assert response.json()['indexed_records'] == len(captured['texts'])
    assert status.json()['state'] == 'ready'
    assert all('EmployeeID' not in item for item in captured['metadata'])
    assert state.vector_engine.matches_provenance(current)
