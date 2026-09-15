"""Cycle 019 forensic contracts for unavailable collaboration analytics."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import network as routes
from src.network_engine import NetworkEngine


def workforce() -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': ['E-sensitive-1', 'E-sensitive-2'],
        'Dept': ['People', 'People'],
        'ManagerID': ['M1', 'M1'],
    })


def test_department_and_reporting_data_never_create_collaboration_edges_or_rankings():
    source = workforce()
    original = source.copy(deep=True)
    engine = NetworkEngine(source)
    source.loc[0, 'Dept'] = 'Changed after construction'
    assert engine.graph is None
    assert engine.get_key_influencers(limit=50) == []
    assert engine.get_isolated_employees(limit=50) == []
    summary = engine.get_network_summary()
    assert summary['success'] is False and summary['available'] is False
    assert 'measured collaboration relationships' in summary['reason'].lower()
    assert 'E-sensitive' not in str(summary)
    pd.testing.assert_frame_equal(engine.df, original)


def test_network_api_returns_only_unavailability_and_never_employee_identifiers():
    class State:
        def has_data(self):
            return True

        def load_from_database(self):
            return False

    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = State
    app.include_router(routes.router)
    with TestClient(app) as client:
        summary = client.get('/api/network/summary')
        assert summary.status_code == 200
        assert summary.json()['available'] is False
        assert 'E-sensitive' not in summary.text
        for path in ('/api/network/influencers', '/api/network/isolated'):
            response = client.get(path)
            assert response.status_code == 409
            assert 'observed relationship data' in response.json()['detail'].lower()
            assert 'E-sensitive' not in response.text


def test_network_api_validates_deprecated_limit_without_weakening_unavailable_boundary():
    class State:
        def has_data(self):
            return True

        def load_from_database(self):
            return False

    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = State
    app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.get('/api/network/influencers?limit=0').status_code == 422
        assert client.get('/api/network/isolated?limit=51').status_code == 422

