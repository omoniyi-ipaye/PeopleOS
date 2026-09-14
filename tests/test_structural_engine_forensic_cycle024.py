"""Cycle 024 forensic contracts for descriptive StructuralEngine behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import structural as routes
from src.structural_engine import StructuralEngine


def workforce(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        'EmployeeID': [f'E{i}' for i in range(rows)],
        'Attrition': [0] * rows,
        'Dept': ['A'] * rows,
        'Tenure': [4.0] * rows,
        'YearsInCurrentRole': [2.0] * rows,
        'YearsSinceLastPromotion': [1.0] * rows,
        'JobLevel': [2] * rows,
        'ManagerID': [None] * rows,
    })


def client_for(state: SimpleNamespace) -> TestClient:
    app = FastAPI()
    app.dependency_overrides[routes.get_app_state] = lambda: state
    app.include_router(routes.router)
    return TestClient(app)


def test_engine_does_not_mutate_source_and_returns_explicit_quality_counts():
    frame = workforce(4)
    original = frame.copy(deep=True)
    engine = StructuralEngine(frame)
    engine.analyze_all()
    pd.testing.assert_frame_equal(frame, original)
    quality = engine._get_data_quality()
    assert quality['source_population'] == 4
    assert quality['role_duration_observations'] == 4
    assert quality['promotion_duration_observations'] == 4


def test_missing_role_duration_is_unavailable_not_zero():
    frame = workforce(3).assign(Tenure=[3.0, np.nan, 0.0], YearsInCurrentRole=[1.0, 1.0, 0.0])
    result = StructuralEngine(frame).calculate_stagnation_index()
    assert result['StagnationIndex'].notna().sum() == 1
    assert result.loc[1, 'StagnationCategory'] == 'Unavailable'
    assert result.loc[2, 'StagnationCategory'] == 'Unavailable'


def test_role_duration_above_tenure_is_unavailable():
    frame = workforce(2).assign(Tenure=[3.0, 3.0], YearsInCurrentRole=[4.0, -1.0])
    result = StructuralEngine(frame).calculate_stagnation_index()
    assert result['StagnationIndex'].isna().all()
    assert set(result['StagnationCategory']) == {'Unavailable'}


def test_future_hire_date_relative_to_snapshot_invalidates_role_duration():
    frame = workforce(2).assign(
        HireDate=['2026-01-01', '2024-01-01'], SnapshotDate=['2025-01-01'] * 2
    )
    result = StructuralEngine(frame)
    scores = result.calculate_stagnation_index()
    assert scores.loc[0, 'StagnationIndex'] is np.nan or pd.isna(scores.loc[0, 'StagnationIndex'])
    assert scores.loc[1, 'StagnationIndex'] == pytest.approx(0.5)
    assert result._get_data_quality()['invalid_hire_dates'] == 1


def test_invalid_date_lexeme_is_disclosed_and_fails_closed():
    frame = workforce(2).assign(HireDate=['not-a-date', '2024-01-01'], SnapshotDate=['2025-01-01'] * 2)
    engine = StructuralEngine(frame)
    assert pd.isna(engine.calculate_stagnation_index().loc[0, 'StagnationIndex'])
    assert engine._get_data_quality()['date_parse_failures']['HireDate'] == 1


def test_promotion_before_hire_is_invalid_for_promotion_duration():
    frame = workforce(2).assign(
        HireDate=['2020-01-01', '2020-01-01'],
        PromotionDate=['2019-01-01', '2023-01-01'],
        SnapshotDate=['2025-01-01'] * 2,
        YearsSinceLastPromotion=[2.0, 2.0],
    )
    engine = StructuralEngine(frame)
    assert pd.isna(engine.df.loc[0, 'YearsSinceLastPromotion'])
    assert engine.df.loc[1, 'YearsSinceLastPromotion'] == pytest.approx(2.0)
    assert engine._get_data_quality()['invalid_promotion_dates'] == 1


def test_promotion_after_snapshot_is_invalid():
    frame = workforce(2).assign(
        HireDate=['2020-01-01'] * 2,
        PromotionDate=['2026-01-01', '2023-01-01'],
        SnapshotDate=['2025-01-01'] * 2,
        YearsSinceLastPromotion=[0.0, 2.0],
    )
    engine = StructuralEngine(frame)
    assert pd.isna(engine.df.loc[0, 'YearsSinceLastPromotion'])
    assert engine._get_data_quality()['invalid_promotion_dates'] == 1


def test_promotion_duration_mismatch_with_valid_date_is_not_silently_accepted():
    frame = workforce(2).assign(
        HireDate=['2020-01-01'] * 2,
        PromotionDate=['2023-01-01'] * 2,
        SnapshotDate=['2025-01-01'] * 2,
        YearsSinceLastPromotion=[0.0, 2.0],
    )
    engine = StructuralEngine(frame)
    assert pd.isna(engine.df.loc[0, 'YearsSinceLastPromotion'])
    assert engine.df.loc[1, 'YearsSinceLastPromotion'] == pytest.approx(2.0)
    assert engine._get_data_quality()['promotion_duration_date_mismatches'] == 1


def test_all_missing_promotion_observations_fail_closed():
    result = StructuralEngine(workforce(20).assign(YearsSinceLastPromotion=np.nan)).get_promotion_bottlenecks()
    assert result['available'] is False
    assert 'valid' in result['reason'].lower()


def test_reporting_links_exclude_self_and_dangling_links_with_counts():
    frame = workforce(5).assign(ManagerID=['E1', 'E1', 'E1', 'missing', None])
    engine = StructuralEngine(frame)
    spans = engine.calculate_span_of_control()
    assert spans.set_index('ManagerID').loc['E1', 'DirectReports'] == 2
    assert not spans['ManagerID'].eq('missing').any()
    quality = engine._get_data_quality()['reporting_links']
    assert quality['self_links_excluded'] == 1
    assert quality['dangling_links_excluded'] == 1
    assert quality['valid_links_used'] == 2


def test_reporting_cycle_links_are_excluded_instead_of_counted():
    frame = workforce(5).assign(ManagerID=['E1', 'E0', None, None, None])
    engine = StructuralEngine(frame)
    assert engine.calculate_span_of_control().empty
    quality = engine._get_data_quality()['reporting_links']
    assert quality['cycle_links_excluded'] == 2
    assert quality['valid_links_used'] == 0


def test_acyclic_reporting_links_are_counted_once():
    frame = workforce(5).assign(ManagerID=['E1', 'E2', 'E3', 'E4', None])
    spans = StructuralEngine(frame).calculate_span_of_control()
    assert spans.set_index('ManagerID')['DirectReports'].to_dict() == {'E1': 1, 'E2': 1, 'E3': 1, 'E4': 1}


@pytest.mark.parametrize(
    ('reports', 'expected'),
    [(3, 'Under-Leveraged'), (4, 'Optimal'), (8, 'Optimal'), (9, 'Stretched'),
     (11, 'Stretched'), (12, 'Overloaded'), (14, 'Overloaded'), (15, 'Critical')],
)
def test_span_thresholds_have_explicit_inclusive_boundaries(reports, expected):
    frame = workforce(reports + 1)
    frame['ManagerID'] = [None] + ['E0'] * reports
    span = StructuralEngine(frame).calculate_span_of_control()
    assert span.loc[span['ManagerID'].eq('E0'), 'SpanCategory'].iloc[0] == expected


def test_span_no_longer_emits_numeric_burnout_proxy():
    frame = workforce(16)
    frame['ManagerID'] = ['E0'] * 15 + [None]
    span = StructuralEngine(frame).calculate_span_of_control()
    assert 'BurnoutRiskScore' not in span.columns
    assert span.loc[span['ManagerID'].eq('E0'), 'MetricSemantics'].iloc[0].endswith('not_burnout_prediction')


def test_canonical_reporting_analysis_has_no_manager_records_or_burnout_claim():
    frame = workforce(16)
    frame['ManagerID'] = ['E0'] * 15 + [None]
    result = StructuralEngine(frame)._analyze_reporting_span()
    assert 'at_risk_managers' not in result
    assert result['summary']['at_risk_count'] == 1
    assert 'burnout' in ' '.join(result['scientific_limits']).lower()
    assert result['metric_semantics'].startswith('recorded_reporting_span_screening')


def test_legacy_burnout_method_is_a_safe_compatibility_alias():
    result = StructuralEngine(workforce(2)).analyze_manager_burnout_risk()
    assert result['available'] is False
    assert result['metric_semantics'] == 'recorded_reporting_span_screening_not_burnout_prediction'


def test_role_duration_hotspots_report_coverage_and_not_employee_ranking():
    frame = workforce(10).assign(Tenure=[4.0] * 10, YearsInCurrentRole=[4.0] * 10)
    result = StructuralEngine(frame).identify_stagnation_hotspots()
    assert result['population']['coverage'] == 1.0
    assert result['metric_semantics'] == 'aggregate_role_duration_screening_not_employee_performance_determination'


def test_missing_department_is_an_explicit_unknown_promotion_cohort():
    frame = workforce(20).assign(Dept=[None] * 10 + ['A'] * 10, YearsSinceLastPromotion=[1.0] * 20)
    result = StructuralEngine(frame).get_promotion_bottlenecks()
    assert result['population']['eligible_population'] == 20
    assert result['data_quality']['promotion_duration_observations'] == 20


def test_zero_promotion_baseline_does_not_create_infinite_relative_gap():
    frame = workforce(20).assign(Dept=['A'] * 10 + ['B'] * 10, YearsSinceLastPromotion=[0.0] * 20)
    result = StructuralEngine(frame).get_promotion_bottlenecks()
    json.dumps(result, allow_nan=False)
    assert result['bottlenecks'] == []


def test_promotion_output_labels_observational_limits():
    frame = workforce(40).assign(Gender=['A'] * 20 + ['B'] * 20)
    result = StructuralEngine(frame).audit_promotion_velocity()
    assert 'not_promotion_readiness' in result['metric_semantics']
    assert any('unadjusted' in result['methodology'].lower() for _ in [0])


def test_structural_analysis_route_redacts_identifiers_and_sets_governance_boundary():
    frame = workforce(16)
    frame['ManagerID'] = ['E0'] * 15 + [None]
    state = SimpleNamespace(
        structural_engine=StructuralEngine(frame),
        has_data=lambda: True,
        load_from_database=lambda: False,
    )
    with client_for(state) as client:
        response = client.get('/api/structural/analysis')
    assert response.status_code == 200
    body = response.json()
    assert 'E0' not in response.text
    assert 'not burnout' in body['governance']
    assert 'promotion-readiness' in body['governance']


def test_span_route_redacts_records_and_keeps_non_causal_semantics():
    frame = workforce(16)
    frame['ManagerID'] = ['E0'] * 15 + [None]
    state = SimpleNamespace(
        structural_engine=StructuralEngine(frame),
        has_data=lambda: True,
        load_from_database=lambda: False,
    )
    with client_for(state) as client:
        response = client.get('/api/structural/span-of-control/analysis')
    assert response.status_code == 200
    assert 'E0' not in response.text
    assert 'causal_health_finding' in response.json()['metric_semantics']


def test_deprecated_individual_structural_routes_remain_blocked():
    state = SimpleNamespace(
        structural_engine=StructuralEngine(workforce(2)),
        has_data=lambda: True,
        load_from_database=lambda: False,
    )
    with client_for(state) as client:
        assert client.get('/api/structural/stagnation').status_code == 403
        assert client.get('/api/structural/span-of-control').status_code == 403
        assert client.get('/api/structural/employee/E0/stagnation').status_code == 403
