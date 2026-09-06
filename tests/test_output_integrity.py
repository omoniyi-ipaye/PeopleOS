"""Regression tests for PeopleOS output-integrity contracts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from api.routes.experience import _has_measured_signals, _safe_index
from api.routes.scenario import _sanitize_result
from src.agent.aggregator import EvidenceAggregator
from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.experience_engine import ExperienceEngine
from src.fairness_engine import FairnessEngine
from src.population import active_population, normalize_attrition, resolve_current_population
from src.preprocessor import Preprocessor
from src.serialization import json_safe


def _base_rows(n=40):
    return pd.DataFrame({
        'EmployeeID': [f'E{i:03d}' for i in range(n)],
        'Dept': ['A'] * (n // 2) + ['B'] * (n - n // 2),
        'Tenure': np.linspace(.5, 10, n),
        'Salary': np.linspace(50_000, 100_000, n),
        'LastRating': np.linspace(2.5, 4.5, n),
        'Age': np.linspace(24, 58, n),
        'Gender': ['Female', 'Male'] * (n // 2),
        'Attrition': [0] * (n - 8) + [1] * 8,
    })


def test_latest_snapshot_is_single_current_employee_observation():
    older = _base_rows(20)
    older['SnapshotDate'] = '2025-12-31'
    newer = older.copy()
    newer['SnapshotDate'] = '2026-06-30'
    newer.loc[0, 'Salary'] = 123_456
    history = pd.concat([older, newer], ignore_index=True)

    current, resolution = resolve_current_population(history)
    assert len(current) == 20
    assert resolution.source_rows == 40
    assert resolution.snapshot_history is True
    assert resolution.as_of_date == '2026-06-30'
    assert current.loc[current.EmployeeID == 'E000', 'Salary'].iloc[0] == 123_456


def test_attrition_normalization_fails_unknown_labels_to_missing():
    values = pd.Series(['Active', 'left', '0', '1', 'maybe'])
    normalized = normalize_attrition(values)
    assert normalized.iloc[:4].tolist() == [0, 1, 0, 1]
    assert pd.isna(normalized.iloc[4])


def test_headcount_is_active_population_not_record_count():
    data = _base_rows(40)
    engine = AnalyticsEngine(data)
    assert engine.get_record_count() == 40
    assert engine.get_headcount() == 32
    assert len(active_population(data)) == 32
    assert engine.get_observed_attrition_share() == 0.2
    summary = engine.get_summary_statistics()
    assert summary['headcount'] == 32
    assert summary['record_count'] == 40
    assert summary['turnover_rate_semantics'] == 'observed_attrition_share_not_period_turnover'


def test_current_compensation_excludes_departed_and_invalid_salary():
    data = _base_rows(40)
    data.loc[0, 'Salary'] = -1
    data.loc[data['Attrition'] == 1, 'Salary'] = 1_000_000
    engine = CompensationEngine(data)
    summary = engine.get_compensation_summary()
    assert summary['headcount'] == 31
    assert summary['max_salary'] < 1_000_000
    assert summary['population'] == 'current_active_employees_with_valid_positive_salary'
    equity = engine.calculate_pay_equity_score()
    assert (equity['MetricSemantics'] == 'salary_dispersion_consistency_not_adjusted_pay_equity').all()


def test_four_fifths_uses_favorable_retention_rate_and_suppresses_small_groups():
    a = pd.DataFrame({'EmployeeID': [f'A{i}' for i in range(20)], 'Gender': ['A'] * 20, 'Attrition': [1] * 2 + [0] * 18})
    b = pd.DataFrame({'EmployeeID': [f'B{i}' for i in range(20)], 'Gender': ['B'] * 20, 'Attrition': [1] * 8 + [0] * 12})
    tiny = pd.DataFrame({'EmployeeID': [f'C{i}' for i in range(4)], 'Gender': ['C'] * 4, 'Attrition': [0] * 4})
    engine = FairnessEngine(pd.concat([a, b, tiny], ignore_index=True))
    result = engine.calculate_four_fifths_rule('Attrition', favorable=False)
    gender = result[result['attribute'] == 'Gender'].set_index('group')
    assert 'C' not in gender.index
    assert abs(gender.loc['A', 'favorable_rate'] - .9) < 1e-9
    assert abs(gender.loc['B', 'favorable_rate'] - .6) < 1e-9
    assert abs(gender.loc['B', 'adverse_impact_ratio'] - (2 / 3)) < 1e-9
    assert bool(gender.loc['B', 'passes_4_5_rule']) is False


def test_preprocessor_reuses_training_statistics_and_unknown_category():
    train = _base_rows(30).drop(columns=['Attrition'])
    train['Dept'] = ['A'] * 15 + ['B'] * 15
    train.loc[0, 'Salary'] = np.nan
    p = Preprocessor()
    _, metadata = p.fit_transform(train, target_column='__no_target__')
    assert metadata['fit_scope'] == 'training_population'
    train_salary_median = p.impute_values['Salary']

    holdout = train.iloc[:3].copy()
    holdout['Dept'] = ['NEW', 'NEW', 'NEW']
    holdout['Salary'] = [np.nan, np.nan, np.nan]
    transformed = p.transform(holdout, target_column='__no_target__')
    assert p.impute_values['Salary'] == train_salary_median
    assert transformed['Dept'].notna().all()


def test_evidence_quality_penalizes_assumptions_and_partial_tools():
    assumed = EvidenceItem(
        kind=EvidenceKind.ASSUMED,
        claim='Assumed replacement cost multiplier',
        source_tool='scenario.assumption',
        value=1.5,
        metric='replacement_cost_multiplier',
        confidence=1.0,
    )
    partial = ToolResult(
        tool_id='scenario.assumption',
        status=ToolResultStatus.PARTIAL,
        summary='Only configured assumptions are available',
        evidence=[assumed],
        warnings=['No validated causal estimate'],
    )
    bundle = EvidenceAggregator().aggregate('What will happen?', [partial])
    assert bundle.sufficiency.value == 'insufficient'
    assert bundle.overall_confidence < 0.5
    assert bundle.coverage_score < 0.5


def test_evidence_contradiction_requires_same_scope_and_material_difference():
    one = EvidenceItem(kind=EvidenceKind.DERIVED, claim='A', source_tool='one', value=.20, metric='attrition_share', metadata={'department': 'A'})
    rounding = EvidenceItem(kind=EvidenceKind.DERIVED, claim='A2', source_tool='two', value=.201, metric='attrition_share', metadata={'department': 'A'})
    other_scope = EvidenceItem(kind=EvidenceKind.DERIVED, claim='B', source_tool='three', value=.40, metric='attrition_share', metadata={'department': 'B'})
    result = [
        ToolResult(tool_id='one', status=ToolResultStatus.SUCCESS, summary='ok', evidence=[one]),
        ToolResult(tool_id='two', status=ToolResultStatus.SUCCESS, summary='ok', evidence=[rounding]),
        ToolResult(tool_id='three', status=ToolResultStatus.SUCCESS, summary='ok', evidence=[other_scope]),
    ]
    bundle = EvidenceAggregator().aggregate('Compare', result)
    assert bundle.contradictions == []


def test_experience_api_refuses_proxy_derived_exi_without_measured_signals():
    data = _base_rows(40)
    engine = ExperienceEngine(data)
    state = SimpleNamespace(experience_engine=engine, raw_df=data)
    assert _has_measured_signals(state) is False
    result = _safe_index(state)
    assert result.available is False
    assert result.overall_exi is None
    assert 'will not infer experience' in (result.reason or '')


def test_scenario_output_is_forced_to_exploratory_semantics():
    result = SimpleNamespace(
        assumptions=['Average salary: $80,000'],
        confidence_level='High',
        confidence_score=.95,
        recommendation='Proceed',
        alternative_actions=['Target high-risk employees only', 'Phase implementation'],
    )
    sanitized = _sanitize_result(result)
    assert sanitized.confidence_level == 'Exploratory'
    assert sanitized.confidence_score <= .5
    assert 'validate' in sanitized.recommendation.lower()
    assert all('high-risk' not in item.lower() for item in sanitized.alternative_actions)


def test_json_safe_normalizes_numpy_scalars_and_nonfinite_numbers_recursively():
    payload = {
        'healthy': np.bool_(True),
        'count': np.int64(7),
        'score': np.float64(.75),
        'undefined': float('nan'),
        'positive_infinity': float('inf'),
        'negative_infinity': np.float64('-inf'),
        'nested': [np.bool_(False), {'value': np.float32(2.5)}],
    }
    safe = json_safe(payload)
    assert safe['healthy'] is True
    assert safe['count'] == 7
    assert safe['score'] == .75
    assert safe['undefined'] is None
    assert safe['positive_infinity'] is None
    assert safe['negative_infinity'] is None
    assert safe['nested'] == [False, {'value': 2.5}]
