"""Known-answer tests for the governed downstream analytical runtime."""
import pandas as pd
import pytest

from src.agent.analysis_sandbox import AnalysisSpec, CohortFilter, GovernedAnalysisSandbox


def frame():
    rows = []
    for i in range(30):
        rows.append({
            'EmployeeID': f'E{i:03d}',
            'ManagerID': 'E000',
            'Dept': 'Engineering' if i < 15 else 'Sales',
            'Location': 'Madrid' if i % 2 else 'Barcelona',
            'JobLevel': 'L3' if i % 3 else 'L4',
            'JobTitle': 'Analyst' if i % 4 else 'Manager',
            'Gender': 'Female' if i % 2 else 'Male',
            'Salary': 60000 + i * 1000,
            'Age': 26 + (i % 10),
            'Tenure': 1 + (i % 5),
            'LastRating': 3 + (i % 3),
            'Attrition': 1 if i in {12, 13, 14, 27, 28, 29} else 0,
        })
    return pd.DataFrame(rows)


def test_group_summary_is_deterministic_and_aggregate_only():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='group_summary', group_by='Dept', measure='Salary', statistic='mean'))
    assert result['available'] is True
    assert result['population'] == 'active'
    groups = {row['group']: row for row in result['output']['groups']}
    assert groups['Engineering']['eligible_count'] == 12
    assert groups['Sales']['eligible_count'] == 12
    assert groups['Engineering']['value'] == pytest.approx(65500)
    assert 'EmployeeID' not in str(result)


def test_identifier_columns_cannot_be_requested_even_for_aggregates():
    sandbox = GovernedAnalysisSandbox(frame())
    with pytest.raises(ValueError, match='Identifier-like'):
        sandbox.run(AnalysisSpec(operation='group_summary', group_by='EmployeeID', statistic='count'))


def test_identifier_columns_cannot_be_used_as_filters():
    sandbox = GovernedAnalysisSandbox(frame())
    with pytest.raises(ValueError, match='Identifier-like'):
        sandbox.run(AnalysisSpec(operation='group_summary', group_by='Dept', statistic='count', filters=[CohortFilter(column='EmployeeID', operator='eq', value='E001')]))


def test_small_groups_are_suppressed_instead_of_exposed():
    data = frame()
    data.loc[:2, 'Location'] = 'Tiny office'
    result = GovernedAnalysisSandbox(data).run(AnalysisSpec(operation='group_summary', group_by='Location', statistic='count'))
    assert result['available'] is True
    labels = {row['group'] for row in result['output']['groups']}
    assert 'Tiny office' not in labels
    assert result['output']['suppressed_groups'] >= 1


def test_current_population_can_calculate_observed_binary_rate_by_group():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='group_summary', population='current', group_by='Dept', measure='Attrition', statistic='rate'))
    groups = {row['group']: row for row in result['output']['groups']}
    assert groups['Engineering']['value'] == pytest.approx(3 / 15)
    assert groups['Sales']['value'] == pytest.approx(3 / 15)


def test_compare_groups_returns_support_and_difference_not_people():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='compare_groups', group_by='Dept', measure='Salary', statistic='mean', group_a='Engineering', group_b='Sales'))
    assert result['available'] is True
    assert len(result['output']['groups']) == 2
    assert result['output']['difference_b_minus_a'] > 0
    assert all(row['measured_count'] >= 5 for row in result['output']['groups'])


def test_correlation_requires_support_and_reports_n_and_p_value():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='correlation', measure='Salary', second_measure='Tenure'))
    assert result['available'] is True
    assert result['output']['paired_observations'] == 24
    assert 0 <= result['output']['p_value'] <= 1
    assert result['output']['causal'] is False


def test_crosstab_suppresses_small_cells():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='crosstab', group_by='Dept', second_group_by='Location'))
    assert result['available'] is True
    for cell in result['output']['cells']:
        assert cell['count'] is None or cell['count'] == 0 or cell['count'] >= 5


def test_stacked_categorical_and_numeric_filters_define_exact_cohort():
    spec = AnalysisSpec(
        operation='group_summary',
        group_by='Dept',
        measure='Salary',
        statistic='mean',
        filters=[
            CohortFilter(column='Dept', operator='eq', value='Engineering'),
            CohortFilter(column='Location', operator='eq', value='Madrid'),
            CohortFilter(column='Tenure', operator='lte', value=4),
        ],
    )
    result = GovernedAnalysisSandbox(frame()).run(spec)
    assert result['available'] is True
    assert result['filter_context']['population_after_filters'] >= 5
    assert result['filter_context']['excluded_by_filters'] > 0
    assert all(row['eligible_count'] >= 5 for row in result['output']['groups'])
    assert 'EmployeeID' not in str(result)


def test_filtered_cohort_below_privacy_floor_is_unavailable():
    spec = AnalysisSpec(
        operation='group_summary',
        group_by='Dept',
        statistic='count',
        filters=[
            CohortFilter(column='Dept', operator='eq', value='Engineering'),
            CohortFilter(column='Location', operator='eq', value='Madrid'),
            CohortFilter(column='JobTitle', operator='eq', value='Manager'),
            CohortFilter(column='Tenure', operator='lt', value=3),
        ],
    )
    result = GovernedAnalysisSandbox(frame()).run(spec)
    assert result['available'] is False
    assert 'fewer than 5' in result['reason'] or 'No records match' in result['reason']