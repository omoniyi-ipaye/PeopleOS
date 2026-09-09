"""Known-answer tests for the governed downstream analytical runtime."""
import pandas as pd
import pytest

from src.agent.analysis_sandbox import AnalysisSpec, GovernedAnalysisSandbox


def frame():
    rows = []
    for i in range(30):
        rows.append({
            'EmployeeID': f'E{i:03d}',
            'ManagerID': 'E000',
            'Dept': 'Engineering' if i < 15 else 'Sales',
            'Location': 'Madrid' if i % 2 else 'Barcelona',
            'Salary': 60000 + i * 1000,
            'Tenure': 1 + (i % 5),
            'Attrition': 1 if i in {12, 13, 14, 27, 28, 29} else 0,
        })
    return pd.DataFrame(rows)


def test_group_summary_is_deterministic_and_aggregate_only():
    result = GovernedAnalysisSandbox(frame()).run(AnalysisSpec(operation='group_summary', group_by='Dept', measure='Salary', statistic='mean'))
    assert result['available'] is True
    assert result['population'] == 'active'
    groups = {row['group']: row for row in result['output']['groups']}
    # Departed rows are not part of active salary analysis.
    assert groups['Engineering']['eligible_count'] == 12
    assert groups['Sales']['eligible_count'] == 12
    assert groups['Engineering']['value'] == pytest.approx(65500)
    assert 'EmployeeID' not in str(result)


def test_identifier_columns_cannot_be_requested_even_for_aggregates():
    sandbox = GovernedAnalysisSandbox(frame())
    with pytest.raises(ValueError, match='Identifier-like'):
        sandbox.run(AnalysisSpec(operation='group_summary', group_by='EmployeeID', statistic='count'))


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
