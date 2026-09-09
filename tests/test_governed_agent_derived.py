"""Agent-level tests for governed downstream calculations."""
from types import SimpleNamespace

import pandas as pd

from src.agent.derived_analysis import plan_derived_analysis
from src.agent.governed_agent import GovernedPeopleIntelligenceAgent


def workforce():
    return pd.DataFrame({
        'EmployeeID': [f'E{i:02d}' for i in range(20)],
        'Dept': ['Engineering'] * 10 + ['Sales'] * 10,
        'Location': ['Madrid', 'Barcelona'] * 10,
        'Salary': [60000 + i * 1000 for i in range(20)],
        'Tenure': [1 + (i % 5) for i in range(20)],
        'LastRating': [3 + (i % 3) for i in range(20)],
        'Age': [28 + i for i in range(20)],
        'Attrition': [0] * 8 + [1] * 2 + [0] * 8 + [1] * 2,
    })


def state():
    return SimpleNamespace(
        raw_df=workforce(), analytics_engine=None, compensation_engine=None,
        fairness_engine=None, experience_engine=None, structural_engine=None,
        model_metrics={}, risk_scores=None, ml_engine=None, llm_client=None,
    )


def test_common_people_questions_compile_to_typed_specs():
    salary = plan_derived_analysis('What is average salary by department?')
    assert salary is not None and salary.operation == 'group_summary'
    assert salary.measure == 'Salary' and salary.group_by == 'Dept' and salary.statistic == 'mean'

    attrition = plan_derived_analysis('Show observed attrition share by department')
    assert attrition is not None and attrition.population == 'current'
    assert attrition.measure == 'Attrition' and attrition.statistic == 'rate'

    correlation = plan_derived_analysis('What is the correlation between salary and tenure?')
    assert correlation is not None and correlation.operation == 'correlation'


def test_causal_or_consequential_language_never_routes_to_sandbox():
    assert plan_derived_analysis('Why is attrition high by department?') is None
    assert plan_derived_analysis('Rank employees by salary') is None
    assert plan_derived_analysis('Who should we fire by department?') is None


def test_agent_runs_derived_salary_analysis_without_llm_or_row_output():
    agent = GovernedPeopleIntelligenceAgent(state())
    agent.audit.record = lambda **kwargs: None
    answer = agent.investigate('What is average salary by department?', dataset_version='dataset-1')
    assert answer.status == 'complete'
    assert answer.model is None
    assert answer.tools_used == ['workforce.derived_analysis']
    assert 'Mean salary by department' in answer.answer
    assert 'Engineering:' in answer.answer and 'Sales:' in answer.answer
    evidence = answer.evidence.evidence_items()
    assert len(evidence) == 1
    assert evidence[0].kind.value == 'derived'
    assert evidence[0].dataset_version == 'dataset-1'
    assert 'EmployeeID' not in str(evidence[0].value)


def test_agent_runs_observed_attrition_share_on_current_population():
    agent = GovernedPeopleIntelligenceAgent(state())
    agent.audit.record = lambda **kwargs: None
    answer = agent.investigate('Observed attrition share by department', dataset_version='dataset-1')
    assert answer.status == 'complete'
    assert '20.0%' in answer.answer
    evidence = answer.evidence.evidence_items()[0]
    assert evidence.metadata['analysis_spec']['population'] == 'current'


def test_unsupported_freeform_code_request_does_not_enter_runtime():
    assert plan_derived_analysis('Run arbitrary Python to inspect every employee row') is None
