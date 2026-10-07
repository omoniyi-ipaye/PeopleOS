"""System read-catalog and privacy-boundary tests for the PeopleOS agent."""

import json
from types import SimpleNamespace

import pandas as pd

from src.agent.access import compact_for_agent_context, profile_frame, redact_for_agent
from src.agent.evidence import EvidenceBundle, EvidenceItem, EvidenceKind, EvidenceSufficiency, ToolResult, ToolResultStatus
from src.agent.orchestrator import PeopleIntelligenceAgent
from src.agent.selector import TOOL_PLAN_PROMPT_PREFIX
from src.agent.system_tools import RuntimeStatusTool
from src.agent.tools import ToolContext
from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine


def _state(llm_client=None):
    frame = pd.DataFrame({
        'EmployeeID': [f'SYN{i:03d}' for i in range(40)],
        'Dept': ['Engineering'] * 20 + ['Operations'] * 20,
        'Attrition': [0] * 16 + [1] * 4 + [0] * 16 + [1] * 4,
        'Salary': [60000] * 20 + [90000] * 20,
        'Tenure': [2.0] * 40,
        'Age': [35.0] * 40,
        'LastRating': [4.0] * 40,
        'Gender': ['A', 'B'] * 20,
    })
    return SimpleNamespace(
        raw_df=frame,
        analytics_engine=AnalyticsEngine(frame),
        compensation_engine=CompensationEngine(frame),
        fairness_engine=None,
        experience_engine=None,
        structural_engine=None,
        succession_engine=None,
        survival_engine=None,
        team_dynamics_engine=None,
        scenario_engine=None,
        quality_of_hire_engine=None,
        sentiment_engine=None,
        vector_engine=None,
        ml_engine=None,
        model_metrics=None,
        risk_scores=None,
        nlp_results=None,
        runtime_provenance={'workspace_id': 'local', 'dataset_id': 'fixture', 'dataset_version': 'v1'},
        features_enabled={'predictive': False, 'nlp': False, 'llm': bool(llm_client)},
        llm_client=llm_client,
    )


def test_complete_snapshot_profile_never_contains_row_values():
    frame = _state().raw_df
    result = profile_frame(frame)
    rendered = json.dumps(result)

    assert result['full_scan'] is True
    assert result['records_scanned'] == 40
    assert result['raw_values_included'] is False
    assert result['employee_rows_included'] is False
    assert 'SYN000' not in rendered
    assert 'EmployeeID' in result['protected_fields']


def test_agent_redaction_removes_identified_rows_but_keeps_safe_aggregates():
    cleaned, redactions = redact_for_agent({
        'records': [{'EmployeeID': 'SYN000', 'risk_score': 0.9}],
        'department_totals': {'Engineering': 20},
    })

    assert redactions >= 1
    assert 'SYN000' not in json.dumps(cleaned)
    assert cleaned['department_totals'] == {'Engineering': 20}

    frame_value, frame_redactions = redact_for_agent(_state().raw_df)
    assert frame_value is None
    assert frame_redactions == 1


def test_model_context_compacts_large_engine_payload_after_redaction():
    compacted, redactions = compact_for_agent_context({
        'summary': 'A useful engine summary remains available to the narrative model.',
        'kaplan_meier_by_dept': [
            {'month': month, 'survival': 0.9, 'EmployeeID': f'SYN{month:03d}'}
            for month in range(500)
        ],
    }, max_chars=900)

    assert len(json.dumps(compacted, separators=(',', ':'))) <= 900
    assert 'SYN000' not in json.dumps(compacted)
    assert 'summary' in compacted
    assert redactions >= 1


def test_catalog_reports_runtime_availability_and_read_contracts():
    descriptors = {
        item['tool_id']: item
        for item in PeopleIntelligenceAgent(_state()).registry.list_descriptors()
    }

    assert descriptors['system.data_profile']['runtime_available'] is True
    assert descriptors['system.data_profile']['data_scope'] == 'schema'
    assert descriptors['workforce.analytics_detail']['runtime_available'] is True
    assert descriptors['workforce.predictive_detail']['availability'] == 'unavailable'
    assert descriptors['workforce.predictive_detail']['runtime_available'] is False
    assert descriptors['workforce.semantic_search']['runtime_available'] is False
    assert all(item['read_only'] for item in descriptors.values())


def test_runtime_status_separates_initialized_engine_from_active_model():
    runtime = _state()
    runtime.ml_engine = object()
    runtime.model_provenance = None
    runtime.vector_engine = SimpleNamespace(is_initialized=True)

    result = RuntimeStatusTool(runtime).execute(ToolContext(request_id='runtime-status'))
    value = result.evidence[0].value

    assert value['active_model'] is False
    assert value['model_ready'] is False
    assert value['semantic_search_prepared'] is True


class SelectingLLM:
    is_available = True
    model = 'controlled-tool-selector'

    def __init__(self):
        self.selector_request = None
        self.prompts = []

    def generate(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if prompt.startswith(TOOL_PLAN_PROMPT_PREFIX):
            self.selector_request = json.loads(prompt.split('REQUEST_DATA:\n', 1)[1])
            return json.dumps({'tool_ids': ['system.data_profile']})
        # Force the normal verified fallback for this test; the assertion is
        # about tool access and redaction, not model prose quality.
        return '{}'


def test_agentic_second_pass_runs_after_analysis_and_uses_only_redacted_context():
    llm = SelectingLLM()
    runtime = _state(llm)
    result = PeopleIntelligenceAgent(runtime).investigate(
        'What should the People team focus on across the workforce?',
        dataset_version='v1',
        agentic=True,
        record_audit=False,
    )

    assert llm.selector_request is not None
    assert 'system.data_profile' not in llm.selector_request['completed_tool_ids']
    assert 'SYN000' not in json.dumps(llm.selector_request)
    assert result.tools_used[-1] == 'system.data_profile'
    assert any(item.metric == 'dataset_profile' for item in result.evidence.evidence_items())
    assert all('SYN000' not in json.dumps(item.model_dump(mode='json')) for item in result.evidence.evidence_items())
    assert len([prompt for prompt in llm.prompts if prompt.startswith(TOOL_PLAN_PROMPT_PREFIX)]) == 1


def test_narrative_uses_short_model_citation_keys_then_restores_canonical_ids():
    class NarrativeLLM:
        is_available = True
        model = 'controlled-narrative-fixture'

        def generate(self, prompt, **kwargs):
            return json.dumps({
                'answer': 'The current workforce has useful verified context [e1].',
                'evidence_ids': ['e1'],
                'next_step': 'review_coverage',
            })

    llm = NarrativeLLM()
    agent = PeopleIntelligenceAgent(_state(llm))
    canonical = EvidenceItem(
        kind=EvidenceKind.OBSERVED,
        claim='Current active employee count: 40',
        source_tool='workforce.summary',
        metric='headcount',
        value=40,
    )
    bundle = EvidenceBundle(
        question='What is current headcount?',
        tool_results=[ToolResult(
            tool_id='workforce.summary',
            status=ToolResultStatus.SUCCESS,
            summary='Summary available.',
            evidence=[canonical],
        )],
        sufficiency=EvidenceSufficiency.SUFFICIENT,
    )
    result = agent._synthesize('What is current headcount?', 'baseline workforce context', bundle, required_metrics=['headcount'])

    assert result.mode == 'grounded_llm'
    assert '[e1]' not in result.answer
    assert result.cited_evidence_ids
    assert result.cited_evidence_ids[0].startswith('ev_')
    assert result.cited_evidence_ids[0] in result.answer


def test_failed_optional_selection_returns_verified_result_without_a_second_model_wait():
    class FailingLLM:
        is_available = True
        model = 'controlled-timeout-fixture'

        def __init__(self):
            self.calls = 0

        def generate(self, _prompt, **_kwargs):
            self.calls += 1
            return 'not-json'

    llm = FailingLLM()
    result = PeopleIntelligenceAgent(_state(llm)).investigate(
        'What should the People team focus on across the workforce?',
        dataset_version='v1',
        agentic=True,
        record_audit=False,
    )

    assert llm.calls == 1
    assert result.model is None
    assert 'optional AI exploration did not complete' in result.answer
    assert any('initial governed plan was retained' in warning for warning in result.warnings)
