"""Transport contract tests; these do not claim live model validation."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llm_client import LLMClient, LLMClientError


def client_with_models(monkeypatch, models, model='gemma3:4b'):
    transport = Mock()
    transport.list.return_value = models
    factory = Mock(return_value=transport)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setattr('src.llm_client.load_config', lambda: {'ollama': {
        'model': model, 'host': 'http://localhost:11434', 'timeout': 12}})
    return LLMClient(), transport, factory


@pytest.mark.parametrize('models', [[], [{'name': 'gemma3:1b'}]])
def test_reachable_server_without_configured_model_is_unavailable(monkeypatch, models):
    client, transport, factory = client_with_models(monkeypatch, {'models': models})
    assert not client.is_available
    assert 'not installed' in client.unavailable_reason
    with pytest.raises(LLMClientError):
        client.generate('test')
    transport.generate.assert_not_called()


@pytest.mark.parametrize('listing', [
    {'models': [{'name': 'gemma3:4b', 'digest': 'sha256:fixture'}]},
    SimpleNamespace(models=[SimpleNamespace(model='gemma3:4b', digest='sha256:fixture')]),
])
def test_timeout_and_exact_installed_model_digest(monkeypatch, listing):
    client, transport, factory = client_with_models(monkeypatch, listing)
    assert client.is_available
    assert client.model_digest == 'sha256:fixture'
    factory.assert_called_once_with(host='http://localhost:11434', timeout=12)


def test_latest_alias(monkeypatch):
    client, _, _ = client_with_models(monkeypatch, {'models': [{'name': 'gemma3:latest'}]}, 'gemma3')
    assert client.is_available


def test_request_failure_remains_explicit(monkeypatch):
    client, transport, _ = client_with_models(monkeypatch, {'models': [{'name': 'gemma3:4b'}]})
    transport.generate.side_effect = TimeoutError('bounded request timed out')
    with pytest.raises(LLMClientError, match='timed out'):
        client.generate('test')


def safe_client(monkeypatch):
    from src.safe_llm_client import SafeLLMClient

    transport = Mock()
    transport.list.return_value = {'models': [{'name': 'gemma3:4b', 'digest': 'sha256:fixture'}]}
    transport.generate.return_value = {'response': '{"evidence_ids":["ev_1"],"next_step":"validate_source"}'}
    factory = Mock(return_value=transport)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setattr('src.llm_client.load_config', lambda: {'ollama': {
        'model': 'gemma3:4b', 'host': 'http://localhost:11434', 'timeout': 60,
        'response_max_tokens': 1500}})
    return SafeLLMClient(), transport


def test_governed_selector_uses_json_mode_and_bounded_generation(monkeypatch):
    client, transport = safe_client(monkeypatch)
    prompt = 'Select relevant evidence for a governed PeopleOS investigation. REQUEST_DATA: {}'
    result = client.generate(prompt, options={'temperature': 0.0, 'num_predict': 900})

    assert result == '{"evidence_ids":["ev_1"],"next_step":"validate_source"}'
    transport.generate.assert_called_once_with(
        model='gemma3:4b',
        prompt=prompt,
        format='json',
        options={'num_predict': 256, 'temperature': 0.0},
    )


def test_governed_headcount_selector_compacts_irrelevant_same_source_rows(monkeypatch):
    client, transport = safe_client(monkeypatch)
    request = {
        'question': 'What is current headcount?',
        'plan': 'baseline workforce context',
        'required_metrics': ['headcount'],
        'evidence': [
            {'evidence_id': 'ev_headcount', 'claim': 'Current active employee count: 80', 'source_tool': 'workforce.summary', 'limitations': {}},
            {'evidence_id': 'ev_records', 'claim': 'Source record count: 120', 'source_tool': 'workforce.summary', 'limitations': {}},
            {'evidence_id': 'ev_salary', 'claim': 'Average active-employee salary: 70,000', 'source_tool': 'workforce.summary', 'limitations': {}},
            {'evidence_id': 'ev_age', 'claim': 'Average active-employee age: 36.0 years', 'source_tool': 'workforce.summary', 'limitations': {}},
        ],
        'limitations': [],
        'contradictions': [],
    }
    prompt = ('Select relevant evidence for a governed PeopleOS investigation. Rules.\nREQUEST_DATA:\n'
              + json.dumps(request))
    client.generate(prompt, options={'temperature': 0.0})

    sent = transport.generate.call_args.kwargs['prompt']
    sent_request = json.loads(sent.split('\nREQUEST_DATA:\n', 1)[1])
    assert [item['evidence_id'] for item in sent_request['evidence']] == ['ev_headcount']
    assert 'ev_salary' not in sent and 'ev_age' not in sent and 'ev_records' not in sent


def test_selector_compaction_fails_safe_for_unknown_required_metric(monkeypatch):
    client, transport = safe_client(monkeypatch)
    request = {
        'question': 'What is median salary?',
        'plan': 'compensation evidence requested',
        'required_metrics': ['salary_median'],
        'evidence': [
            {'evidence_id': 'ev_headcount', 'claim': 'Current active employee count: 80', 'source_tool': 'workforce.summary', 'limitations': {}},
            {'evidence_id': 'ev_salary', 'claim': 'Average active-employee salary: 70,000', 'source_tool': 'workforce.summary', 'limitations': {}},
        ],
        'limitations': [],
        'contradictions': [],
    }
    prompt = ('Select relevant evidence for a governed PeopleOS investigation. Rules.\nREQUEST_DATA:\n'
              + json.dumps(request))
    client.generate(prompt, options={'temperature': 0.0})
    assert transport.generate.call_args.kwargs['prompt'] == prompt


def test_safe_client_non_selector_keeps_standard_generation_path(monkeypatch):
    client, transport = safe_client(monkeypatch)
    transport.generate.return_value = {'response': 'ordinary response'}
    assert client.generate('ordinary prompt', options={'temperature': 0.5}) == 'ordinary response'
    transport.generate.assert_called_once_with(
        model='gemma3:4b',
        prompt='ordinary prompt',
        options={'num_predict': 1500, 'temperature': 0.5},
    )


def test_acceptance_oracle_rejects_changed_numbers_injected_labels_and_missing_sources():
    from scripts.validate_local_llm import answer_matches_known_values
    headcount = '- Current active employee count: 80 [ev_synthetic_headcount; workforce.summary]'
    span = '- Average manager span of control: 5 [ev_synthetic_span; workforce.organization_structure]'
    assert answer_matches_known_values(headcount)
    assert answer_matches_known_values(headcount + '\n' + span, include_span=True)
    assert not answer_matches_known_values(headcount.replace(': 80 ', ': 81 '))
    assert not answer_matches_known_values(headcount + '\n999999')
    assert not answer_matches_known_values(headcount.replace('Current active employee count', 'Injected label'))
    assert not answer_matches_known_values(headcount, include_span=True)
