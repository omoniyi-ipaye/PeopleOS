"""Transport contract tests; these do not claim live model validation."""
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


def test_governed_selector_uses_json_mode_and_bounded_generation(monkeypatch):
    from src.safe_llm_client import SafeLLMClient

    transport = Mock()
    transport.list.return_value = {'models': [{'name': 'gemma3:4b', 'digest': 'sha256:fixture'}]}
    transport.generate.return_value = {'response': '{"evidence_ids":["ev_1"],"next_step":"validate_source"}'}
    factory = Mock(return_value=transport)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setattr('src.llm_client.load_config', lambda: {'ollama': {
        'model': 'gemma3:4b', 'host': 'http://localhost:11434', 'timeout': 60,
        'response_max_tokens': 1500}})

    client = SafeLLMClient()
    prompt = 'Select relevant evidence for a governed PeopleOS investigation. REQUEST_DATA: {}'
    result = client.generate(prompt, options={'temperature': 0.0, 'num_predict': 900})

    assert result == '{"evidence_ids":["ev_1"],"next_step":"validate_source"}'
    transport.generate.assert_called_once_with(
        model='gemma3:4b',
        prompt=prompt,
        format='json',
        options={'num_predict': 256, 'temperature': 0.0},
    )


def test_safe_client_non_selector_keeps_standard_generation_path(monkeypatch):
    from src.safe_llm_client import SafeLLMClient

    transport = Mock()
    transport.list.return_value = {'models': [{'name': 'gemma3:4b', 'digest': 'sha256:fixture'}]}
    transport.generate.return_value = {'response': 'ordinary response'}
    factory = Mock(return_value=transport)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setattr('src.llm_client.load_config', lambda: {'ollama': {
        'model': 'gemma3:4b', 'host': 'http://localhost:11434', 'timeout': 60,
        'response_max_tokens': 1500}})

    client = SafeLLMClient()
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
