"""Contracts for the optional local-only column mapping assistant."""

import pandas as pd

from src.column_mapping_llm import _parse_json_response, suggest_column_mappings


def test_parser_accepts_fenced_json_with_short_preamble():
    payload = _parse_json_response(
        'Here is the safe schema suggestion.\n```json\n'
        '{"mappings":[{"source":"worker_code","target":"EmployeeID"}]}\n'
        '```\n'
    )
    assert payload[0]['target'] == 'EmployeeID'


def test_local_mapping_prompt_contains_metadata_not_cell_values(monkeypatch):
    class FakeClient:
        is_available = True

        def __init__(self):
            self.prompt = None
            self.kwargs = None

        def generate(self, prompt, **kwargs):
            self.prompt = prompt
            self.kwargs = kwargs
            return '{"mappings":[{"source":"worker_code","target":"EmployeeID","confidence":0.9,"reason":"Identifier"}]}'

    client = FakeClient()
    monkeypatch.setattr('src.column_mapping_llm.LLMClient', lambda **kwargs: client)
    frame = pd.DataFrame({'worker_code': ['private-id-1', 'private-id-2'], 'Dept': ['People', 'Finance']})

    result = suggest_column_mappings(frame, existing_mapping={'Dept': 'Dept'})

    assert result['used'] is True
    assert result['mappings'] == {'worker_code': 'EmployeeID'}
    assert client.kwargs == {'format': 'json', 'options': {'temperature': 0, 'num_predict': 1200}}
    assert 'private-id-1' not in client.prompt
    assert 'Finance' not in client.prompt
