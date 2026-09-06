"""Tests for the transport-level PeopleOS LLM safety boundary."""

import pytest

from src.agent.policy import HRAdvicePolicy
from src.llm_client import LLMClientError
from src.safe_llm_client import _GuardedOllamaClient


class FakeOllama:
    def __init__(self, text: str):
        self.text = text

    def generate(self, *args, **kwargs):
        return {"response": self.text}

    def list(self):
        return {"models": []}


def test_guarded_transport_allows_neutral_hr_analysis():
    client = _GuardedOllamaClient(
        FakeOllama("The termination rate increased from 4% to 6%; review systemic drivers."),
        HRAdvicePolicy(),
    )
    response = client.generate(model="fake", prompt="analyze")
    assert "termination rate" in response["response"]


def test_guarded_transport_blocks_punitive_recommendation():
    client = _GuardedOllamaClient(
        FakeOllama("You should terminate the highest-risk employee."),
        HRAdvicePolicy(),
    )
    with pytest.raises(LLMClientError):
        client.generate(model="fake", prompt="analyze")


def test_guarded_transport_proxies_non_generation_methods():
    client = _GuardedOllamaClient(FakeOllama("safe"), HRAdvicePolicy())
    assert client.list() == {"models": []}
