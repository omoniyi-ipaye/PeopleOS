"""Regression tests for the Strategic Advisor safety boundary."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes.advisor import _generate_safe


class FakeLLMClient:
    def __init__(self, response: str, is_valid: bool, cleaned: str = ""):
        self.response = response
        self.is_valid = is_valid
        self.cleaned = cleaned or response

    def generate(self, prompt: str) -> str:
        return self.response

    def _validate_response(self, response: str):
        assert response == self.response
        return self.is_valid, self.cleaned


def test_generate_safe_returns_validated_output():
    state = SimpleNamespace(
        llm_client=FakeLLMClient(
            response="raw answer",
            is_valid=True,
            cleaned="safe answer",
        )
    )

    result = _generate_safe(state, "question")

    assert result == "safe answer"


def test_generate_safe_blocks_prohibited_output():
    state = SimpleNamespace(
        llm_client=FakeLLMClient(
            response="Recommend termination immediately",
            is_valid=False,
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        _generate_safe(state, "question")

    assert exc_info.value.status_code == 422
    assert "blocked" in exc_info.value.detail.lower()
