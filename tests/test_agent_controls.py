"""Regression tests for PeopleOS agent operational controls."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from api.security import _bearer_token
from src.agent.audit import AgentAuditLogger
from src.agent.evidence import ToolResult, ToolResultStatus
from src.agent.people_tools import FairnessOutcomeTool
from src.agent.tools import ToolContext


def test_bearer_token_parser_rejects_non_bearer_and_empty_tokens():
    assert _bearer_token(None) is None
    assert _bearer_token("") is None
    assert _bearer_token("Basic abc") is None
    assert _bearer_token("Bearer ") is None
    assert _bearer_token("Bearer secret") == "secret"


def test_agent_audit_stores_question_hash_not_plaintext_or_evidence(tmp_path):
    path = tmp_path / "audit.jsonl"
    logger = AgentAuditLogger(str(path))
    question = "Which employees are most likely to leave?"
    result = ToolResult(
        tool_id="workforce.summary",
        status=ToolResultStatus.SUCCESS,
        summary="safe aggregate summary",
    )

    logger.record(
        request_id="pia_test",
        question=question,
        status="complete",
        confidence=0.9,
        tools_used=["workforce.summary"],
        tool_results=[result],
        model=None,
        policy_id="hr_advice.v1",
        policy_blocked=False,
    )

    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert question not in raw
    assert "employees" not in raw.lower()
    assert "answer" not in record
    assert "evidence" not in record
    assert record["question_sha256"] == logger.question_hash(question)
    assert record["tool_statuses"] == {"workforce.summary": "success"}


class FakeFairnessEngine:
    min_group_size = 10

    def calculate_demographic_parity(self, outcome_col):
        assert outcome_col == "Attrition"
        # Mirrors the production engine contract: small groups are already
        # suppressed, and the count of suppressed groups is carried forward.
        return pd.DataFrame([
            {
                "attribute": "Gender",
                "group": "A",
                "rate": 0.20,
                "count": 25,
                "disparity": 0.08,
                "parity_ratio": 0.70,
                "suppressed_group_count": 1,
                "overall_known_outcome_count": 29,
                "attribute_observed_count": 29,
                "attribute_coverage": 1.0,
            },
        ])


def test_fairness_tool_preserves_engine_small_group_suppression():
    state = SimpleNamespace(
        fairness_engine=FakeFairnessEngine(),
        raw_df=pd.DataFrame({"Attrition": [0, 1]}),
    )
    result = FairnessOutcomeTool(state).execute(ToolContext(request_id="test"))

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.evidence) == 1
    assert result.evidence[0].metadata["group"] == "A"
    assert result.metadata["suppressed_group_count"] == 1
    assert "suppressed" in " ".join(result.warnings).lower()
    assert all(record["group"] != "B" for record in result.metadata["eligible_groups"])
