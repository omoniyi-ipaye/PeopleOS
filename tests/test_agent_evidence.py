"""Tests for canonical PeopleOS agent evidence contracts."""

import numpy as np

from src.agent.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    ToolResult,
    ToolResultStatus,
)


def test_evidence_bundle_flattens_tool_evidence():
    item = EvidenceItem(
        kind=EvidenceKind.DERIVED,
        claim="Engineering observed attrition share is above baseline",
        source_tool="workforce.attrition",
        value=0.187,
        metric="observed_attrition_share",
        confidence=0.91,
    )
    result = ToolResult(
        tool_id="workforce.attrition",
        status=ToolResultStatus.SUCCESS,
        summary="Observed attrition analysis completed",
        evidence=[item],
    )
    bundle = EvidenceBundle(
        question="Why is Engineering attrition elevated?",
        tool_results=[result],
        overall_confidence=0.91,
    )

    assert bundle.evidence_items() == [item]
    assert bundle.has_failures() is False


def test_evidence_bundle_detects_blocked_or_failed_tools():
    bundle = EvidenceBundle(
        question="What action should we take?",
        tool_results=[
            ToolResult(
                tool_id="policy.employment_action",
                status=ToolResultStatus.BLOCKED,
                summary="Action blocked by policy",
            )
        ],
    )

    assert bundle.has_failures() is True


def test_evidence_contract_normalizes_numpy_scalars_recursively():
    item = EvidenceItem(
        kind=EvidenceKind.DERIVED,
        claim="Fairness screening completed",
        source_tool="workforce.fairness",
        value=np.bool_(True),
        metadata={
            "passes_screen": np.bool_(False),
            "sample_size": np.int64(24),
            "ratio": np.float64(0.82),
            "nested": [np.bool_(True), {"count": np.int32(3)}],
        },
    )
    result = ToolResult(
        tool_id="workforce.fairness",
        status=ToolResultStatus.SUCCESS,
        summary="Fairness screening completed",
        evidence=[item],
        metadata={"eligible": np.bool_(True)},
    )

    payload = result.model_dump(mode="json")
    assert payload["evidence"][0]["value"] is True
    assert payload["evidence"][0]["metadata"]["passes_screen"] is False
    assert payload["evidence"][0]["metadata"]["sample_size"] == 24
    assert payload["evidence"][0]["metadata"]["ratio"] == 0.82
    assert payload["evidence"][0]["metadata"]["nested"] == [True, {"count": 3}]
    assert payload["metadata"]["eligible"] is True
