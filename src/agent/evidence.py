"""Canonical evidence contracts for PeopleOS agent workflows.

The agent layer must reason over structured evidence rather than opaque prose.
These models are intentionally domain-neutral enough to wrap existing PeopleOS
analytics engines without changing their internal implementations first.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceKind(str, Enum):
    """How strongly an evidence item is grounded in the underlying system."""

    OBSERVED = "observed"
    DERIVED = "derived"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"


class ToolResultStatus(str, Enum):
    """Execution status for a governed PeopleOS tool call."""

    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class EvidenceItem(BaseModel):
    """One traceable fact, metric, inference, or material unknown."""

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex}")
    kind: EvidenceKind
    claim: str
    source_tool: str
    value: Any = None
    metric: Optional[str] = None
    source_ref: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Canonical result returned by every governed PeopleOS agent tool."""

    result_id: str = Field(default_factory=lambda: f"tr_{uuid4().hex}")
    tool_id: str
    status: ToolResultStatus
    summary: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    duration_ms: Optional[float] = Field(default=None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """Evidence assembled for one People Intelligence Agent investigation."""

    bundle_id: str = Field(default_factory=lambda: f"eb_{uuid4().hex}")
    question: str
    tool_results: List[ToolResult] = Field(default_factory=list)
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contradictions: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    verification_notes: List[str] = Field(default_factory=list)

    def evidence_items(self) -> List[EvidenceItem]:
        """Flatten all evidence items while preserving tool execution records."""
        return [
            item
            for result in self.tool_results
            for item in result.evidence
        ]

    def has_failures(self) -> bool:
        """Return whether any tool execution failed or was blocked."""
        return any(
            result.status in {ToolResultStatus.FAILED, ToolResultStatus.BLOCKED}
            for result in self.tool_results
        )
