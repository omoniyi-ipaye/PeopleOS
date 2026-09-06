"""Canonical evidence contracts for PeopleOS agent workflows."""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceKind(str, Enum):
    OBSERVED = 'observed'
    DERIVED = 'derived'
    ASSUMED = 'assumed'
    UNKNOWN = 'unknown'


class ToolResultStatus(str, Enum):
    SUCCESS = 'success'
    PARTIAL = 'partial'
    BLOCKED = 'blocked'
    FAILED = 'failed'


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = 'sufficient'
    LIMITED = 'limited'
    INSUFFICIENT = 'insufficient'


class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f'ev_{uuid4().hex}')
    kind: EvidenceKind
    claim: str
    source_tool: str
    value: Any = None
    metric: Optional[str] = None
    source_ref: Optional[str] = None
    # Compatibility field. This is a bounded evidence-quality weight, not a
    # calibrated probability that the claim is true.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_kind: str = 'heuristic_evidence_quality'
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    result_id: str = Field(default_factory=lambda: f'tr_{uuid4().hex}')
    tool_id: str
    status: ToolResultStatus
    summary: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    duration_ms: Optional[float] = Field(default=None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    bundle_id: str = Field(default_factory=lambda: f'eb_{uuid4().hex}')
    question: str
    tool_results: List[ToolResult] = Field(default_factory=list)
    # Compatibility name retained for clients; semantically this is an
    # investigation evidence-quality heuristic, not statistical confidence.
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_kind: str = 'heuristic_evidence_quality'
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency: EvidenceSufficiency = EvidenceSufficiency.INSUFFICIENT
    contradictions: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    verification_notes: List[str] = Field(default_factory=list)
    provenance: Dict[str, Optional[str]] = Field(default_factory=dict)

    def evidence_items(self) -> List[EvidenceItem]:
        return [item for result in self.tool_results for item in result.evidence]

    def has_failures(self) -> bool:
        return any(result.status in {ToolResultStatus.FAILED, ToolResultStatus.BLOCKED} for result in self.tool_results)

    def can_synthesize(self) -> bool:
        return self.sufficiency != EvidenceSufficiency.INSUFFICIENT and bool(self.evidence_items())
