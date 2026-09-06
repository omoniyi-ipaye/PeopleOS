"""Canonical evidence contracts for PeopleOS agent workflows."""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _json_safe(value: Any) -> Any:
    """Normalize analytical scalar/container types into JSON-safe Python values.

    Evidence adapters may receive NumPy/Pandas scalars from deterministic engines.
    The public evidence contract must never leak those implementation types into
    FastAPI/Pydantic serialization.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # NumPy scalar types (np.bool_, np.int64, np.float64, etc.) and several
    # Pandas scalar wrappers expose ``item`` to return the native Python scalar.
    item_method = getattr(value, 'item', None)
    if callable(item_method):
        try:
            native = item_method()
            if native is not value:
                return _json_safe(native)
        except (TypeError, ValueError):
            pass

    # Pandas timestamps and similar date-like values commonly expose isoformat.
    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        try:
            return isoformat()
        except (TypeError, ValueError):
            pass

    # Preserve strings for unknown analytical labels rather than failing the
    # entire investigation response at the serialization boundary.
    return str(value)


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

    @field_validator('value', 'metadata', mode='before')
    @classmethod
    def normalize_analytical_values(cls, value: Any) -> Any:
        return _json_safe(value)


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

    @field_validator('metadata', mode='before')
    @classmethod
    def normalize_metadata(cls, value: Any) -> Any:
        return _json_safe(value)


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
