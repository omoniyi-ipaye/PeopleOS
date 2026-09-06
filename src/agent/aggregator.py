"""Evidence aggregation, confidence, sufficiency and contradiction checks."""

from collections import defaultdict
from typing import Iterable, List, Optional

from src.agent.evidence import EvidenceBundle, EvidenceItem, EvidenceSufficiency, ToolResult, ToolResultStatus


class EvidenceAggregator:
    """Build a canonical evidence bundle from governed tool results."""

    def aggregate(
        self,
        question: str,
        results: Iterable[ToolResult],
        *,
        workspace_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> EvidenceBundle:
        results = list(results)
        items: List[EvidenceItem] = [item for result in results for item in result.evidence]
        unknowns: List[str] = []
        notes: List[str] = []

        for result in results:
            if result.status in {ToolResultStatus.PARTIAL, ToolResultStatus.BLOCKED, ToolResultStatus.FAILED}:
                reason = result.error or "; ".join([w for w in result.warnings if w]) or result.summary
                unknowns.append(f"{result.tool_id}: {reason}")
            if result.status == ToolResultStatus.SUCCESS:
                notes.append(f"{result.tool_id} completed successfully")

        successful = sum(r.status == ToolResultStatus.SUCCESS for r in results)
        coverage = successful / len(results) if results else 0.0
        confidence = self._confidence(items, results)
        contradictions = self._detect_contradictions(items)
        if contradictions:
            confidence = max(0.0, confidence - min(0.25, 0.05 * len(contradictions)))

        if not items or coverage < 0.5 or confidence < 0.45:
            sufficiency = EvidenceSufficiency.INSUFFICIENT
        elif coverage < 1.0 or unknowns or contradictions or confidence < 0.70:
            sufficiency = EvidenceSufficiency.LIMITED
        else:
            sufficiency = EvidenceSufficiency.SUFFICIENT

        return EvidenceBundle(
            question=question,
            tool_results=results,
            overall_confidence=round(confidence, 3),
            coverage_score=round(coverage, 3),
            sufficiency=sufficiency,
            contradictions=contradictions,
            unknowns=unknowns,
            verification_notes=notes,
            provenance={
                "workspace_id": workspace_id,
                "dataset_version": dataset_version,
                "model_version": model_version,
            },
        )

    def _confidence(self, items: List[EvidenceItem], results: List[ToolResult]) -> float:
        if not results:
            return 0.0
        if not items:
            return 0.35 if any(r.status == ToolResultStatus.SUCCESS for r in results) else 0.1
        evidence_confidence = sum(item.confidence for item in items) / len(items)
        successful = sum(r.status == ToolResultStatus.SUCCESS for r in results)
        coverage = successful / len(results)
        return max(0.0, min(1.0, evidence_confidence * 0.8 + coverage * 0.2))

    def _detect_contradictions(self, items: List[EvidenceItem]) -> List[str]:
        grouped = defaultdict(list)
        for item in items:
            if not item.metric:
                continue
            scope = tuple(sorted((item.metadata or {}).items()))
            try:
                hash(scope)
            except TypeError:
                scope = ()
            grouped[(item.metric, scope)].append(item)

        contradictions: List[str] = []
        for (metric, _scope), metric_items in grouped.items():
            values = [item.value for item in metric_items if isinstance(item.value, (int, float, str, bool))]
            if len(values) > 1 and len(set(values)) > 1:
                sources = {item.source_tool for item in metric_items}
                if len(sources) > 1:
                    contradictions.append(f"Conflicting values observed for metric '{metric}' across tools.")
        return contradictions
