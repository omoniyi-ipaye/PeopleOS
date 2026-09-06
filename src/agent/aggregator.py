"""Evidence aggregation, confidence and contradiction checks."""

from collections import defaultdict
from typing import Iterable, List

from src.agent.evidence import EvidenceBundle, EvidenceItem, ToolResult, ToolResultStatus


class EvidenceAggregator:
    """Build a canonical evidence bundle from governed tool results."""

    def aggregate(self, question: str, results: Iterable[ToolResult]) -> EvidenceBundle:
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

        confidence = self._confidence(items, results)
        contradictions = self._detect_contradictions(items)
        if contradictions:
            confidence = max(0.0, confidence - min(0.25, 0.05 * len(contradictions)))

        return EvidenceBundle(
            question=question,
            tool_results=results,
            overall_confidence=round(confidence, 3),
            contradictions=contradictions,
            unknowns=unknowns,
            verification_notes=notes,
        )

    def _confidence(self, items: List[EvidenceItem], results: List[ToolResult]) -> float:
        if not results:
            return 0.0
        if not items:
            return 0.35 if any(r.status == ToolResultStatus.SUCCESS for r in results) else 0.1

        evidence_confidence = sum(item.confidence for item in items) / len(items)
        successful = sum(r.status == ToolResultStatus.SUCCESS for r in results)
        coverage = successful / len(results)
        # Evidence quality carries more weight than execution coverage, but partial
        # tool failure must still reduce the overall confidence presented to users.
        return max(0.0, min(1.0, evidence_confidence * 0.8 + coverage * 0.2))

    def _detect_contradictions(self, items: List[EvidenceItem]) -> List[str]:
        """Detect incompatible values for the same metric/scope when obvious.

        This is deliberately conservative. More sophisticated statistical conflict
        detection can be added later without changing the bundle contract.
        """
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
                # Different tools can legitimately provide values at different
                # granularities; only flag when the source tools differ.
                sources = {item.source_tool for item in metric_items}
                if len(sources) > 1:
                    contradictions.append(f"Conflicting values observed for metric '{metric}' across tools.")
        return contradictions
