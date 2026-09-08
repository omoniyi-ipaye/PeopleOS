"""Evidence aggregation, heuristic quality, sufficiency and contradiction checks.

PeopleOS evidence quality is an operational heuristic, not a calibrated probability
that an answer is true. It combines tool contribution, evidence provenance/kind,
known gaps and cross-tool consistency so synthesis can fail closed when support is
thin.
"""

from collections import defaultdict
from math import isclose
from typing import Iterable, List, Optional

from src.agent.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    EvidenceSufficiency,
    ToolResult,
    ToolResultStatus,
)


_KIND_WEIGHT = {
    EvidenceKind.OBSERVED: 1.0,
    EvidenceKind.DERIVED: 0.9,
    EvidenceKind.ASSUMED: 0.4,
    EvidenceKind.UNKNOWN: 0.0,
}

_STATUS_COVERAGE = {
    ToolResultStatus.SUCCESS: 1.0,
    ToolResultStatus.PARTIAL: 0.45,
    ToolResultStatus.BLOCKED: 0.0,
    ToolResultStatus.FAILED: 0.0,
}


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
        unknowns: List[str] = []
        notes: List[str] = []
        clean_results = []
        seen_results, seen_evidence = set(), set()
        for result in results:
            if result.result_id in seen_results:
                continue
            seen_results.add(result.result_id)
            evidence = []
            for item in result.evidence:
                usable = (result.status in {ToolResultStatus.SUCCESS, ToolResultStatus.PARTIAL}
                          and item.source_tool == result.tool_id
                          and not (item.metric and item.value is None)
                          and not (dataset_version and item.dataset_version and item.dataset_version != dataset_version)
                          and not (model_version and item.model_version and item.model_version != model_version))
                if not usable:
                    unknowns.append(f'{result.tool_id}: discarded unavailable or out-of-scope evidence')
                    continue
                if item.evidence_id not in seen_evidence:
                    evidence.append(item)
                    seen_evidence.add(item.evidence_id)
            clean_results.append(result.model_copy(update={'evidence': evidence}))
        results = clean_results
        items: List[EvidenceItem] = [item for result in results for item in result.evidence]

        for result in results:
            # Caveats remain material even when a tool calculates successfully.
            unknowns.extend(f"{result.tool_id}: {warning}" for warning in result.warnings if warning)
            if result.status in {ToolResultStatus.PARTIAL, ToolResultStatus.BLOCKED, ToolResultStatus.FAILED}:
                # Exception text is diagnostic data and can contain filesystem,
                # connector or credential details. Surface the governed summary
                # and explicit safe warnings; keep raw errors inside the typed
                # result for local diagnosis and out of the rendered answer.
                reason = "; ".join(w for w in result.warnings if w) or result.summary
                detail = f"{result.tool_id}: {reason}"
                if detail not in unknowns:
                    unknowns.append(detail)
            elif result.status == ToolResultStatus.SUCCESS:
                notes.append(f"{result.tool_id} completed successfully")
                if not result.evidence:
                    # Successful empty results may be meaningful (for example no
                    # hotspots), but they do not provide positive support for a claim.
                    notes.append(f"{result.tool_id} returned no positive evidence items")

        coverage = self._coverage(results)
        quality = self._quality(items, coverage)
        contradictions = self._detect_contradictions(items)

        # Known gaps and contradictions reduce the heuristic quality score. These
        # penalties are intentionally conservative and explicitly non-probabilistic.
        if unknowns:
            quality -= min(0.20, 0.04 * len(unknowns))
        if contradictions:
            quality -= min(0.30, 0.08 * len(contradictions))
        quality = max(0.0, min(1.0, quality))

        has_only_assumed = bool(items) and all(item.kind in {EvidenceKind.ASSUMED, EvidenceKind.UNKNOWN} for item in items)
        has_supported_item = any(item.kind in {EvidenceKind.OBSERVED, EvidenceKind.DERIVED} for item in items)

        if not has_supported_item or coverage < 0.5 or quality < 0.45 or has_only_assumed:
            sufficiency = EvidenceSufficiency.INSUFFICIENT
        elif coverage < 0.8 or unknowns or contradictions or quality < 0.70:
            sufficiency = EvidenceSufficiency.LIMITED
        else:
            sufficiency = EvidenceSufficiency.SUFFICIENT

        notes.append("overall_confidence is a heuristic evidence-quality score, not a probability of truth")

        return EvidenceBundle(
            question=question,
            tool_results=results,
            overall_confidence=round(quality, 3),
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

    def _coverage(self, results: List[ToolResult]) -> float:
        if not results:
            return 0.0
        return sum(_STATUS_COVERAGE[result.status] if result.evidence else 0.0 for result in results) / len(results)

    def _quality(self, items: List[EvidenceItem], coverage: float) -> float:
        if not items:
            # A successfully completed investigation with no positive evidence is
            # useful operationally but must not look like strong support.
            return min(0.35, coverage * 0.35)
        weighted = []
        for item in items:
            kind_weight = _KIND_WEIGHT.get(item.kind, 0.0)
            weighted.append(float(item.confidence) * kind_weight)
        evidence_quality = sum(weighted) / len(weighted)
        return max(0.0, min(1.0, evidence_quality * 0.85 + coverage * 0.15))

    def _scope_key(self, item: EvidenceItem) -> tuple:
        metadata = item.metadata or {}
        # Only fields that materially define the population/scope should be used.
        keys = (
            "population",
            "department",
            "attribute",
            "group",
            "segment",
            "cohort",
            "time_window",
            "outcome",
            "unit",
            "currency",
        )
        scope = tuple((key, str(metadata[key])) for key in keys if key in metadata)
        return (item.dataset_version, scope)

    @staticmethod
    def _meaningfully_different(left, right) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return left != right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            # Ignore tiny rounding differences; contradictions should represent
            # materially different claims, not serialization precision.
            return not isclose(float(left), float(right), rel_tol=0.01, abs_tol=1e-6)
        return str(left) != str(right)

    def _detect_contradictions(self, items: List[EvidenceItem]) -> List[str]:
        grouped = defaultdict(list)
        for item in items:
            if not item.metric or item.value is None:
                continue
            grouped[(item.metric, self._scope_key(item))].append(item)

        contradictions: List[str] = []
        for (metric, _scope), metric_items in grouped.items():
            sources = {item.source_tool for item in metric_items}
            if len(sources) < 2:
                continue
            values = [item.value for item in metric_items]
            different = any(
                self._meaningfully_different(values[i], values[j])
                for i in range(len(values))
                for j in range(i + 1, len(values))
            )
            if different:
                contradictions.append(f"Materially conflicting values observed for metric '{metric}' across tools in the same scope.")
        return contradictions
