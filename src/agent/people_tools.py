"""Additional governed aggregate evidence tools for People Intelligence."""

from time import perf_counter
from typing import Any, Dict, List

import pandas as pd

from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.agent.tools import ToolContext


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


class FairnessOutcomeTool:
    """Aggregate outcome disparity evidence with small-group suppression."""

    tool_id = "workforce.fairness"
    description = "Aggregate attrition outcome disparity across sufficiently large groups."
    min_group_size = 10

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "fairness_engine", None)
        raw_df = getattr(self.state, "raw_df", None)
        if engine is None or raw_df is None or "Attrition" not in raw_df.columns:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary="Fairness outcome analysis is unavailable for this dataset.",
                warnings=["Attrition outcomes and protected-group fields are required."],
                duration_ms=_elapsed_ms(started),
            )

        try:
            frame = engine.calculate_demographic_parity("Attrition")
            if frame is None or frame.empty:
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.PARTIAL,
                    summary="No eligible protected-group disparity evidence is available.",
                    warnings=["This can mean protected-group values, known outcomes, or minimum group support are insufficient; it must not be interpreted as evidence of parity."],
                    duration_ms=_elapsed_ms(started),
                )

            # The engine already enforces minimum group size. Preserve its
            # suppression ledger rather than recalculating from the filtered
            # frame (which would incorrectly report zero suppressed groups).
            if "suppressed_group_count" in frame.columns:
                suppressed = int(
                    frame.groupby("attribute", observed=True)["suppressed_group_count"].max().fillna(0).sum()
                )
            else:
                suppressed = 0

            evidence: List[EvidenceItem] = []
            records: List[Dict[str, Any]] = []
            for _, row in frame.iterrows():
                disparity_value = row.get("disparity")
                disparity = None if pd.isna(disparity_value) else float(disparity_value)
                parity_value = row.get("parity_ratio")
                record = {
                    "attribute": str(row.get("attribute")),
                    "group": str(row.get("group")),
                    "rate": None if pd.isna(row.get("rate")) else float(row.get("rate")),
                    "count": int(row.get("count", 0)),
                    "disparity": disparity,
                    "outcome_rate_ratio_to_overall": None if pd.isna(parity_value) else float(parity_value),
                    "overall_known_outcome_count": int(row.get("overall_known_outcome_count", 0) or 0),
                    "attribute_observed_count": int(row.get("attribute_observed_count", 0) or 0),
                    "attribute_coverage": None if pd.isna(row.get("attribute_coverage")) else float(row.get("attribute_coverage")),
                }
                records.append(record)
                if disparity is not None and disparity >= 0.05:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.DERIVED,
                        claim=(
                            f"{record['attribute']} group '{record['group']}' observed attrition rate differs "
                            f"from the overall known-outcome rate by {disparity:.1%} in absolute terms."
                        ),
                        source_tool=self.tool_id,
                        value=disparity,
                        metric="attrition_outcome_disparity",
                        confidence=1.0,
                        dataset_version=context.dataset_version,
                        metadata={
                            "attribute": record["attribute"],
                            "group": record["group"],
                            "group_size": record["count"],
                            "measured_count": record["attribute_observed_count"],
                            "eligible_count": record["overall_known_outcome_count"],
                            "attribute_coverage": record["attribute_coverage"],
                            "outcome_rate_ratio_to_overall": record["outcome_rate_ratio_to_overall"],
                            "confidence_basis": "deterministic_descriptive_disparity",
                            "not_legal_or_causal_fairness_determination": True,
                        },
                    ))

            warnings = [
                "Outcome disparity is a descriptive screening measure; it does not establish discrimination, bias, or causation."
            ]
            if suppressed:
                warnings.append(
                    f"{suppressed} small group(s) were suppressed because group size was below {engine.min_group_size}."
                )

            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS if evidence else ToolResultStatus.PARTIAL,
                summary=(
                    "Aggregate fairness outcome disparities calculated."
                    if evidence else
                    "Eligible groups were measured, but no disparity crossed the agent evidence threshold; this is not proof of parity."
                ),
                evidence=evidence,
                warnings=warnings,
                duration_ms=_elapsed_ms(started),
                metadata={"eligible_groups": records, "minimum_group_size": engine.min_group_size, "suppressed_group_count": suppressed},
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary="Fairness outcome analysis failed.",
                error=str(exc),
                duration_ms=_elapsed_ms(started),
            )


class EmployeeExperienceTool:
    """Aggregate Employee Experience evidence without employee-level output."""

    tool_id = "workforce.employee_experience"
    description = "Aggregate configured experience composite, engagement segments and measured signals."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "experience_engine", None)
        if engine is None:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary="Employee-experience analysis is unavailable.",
                warnings=["Experience engine is not initialized."],
                duration_ms=_elapsed_ms(started),
            )

        try:
            analysis = engine.analyze_all()
            evidence: List[EvidenceItem] = []
            metadata: Dict[str, Any] = {}
            warnings = list(analysis.get("warnings", []) or [])

            index = analysis.get("experience_index", {}) or {}
            if index.get("available", True):
                overall = index.get("overall_exi")
                if overall is not None:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.ASSUMED,
                        claim=f"Configured Employee Experience Index: {float(overall):.1f}/100.",
                        source_tool=self.tool_id,
                        value=float(overall),
                        metric="employee_experience_index",
                        confidence=1.0,
                        dataset_version=context.dataset_version,
                        metadata={
                            "confidence_basis": "deterministic_configured_composite",
                            "configured_composite_not_validated_outcome": True,
                            "not_probability": True,
                        },
                    ))
                # Grouped summaries are safe aggregate context; do not carry any
                # employee-level arrays returned by other experience sub-analyses.
                metadata["experience_index"] = {
                    key: value for key, value in index.items()
                    if key not in {"employees", "employee_scores", "at_risk_employees"}
                }

            segments = analysis.get("segments", {}) or {}
            segment_counts = segments.get("segments") or segments.get("distribution") or {}
            if isinstance(segment_counts, list):
                segment_counts = {item['segment']: item for item in segment_counts if isinstance(item, dict) and 'segment' in item}
            if isinstance(segment_counts, dict):
                metadata["segments"] = segment_counts
                for segment, value in segment_counts.items():
                    count = value.get("count") if isinstance(value, dict) else value
                    if isinstance(count, (int, float)):
                        evidence.append(EvidenceItem(
                            kind=EvidenceKind.ASSUMED,
                            claim=f"Configured '{segment}' experience segment count: {int(count)}.",
                            source_tool=self.tool_id,
                            value=int(count),
                            metric="experience_segment_count",
                            confidence=1.0,
                            dataset_version=context.dataset_version,
                            metadata={
                                "segment": str(segment),
                                "confidence_basis": "deterministic_configured_threshold",
                                "configured_segment_not_validated_outcome": True,
                            },
                        ))

            summary = analysis.get("summary", {}) or {}
            if isinstance(summary, dict):
                safe_summary = {
                    key: value for key, value in summary.items()
                    if not any(token in key.lower() for token in ("employee", "person", "name", "id"))
                }
                metadata["summary"] = safe_summary

            signals = analysis.get("signals", {}) or {}
            metadata["signals"] = signals

            warnings.append(
                "EXI weights and segment thresholds are configured constructs. Treat them as deterministic summaries of measured inputs, not validated employee outcomes or probabilities."
            )
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL if evidence else ToolResultStatus.PARTIAL,
                summary="Configured employee-experience composites calculated with construct limitations preserved." if evidence else "Measured employee-experience evidence is unavailable.",
                evidence=evidence,
                warnings=warnings,
                duration_ms=_elapsed_ms(started),
                metadata=metadata,
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary="Employee-experience analysis failed.",
                error=str(exc),
                duration_ms=_elapsed_ms(started),
            )
