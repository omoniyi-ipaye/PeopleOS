"""Additional governed aggregate evidence tools for People Intelligence."""

from time import perf_counter
from typing import Any, Dict, List

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
                    summary="No protected-group disparity evidence is available.",
                    duration_ms=_elapsed_ms(started),
                )

            eligible = frame[frame["count"] >= self.min_group_size].copy()
            suppressed = int(len(frame) - len(eligible))
            evidence: List[EvidenceItem] = []
            records: List[Dict[str, Any]] = []
            for _, row in eligible.iterrows():
                record = {
                    "attribute": str(row.get("attribute")),
                    "group": str(row.get("group")),
                    "rate": float(row.get("rate", 0.0)),
                    "count": int(row.get("count", 0)),
                    "disparity": float(row.get("disparity", 0.0)),
                    "parity_ratio": None if row.get("parity_ratio") is None else float(row["parity_ratio"]),
                }
                records.append(record)
                if record["disparity"] >= 0.05:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.DERIVED,
                        claim=(
                            f"{record['attribute']} group '{record['group']}' has an attrition-rate "
                            f"disparity of {record['disparity']:.1%} from the workforce baseline."
                        ),
                        source_tool=self.tool_id,
                        value=record["disparity"],
                        metric="attrition_outcome_disparity",
                        confidence=0.85,
                        dataset_version=context.dataset_version,
                        metadata={
                            "attribute": record["attribute"],
                            "group": record["group"],
                            "group_size": record["count"],
                            "parity_ratio": record["parity_ratio"],
                        },
                    ))

            warnings = []
            if suppressed:
                warnings.append(
                    f"{suppressed} small group(s) were suppressed because group size was below {self.min_group_size}."
                )

            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary="Aggregate fairness outcome analysis completed.",
                evidence=evidence,
                warnings=warnings,
                duration_ms=_elapsed_ms(started),
                metadata={"eligible_groups": records, "minimum_group_size": self.min_group_size},
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
    """Aggregate Employee Experience Index evidence without employee-level output."""

    tool_id = "workforce.employee_experience"
    description = "Aggregate experience index, engagement segments and experience drivers."

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
                        kind=EvidenceKind.DERIVED,
                        claim=f"Overall Employee Experience Index is {float(overall):.1f}/100.",
                        source_tool=self.tool_id,
                        value=float(overall),
                        metric="employee_experience_index",
                        confidence=0.85,
                        dataset_version=context.dataset_version,
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
                            kind=EvidenceKind.DERIVED,
                            claim=f"{segment} experience segment count: {int(count)}.",
                            source_tool=self.tool_id,
                            value=int(count),
                            metric="experience_segment_count",
                            confidence=0.9,
                            dataset_version=context.dataset_version,
                            metadata={"segment": str(segment)},
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

            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS if evidence else ToolResultStatus.PARTIAL,
                summary="Aggregate employee-experience evidence calculated." if evidence else "Measured employee-experience evidence is unavailable.",
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
