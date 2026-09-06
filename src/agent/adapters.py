"""Adapters exposing existing PeopleOS engines as governed aggregate agent tools.

The adapters intentionally avoid raw employee records. They return aggregate,
traceable evidence suitable for synthesis by the People Intelligence Agent.
"""

from time import perf_counter
from typing import Any, Dict, List

import pandas as pd

from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.agent.tools import ToolContext


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


class WorkforceSummaryTool:
    tool_id = "workforce.summary"
    description = "Aggregate workforce headcount, turnover, tenure, rating and salary metrics."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "analytics_engine", None)
        if engine is None:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.BLOCKED,
                summary="Workforce analytics are unavailable because no dataset is loaded.",
                error="analytics_engine_unavailable",
                duration_ms=_elapsed_ms(started),
            )
        try:
            stats = engine.get_summary_statistics()
            evidence: List[EvidenceItem] = []
            metric_labels = {
                "headcount": "Total workforce records",
                "active_count": "Active employee count",
                "turnover_rate": "Overall turnover rate",
                "salary_mean": "Average active-employee salary",
                "tenure_mean": "Average active-employee tenure",
                "lastrating_mean": "Average active-employee rating",
                "department_count": "Department count",
            }
            for metric, label in metric_labels.items():
                value = stats.get(metric)
                if value is not None:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.DERIVED,
                        claim=f"{label}: {value}",
                        source_tool=self.tool_id,
                        value=value,
                        metric=metric,
                        confidence=1.0,
                        dataset_version=context.dataset_version,
                    ))
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary="Aggregate workforce summary calculated.",
                evidence=evidence,
                duration_ms=_elapsed_ms(started),
                metadata={"metrics": stats},
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary="Workforce summary calculation failed.",
                error=str(exc),
                duration_ms=_elapsed_ms(started),
            )


class DepartmentRiskTool:
    tool_id = "workforce.department_risk"
    description = "Identify aggregate department turnover hotspots without exposing employees."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "analytics_engine", None)
        if engine is None:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.BLOCKED,
                summary="Department risk analysis is unavailable.",
                error="analytics_engine_unavailable",
                duration_ms=_elapsed_ms(started),
            )
        try:
            threshold = context.parameters.get("turnover_threshold")
            frame = engine.get_high_risk_departments(threshold=threshold)
            if frame is None or frame.empty:
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.SUCCESS,
                    summary="No departments exceed the configured turnover-risk threshold.",
                    evidence=[],
                    duration_ms=_elapsed_ms(started),
                    metadata={"departments": []},
                )
            records = frame.head(10).replace({pd.NA: None}).to_dict("records")
            evidence = []
            for row in records:
                dept = str(row.get("Dept", "Unknown"))
                turnover = row.get("Turnover_Rate")
                headcount = row.get("Headcount")
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.DERIVED,
                    claim=f"{dept} turnover rate is {turnover}",
                    source_tool=self.tool_id,
                    value=turnover,
                    metric="department_turnover_rate",
                    confidence=0.95,
                    dataset_version=context.dataset_version,
                    metadata={"department": dept, "active_headcount": headcount},
                ))
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary=f"Identified {len(records)} department turnover hotspot(s).",
                evidence=evidence,
                duration_ms=_elapsed_ms(started),
                metadata={"departments": records},
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.FAILED,
                summary="Department risk analysis failed.",
                error=str(exc),
                duration_ms=_elapsed_ms(started),
            )


class RetentionRiskTool:
    tool_id = "workforce.retention_risk"
    description = "Aggregate predictive attrition-risk distribution and model quality."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        scores = getattr(self.state, "risk_scores", None)
        if scores is None or len(scores) == 0:
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.PARTIAL,
                summary="Predictive retention risk is unavailable for the current dataset.",
                warnings=["Attrition-labelled historical data or a trained model may be missing."],
                duration_ms=_elapsed_ms(started),
            )
        try:
            counts = scores["risk_category"].value_counts().to_dict()
            total = int(len(scores))
            high = int(counts.get("High", 0))
            medium = int(counts.get("Medium", 0))
            low = int(counts.get("Low", 0))
            mean_risk = float(scores["risk_score"].mean()) if "risk_score" in scores.columns else None
            metrics = getattr(self.state, "model_metrics", None) or {}
            model_f1 = metrics.get("f1")
            confidence = float(max(0.5, min(0.98, model_f1))) if isinstance(model_f1, (int, float)) else 0.75
            evidence = [
                EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"High-risk population: {high} of {total}", source_tool=self.tool_id, value=high, metric="high_risk_count", confidence=confidence, dataset_version=context.dataset_version),
                EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Medium-risk population: {medium} of {total}", source_tool=self.tool_id, value=medium, metric="medium_risk_count", confidence=confidence, dataset_version=context.dataset_version),
                EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Low-risk population: {low} of {total}", source_tool=self.tool_id, value=low, metric="low_risk_count", confidence=confidence, dataset_version=context.dataset_version),
            ]
            if mean_risk is not None:
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Mean predicted attrition risk: {mean_risk:.3f}", source_tool=self.tool_id, value=mean_risk, metric="mean_risk_score", confidence=confidence, dataset_version=context.dataset_version))
            if model_f1 is not None:
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Current predictive model F1 score: {model_f1}", source_tool=self.tool_id, value=model_f1, metric="model_f1", confidence=1.0, dataset_version=context.dataset_version))
            return ToolResult(
                tool_id=self.tool_id,
                status=ToolResultStatus.SUCCESS,
                summary="Aggregate retention-risk distribution calculated.",
                evidence=evidence,
                duration_ms=_elapsed_ms(started),
                metadata={"distribution": {"High": high, "Medium": medium, "Low": low, "total": total}, "model_metrics": metrics},
            )
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary="Retention-risk analysis failed.", error=str(exc), duration_ms=_elapsed_ms(started))


class CompensationEquityTool:
    tool_id = "workforce.compensation_equity"
    description = "Aggregate compensation equity, salary spread and salary/attrition relationship."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "compensation_engine", None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary="Compensation analysis is unavailable.", warnings=["Salary data may be missing."], duration_ms=_elapsed_ms(started))
        try:
            evidence: List[EvidenceItem] = []
            warnings: List[str] = []
            metadata: Dict[str, Any] = {}
            equity = engine.calculate_pay_equity_score()
            if equity is not None and not equity.empty:
                records = equity.head(10).to_dict("records")
                metadata["department_equity"] = records
                for row in records:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.DERIVED,
                        claim=f"{row.get('Dept', 'Unknown')} pay-equity score is {row.get('EquityScore')}",
                        source_tool=self.tool_id,
                        value=row.get("EquityScore"),
                        metric="pay_equity_score",
                        confidence=0.9,
                        dataset_version=context.dataset_version,
                        metadata={"department": row.get("Dept"), "status": row.get("Status"), "headcount": row.get("Headcount")},
                    ))
            correlation = engine.correlate_salary_with_attrition()
            if correlation:
                metadata["salary_attrition_correlation"] = correlation
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.DERIVED,
                    claim=correlation.get("interpretation", "Salary/attrition relationship calculated."),
                    source_tool=self.tool_id,
                    value=correlation.get("correlation"),
                    metric="salary_attrition_correlation",
                    confidence=0.9 if correlation.get("is_significant") else 0.65,
                    dataset_version=context.dataset_version,
                    metadata={"p_value": correlation.get("p_value"), "is_significant": correlation.get("is_significant")},
                ))
            if getattr(engine, "has_gender", False):
                gap = engine.calculate_gender_pay_gap()
                metadata["gender_pay_gap"] = gap
                unadjusted = gap.get("unadjusted", {}) if isinstance(gap, dict) else {}
                if unadjusted:
                    evidence.append(EvidenceItem(
                        kind=EvidenceKind.DERIVED,
                        claim=f"Unadjusted gender pay gap is {unadjusted.get('gap_percentage')}%",
                        source_tool=self.tool_id,
                        value=unadjusted.get("gap_percentage"),
                        metric="gender_pay_gap_pct",
                        confidence=0.9 if unadjusted.get("is_significant") else 0.7,
                        dataset_version=context.dataset_version,
                        metadata={"p_value": unadjusted.get("p_value"), "is_significant": unadjusted.get("is_significant")},
                    ))
                warnings.extend(gap.get("warnings", []) if isinstance(gap, dict) else [])
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary="Compensation equity evidence calculated.", evidence=evidence, warnings=warnings, duration_ms=_elapsed_ms(started), metadata=metadata)
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary="Compensation equity analysis failed.", error=str(exc), duration_ms=_elapsed_ms(started))


class OrganizationStructureTool:
    tool_id = "workforce.organization_structure"
    description = "Aggregate span-of-control and role-stagnation hotspots."

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        engine = getattr(self.state, "structural_engine", None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary="Organization-structure analysis is unavailable.", warnings=["ManagerID and/or role-tenure fields may be missing."], duration_ms=_elapsed_ms(started))
        try:
            evidence: List[EvidenceItem] = []
            metadata: Dict[str, Any] = {}
            burnout = engine.analyze_manager_burnout_risk()
            if burnout.get("available"):
                summary = burnout.get("summary", {})
                metadata["span_of_control"] = {"summary": summary, "department_summary": burnout.get("department_summary", [])}
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Managers above span warning threshold: {summary.get('at_risk_count', 0)}", source_tool=self.tool_id, value=summary.get("at_risk_count", 0), metric="manager_span_risk_count", confidence=0.95, dataset_version=context.dataset_version))
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Average manager span of control: {summary.get('avg_span')}", source_tool=self.tool_id, value=summary.get("avg_span"), metric="average_span_of_control", confidence=0.95, dataset_version=context.dataset_version))
            stagnation = engine.identify_stagnation_hotspots()
            if stagnation.get("available"):
                summary = stagnation.get("summary", {})
                metadata["stagnation"] = {"summary": summary, "hotspots": stagnation.get("hotspots", [])}
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Critical role-stagnation count: {summary.get('critical_count', 0)}", source_tool=self.tool_id, value=summary.get("critical_count", 0), metric="critical_stagnation_count", confidence=0.9, dataset_version=context.dataset_version))
            if not evidence:
                return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary="Structural engines ran but the dataset lacks enough fields for aggregate findings.", warnings=[burnout.get("reason", "") if isinstance(burnout, dict) else "", stagnation.get("reason", "") if isinstance(stagnation, dict) else ""], duration_ms=_elapsed_ms(started), metadata=metadata)
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary="Organization-structure evidence calculated.", evidence=evidence, duration_ms=_elapsed_ms(started), metadata=metadata)
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary="Organization-structure analysis failed.", error=str(exc), duration_ms=_elapsed_ms(started))
