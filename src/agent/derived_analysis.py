"""Deterministic routing into the governed downstream analysis runtime."""
from __future__ import annotations

import re
from typing import Any, Optional

from src.agent.analysis_sandbox import AnalysisSpec, GovernedAnalysisSandbox
from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.agent.tools import ToolContext

_MEASURES = {
    'salary': 'Salary', 'pay': 'Salary', 'compensation': 'Salary',
    'age': 'Age', 'tenure': 'Tenure', 'rating': 'LastRating',
    'performance rating': 'LastRating', 'performance': 'LastRating',
}
_GROUPS = {
    'department': 'Dept', 'departments': 'Dept', 'team': 'Dept', 'teams': 'Dept', 'function': 'Dept', 'functions': 'Dept',
    'location': 'Location', 'locations': 'Location', 'office': 'Location', 'offices': 'Location',
    'gender': 'Gender', 'job title': 'JobTitle', 'role': 'JobTitle', 'job level': 'JobLevel', 'level': 'JobLevel',
}


def plan_derived_analysis(question: str) -> Optional[AnalysisSpec]:
    """Translate explicit aggregate analytical language into a bounded spec.

    This grammar intentionally covers common People-team follow-ups rather than
    guessing at arbitrary intent. Unsupported requests continue through the
    normal fail-closed agent path.
    """
    q = re.sub(r'\s+', ' ', question.lower().strip())
    if re.search(r'\b(why|cause[sd]?|causal|because|fire|terminate|dismiss|rank employees?|which employees?|who should)\b', q):
        return None
    if re.search(r'\b(last|this|next|previous)\s+(month|quarter|year|week)\b|\b20\d{2}\b|\bq[1-4]\b', q):
        return None

    measure_words = '|'.join(sorted((re.escape(k) for k in _MEASURES), key=len, reverse=True))
    group_words = '|'.join(sorted((re.escape(k) for k in _GROUPS), key=len, reverse=True))

    # Average/median/sum measure by a supported aggregate dimension.
    match = re.search(rf'\b(average|mean|median|total|sum)\s+(?:active[- ]employee\s+)?({measure_words})\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        statistic = {'average': 'mean', 'mean': 'mean', 'median': 'median', 'total': 'sum', 'sum': 'sum'}[match.group(1)]
        return AnalysisSpec(operation='group_summary', population='active', group_by=_GROUPS[match.group(3)], measure=_MEASURES[match.group(2)], statistic=statistic)

    # Headcount/count by a supported dimension.
    match = re.search(rf'\b(?:headcount|employee count|people count|count)\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        return AnalysisSpec(operation='group_summary', population='active', group_by=_GROUPS[match.group(1)], statistic='count')

    # Recorded attrition/departure share by group uses the current population,
    # not the active-only population.
    match = re.search(rf'\b(?:recorded |observed )?(?:attrition|departure)\s+(?:share|rate|percentage)\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        return AnalysisSpec(operation='group_summary', population='current', group_by=_GROUPS[match.group(1)], measure='Attrition', statistic='rate')

    # Pairwise numeric relationship. This remains explicitly non-causal.
    match = re.search(rf'\b(?:correlation|relationship|association)\s+between\s+({measure_words})\s+and\s+({measure_words})\b', q)
    if match:
        return AnalysisSpec(operation='correlation', population='active', measure=_MEASURES[match.group(1)], second_measure=_MEASURES[match.group(2)])

    # Crosstab / distribution of one supported categorical dimension by another.
    match = re.search(rf'\b(?:crosstab|cross[- ]tab|distribution)\s+(?:of\s+)?({group_words})\s+(?:by|across)\s+({group_words})\b', q)
    if match and _GROUPS[match.group(1)] != _GROUPS[match.group(2)]:
        return AnalysisSpec(operation='crosstab', population='active', group_by=_GROUPS[match.group(1)], second_group_by=_GROUPS[match.group(2)])

    return None


class GovernedDerivedAnalysisTool:
    tool_id = 'workforce.derived_analysis'
    description = 'Runs a typed, aggregate-only downstream calculation with minimum-support and identifier controls.'

    def __init__(self, state: Any):
        self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        spec_payload = context.parameters.get('analysis_spec')
        if spec_payload is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.BLOCKED, summary='No governed analysis specification was supplied.', warnings=['Downstream analysis was not executed.'])
        try:
            spec = AnalysisSpec.model_validate(spec_payload)
            frame = getattr(self.state, 'raw_df', None)
            if frame is None:
                return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary='No active workforce data is available for downstream analysis.', warnings=['Add workforce data before running derived analysis.'])
            result = GovernedAnalysisSandbox(frame).run(spec)
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.BLOCKED, summary='The requested downstream analysis did not pass the governed calculation contract.', warnings=[str(exc)])

        if not result.get('available'):
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary='The downstream analysis is unavailable with the current support.', warnings=[str(result.get('reason') or 'Insufficient measured support.')], metadata={'analysis_spec': spec.model_dump(), 'analysis_result': result})

        evidence = EvidenceItem(
            kind=EvidenceKind.DERIVED,
            claim='Governed downstream aggregate analysis completed.',
            source_tool=self.tool_id,
            metric='derived_analysis',
            value=result['output'],
            dataset_version=context.dataset_version,
            confidence=1.0,
            metadata={
                'analysis_spec': spec.model_dump(),
                'population': result['population'],
                'population_count': result['population_count'],
                'semantics': result['semantics'],
            },
        )
        return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary='Governed downstream aggregate analysis completed.', evidence=[evidence], metadata={'analysis_spec': spec.model_dump()})
