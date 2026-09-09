"""Deterministic routing into the governed downstream analysis runtime."""
from __future__ import annotations

import re
from typing import Any, Optional

from src.agent.analysis_sandbox import AnalysisSpec, CohortFilter, GovernedAnalysisSandbox
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


def _cohort_filters(question: str) -> list[CohortFilter]:
    """Extract explicit cohort constraints without guessing unknown entities."""
    q = re.sub(r'\s+', ' ', question.strip())
    filters: list[CohortFilter] = []

    categorical_patterns = [
        ('Dept', r'\b(?:department|team|function)\s*(?:=|is|of)?\s*["“]?([A-Za-z][A-Za-z0-9 &/._-]{1,50})["”]?(?=\s*(?:,|and|with|where|who|having|$))'),
        ('Location', r'\b(?:location|office|city|country)\s*(?:=|is|of)?\s*["“]?([A-Za-z][A-Za-z0-9 &/._-]{1,50})["”]?(?=\s*(?:,|and|with|where|who|having|$))'),
        ('JobLevel', r'\b(?:job\s+level|level)\s*(?:=|is)?\s*["“]?([A-Za-z0-9._-]{1,30})["”]?'),
        ('JobTitle', r'\b(?:job\s+title|role)\s*(?:=|is)?\s*["“]?([A-Za-z][A-Za-z0-9 &/._-]{1,60})["”]?(?=\s*(?:,|and|with|where|who|having|$))'),
        ('Gender', r'\bgender\s*(?:=|is)?\s*["“]?([A-Za-z][A-Za-z -]{1,30})["”]?'),
    ]
    for column, pattern in categorical_patterns:
        match = re.search(pattern, q, re.I)
        if match:
            filters.append(CohortFilter(column=column, operator='eq', value=match.group(1).strip()))

    # Natural People-language shortcuts: "Engineering in Madrid" and "women in Sales".
    known_lead = re.search(r'\b([A-Z][A-Za-z0-9 &/._-]{1,40})\s+(?:employees?|people|staff)\s+in\s+([A-Z][A-Za-z0-9 &/._-]{1,40})\b', q)
    if known_lead and not any(f.column == 'Dept' for f in filters) and not any(f.column == 'Location' for f in filters):
        filters.extend([
            CohortFilter(column='Dept', operator='eq', value=known_lead.group(1).strip()),
            CohortFilter(column='Location', operator='eq', value=known_lead.group(2).strip()),
        ])

    for word, value in [('women', 'Female'), ('female', 'Female'), ('men', 'Male'), ('male', 'Male')]:
        if re.search(rf'\b{word}\b', q, re.I) and not any(f.column == 'Gender' for f in filters):
            filters.append(CohortFilter(column='Gender', operator='eq', value=value))
            break

    numeric_columns = {
        'tenure': 'Tenure', 'age': 'Age', 'salary': 'Salary', 'pay': 'Salary',
        'rating': 'LastRating', 'performance rating': 'LastRating',
    }
    numeric_words = '|'.join(sorted(map(re.escape, numeric_columns), key=len, reverse=True))
    comparisons = [
        ('lte', r'(?:under|below|less than|at most|up to)\s*([0-9]+(?:\.[0-9]+)?)'),
        ('gte', r'(?:over|above|more than|at least)\s*([0-9]+(?:\.[0-9]+)?)'),
        ('lt', r'<\s*([0-9]+(?:\.[0-9]+)?)'),
        ('gt', r'>\s*([0-9]+(?:\.[0-9]+)?)'),
    ]
    for metric_match in re.finditer(rf'\b({numeric_words})\b([^,.;]*)', q, re.I):
        column = numeric_columns[metric_match.group(1).lower()]
        tail = metric_match.group(2)
        for operator, pattern in comparisons:
            comp = re.search(pattern, tail, re.I)
            if comp:
                filters.append(CohortFilter(column=column, operator=operator, value=float(comp.group(1))))
                break

    # Deduplicate exact filters while preserving order.
    unique: list[CohortFilter] = []
    seen = set()
    for item in filters:
        key = (item.column, item.operator, str(item.value).casefold())
        if key not in seen:
            seen.add(key); unique.append(item)
    return unique[:8]


def plan_derived_analysis(question: str) -> Optional[AnalysisSpec]:
    """Translate explicit aggregate analytical language into a bounded spec."""
    q = re.sub(r'\s+', ' ', question.lower().strip())
    if re.search(r'\b(why|cause[sd]?|causal|because|fire|terminate|dismiss|rank employees?|which employees?|who should)\b', q):
        return None
    if re.search(r'\b(last|this|next|previous)\s+(month|quarter|year|week)\b|\b20\d{2}\b|\bq[1-4]\b', q):
        return None

    filters = _cohort_filters(question)
    measure_words = '|'.join(sorted((re.escape(k) for k in _MEASURES), key=len, reverse=True))
    group_words = '|'.join(sorted((re.escape(k) for k in _GROUPS), key=len, reverse=True))

    match = re.search(rf'\b(average|mean|median|total|sum)\s+(?:active[- ]employee\s+)?({measure_words})\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        statistic = {'average': 'mean', 'mean': 'mean', 'median': 'median', 'total': 'sum', 'sum': 'sum'}[match.group(1)]
        return AnalysisSpec(operation='group_summary', population='active', filters=filters, group_by=_GROUPS[match.group(3)], measure=_MEASURES[match.group(2)], statistic=statistic)

    match = re.search(rf'\b(?:headcount|employee count|people count|count)\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        return AnalysisSpec(operation='group_summary', population='active', filters=filters, group_by=_GROUPS[match.group(1)], statistic='count')

    match = re.search(rf'\b(?:recorded |observed )?(?:attrition|departure)\s+(?:share|rate|percentage)\s+(?:by|across)\s+({group_words})\b', q)
    if match:
        return AnalysisSpec(operation='group_summary', population='current', filters=filters, group_by=_GROUPS[match.group(1)], measure='Attrition', statistic='rate')

    match = re.search(rf'\b(?:correlation|relationship|association)\s+between\s+({measure_words})\s+and\s+({measure_words})\b', q)
    if match:
        return AnalysisSpec(operation='correlation', population='active', filters=filters, measure=_MEASURES[match.group(1)], second_measure=_MEASURES[match.group(2)])

    match = re.search(rf'\b(?:crosstab|cross[- ]tab|distribution)\s+(?:of\s+)?({group_words})\s+(?:by|across)\s+({group_words})\b', q)
    if match and _GROUPS[match.group(1)] != _GROUPS[match.group(2)]:
        return AnalysisSpec(operation='crosstab', population='active', filters=filters, group_by=_GROUPS[match.group(1)], second_group_by=_GROUPS[match.group(2)])

    # Filtered one-number summaries, e.g. "average salary for department Engineering, location Madrid, tenure under 2".
    match = re.search(rf'\b(average|mean|median|total|sum)\s+({measure_words})\b', q)
    if match and filters:
        statistic = {'average': 'mean', 'mean': 'mean', 'median': 'median', 'total': 'sum', 'sum': 'sum'}[match.group(1)]
        # A synthetic single cohort label lets the sandbox retain the same governed group-summary path.
        return AnalysisSpec(operation='group_summary', population='active', filters=filters, group_by=filters[0].column, measure=_MEASURES[match.group(2)], statistic=statistic, max_groups=20)

    return None


class GovernedDerivedAnalysisTool:
    tool_id = 'workforce.derived_analysis'
    description = 'Runs a typed, aggregate-only downstream calculation with cohort filters, minimum-support and identifier controls.'

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
                'filter_context': result.get('filter_context', {}),
                'population': result['population'],
                'population_count': result['population_count'],
                'semantics': result['semantics'],
            },
        )
        return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary='Governed downstream aggregate analysis completed.', evidence=[evidence], metadata={'analysis_spec': spec.model_dump()})
