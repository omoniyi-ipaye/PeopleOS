"""PeopleOS agent extension for governed downstream analytical follow-ups."""
from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from src.agent.derived_analysis import GovernedDerivedAnalysisTool, plan_derived_analysis
from src.agent.evidence import EvidenceSufficiency, ToolResultStatus
from src.agent.orchestrator import AgentAnswer, PeopleIntelligenceAgent
from src.agent.policy import PolicyViolation
from src.agent.tools import ToolContext


class GovernedPeopleIntelligenceAgent(PeopleIntelligenceAgent):
    """Adds bounded derived calculations without changing the stable base agent."""

    def __init__(self, state: Any):
        super().__init__(state)
        self.derived_tool = GovernedDerivedAnalysisTool(state)

    def investigate(
        self,
        question: str,
        *,
        actor_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> AgentAnswer:
        spec = plan_derived_analysis(question)
        if spec is None:
            return super().investigate(
                question,
                actor_id=actor_id,
                workspace_id=workspace_id,
                dataset_version=dataset_version,
                model_version=model_version,
            )

        request_id = f"pia_{uuid4().hex}"
        context = ToolContext(
            request_id=request_id,
            actor_id=actor_id,
            workspace_id=workspace_id,
            dataset_version=dataset_version,
            parameters={'analysis_spec': spec.model_dump()},
        )
        result = self.derived_tool.execute(context)
        bundle = self.aggregator.aggregate(
            question,
            [result],
            workspace_id=workspace_id,
            dataset_version=dataset_version,
            model_version=model_version,
        )
        warnings = list(result.warnings)

        if result.status == ToolResultStatus.SUCCESS and result.evidence:
            bundle.sufficiency = EvidenceSufficiency.SUFFICIENT
            answer = self._render_derived_answer(spec.model_dump(), result.evidence[0].value)
            status = 'complete'
        else:
            bundle.sufficiency = EvidenceSufficiency.INSUFFICIENT
            reason = result.warnings[0] if result.warnings else 'The current data does not support this calculation.'
            bundle.unknowns.append(reason)
            answer = f"PeopleOS cannot calculate that reliably from the current measured population. {reason}"
            status = 'insufficient'

        try:
            answer = self.policy.enforce_text(answer)
        except PolicyViolation:
            answer = 'PeopleOS blocked this derived output because it crossed the employment-action policy boundary.'
            status = 'insufficient'
            bundle.sufficiency = EvidenceSufficiency.INSUFFICIENT
            warnings.append('Derived output was blocked by HR advice policy.')

        response = AgentAnswer(
            request_id=request_id,
            question=question,
            answer=answer,
            status=status,
            confidence=float(bundle.overall_confidence or (1.0 if status == 'complete' else 0.0)),
            tools_used=[self.derived_tool.tool_id],
            model=None,
            evidence=bundle,
            warnings=warnings,
        )
        try:
            self.audit.record(
                request_id=request_id,
                question=question,
                status=status,
                confidence=response.confidence,
                tools_used=response.tools_used,
                tool_results=[result],
                model=None,
                policy_id=self.policy.policy_id,
                policy_blocked=False,
                workspace_id=workspace_id,
                dataset_version=dataset_version,
                actor_id=actor_id,
            )
        except Exception as exc:
            response.warnings.append(f"Audit record could not be written: {exc}")
        return response

    @staticmethod
    def _render_derived_answer(spec: dict[str, Any], output: dict[str, Any]) -> str:
        operation = spec['operation']
        statistic = spec.get('statistic', 'count')
        measure = spec.get('measure')
        group = spec.get('group_by')

        def label(column: Optional[str]) -> str:
            return {
                'Dept': 'department', 'Location': 'location', 'Gender': 'gender',
                'JobTitle': 'role', 'JobLevel': 'job level', 'Salary': 'salary',
                'Tenure': 'tenure', 'Age': 'age', 'LastRating': 'performance rating',
                'Attrition': 'recorded attrition',
            }.get(column or '', (column or 'measure').replace('_', ' ').lower())

        def value_text(value: float) -> str:
            if statistic == 'rate': return f"{value:.1%}"
            if measure == 'Salary' or statistic == 'sum': return f"{value:,.0f}"
            if statistic == 'count': return f"{int(round(value)):,}"
            return f"{value:.2f}"

        if operation == 'group_summary':
            rows = output.get('groups', [])
            descriptor = 'headcount' if statistic == 'count' else f"{statistic} {label(measure)}"
            parts = [f"{row['group']}: {value_text(float(row['value']))} (n={row['measured_count']})" for row in rows[:8]]
            suppressed = int(output.get('suppressed_groups', 0) or 0)
            suffix = f" {suppressed} smaller group{'s were' if suppressed != 1 else ' was'} hidden because there was not enough support." if suppressed else ''
            return f"{descriptor.capitalize()} by {label(group)} — " + '; '.join(parts) + '.' + suffix

        if operation == 'compare_groups':
            rows = output.get('groups', [])
            if len(rows) >= 2:
                difference = float(output['difference_b_minus_a'])
                return f"{label(measure).capitalize()} comparison — {rows[0]['group']}: {value_text(float(rows[0]['value']))} (n={rows[0]['measured_count']}); {rows[1]['group']}: {value_text(float(rows[1]['value']))} (n={rows[1]['measured_count']}). Difference (second minus first): {value_text(difference)}."

        if operation == 'correlation':
            r = float(output['correlation']); p = float(output['p_value']); n = int(output['paired_observations'])
            p_text = 'p<0.001' if p < .001 else f"p={p:.3f}"
            return f"The observed relationship between {label(spec.get('measure'))} and {label(spec.get('second_measure'))} is r={r:.2f} (n={n}, {p_text}). This is an association in the measured data, not evidence that one factor caused the other."

        if operation == 'crosstab':
            cells = output.get('cells', [])
            shown = sum(cell.get('count') is not None and cell.get('count') != 0 for cell in cells)
            suppressed = int(output.get('suppressed_cells', 0) or 0)
            return f"PeopleOS calculated the {label(spec.get('group_by'))} by {label(spec.get('second_group_by'))} distribution. {shown} supported cells are available in the evidence details; {suppressed} small cells were hidden to protect against over-interpreting tiny groups."

        return 'PeopleOS completed a governed aggregate calculation. Open the evidence details to inspect the measured support.'
