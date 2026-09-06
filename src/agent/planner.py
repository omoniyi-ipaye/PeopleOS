"""Deterministic first-stage planner for People Intelligence investigations.

The planner intentionally does not let the LLM invent tool names. It maps a user
question to an allowlisted evidence plan; future probabilistic planning can be
added behind the same registry and policy boundaries.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InvestigationPlan:
    tool_ids: List[str]
    rationale: str


class EvidencePlanner:
    """Select the smallest useful set of aggregate tools for a question."""

    def plan(self, question: str) -> InvestigationPlan:
        q = question.lower()
        tools = ["workforce.summary"]
        reasons = ["baseline workforce context"]

        def add(tool_id: str, reason: str) -> None:
            if tool_id not in tools:
                tools.append(tool_id)
                reasons.append(reason)

        if any(term in q for term in ["turnover", "attrition", "retention", "leave", "flight risk", "risk"]):
            add("workforce.retention_risk", "retention/attrition evidence requested")
            add("workforce.department_risk", "department hotspot context supports retention analysis")

        if any(term in q for term in ["department", "team", "function", "hotspot"]):
            add("workforce.department_risk", "department-level evidence requested")

        if any(term in q for term in ["salary", "pay", "compensation", "equity", "equal pay", "gender gap"]):
            add("workforce.compensation_equity", "compensation/equity evidence requested")

        if any(term in q for term in ["fairness", "bias", "disparity", "protected group", "adverse impact"]):
            add("workforce.fairness", "fairness/disparity evidence requested")

        if any(term in q for term in ["experience", "engagement", "enps", "pulse", "work-life", "work life", "employee sentiment"]):
            add("workforce.employee_experience", "employee-experience evidence requested")

        if any(term in q for term in ["manager", "span", "structure", "stagnation", "promotion", "org design", "organization design"]):
            add("workforce.organization_structure", "organization-structure evidence requested")

        # Broad strategic questions benefit from the major systemic lenses without
        # exposing employee-level data.
        if any(term in q for term in ["workforce health", "people health", "what should we do", "strategic", "executive", "overall"]):
            add("workforce.retention_risk", "strategic workforce health lens")
            add("workforce.department_risk", "strategic hotspot lens")
            add("workforce.compensation_equity", "strategic compensation lens")
            add("workforce.fairness", "strategic fairness lens")
            add("workforce.employee_experience", "strategic employee-experience lens")
            add("workforce.organization_structure", "strategic structure lens")

        return InvestigationPlan(tool_ids=tools, rationale="; ".join(reasons))
