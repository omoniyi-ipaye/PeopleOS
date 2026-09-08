"""Deterministic first-stage planner for People Intelligence investigations.

The planner intentionally does not let the LLM invent tool names. It maps a user
question to an allowlisted evidence plan; future probabilistic planning can be
added behind the same registry and policy boundaries.
"""

from dataclasses import dataclass, field
import re
from typing import List


@dataclass(frozen=True)
class InvestigationPlan:
    tool_ids: List[str]
    rationale: str
    limitations: List[str] = field(default_factory=list)
    supported: bool = True
    required_metrics: List[str] = field(default_factory=list)


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

        limitations = []
        supported = len(tools) > 1 or any(term in q for term in [
            "headcount", "workforce", "employee count", "how many employees", "tenure", "average age", "rating"
        ])
        if not supported:
            limitations.append("No registered analysis matches this question. Ask about workforce counts, compensation, observed attrition, experience, fairness, or organizational structure.")
        if re.search(r"\b(why|cause[sd]?|causal|because)\b", q):
            limitations.append("These observational aggregates cannot establish causes or explain why an outcome occurred.")
        if re.search(r"\b(last|this|next|previous)\s+(month|quarter|year|week)\b|\b20\d{2}\b|\bq[1-4]\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q):
            limitations.append("This investigation does not apply the requested time filter; evidence describes the loaded current snapshot.")
        if "turnover" in q:
            limitations.append("Observed attrition share is not a period turnover rate; exposure and dated departures are required for period turnover.")
        if any(term in q for term in ["department", "team", "function", "engineering", "sales", "marketing"]):
            limitations.append("This plan returns workforce-wide and available department aggregates; it does not filter the dataset to a named team.")
        required_metrics = []
        for terms, metric in [
            (["headcount", "how many employees", "employee count"], "headcount"),
            (["average salary", "mean salary"], "salary_mean"),
            (["average age", "mean age"], "age_mean"),
            (["average tenure", "mean tenure"], "tenure_mean"),
            (["average rating", "mean rating", "average performance rating"], "lastrating_mean"),
            (["enps"], "enps"),
            (["payroll", "salary budget"], "annual_payroll"),
        ]:
            if any(term in q for term in terms):
                required_metrics.append(metric)
        # Explicit statistics are separate evidence contracts: a measured mean
        # never answers a requested median, minimum, maximum, sum or percentile.
        # Require every explicitly named statistic; unavailable variants fail
        # closed in the orchestrator instead of substituting nearby evidence.
        statistic_patterns = {
            "mean": r"\b(mean|average|avg)\b", "median": r"\bmedian\b",
            "min": r"\b(minimum|min|lowest)\b", "max": r"\b(maximum|max|highest)\b",
            "total": r"\b(total|sum|combined)\b", "std": r"\b(standard deviation|std)\b",
            "variance": r"\bvariance\b", "range": r"\brange\b",
            "percentile": r"\b(percentile|percentiles|p(?:10|25|50|75|90|95|99))\b",
        }
        statistic_columns = {
            "salary": r"\b(salary|salaries|pay|compensation|payroll)\b",
            "age": r"\b(ages?|old)\b", "tenure": r"\btenure\b",
            "lastrating": r"\bratings?\b",
        }
        statistics = [name for name, pattern in statistic_patterns.items() if re.search(pattern, q)]
        columns = [name for name, pattern in statistic_columns.items() if re.search(pattern, q)]
        for column in columns:
            for statistic in statistics:
                required_metrics.append(f"{column}_{statistic}")
        if re.search(r"\b(headcount|employee count|employees)\b", q) and not columns:
            required_metrics.extend("headcount" if statistic == "total" else f"headcount_{statistic}" for statistic in statistics)
        required_metrics = list(dict.fromkeys(required_metrics))

        # A generic workforce keyword is not evidence that an arbitrary metric
        # or population restriction has been implemented. Until typed filters are
        # available, treat unfamiliar summary-query terms conservatively.
        summary_words = set("what is are was were the our my a an and of for in about tell me show give please can you do we have how many employees employee people workforce current currently active total headcount count average mean age salary tenure performance rating overview summary statistics company organization organisation now today".split())
        summary_words.update({"q1", "q2", "q3", "q4"})
        tokens = set(re.findall(r"[a-z]+[0-9]*", q))
        if len(tools) == 1 and not required_metrics and not re.search(r"\b(summary|overview|statistics)\b", q):
            supported = False
            limitations.append("The requested workforce metric is not supported by the registered summary tool.")
        if len(tools) == 1 and required_metrics and tokens - summary_words:
            limitations.append("The requested population or time scope is not applied; the available summary is for the whole current workforce only.")
        if re.search(r"\b(women|men|female|male|nonbinary|part.time|full.time|contractors?|remote|onsite)\b|\b(in|within|among)\s+(?!our\b|the workforce\b|the company\b|the organization\b)\w+", q):
            limitations.append("Requested subgroup filters are not applied by this investigation; aggregate evidence must not be interpreted as that subgroup's result.")
        if re.search(r"\b(absenteeism|absence|absences|overtime|vacancies|vacancy|recruitment|productivity)\b", q):
            supported = False
            limitations.append("No registered evidence tool measures the requested outcome in this investigation.")
        return InvestigationPlan(required_metrics=required_metrics, tool_ids=tools if supported else [], rationale="; ".join(reasons), limitations=limitations, supported=supported)
