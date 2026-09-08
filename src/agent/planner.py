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
    # True when some useful aggregate context exists but the user's requested
    # conclusion cannot be answered by the registered contracts. The
    # orchestrator may show that context, but must abstain from synthesis.
    must_abstain: bool = False


class EvidencePlanner:
    """Select the smallest useful set of aggregate tools for a question."""

    @staticmethod
    def _canonical_whole_workforce_question(question: str) -> str:
        """Recognize complete, unambiguous paraphrases, never strip modifiers.

        A global synonym replacement could erase a named scope or a negation.
        These closed grammars consume the entire question; anything else goes
        unchanged through the conservative scope and unsupported-term gates.
        """
        whole_scope = (
            r"(?:for|across|among|within|in)\s+"
            r"(?:(?:the|our)\s+)?(?:(?:whole|entire)\s+)?"
            r"(?:workforce|company|organization|organisation)"
        )
        prefix = r"(?:please\s+)?(?:(?:can|could)\s+you\s+)?"
        ending = r"\s*[?.!]?"
        if re.fullmatch(
            prefix + r"(?:summari[sz]e|give\s+(?:me|us)\s+a\s+summary\s+of|show(?:\s+me)?)"
            r"\s+(?:pay|salaries|compensation)\s+" + whole_scope + ending,
            question,
        ):
            return "summarise pay for our workforce"
        measure = re.fullmatch(
            prefix + r"(?:what\s+is|show(?:\s+me)?|tell\s+me)\s+(?:the\s+|our\s+)?"
            r"(?P<stat>average|mean)\s+"
            r"(?P<metric>pay|salary|age|tenure|performance\s+rating)\s+"
            + whole_scope + ending,
            question,
        )
        if measure:
            metric = {"pay": "salary", "performance rating": "rating"}.get(
                measure["metric"], measure["metric"]
            )
            return f'what is {measure["stat"]} {metric} for our workforce'
        if re.fullmatch(
            prefix + r"how\s+many\s+(?:people|employees|staff\s+members)\s+"
            r"(?:do\s+we\s+(?:currently\s+)?have|(?:currently\s+)?work\s+here)" + ending,
            question,
        ):
            return "what is current headcount"
        return question

    def plan(self, question: str) -> InvestigationPlan:
        q = self._canonical_whole_workforce_question(question.lower().strip())

        risk_or_departure = bool(re.search(
            r"\b(?:risk(?:\s+score)?|flight\s+risk|likely\s+to\s+leave|will\s+leave|attrition|turnover)\b", q
        ))
        aggregate_dimension = bool(re.search(
            r"\b(?:departments?|teams?|functions?|groups?|cohorts?|demographics?|segments?)\b", q
        ))
        singular_person = bool(re.search(
            r"\b(?:employee(?!s\b)|worker(?!s\b)|staff\s+member|he|she|him|her)\b|\bemployee\s+[a-z]*\d+[a-z0-9_-]*\b",
            q,
        ))
        named_person_pattern = bool(re.search(
            r"\b(?:is|will|assess|evaluate|score)\s+(?!(?:(?:the|our)\s+)?(?:workforce|company|organization|team|department)\b)"
            r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?\s+(?:likely\s+to\s+leave|at\s+(?:attrition\s+)?risk|attrition\s+risk)\b"
            r"|\bwhat\s+is\s+(?!(?:(?:the|our)\s+)?(?:workforce|company|organization)\b)"
            r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?\s+risk\s+score\b",
            q,
        ))
        population_ranking = bool(
            re.search(r"\brank\b[^?.]{0,60}\b(?:workforce|employees?|people|staff|workers?)\b", q)
            and not aggregate_dimension
        )
        individualized_risk = risk_or_departure and (
            singular_person or named_person_pattern or population_ranking
        )

        # Refuse requests for employee-level disclosure or consequential action
        # before selecting any evidence tools. PeopleOS is an aggregate,
        # read-only investigator; model availability must not change this gate.
        if individualized_risk or re.search(
            r"\b(?:who|which)\b[^?.]{0,80}\b(?:employees?|people|staff|individuals?)\b"
            r"|\bwho\b[^?.]{0,80}\b(?:likely\s+to\s+leave|flight\s+risks?|highest\s+risk|attrition\s+risk)\b"
            r"|\b(?:list|name|identify|reveal)\b[^?.]{0,80}\b(?:employee names?|employees?|people|staff|individuals?)\b"
            r"|\b(?:show|give|list|name|identify|reveal)\b[^?.]{0,80}\b(?:names?\s+of\s+)?(?:top\s+|highest\s+)?(?:flight\s+risks?|at-risk\s+(?:employees?|people|staff)|employees?\s+(?:most\s+likely\s+to\s+leave|with\s+the\s+highest\s+risk))\b"
            r"|\bshow\b[^?.]{0,80}\b(?:employee names?|named employees?|individual employees?)\b"
            r"|\bwho\b[^?.]{0,50}\b(?:fire|terminate|dismiss|discipline|demote)\b"
            r"|\b(?:fire|terminate|dismiss|discipline|demote|rank)\b[^?.]{0,80}\b(?:employees?|people|staff|individuals?|him|her|them)\b",
            q,
        ):
            limitation = (
                "PeopleOS does not identify, rank, or recommend consequential employment action about individuals; "
                "use aggregate evidence with authorized human review."
            )
            return InvestigationPlan(tool_ids=[], rationale="request blocked at the agent boundary", limitations=[limitation], supported=False)

        # A negated domain is not an instruction to run that domain's tools.
        # This avoids an unavailable optional tool turning a valid request into
        # an apparently failed investigation.
        routing_q = re.sub(
            r"\b(?:do not|don't|dont|exclude|skip|without|ignore)\s+(?:(?:analy[sz]e|review|include|show)\s+)?"
            r"(?:attrition|retention|turnover|risk|compensation|salary|pay|fairness|bias|experience|engagement)\b",
            " ",
            q,
        )
        tools = ["workforce.summary"]
        reasons = ["baseline workforce context"]

        def add(tool_id: str, reason: str) -> None:
            if tool_id not in tools:
                tools.append(tool_id)
                reasons.append(reason)

        observed_attrition_metric = bool(re.search(
            r"\b(?:(?:recorded|observed)\s+)?(?:attrition|departure)\s+(?:share|percentage)\b", routing_q
        ))
        if not observed_attrition_metric and any(term in routing_q for term in ["turnover", "attrition", "retention", "leave", "flight risk", "risk"]):
            add("workforce.retention_risk", "retention/attrition evidence requested")
            add("workforce.department_risk", "department hotspot context supports retention analysis")

        if any(term in routing_q for term in ["department", "team", "function", "hotspot"]):
            add("workforce.department_risk", "department-level evidence requested")

        if any(term in routing_q for term in ["salary", "pay", "compensation", "equity", "equal pay", "gender gap"]):
            add("workforce.compensation_equity", "compensation/equity evidence requested")

        if any(term in routing_q for term in ["fairness", "bias", "disparity", "protected group", "adverse impact"]):
            add("workforce.fairness", "fairness/disparity evidence requested")

        if any(term in routing_q for term in ["experience", "engagement", "enps", "pulse", "work-life", "work life", "employee sentiment"]):
            add("workforce.employee_experience", "employee-experience evidence requested")

        if any(term in routing_q for term in ["manager", "span", "structure", "stagnation", "promotion", "org design", "organization design", "burnout"]):
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

        headcount_requested = bool(re.search(
            r"\b(?:headcount|employee count|workforce size|staff count)\b|\bhow (?:many|large|big)\s+(?:is\s+)?(?:our\s+|the\s+)?workforce\b|"
            r"\b(?:how many|number of)\s+(?:our\s+)?(?:(?:active|current|currently|total)\s+)*(?:employees|staff(?: members?)?)\b", q
        ))
        limitations = []
        must_abstain = False
        supported = headcount_requested or observed_attrition_metric or len(tools) > 1 or any(term in q for term in [
            "headcount", "workforce", "employee count", "how many employees", "tenure", "average age", "rating"
        ])
        if not supported:
            limitations.append("No registered analysis matches this question. Ask about workforce counts, compensation, observed attrition, experience, fairness, or organizational structure.")
        if re.search(r"\b(why|cause[sd]?|causal|because)\b", q):
            limitations.append("These observational aggregates cannot establish causes or explain why an outcome occurred.")
        if re.search(r"\b(last|this|next|previous)\s+(month|quarter|year|week)\b|\b20\d{2}\b|\bq[1-4]\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q):
            limitations.append("This investigation does not apply the requested time filter; evidence describes the loaded current snapshot.")
            must_abstain = True
        if "turnover" in q:
            limitations.append("Observed attrition share is not a period turnover rate; exposure and dated departures are required for period turnover.")
        if re.search(r"\b(?:for|in|within|among|excluding|except|between)\s+(?:the\s+)?(?:[a-z]+\s+){0,2}(?:department|team|function)\b", q) or any(term in q for term in ["engineering", "sales", "marketing", "operations"]):
            limitations.append("This plan returns workforce-wide and available department aggregates; it does not filter the dataset to a named team.")
            must_abstain = True
        required_metrics = ["headcount"] if headcount_requested else []
        if observed_attrition_metric:
            required_metrics.append("observed_attrition_share")
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
        summary_words = set("what is are was were the our my a an and of for in about tell me show give please can you do we have how many employees employee people workforce staff members work here large big size current currently active total headcount count number average mean age salary tenure performance rating overview summary statistics company organization organisation now today just not analyze analyse attrition departure observed recorded share percentage".split())
        summary_words.update({"q1", "q2", "q3", "q4"})
        tokens = set(re.findall(r"[a-z]+[0-9]*", routing_q))
        # Every meaningful word must belong to the supported aggregate question
        # vocabulary. Previously only single-tool summaries checked unknown
        # words, allowing "Finance salary and attrition" to bypass scope gates.
        # Unknown names/modifiers require clarification, regardless of the
        # number of selected tools or where those terms occur in the question.
        aggregate_words = summary_words | set("""
            review summarize summarise explain compare rank assess aggregate level which why elevated
            low high highest lowest most least top across by at to with from
            has does should us it its health overall strategic executive
            insights insight trends trend analysis analytics metric metrics
            department departments team teams function functions hotspot hotspots
            risk risks score scores flight likely leave retention turnover
            compensation salaries pay equity equal gender gap fairness bias
            disparity protected group groups adverse impact demographic demographics
            experience engagement enps pulse work life employee sentiment
            manager managers span structure stagnation promotion org design
            organization organisation burnout organizational organisational
            median minimum min maximum max sum combined std standard deviation
            variance range percentile percentiles p10 p25 p50 p75 p90 p95 p99
            payroll budget avg ages old ratings annual rate rates count counts
            reason reasons cause causes caused causal because correlation
            correlation association observed recorded departure departures
            distribution distributions statistic statistics
        """.split())
        unresolved = set(re.findall(r"\b\w+\b", routing_q)) - aggregate_words
        if supported and unresolved:
            limitations.append("Unrecognized population scope or analysis terms require clarification; no whole-workforce result will be substituted.")
            must_abstain = True
        if len(tools) == 1 and not required_metrics and not re.search(r"\b(summary|overview|statistics)\b", q):
            supported = False
            limitations.append("The requested workforce metric is not supported by the registered summary tool.")
        if len(tools) == 1 and required_metrics and tokens - summary_words:
            limitations.append("The requested population or time scope is not applied; the available summary is for the whole current workforce only.")
            must_abstain = True
        # Named scopes are open-ended (Finance, Madrid, a newly uploaded office,
        # etc.). Never use a fixed list of department names to decide whether a
        # filter was requested. None of these tools accepts a typed row filter.
        scope_clause = re.search(
            r"\b(?:for|within|among|excluding|except|between|versus|vs)\s+(.+?)(?:[?.;]|$)", routing_q
        )
        whole_workforce = r"(?:(?:the|our|all|entire|whole)\s+)*(?:current\s+|active\s+)?(?:workforce|company|organization|organisation|employees|staff|people)(?:\s+(?:overall|as a whole))?"
        # "for attrition disparity" names an analysis, not a row population.
        analysis_topic = r"(?:attrition disparity|pay equity|compensation equity|workforce health)"
        if (scope_clause and not re.fullmatch(f"(?:{whole_workforce}|{analysis_topic})", scope_clause.group(1).strip())) or re.search(
            r"\b\w+['’]s\s+(?:average\s+|mean\s+)?(?:salary|pay|headcount|tenure|attrition)\b"
            r"|\b(?:salary|pay|headcount|tenure|attrition)\b[^?.;]*\bby\s+(?:location|office|country|city|department|team|gender|age)\b",
            routing_q,
        ):
            limitations.append("The requested population scope is not applied; these tools provide whole-workforce evidence and cannot answer a filtered or grouped request.")
            must_abstain = True
        if re.search(r"\b(women|men|female|male|nonbinary|part.time|full.time|contractors?|remote|onsite)\b|\b(in|within|among)\s+(?!our\b|the workforce\b|the company\b|the organization\b)\w+", q):
            limitations.append("Requested subgroup filters are not applied by this investigation; aggregate evidence must not be interpreted as that subgroup's result.")
            must_abstain = True
        if re.search(r"\b(absenteeism|absence|absences|overtime|vacancies|vacancy|recruitment|productivity)\b", q):
            supported = False
            limitations.append("No registered evidence tool measures the requested outcome in this investigation.")
        if re.search(r"\bregrettable\s+(?:attrition|turnover|departures?)\b", q):
            supported = False
            limitations.append("Regrettable attrition is not measured by the registered evidence tools.")
        required_metrics = list(dict.fromkeys(required_metrics))
        # Do not display a whole-workforce number beside a scoped question:
        # callers may read the number and miss the warning. An unsupported
        # scope returns an explicit refusal with no substitute aggregate.
        return InvestigationPlan(required_metrics=required_metrics, tool_ids=tools if supported and not must_abstain else [], rationale="; ".join(reasons), limitations=limitations, supported=supported, must_abstain=must_abstain)
