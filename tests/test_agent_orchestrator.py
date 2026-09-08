"""End-to-end tests for the governed People Intelligence Agent."""

import pandas as pd
from types import SimpleNamespace
from src.platform.provenance import frame_fingerprint

from src.agent.orchestrator import PeopleIntelligenceAgent
from src.agent.policy import HRAdvicePolicy
from src.agent.registry import ToolRegistry


class FakeAnalyticsEngine:
    def get_summary_statistics(self):
        return {
            "headcount": 100,
            "active_count": 90,
            "observed_attrition_share": 0.10,
            "salary_mean": 70000.0,
            "tenure_mean": 3.2,
            "lastrating_mean": 4.1,
            "department_count": 3,
        }

    def get_high_risk_departments(self, threshold=None):
        return pd.DataFrame([
            {"Dept": "Engineering", "Observed_Attrition_Share": 0.22, "Headcount": 40},
        ])


class FakeLLM:
    model = "fake-model"
    is_available = True

    def __init__(self, response):
        self.response = response

    def generate(self, prompt, **kwargs):
        return self.response


class FakeState:
    analytics_engine = FakeAnalyticsEngine()
    compensation_engine = None
    structural_engine = None
    model_metrics = {"f1": 0.86}
    risk_scores = pd.DataFrame({
        "risk_category": ["High", "High", "Medium", "Low"],
        "risk_score": [0.82, 0.79, 0.61, 0.20],
    })

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.raw_df = pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(100)], 'Attrition':[0]*90+[1]*10})
        self.risk_scores = pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(90)], 'risk_score':([.82,.79,.61,.20]*23)[:90]})
        self.ml_engine = SimpleNamespace(is_trained=True, get_risk_category=lambda score:'High' if score>=.7 else 'Medium' if score>=.4 else 'Low')
        self.runtime_provenance = {'workspace_id':'local','dataset_id':'fixture','generation':'one','current_fingerprint':frame_fingerprint(self.raw_df)}
        self.model_provenance = {**self.runtime_provenance,'model_id':'fixture-model'}


def test_agent_runs_plan_tools_aggregates_and_falls_back_without_llm():
    answer = PeopleIntelligenceAgent(FakeState()).investigate(
        "Why is attrition elevated?"
    )

    assert answer.status == "partial"
    assert "cannot establish causes" in " ".join(answer.warnings)
    assert answer.model is None
    assert answer.confidence > 0.7
    assert "workforce.summary" in answer.tools_used
    assert "workforce.retention_risk" in answer.tools_used
    assert "workforce.department_risk" in answer.tools_used
    assert 'Observed attrition share:' in answer.answer
    assert '(source department label: "Engineering")' in answer.answer
    assert "EmployeeID" not in answer.answer


def test_agent_blocks_punitive_llm_synthesis_and_returns_safe_evidence():
    state = FakeState(FakeLLM("Fire the highest-risk employees immediately."))
    answer = PeopleIntelligenceAgent(state).investigate("Why is attrition elevated?")

    assert answer.model is None
    assert "blocked" in " ".join(answer.warnings).lower()
    assert "Fire the highest-risk" not in answer.answer
    assert "underlying aggregate evidence" in answer.answer


def test_policy_does_not_false_positive_on_firewall_word():
    policy = HRAdvicePolicy()
    assert policy.evaluate_text("Review the firewall configuration.").allowed is True
    assert policy.evaluate_text("Fire the employee.").allowed is False


def test_registry_rejects_unregistered_tool():
    registry = ToolRegistry([])
    try:
        registry.get("filesystem.shell")
        assert False, "Expected unregistered tool lookup to fail"
    except KeyError:
        pass
