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
            "headcount": 90,
            "record_count": 100,
            "active_count": 90,
            "observed_attrition_share": 0.10,
            "attrition_known_count": 100,
            "salary_mean": 70000.0,
            "salary_observations": 90,
            "salary_excluded_count": 0,
            "tenure_mean": 3.2,
            "tenure_observations": 90,
            "tenure_excluded_count": 0,
            "age_mean": None,
            "age_observations": 0,
            "age_excluded_count": 90,
            "lastrating_mean": 4.1,
            "lastrating_observations": 90,
            "lastrating_excluded_count": 0,
            "department_count": 3,
        }

    def get_department_aggregates(self):
        return pd.DataFrame([
            {
                "Dept": "Engineering",
                "Total_Records": 40,
                "Outcome_Observations": 40,
                "Headcount": 31,
                "Observed_Attrition_Share": 0.22,
            },
        ])

    def get_high_risk_departments(self, threshold=None):
        return pd.DataFrame([
            {
                "Dept": "Engineering",
                "Total_Records": 40,
                "Outcome_Observations": 40,
                "Observed_Attrition_Share": 0.22,
                "Headcount": 31,
            },
        ])


class FakeLLM:
    model = "fake-model"
    is_available = True

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return self.response


class FakeState:
    analytics_engine = FakeAnalyticsEngine()
    compensation_engine = None
    structural_engine = None
    model_metrics = {"f1": 0.86, "future_departure_validated": False}

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.raw_df = pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(100)], 'Attrition':[0]*90+[1]*10})
        self.risk_scores = pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(90)], 'risk_score':([.82,.79,.61,.20]*23)[:90]})
        self.ml_engine = SimpleNamespace(is_trained=True, get_risk_category=lambda score:'High' if score>=.7 else 'Medium' if score>=.4 else 'Low')
        self.runtime_provenance = {'workspace_id':'local','dataset_id':'fixture','generation':'one','current_fingerprint':frame_fingerprint(self.raw_df)}
        self.model_provenance = {**self.runtime_provenance,'model_id':'fixture-model'}


def test_causal_attrition_question_abstains_without_substitute_descriptive_numbers():
    answer = PeopleIntelligenceAgent(FakeState()).investigate(
        "Why is attrition elevated?"
    )

    assert answer.status == "insufficient"
    assert "cannot establish causes" in " ".join(answer.warnings)
    assert answer.model is None
    assert answer.tools_used == []
    assert 'Observed attrition share:' not in answer.answer
    assert "EmployeeID" not in answer.answer
    assert "will not infer" in answer.answer.lower()


def test_agent_blocks_punitive_llm_synthesis_and_returns_safe_evidence():
    llm = FakeLLM("Fire the highest-risk employees immediately.")
    state = FakeState(llm)
    answer = PeopleIntelligenceAgent(state).investigate("Show department hotspots.")

    assert llm.calls == 1
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
