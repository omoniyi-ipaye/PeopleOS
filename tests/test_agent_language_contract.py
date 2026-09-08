"""Real-engine language acceptance with independently calculated answers.

No LLM is used here: these tests establish routing and deterministic evidence,
not local-model quality. Unknown qualifiers must never become global answers.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent.orchestrator import PeopleIntelligenceAgent
from src.agent.planner import EvidencePlanner
from src.analytics_engine import AnalyticsEngine
from src.platform.provenance import frame_fingerprint


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("PEOPLEOS_AGENT_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    # Ten active people, two departed. Departed salaries deliberately differ.
    frame = pd.DataFrame({
        "EmployeeID": [f"LANG{i}" for i in range(12)],
        "Attrition": [0] * 10 + [1, 1],
        "Dept": ["Research"] * 12,
        "Salary": [60000] * 5 + [80000] * 5 + [200000, 200000],
        "Age": [30] * 5 + [50] * 5 + [70, 70],
        "Tenure": [2] * 5 + [6] * 5 + [20, 20],
        "LastRating": [3] * 5 + [5] * 5 + [1, 1],
    })
    return SimpleNamespace(
        raw_df=frame, analytics_engine=AnalyticsEngine(frame), llm_client=None,
        has_data=lambda: True,
        runtime_provenance={"workspace_id": "local", "dataset_id": "language",
                            "current_fingerprint": frame_fingerprint(frame)},
    )


@pytest.mark.parametrize("question,metric,expected", [
    ("Summarise pay across the whole workforce", "salary_mean", 70000),
    ("Please summarize compensation across our entire company.", "salary_mean", 70000),
    ("Could you give us a summary of salaries for the workforce?", "salary_mean", 70000),
    ("Show me pay within our whole organisation", "salary_mean", 70000),
    ("What is the average pay across our entire workforce?", "salary_mean", 70000),
    ("Show me mean salary among the whole company", "salary_mean", 70000),
    ("Tell me average age across our workforce", "age_mean", 40),
    ("What is our mean tenure within the whole organization?", "tenure_mean", 4),
    ("What is the average performance rating across the workforce?", "lastrating_mean", 4),
    ("How many people do we currently have?", "headcount", 10),
    ("Please tell me average salary across the company!", "salary_mean", 70000),
    ("How many staff members currently work here?", "headcount", 10),
])
def test_complete_paraphrases_preserve_real_metric_values(state, question, metric, expected):
    plan = EvidencePlanner().plan(question)
    assert plan.supported and not plan.must_abstain
    result = PeopleIntelligenceAgent(state).investigate(question)
    assert result.status in {"complete", "partial"}
    evidence = {item.metric: item.value for item in result.evidence.evidence_items()}
    assert evidence[metric] == pytest.approx(expected)
    assert result.model is None


@pytest.mark.parametrize("question", [
    "Summarise pay across the whole workforce except Research",
    "Summarise pay across the whole workforce last year",
    "Summarise pay across the whole workforce in 2025",
    "Summarise pay across the whole workforce by location",
    "Summarise pay across the whole workforce for contractors",
    "Summarise pay across the whole workforce excluding women",
    "Summarise pay across the whole workforce in Zürich",
    "Summarise pay across the whole workforce; only remote employees",
    "Summarise pay across the whole workforce and predict next year",
    "What is the average pay across our entire workforce under 30?",
    "What is average salary across the Research workforce?",
    "How many people do we currently have in Lagos?",
    "How many people do we currently have excluding departed staff?",
    "Don't summarise pay across the whole workforce",
    "Do not show mean salary among the whole company",
    "What is the median pay across the whole workforce?",
    "Summarise payroll taxes across the whole workforce",
])
def test_extra_qualifiers_never_receive_substitute_global_evidence(state, question):
    result = PeopleIntelligenceAgent(state).investigate(question)
    assert result.status == "insufficient"
    assert result.tools_used == []
    assert result.evidence.evidence_items() == []
    assert result.model is None
