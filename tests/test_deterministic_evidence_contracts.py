from types import SimpleNamespace

import pandas as pd

from src.agent.adapters import CompensationEquityTool, DepartmentRiskTool
from src.agent.evidence import EvidenceKind, ToolResultStatus
from src.agent.people_tools import EmployeeExperienceTool
from src.agent.tools import ToolContext
from src.fairness_engine import FairnessEngine


def _context():
    return ToolContext(request_id="test", dataset_version="dataset-v1")


def test_impossible_age_is_not_bucketed_into_protected_age_group():
    frame = pd.DataFrame({
        "EmployeeID": [f"E{i}" for i in range(20)],
        "Age": [25] * 10 + [999] * 10,
        "Gender": ["F"] * 10 + ["M"] * 10,
        "Attrition": [0, 1] * 10,
    })
    engine = FairnessEngine(frame)
    parity = engine.calculate_demographic_parity("Attrition")
    age_rows = parity[parity["attribute"] == "Age_Group"]
    assert set(age_rows["group"].astype(str)) == {"Under 30"}
    assert "60+" not in set(age_rows["group"].astype(str))


def test_equalized_odds_requires_support_per_outcome_class():
    frame = pd.DataFrame({
        "EmployeeID": [f"E{i}" for i in range(20)],
        "Gender": ["F"] * 20,
        "Attrition": [1] + [0] * 19,
    })
    predictions = pd.DataFrame({
        "EmployeeID": [f"E{i}" for i in range(20)],
        "predicted": [1] + [0] * 19,
    })
    engine = FairnessEngine(frame, predictions)
    result = engine.calculate_equalized_odds("Attrition")
    row = result[result["attribute"] == "Gender"].iloc[0]
    assert pd.isna(row["positive_n"])
    assert pd.isna(row["negative_n"])
    assert bool(row["class_counts_suppressed"]) is True
    assert pd.isna(row["tpr"])
    assert row["tpr_available"] == False
    assert row["fpr_available"] == True


class _NoOutcomeAnalytics:
    def get_department_aggregates(self):
        return pd.DataFrame([
            {
                "Dept": "People",
                "Total_Records": 12,
                "Outcome_Observations": 0,
                "Headcount": 12,
                "Observed_Attrition_Share": None,
            }
        ])

    def get_high_risk_departments(self, threshold=None):
        raise AssertionError("Hotspot filtering must not run when outcomes are unknown")


def test_missing_department_outcomes_are_not_reported_as_no_hotspots():
    tool = DepartmentRiskTool(SimpleNamespace(analytics_engine=_NoOutcomeAnalytics()))
    result = tool.execute(_context())
    assert result.status == ToolResultStatus.PARTIAL
    assert "unavailable" in result.summary.lower()
    assert result.evidence == []
    assert any("not interpreting missing outcome data" in warning for warning in result.warnings)


class _CompensationEngine:
    warnings = []

    def calculate_pay_equity_score(self):
        return pd.DataFrame([
            {
                "Dept": "People",
                "CV": 0.2,
                "Gini": 0.1,
                "EquityScore": 0.85,
                "SalaryDispersionScore": 0.85,
                "Headcount": 20,
                "Status": "Low dispersion",
            }
        ])

    def correlate_salary_with_attrition(self):
        return {"available": False, "reason": "not measured"}

    def calculate_gender_pay_gap(self):
        return {"available": False, "reason": "not measured"}


def test_agent_uses_primitive_compensation_dispersion_metrics_not_custom_score():
    tool = CompensationEquityTool(SimpleNamespace(compensation_engine=_CompensationEngine()))
    result = tool.execute(_context())
    metrics = {item.metric for item in result.evidence}
    assert "department_salary_cv" in metrics
    assert "department_salary_gini" in metrics
    assert "salary_dispersion_consistency_score" not in metrics


class _ExperienceEngine:
    def analyze_all(self):
        return {
            "experience_index": {"available": True, "overall_exi": 72.5},
            "segments": {"segments": {"Thriving": {"count": 8}}},
            "summary": {},
            "signals": {},
            "warnings": [],
        }


def test_configured_experience_composites_are_assumption_evidence():
    tool = EmployeeExperienceTool(SimpleNamespace(experience_engine=_ExperienceEngine()))
    result = tool.execute(_context())
    assert result.status == ToolResultStatus.PARTIAL
    assert result.evidence
    assert all(item.kind == EvidenceKind.ASSUMED for item in result.evidence)
    assert any("configured constructs" in warning.lower() for warning in result.warnings)
