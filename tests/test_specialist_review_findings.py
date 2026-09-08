"""Known-answer regression contracts converted from the independent specialist findings.

Passing tests now verify corrected behavior. Original review baseline: d96ef3c.
Synthetic inputs and controlled model failures isolate each integrity boundary.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def workforce(n=20, prefix="E"):
    return pd.DataFrame({
        "EmployeeID": [f"{prefix}{i}" for i in range(n)], "Dept": "A",
        "Tenure": 2.0, "Salary": 60000.0, "LastRating": 4.0,
        "Age": 30, "Gender": ["Female" if i % 2 else "Male" for i in range(n)],
        "JobTitle": "Analyst", "Location": "Madrid", "HireDate": "2024-01-01",
        "ManagerID": f"{prefix}0", "Attrition": 0,
    })


def context():
    from src.agent.tools import ToolContext
    return ToolContext(request_id="specialist-review", dataset_version="dataset-B")


def test_regression_sparse_status_column_becoming_all_active():
    from src.data_loader import DataLoader
    from src.analytics_engine import AnalyticsEngine
    raw = workforce()
    raw["Attrition"] = [1] + [None] * 19
    loader = DataLoader()
    loader.min_rows = 1
    loaded = loader._validate_data_quality(raw)
    assert loaded.Attrition.isna().sum() == 19
    assert loaded.Attrition.eq(1).sum() == 1
    assert AnalyticsEngine(loaded).get_headcount() == 0


def test_regression_csv_identifier_leading_zero_loss(tmp_path):
    from src.data_loader import DataLoader
    frame = workforce()
    frame["EmployeeID"] = [f"{i:04d}" for i in range(20)]
    frame["ManagerID"] = "0000"
    path = tmp_path / "ids.csv"
    frame.to_csv(path, index=False)
    loader = DataLoader()
    loader.min_rows = 1
    loaded = loader.load(str(path))
    assert str(loaded.iloc[1]["EmployeeID"]) == "0001"
    assert loaded.ManagerID.eq("0000").all()


def test_regression_activation_retaining_previous_survey_and_nlp(monkeypatch):
    from src.platform import runtime_loader
    from src.sentiment_engine import SentimentEngine
    old_survey = pd.DataFrame({"EmployeeID": ["OLD0"], "SurveyDate": ["2026-01-01"], "eNPSScore": [10]})
    state = SimpleNamespace(
        preprocessor=SimpleNamespace(fit_transform=lambda frame, **kw: (frame.copy(), {})),
        data_loader=SimpleNamespace(features_enabled={}), features_enabled={},
        enps_df=old_survey, onboarding_df=None, nlp_results={"dataset": "OLD"},
    )
    # Keep activation's real reset logic; replace optional expensive engine init.
    def initialize(s):
        s.sentiment_engine = SentimentEngine(s.raw_df, s.enps_df, s.onboarding_df)
    monkeypatch.setattr(runtime_loader, "_initialize_read_only_engines", initialize)
    runtime_loader.activate_dataframe(state, workforce(prefix="NEW"))
    assert state.sentiment_engine.calculate_enps()['available'] is False
    assert state.enps_df is None and state.nlp_results is None
    assert state.model_provenance is None and state.scenario_cache == {}


def test_regression_scenario_cache_surviving_dataset_change(monkeypatch):
    from api.routes import scenario
    from src.scenario_engine import ScenarioEngine
    from fastapi import HTTPException
    from src.platform.provenance import frame_fingerprint
    old = ScenarioEngine(workforce(prefix="OLD"))
    result = old.simulate_compensation_change("percentage", {"scope": "all"}, 5)
    old_state = SimpleNamespace(raw_df=workforce(prefix='OLD'), scenario_cache={})
    old_state.runtime_provenance = {'workspace_id':'local', 'dataset_id':'A', 'generation':'old', 'current_fingerprint':frame_fingerprint(old_state.raw_df)}
    response = scenario._convert_result(result, old_state)
    assert response.provenance['dataset_id'] == 'A'
    assert asyncio.run(scenario.get_scenario(result.scenario_id, state=old_state)).affected_employees == 20
    new_state = SimpleNamespace(raw_df=workforce(3, prefix='NEW'), scenario_cache={})
    new_state.runtime_provenance = {'workspace_id':'local', 'dataset_id':'B', 'generation':'new', 'current_fingerprint':frame_fingerprint(new_state.raw_df)}
    with pytest.raises(HTTPException) as error:
        asyncio.run(scenario.get_scenario(result.scenario_id, state=new_state))
    assert error.value.status_code == 404


def test_regression_experience_agent_success_with_no_measured_evidence():
    from src.experience_engine import ExperienceEngine
    from src.agent.people_tools import EmployeeExperienceTool
    frame = workforce()
    frame["Pulse_Score"] = 4
    engine = ExperienceEngine(frame)
    assert engine.calculate_experience_index()["overall_exi"] == 75
    result = EmployeeExperienceTool(SimpleNamespace(experience_engine=engine)).execute(context())
    assert result.status.value == "success"
    assert any(item.metric == 'employee_experience_index' and item.value == 75 for item in result.evidence)
    assert any(item.metric == 'experience_segment_count' for item in result.evidence)


def test_regression_fairness_agent_failure_for_all_retained():
    from src.fairness_engine import FairnessEngine
    from src.agent.people_tools import FairnessOutcomeTool
    frame = workforce()
    result = FairnessOutcomeTool(SimpleNamespace(raw_df=frame, fairness_engine=FairnessEngine(frame))).execute(context())
    assert result.status.value == 'success'
    assert result.error is None
    assert all(row['parity_ratio'] is None for row in result.metadata['eligible_groups'])


def test_regression_agent_accepting_wrong_population_and_invalid_score():
    from src.agent.adapters import RetentionRiskTool
    state = SimpleNamespace(raw_df=workforce(1, "NEW"), model_metrics={}, ml_engine=None,
        risk_scores=pd.DataFrame({"EmployeeID": ["OLD0"], "risk_score": [2.0], "risk_category": ["Low"]}))
    result = RetentionRiskTool(state).execute(context())
    assert result.status.value == 'partial'
    assert result.evidence == []
    assert result.warnings


def test_regression_onboarding_warning_total_capped_at_ten():
    from src.sentiment_engine import SentimentEngine
    frame = workforce(15)
    surveys = pd.DataFrame({"EmployeeID": frame.EmployeeID, "SurveyType": "30-day",
        "SurveyDate": "2026-01-01", "OverallScore": 2.0})
    engine = SentimentEngine(frame, onboarding_df=surveys)
    assert engine.analyze_onboarding_trajectory()["summary"]["at_risk_count"] == 15
    assert engine.detect_early_warnings()["summary"]["total_at_risk"] == 15


def test_regression_onboarding_health_using_superseded_response():
    from src.sentiment_engine import SentimentEngine
    surveys = pd.DataFrame({"EmployeeID": ["E0", "E0"], "SurveyType": ["30-day"] * 2,
        "SurveyDate": ["2026-01-01", "2026-02-01"], "OverallScore": [1, 5]})
    engine = SentimentEngine(workforce(), onboarding_df=surveys)
    assert engine.analyze_onboarding_trajectory()["trajectories"][0]["scores"]["30-day"] == 5
    assert engine.get_onboarding_health()["by_survey_type"][0]["avg_score"] == 5


def test_regression_qoh_source_grade_from_one_measured_person():
    from src.quality_of_hire_engine import QualityOfHireEngine
    from api.routes.quality_of_hire import _safe_source
    frame = workforce()
    frame["HireSource"] = "Referral"
    frame["LastRating"] = [5] + [None] * 19
    frame["Attrition"] = None
    row = QualityOfHireEngine(frame).calculate_source_effectiveness().iloc[0].to_dict()
    assert row["hire_count"] == 20 and row["performance_observations"] == 1
    assert pd.isna(row["quality_score"]) and row["grade"] == "Unavailable"
    response = _safe_source(row).model_dump()
    assert response["performance_observations"] == 1 


def test_regression_composites_changing_with_component_availability():
    from src.quality_of_hire_engine import QualityOfHireEngine
    frame = workforce(40)
    frame["HireSource"] = ["Measured retention"] * 20 + ["Unknown retention"] * 20
    frame["Attrition"] = [0] * 20 + [None] * 20
    rows = QualityOfHireEngine(frame).calculate_source_effectiveness().set_index("HireSource")
    assert rows.loc["Measured retention", "quality_score"] == 85.7
    assert pd.isna(rows.loc["Unknown retention", "quality_score"])
    assert rows.loc["Measured retention", "quality_weights"] == rows.loc["Unknown retention", "quality_weights"]
    assert rows["avg_performance"].tolist() == [4, 4]


def test_regression_scenario_positive_payback_for_negative_return():
    from src.scenario_engine import ScenarioEngine
    result = ScenarioEngine(workforce()).simulate_headcount_change("expansion", {"scope": "all"}, change_count=1)
    assert result.cost_impact.net_impact < 0
    assert result.payback_months is None
    # With a continuing negative modeled annual net, payback is not reached.


def test_regression_missing_compa_ratio_labeled_near_reference():
    from src.compensation_engine import CompensationEngine
    frame = workforce()
    frame["CompaRatio"] = np.nan
    result = CompensationEngine(frame).calculate_compa_ratio()
    assert result["CompaRatio"].isna().all()
    assert result["CompaStatus"].eq("Unavailable").all()


def test_regression_missing_department_dropped_from_team_headcounts():
    from src.team_dynamics_engine import TeamDynamicsEngine
    frame = workforce()
    frame.loc[:9, "Dept"] = None
    composition = TeamDynamicsEngine(frame).get_team_composition()
    assert composition.Headcount.sum() == 20
    assert "Unknown" in composition.Dept.values


def test_regression_model_training_labeled_with_other_dataset(monkeypatch, tmp_path):
    from starlette.requests import Request
    from api.routes import platform
    from src.platform.jobs import JobStore
    from src.platform.model_lifecycle import ModelLifecycleService
    from src.platform.workspace import WorkspaceStore
    import src.model_training as training
    store = WorkspaceStore(str(tmp_path / "workspace.json"))
    dataset_a = store.register_dataset(workspace_id="local", source_name="A", content_hash="A",
        row_count=20, columns=list(workforce()))
    store.activate_dataset("local", dataset_a.dataset_id)
    monkeypatch.setattr(platform, "_store", store)
    monkeypatch.setattr(platform, "_jobs", JobStore(str(tmp_path / "jobs.json")))
    monkeypatch.setattr(ModelLifecycleService, "_runtime_artifacts", {})
    monkeypatch.setattr(platform, "require_permission", lambda *args: SimpleNamespace(actor_id="reviewer"))
    seen = []
    def controlled_fit(frame):
        seen.append(frame.EmployeeID.tolist())
        return SimpleNamespace(metrics={}, engine=None)
    monkeypatch.setattr(training, "train_attrition_model", controlled_fit)
    state = SimpleNamespace(raw_df=workforce(prefix="B"), features_enabled={"predictive": True})
    from fastapi import HTTPException
    from src.platform.provenance import frame_fingerprint
    state.runtime_provenance = {'workspace_id':'local', 'dataset_id':'dataset-B', 'generation':'B', 'current_fingerprint':frame_fingerprint(state.raw_df)}
    with pytest.raises(HTTPException) as error:
        asyncio.run(platform.train_model("local", Request({"type": "http", "headers": []}), state=state))
    assert error.value.status_code == 409
    assert seen == []
    assert store.get_workspace('local').models == []


def test_regression_model_marked_active_before_scoring_failure(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from starlette.requests import Request
    from api.routes import platform
    from src.platform.model_lifecycle import ModelLifecycleService
    from src.platform.workspace import WorkspaceStore, ModelState
    store = WorkspaceStore(str(tmp_path / "workspace.json"))
    dataset = store.register_dataset(workspace_id="local", source_name="A", content_hash="A", row_count=20, columns=list(workforce()))
    store.activate_dataset("local", dataset.dataset_id)
    model = store.create_model(workspace_id="local", dataset_id=dataset.dataset_id)
    store.update_model("local", model.model_id, state=ModelState.CANDIDATE)
    called = []
    def broken_transform(*args, **kwargs):
        called.append(True)
        raise ValueError("Controlled scoring failure")
    from src.platform.provenance import frame_fingerprint
    provenance = {'workspace_id':'local', 'dataset_id':dataset.dataset_id, 'generation':'A', 'current_fingerprint':frame_fingerprint(workforce())}
    artifact = SimpleNamespace(metrics={'training_current_fingerprint': provenance['current_fingerprint']}, engine=SimpleNamespace(preprocessor=SimpleNamespace(transform=broken_transform)))
    monkeypatch.setattr(ModelLifecycleService, "_runtime_artifacts", {model.model_id: artifact})
    monkeypatch.setattr(platform, "_store", store)
    monkeypatch.setattr(platform, "require_permission", lambda *args: SimpleNamespace(actor_id="reviewer"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(platform.activate_model("local", model.model_id, Request({"type": "http", "headers": []}),
            state=SimpleNamespace(raw_df=workforce(), active_df=workforce(), ml_engine=None, runtime_provenance=provenance)))
    assert error.value.status_code == 409
    workspace = store.get_workspace("local")
    assert called == [True]
    assert workspace.active_model_id is None
    assert workspace.models[0].state == ModelState.CANDIDATE
