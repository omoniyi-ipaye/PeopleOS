"""Acceptance contracts for durable scenario planning and AI-guided comparison."""

import asyncio
from types import SimpleNamespace

import pandas as pd

from api.routes import scenario
from api.schemas.scenario import ScenarioDrilldownRequest, ScenarioSaveRequest
from src.platform.provenance import frame_fingerprint
from src.platform.workspace import WorkspaceStore
from src.scenario_engine import ScenarioEngine


def _workforce() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EmployeeID": ["E1", "E2", "E3", "E4"],
            "Dept": ["People", "People", "Finance", "Finance"],
            "Salary": [100_000.0, 80_000.0, 90_000.0, 70_000.0],
            "Tenure": [2.0, 3.0, 4.0, 1.0],
            "LastRating": [3.0, 4.0, 4.0, 5.0],
            "Attrition": [0, 0, 1, 0],
        }
    )


def _state(frame: pd.DataFrame, *, store: WorkspaceStore, llm_client=None):
    return SimpleNamespace(
        raw_df=frame,
        scenario_cache={},
        config={"scenario": {"max_scenarios_saved": 10}},
        runtime_provenance={
            "workspace_id": "local",
            "dataset_id": "dataset-test",
            "generation": "generation-test",
            "current_fingerprint": frame_fingerprint(frame),
            "reporting_currency": "USD",
        },
        scenario_engine=object(),
        llm_client=llm_client,
        has_data=lambda: True,
    )


def _two_scenarios(state):
    engine = ScenarioEngine(state.raw_df)
    first = engine.simulate_compensation_change("percentage", {"scope": "all"}, 5, 12)
    second = engine.simulate_headcount_change("expansion", {"scope": "all"}, 2, None, "performance")
    return [scenario._convert_result(item, state) for item in (first, second)]


def test_scenarios_are_saved_and_reopened_from_a_fresh_cache(tmp_path, monkeypatch):
    store = WorkspaceStore(path=str(tmp_path / "registry.json"))
    monkeypatch.setattr(scenario, "_workspace_store", store)
    frame = _workforce()
    state = _state(frame, store=store)
    result = _two_scenarios(state)[0]

    saved = asyncio.run(scenario.save_scenario(ScenarioSaveRequest(scenario_id=result.scenario_id, scenario_name="Annual pay review"), state))
    assert saved.scenario_name == "Annual pay review"

    reopened = _state(frame, store=store)
    loaded = scenario._cached_scenario(reopened, result.scenario_id)
    assert loaded["scenario_name"] == "Annual pay review"
    assert store.get_scenario(result.scenario_id).payload["scenario_name"] == "Annual pay review"


def test_drilldown_falls_back_to_verified_deterministic_explanation(tmp_path, monkeypatch):
    store = WorkspaceStore(path=str(tmp_path / "registry.json"))
    monkeypatch.setattr(scenario, "_workspace_store", store)
    frame = _workforce()
    state = _state(frame, store=store)
    results = _two_scenarios(state)

    response = asyncio.run(scenario.drilldown_scenarios(
        ScenarioDrilldownRequest(scenario_ids=[item.scenario_id for item in results]),
        state,
    ))

    assert response.status == "fallback"
    assert response.model is None
    assert results[0].scenario_name in response.answer
    assert response.headline
    assert "What it means for People" in response.answer
    assert response.people_takeaway
    assert response.use_for
    assert response.validate_next
    assert response.decision_boundary
    assert any("unavailable" in warning.lower() for warning in response.warnings)


def test_drilldown_uses_local_model_only_to_select_evidence(tmp_path, monkeypatch):
    class StubLLM:
        is_available = True
        model = "local-test"

        def generate(self, prompt, **kwargs):
            assert "Do not write prose" in prompt
            return '{"evidence_ids":["impact","outcome"],"focus":"outcome"}'

    store = WorkspaceStore(path=str(tmp_path / "registry.json"))
    monkeypatch.setattr(scenario, "_workspace_store", store)
    frame = _workforce()
    state = _state(frame, store=store, llm_client=StubLLM())
    results = _two_scenarios(state)

    response = asyncio.run(scenario.drilldown_scenarios(
        ScenarioDrilldownRequest(
            scenario_ids=[item.scenario_id for item in results],
            question="Which workforce outcome should we inspect first?",
        ),
        state,
    ))

    assert response.status == "complete"
    assert response.model == "local-test"
    assert response.selected_evidence == ["impact", "outcome"]
    assert response.focus == "outcome"
    assert "Financial view:" in response.answer
    assert "Workforce outcome:" in response.answer
    assert response.focus_label == "Modeled people outcome"
    assert response.headline
    assert response.use_for
    assert response.validate_next
