"""Cycle 022 forensic contracts for ScenarioEngine and its governed route boundary."""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from api.routes import scenario
from fastapi import HTTPException
from src.platform.provenance import frame_fingerprint
from src.scenario_engine import ScenarioEngine


def workforce(salaries=None):
    salaries = salaries or [100_000.0, 50_000.0, 50_000.0, 50_000.0]
    return pd.DataFrame(
        {
            "EmployeeID": [f"E{i}" for i in range(len(salaries))],
            "Dept": "People",
            "Salary": salaries,
            "Tenure": 2.0,
            "LastRating": [2.0, 3.0, 4.0, 5.0][: len(salaries)],
            "Attrition": 0,
        }
    )


def active_state(frame, *, generation="one", dataset_id="dataset-a", limit=100):
    return SimpleNamespace(
        raw_df=frame,
        scenario_cache={},
        config={"scenario": {"max_scenarios_saved": limit}},
        runtime_provenance={
            "workspace_id": "local",
            "dataset_id": dataset_id,
            "generation": generation,
            "current_fingerprint": frame_fingerprint(frame),
        },
        scenario_engine=object(),
        has_data=lambda: True,
    )


def test_compensation_does_not_invent_payback_for_mixed_timing():
    engine = ScenarioEngine(workforce())
    engine.scenario_config = dict(
        engine.scenario_config,
        assumed_compensation_elasticity=0.5,
    )

    result = engine.simulate_compensation_change("percentage", {"scope": "all"}, 10, 12)

    assert result.cost_impact.net_impact < 0
    assert result.payback_months is None
    assert result.cost_semantics["payback_available"] is False
    assert any("recurring cost" in item.lower() and "one-off" in item.lower() for item in result.assumptions)
    assert "recommended" not in result.recommendation.lower()


def test_legacy_reduction_helper_is_aggregate_and_never_selects_people():
    frame = workforce()
    result = ScenarioEngine(frame).simulate_headcount_change(
        "reduction", {"scope": "all"}, change_count=1, selection_criteria="cost"
    )

    # The old helper would select the 100k row for cost reduction. The bounded
    # compatibility path uses only the aggregate cohort mean (62.5k).
    assert result.cost_impact.total_benefit == pytest.approx(62_500)
    assert result.payback_months == 3
    assert result.cost_semantics["payback_available"] is True
    assert any("no employee ranking or selection" in item.lower() for item in result.assumptions)


def test_loss_making_expansion_has_no_payback_and_labels_recurring_salary():
    result = ScenarioEngine(workforce()).simulate_headcount_change(
        "expansion", {"scope": "all"}, change_count=1
    )

    assert result.cost_impact.net_impact < 0
    assert result.payback_months is None
    assert result.cost_semantics["payback_available"] is False
    assert any("multi-period cash-flow schedule" in item.lower() for item in result.assumptions)


def test_route_rejects_mutated_snapshot_before_running_scenario():
    frame = workforce()
    state = active_state(frame)
    frame.loc[0, "Salary"] = 101_000.0

    with pytest.raises(HTTPException) as error:
        asyncio.run(scenario.require_scenario(state))

    assert error.value.status_code == 409
    assert "Runtime data changed" in str(error.value.detail)


def test_cache_is_dataset_bound_and_uses_configured_session_limit():
    frame = workforce()
    state = active_state(frame, limit=2)
    engine = ScenarioEngine(frame)

    results = [
        engine.simulate_compensation_change("percentage", {"scope": "all"}, value)
        for value in (1, 2, 3)
    ]
    for result in results:
        response = scenario._convert_result(result, state)
        assert response.provenance["current_fingerprint"] == frame_fingerprint(frame)
        assert response.cost_semantics["payback_available"] is False

    assert len(state.scenario_cache) == 2
    assert results[0].scenario_id not in state.scenario_cache

    frame.loc[0, "Salary"] = 101_000.0
    with pytest.raises(HTTPException) as error:
        asyncio.run(scenario.get_scenario(results[2].scenario_id, state=state))
    assert error.value.status_code == 409
