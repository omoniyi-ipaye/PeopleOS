"""Cycle 011 forensic contracts for FairnessEngine.

These tests deliberately separate mathematical correctness from interpretive safety.
Fairness outputs are descriptive screening signals, never legal/compliance determinations.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.fairness_engine as fairness_module
from src.fairness_engine import FairnessEngine, FairnessEngineError


def workforce(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "EmployeeID": [f"E{i}" for i in range(n)],
        "Gender": np.resize(["Female", "Male"], n),
        "Age": np.resize([25, 35, 45, 55], n),
        "Dept": np.resize(["A", "B"], n),
        "Attrition": np.resize([0, 0, 0, 1], n),
    })


def predictions_for(frame: pd.DataFrame, *, score: float = 0.25) -> pd.DataFrame:
    return pd.DataFrame({"EmployeeID": frame["EmployeeID"], "risk_score": score})


def test_four_fifths_uses_favorable_retention_known_answer():
    frame = workforce(40)
    frame["Gender"] = ["Female"] * 20 + ["Male"] * 20
    frame["Attrition"] = [1] * 10 + [0] * 10 + [1] * 4 + [0] * 16
    result = FairnessEngine(frame).calculate_four_fifths_rule("Attrition")
    female = result[(result.attribute == "Gender") & (result.group == "Female")].iloc[0]
    male = result[(result.attribute == "Gender") & (result.group == "Male")].iloc[0]
    assert female.favorable_rate == pytest.approx(0.5)
    assert male.favorable_rate == pytest.approx(0.8)
    assert female.reference_favorable_rate == pytest.approx(0.8)
    assert female.adverse_impact_ratio == pytest.approx(0.625)
    assert female.passes_4_5_rule is False or bool(female.passes_4_5_rule) is False


def test_four_fifths_zero_reference_is_unavailable_not_zero_or_pass():
    frame = workforce(40)
    frame["Attrition"] = 1
    result = FairnessEngine(frame).calculate_four_fifths_rule("Attrition")
    assert not result.empty
    assert result["adverse_impact_ratio"].isna().all()
    assert result["passes_4_5_rule"].isna().all()


def test_protected_groups_below_support_floor_are_not_returned():
    frame = workforce(24)
    frame["Gender"] = ["Female"] * 9 + ["Male"] * 15
    result = FairnessEngine(frame).calculate_four_fifths_rule("Attrition")
    gender_rows = result[result.attribute == "Gender"]
    assert set(gender_rows.group) == {"Male"}
    assert int(gender_rows.iloc[0].suppressed_group_count) == 1


def test_prediction_fairness_requires_exact_current_employee_coverage():
    frame = workforce(40)
    partial = predictions_for(frame.iloc[:-1])
    result = FairnessEngine(frame, partial).analyze_prediction_fairness()
    assert result["available"] is False
    assert "coverage" in result["reason"].lower() or "population" in result["reason"].lower()


def test_prediction_fairness_rejects_stale_or_extra_employee_ids():
    frame = workforce(40)
    pred = predictions_for(frame)
    pred.loc[len(pred)] = ["STALE-ID", 0.3]
    result = FairnessEngine(frame, pred).analyze_prediction_fairness()
    assert result["available"] is False


def test_prediction_fairness_rejects_duplicate_employee_ids():
    frame = workforce(40)
    pred = pd.concat([predictions_for(frame), predictions_for(frame).iloc[[0]]], ignore_index=True)
    assert FairnessEngine(frame, pred).analyze_prediction_fairness()["available"] is False


def test_prediction_fairness_is_unavailable_when_no_valid_probability_exists():
    frame = workforce(40)
    pred = predictions_for(frame)
    pred["risk_score"] = np.resize([np.nan, np.inf, -0.1, 1.1], len(pred))
    result = FairnessEngine(frame, pred).analyze_prediction_fairness()
    assert result["available"] is False
    assert result.get("attribute_analysis") is None or len(result.get("attribute_analysis", [])) == 0


def test_prediction_fairness_is_invariant_to_prediction_row_order():
    frame = workforce(40)
    pred = predictions_for(frame)
    pred["risk_score"] = np.linspace(0.05, 0.95, len(pred))
    a = FairnessEngine(frame, pred).analyze_prediction_fairness()["attribute_analysis"]
    b = FairnessEngine(frame, pred.sample(frac=1, random_state=11)).analyze_prediction_fairness()["attribute_analysis"]
    cols = ["attribute", "group", "mean_risk", "count", "difference_from_overall"]
    a = a[cols].sort_values(["attribute", "group"]).reset_index(drop=True)
    b = b[cols].sort_values(["attribute", "group"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_equalized_odds_small_outcome_class_counts_are_not_disclosed_or_reconstructable():
    frame = workforce(40)
    frame["Gender"] = ["Female"] * 20 + ["Male"] * 20
    frame["Attrition"] = [1] * 3 + [0] * 17 + [1] * 10 + [0] * 10
    pred = pd.DataFrame({
        "EmployeeID": frame.EmployeeID,
        "predicted": np.resize([0, 1], len(frame)),
        "risk_score": 0.5,
    })
    result = FairnessEngine(frame, pred).calculate_equalized_odds("Attrition")
    female = result[(result.attribute == "Gender") & (result.group == "Female")].iloc[0]
    assert female.tpr is None or pd.isna(female.tpr)
    # When either outcome class is below the support floor, neither class count
    # may be exposed alongside the total because the small class is reconstructable.
    assert female.positive_n is None or pd.isna(female.positive_n)
    assert female.negative_n is None or pd.isna(female.negative_n)
    assert bool(female.get("class_counts_suppressed", False)) is True


@pytest.mark.parametrize("bad", [0, 1, -1, 1.5, np.nan, np.inf])
def test_invalid_minimum_group_size_configuration_fails_closed(monkeypatch, bad):
    monkeypatch.setattr(fairness_module, "load_config", lambda: {"fairness": {"min_group_size": bad}})
    with pytest.raises(FairnessEngineError):
        FairnessEngine(workforce())


def test_age_group_boundaries_are_exact_and_impossible_ages_are_unavailable():
    frame = workforce(7)
    frame["Age"] = [29, 30, 40, 50, 60, 0, 121]
    engine = FairnessEngine(frame)
    values = engine.df["Age_Group"].astype("object").tolist()
    assert values[:5] == ["Under 30", "30-39", "40-49", "50-59", "60+"]
    assert pd.isna(values[5]) and pd.isna(values[6])


def test_demographic_parity_outputs_are_finite_json_or_null():
    frame = workforce(40)
    result = FairnessEngine(frame).calculate_demographic_parity("Attrition")
    json.dumps(result.to_dict("records"), allow_nan=False)


def test_engine_does_not_mutate_source_frame():
    frame = workforce(40)
    before = frame.copy(deep=True)
    FairnessEngine(frame).analyze_all()
    pd.testing.assert_frame_equal(frame, before)
