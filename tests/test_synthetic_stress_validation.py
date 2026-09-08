from scripts.synthetic_stress_validation import run, synthetic_workforce


def test_generator_is_reproducible_and_drift_is_explicit():
    original = synthetic_workforce(200, seed=41)
    repeated = synthetic_workforce(200, seed=41)
    drifted = synthetic_workforce(200, seed=41, drift=True)
    assert original.equals(repeated)
    assert not original.equals(drifted)
    assert original["EmployeeID"].is_unique


def test_known_answer_stress_contract():
    result = run(rows=1_000, seed=41, workers=4)
    assert result["complete"] is True
    assert result["failed"] == 0
    assert result["passed"] == len(result["checks"])
    assert result["profiles"]["privacy"]["tiny_group_records"] == 7
    assert result["profiles"]["missingness"]["employment_status"] > 0
