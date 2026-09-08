"""Check the real-model harness's fixed-answer judging, not embedding validity.

Actual model execution is scripts/validate_local_embeddings.py and is never
silently skipped or replaced by these tests.
"""
from scripts.validate_local_embeddings import CASES, RECORDS, judge


def test_contract_answers_exist_and_include_difficult_cases():
    ids = {record["id"] for record in RECORDS}
    assert len(ids) == len(RECORDS)
    assert all(case["expected"] in ids for case in CASES)
    assert {case["scope"] for case in CASES} == {"english", "spanish", "cross_language", "negation"}


def test_wrong_top_result_cannot_pass_with_correct_answer_lower_down():
    case = CASES[0]
    assert not judge(case, [{"id": "sales"}, {"id": case["expected"]}])["passed"]
    assert not judge(case, [])["passed"]
    assert judge(case, [{"id": case["expected"]}])["passed"]


def test_returned_nested_provenance_is_detached_without_model_download():
    """Unit-test copy isolation only; the separate harness validates real retrieval."""
    from types import SimpleNamespace
    import numpy as np
    from src.vector_engine import VectorEngine

    engine = VectorEngine.__new__(VectorEngine)
    engine.dimension = 1
    engine.metadata = [{"id": "synthetic", "source": {"snapshot": "original"}}]
    engine.model = SimpleNamespace(encode=lambda _: np.array([[1.0]]))
    engine.index = SimpleNamespace(search=lambda *_: (np.array([[0.0]]), np.array([[0]])))
    result = engine.search("synthetic")
    result[0]["source"]["snapshot"] = "tampered"
    assert engine.search("synthetic")[0]["source"]["snapshot"] == "original"
