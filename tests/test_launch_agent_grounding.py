"""Legacy API and scoped-question acceptance with real aggregate engines.

The adversarial model is a test double, not a live local-model evaluation.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_app_state
from api.routes import advisor, intelligence
from src.agent.orchestrator import PeopleIntelligenceAgent
from src.analytics_engine import AnalyticsEngine
from src.platform.provenance import frame_fingerprint


class AdversarialModel:
    is_available = True
    model = "adversarial-test-double"

    def generate(self, prompt, **kwargs):
        return "Headcount is 9000. Low salary causes attrition."

    def _validate_response(self, response):
        # Proves lexical safety alone does not establish factual grounding.
        return True, response


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("PEOPLEOS_AGENT_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    frame = pd.DataFrame({
        "EmployeeID": [f"SYN{i}" for i in range(100)],
        "Attrition": [0] * 80 + [1] * 20,
        "Dept": ["Finance"] * 50 + ["Legal"] * 50,
        "Location": ["Madrid"] * 50 + ["Lagos"] * 50,
        "Salary": [60000] * 50 + [80000] * 50,
    })
    return SimpleNamespace(
        raw_df=frame, analytics_engine=AnalyticsEngine(frame),
        llm_client=AdversarialModel(), has_data=lambda: True,
        runtime_provenance={
            "workspace_id": "local", "dataset_id": "synthetic",
            "current_fingerprint": frame_fingerprint(frame),
        },
    )


@pytest.fixture
def client(state, monkeypatch):
    workspace = SimpleNamespace(active_dataset_id="synthetic", active_model_id=None)
    store = SimpleNamespace(
        ensure_workspace=lambda workspace_id: workspace,
        open_session=lambda **kwargs: SimpleNamespace(
            dataset_id="synthetic", model_id=None, session_id="session"
        ),
        record_request=lambda *args: None,
    )
    monkeypatch.setattr(intelligence, "_store", store)
    app = FastAPI()

    @app.middleware("http")
    async def identity(request, call_next):
        request.state.peopleos_role = request.headers.get("x-test-role", "owner")
        request.state.peopleos_actor_id = "test-owner"
        return await call_next(request)

    app.include_router(advisor.router)
    app.dependency_overrides[get_app_state] = lambda: state
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("endpoint", ["ask", "summary"])
def test_legacy_endpoints_reject_unverified_model_prose(client, endpoint):
    response = (client.post("/api/advisor/ask", params={"question": "What is headcount?"})
                if endpoint == "ask" else client.get("/api/advisor/summary"))
    assert response.status_code == 200, response.text
    body = response.json()
    output = body.get("answer", body.get("summary"))
    assert "Current active employee count: 80" in output
    assert "9000" not in output
    assert "Low salary causes attrition" not in output
    assert body["evidence"]["provenance"]["dataset_version"] == "synthetic"


@pytest.mark.parametrize("endpoint", ["ask", "summary"])
def test_legacy_endpoints_enforce_governed_permission(client, endpoint):
    response = (client.post("/api/advisor/ask", params={"question": "What is headcount?"}, headers={"x-test-role": "viewer"})
                if endpoint == "ask" else client.get("/api/advisor/summary", headers={"x-test-role": "viewer"}))
    assert response.status_code == 403


def test_legacy_endpoint_enforces_snapshot_integrity(client, state):
    state.raw_df.loc[0, "Salary"] = 1
    response = client.post("/api/advisor/ask", params={"question": "What is headcount?"})
    assert response.status_code == 409


def test_legacy_endpoint_is_usable_without_llm(client, state):
    state.llm_client = None
    response = client.post("/api/advisor/ask", params={"question": "What is headcount?"})
    assert response.status_code == 200
    assert "Current active employee count: 80" in response.json()["answer"]
    assert response.json()["model"] is None


@pytest.mark.parametrize("question", [
    "What is average salary for Finance?",
    "What is average salary for Madrid?",
    "Average salary for Finance/Madrid",
    "What is average salary for Legal?",
    "Compare average salary between Madrid and Lagos.",
    "Show average salary excluding Finance.",
    "What is Finance's average salary?",
    "Show average salary by location.",
    "What is average salary in Madrid?",
    "What was average salary last year?",
    "Finance salary and attrition",
    "Madrid compensation and engagement",
    "Compare Finance and Legal salary",
    "Finance: average salary",
    "Zürich salary and attrition",
])
def test_unsupported_scopes_never_synthesize_a_global_answer(state, question):
    result = PeopleIntelligenceAgent(state).investigate(question)
    assert result.status == "insufficient"
    assert result.model is None
    assert result.tools_used == []
    assert result.evidence.evidence_items() == []
    assert "will not infer the missing answer" in result.answer
    assert any("scope" in warning or "filter" in warning for warning in result.warnings)


@pytest.mark.parametrize("question", ["What is headcount?", "What is average salary for our workforce?"])
def test_explicit_whole_workforce_questions_remain_available(state, question):
    result = PeopleIntelligenceAgent(state).investigate(question)
    assert result.status in {"complete", "partial"}
    assert "requested population scope" not in " ".join(result.warnings).lower()
