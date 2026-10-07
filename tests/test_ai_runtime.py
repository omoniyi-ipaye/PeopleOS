from types import SimpleNamespace

import pytest

from src.platform import ai_runtime


def test_preferences_default_to_deterministic_mode_and_write_owner_choice(tmp_path):
    store = ai_runtime.AIPreferencesStore(str(tmp_path / "ai-preferences.json"))

    assert store.get()["provider"] == "none"
    assert store.get()["enabled"] is False

    saved = store.update(enabled=True, model="qwen3.8:latest")
    assert saved["provider"] == "ollama"
    assert saved["enabled"] is True
    assert store.get()["ollama_model"] == "qwen3.8:latest"
    assert (tmp_path / "ai-preferences.json").stat().st_mode & 0o077 == 0


def test_cloud_model_is_rejected_from_local_setup():
    with pytest.raises(ai_runtime.LocalLLMSetupError, match="downloaded to this computer"):
        ai_runtime._validate_local_model("kimi-k3:cloud")


def test_probe_lists_only_local_models_and_recommends_installed_preference(monkeypatch):
    monkeypatch.setattr(ai_runtime, "find_ollama_binary", lambda: "/usr/local/bin/ollama")
    monkeypatch.setattr(ai_runtime, "_request_json", lambda *args, **kwargs: {
        "models": [
            {"name": "kimi-k3:cloud", "digest": "remote"},
            {"name": "nomic-embed-text:latest", "capabilities": ["embedding"], "digest": "embedding"},
            {"name": "hermes3", "digest": "local", "size": 1234},
        ]
    })

    result = ai_runtime.probe_ollama("http://127.0.0.1:11434", "gemma3:4b")

    assert result["ollama_running"] is True
    assert result["installed_models"] == [{
        "name": "hermes3:latest",
        "digest": "local",
        "size": 1234,
        "remote": False,
    }]
    assert result["recommended_model"] == "hermes3:latest"


def test_prepare_without_explicit_model_does_not_redownload_installed_preferred(monkeypatch, tmp_path):
    store = ai_runtime.AIPreferencesStore(str(tmp_path / "ai-preferences.json"))
    monkeypatch.setattr(ai_runtime, "AIPreferencesStore", lambda: store)
    monkeypatch.setattr(ai_runtime, "ensure_ollama_running", lambda *args, **kwargs: {
        "installed_models": [{"name": "qwen3.8:latest", "remote": False}],
        "selected_model_installed": False,
    })
    monkeypatch.setattr(ai_runtime, "test_ollama_model", lambda host, model: {
        "passed": True,
        "model": model,
        "response": "READY",
        "elapsed_ms": 1.0,
    })
    monkeypatch.setattr(ai_runtime, "pull_ollama_model", lambda *args, **kwargs: pytest.fail("installed model was pulled again"))

    result = ai_runtime.prepare_local_ollama("")

    assert result["model"] == "qwen3.8:latest"
    assert store.get()["enabled"] is True


def test_test_model_uses_no_data_smoke_payload(monkeypatch):
    captured = {}

    def fake_request(*args, **kwargs):
        captured.update(kwargs["payload"])
        return {"model": "hermes3:latest", "response": "READY"}

    monkeypatch.setattr(ai_runtime, "_request_json", fake_request)
    result = ai_runtime.test_ollama_model("http://localhost:11434", "hermes3")

    assert result["passed"] is True
    assert captured["model"] == "hermes3:latest"
    assert captured["think"] is False
    assert "workforce" not in captured["prompt"].lower()


def test_refresh_llm_state_disables_client_when_owner_preference_is_off(monkeypatch, tmp_path):
    store = ai_runtime.AIPreferencesStore(str(tmp_path / "ai-preferences.json"))
    monkeypatch.setattr(ai_runtime, "AIPreferencesStore", lambda: store)
    state = SimpleNamespace(features_enabled={"nlp": True, "llm": True}, raw_df=None, llm_client=object())

    result = ai_runtime.refresh_llm_state(state)

    assert result == {"enabled": False, "available": False, "model": None}
    assert state.llm_client is None
    assert state.features_enabled["llm"] is False
