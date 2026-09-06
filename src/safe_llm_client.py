"""Non-bypassable PeopleOS LLM safety wrapper.

Legacy PeopleOS methods call ``self.client.generate`` directly in several places.
SafeLLMClient wraps the underlying Ollama transport so every generate call is
policy-checked, including inherited legacy methods and future callers that use the
raw client attribute. This keeps the policy boundary below individual routes.
"""

from typing import Any

from src.agent.policy import HRAdvicePolicy, PolicyViolation
from src.llm_client import LLMClient, LLMClientError
from src.logger import get_logger


logger = get_logger("safe_llm_client")


class _GuardedOllamaClient:
    """Proxy that intercepts every model generation before it leaves PeopleOS."""

    def __init__(self, client: Any, policy: HRAdvicePolicy):
        self._client = client
        self._policy = policy

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client.generate(*args, **kwargs)
        text = response.get("response", "") if isinstance(response, dict) else ""
        try:
            self._policy.enforce_text(text)
        except PolicyViolation as exc:
            logger.warning("LLM output blocked by policy boundary: %s", exc)
            raise LLMClientError("LLM output blocked by PeopleOS HR advice policy") from exc
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class SafeLLMClient(LLMClient):
    """Legacy-compatible LLMClient with a transport-level safety boundary."""

    def __init__(self):
        self.policy = HRAdvicePolicy()
        super().__init__()
        if self.client is not None:
            self.client = _GuardedOllamaClient(self.client, self.policy)

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate through the guarded transport and preserve legacy return shape."""
        return super().generate(prompt, **kwargs)
