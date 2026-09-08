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

_SELECTOR_PROMPT_PREFIX = "Select relevant evidence for a governed PeopleOS investigation."
_SELECTOR_MAX_TOKENS = 256


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
        """Generate safely, with strict JSON mode for the governed evidence selector."""
        if prompt.startswith(_SELECTOR_PROMPT_PREFIX):
            if not self.is_available or self.client is None:
                raise LLMClientError("LLM client not available")
            caller_options = kwargs.get("options", {})
            options = {
                "num_predict": min(int(caller_options.get("num_predict", _SELECTOR_MAX_TOKENS)), _SELECTOR_MAX_TOKENS),
                **{key: value for key, value in caller_options.items() if key != "num_predict"},
            }
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",
                    options=options,
                )
                generated = response.get("response", "") if isinstance(response, dict) else ""
            except LLMClientError:
                raise
            except Exception as exc:
                logger.error("Governed selector generation failed: %s", exc)
                raise LLMClientError(f"LLM generation failed: {exc}") from exc
        else:
            generated = super().generate(prompt, **kwargs)

        if isinstance(generated, str):
            structured = generated.strip()
            if structured.startswith("```json\n") and structured.endswith("\n```"):
                return structured[len("```json\n"):-len("\n```")].strip()
        return generated
