"""Non-bypassable PeopleOS LLM safety wrapper.

Legacy PeopleOS methods call ``self.client.generate`` directly in several places.
SafeLLMClient wraps the underlying Ollama transport so every generate call is
policy-checked, including inherited legacy methods and future callers that use the
raw client attribute. This keeps the policy boundary below individual routes.
"""

import json
from typing import Any

from src.agent.policy import HRAdvicePolicy, PolicyViolation
from src.llm_client import LLMClient, LLMClientError
from src.logger import get_logger


logger = get_logger("safe_llm_client")

_SELECTOR_PROMPT_PREFIX = "Select relevant evidence for a governed PeopleOS investigation."
_SELECTOR_DATA_MARKER = "\nREQUEST_DATA:\n"
_SELECTOR_MAX_TOKENS = 256
_SELECTOR_REQUIRED_PREFIXES = {
    "headcount": ("Current active employee count:",),
    "observed_attrition_share": ("Observed attrition share:",),
    "salary_mean": ("Average active-employee salary:",),
    "age_mean": ("Average active-employee age:",),
    "tenure_mean": ("Average active-employee tenure:",),
    "lastrating_mean": ("Average active-employee rating:",),
}


def _compact_selector_prompt(prompt: str) -> str:
    """Reduce a selector prompt only when all requested metrics can be preserved exactly.

    The orchestrator still owns the complete evidence ledger and validates every model-returned
    evidence ID against it. This transport optimization only removes irrelevant same-source
    rows before local inference. If a requested metric cannot be matched safely, the original
    prompt is returned unchanged.
    """
    if _SELECTOR_DATA_MARKER not in prompt:
        return prompt
    prefix, raw_request = prompt.split(_SELECTOR_DATA_MARKER, 1)
    try:
        request = json.loads(raw_request)
    except (TypeError, ValueError, json.JSONDecodeError):
        return prompt
    if not isinstance(request, dict):
        return prompt
    evidence = request.get("evidence")
    required_metrics = request.get("required_metrics")
    if not isinstance(evidence, list) or not isinstance(required_metrics, list) or not required_metrics:
        return prompt

    required_ids: set[str] = set()
    for metric in required_metrics:
        prefixes = _SELECTOR_REQUIRED_PREFIXES.get(metric)
        if not prefixes:
            return prompt
        matches = [item for item in evidence if isinstance(item, dict) and
                   isinstance(item.get("claim"), str) and item["claim"].startswith(prefixes)]
        if not matches:
            return prompt
        for item in matches:
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str):
                required_ids.add(evidence_id)

    # Preserve at least one representative from every source that the orchestrator supplied.
    source_ids: set[str] = set()
    seen_sources: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = item.get("source_tool")
        evidence_id = item.get("evidence_id")
        if isinstance(source, str) and source not in seen_sources and isinstance(evidence_id, str):
            seen_sources.add(source)
            source_ids.add(evidence_id)

    keep_ids = required_ids | source_ids
    compacted = [item for item in evidence if isinstance(item, dict) and item.get("evidence_id") in keep_ids]
    if not compacted or len(compacted) >= len(evidence):
        return prompt
    request["evidence"] = compacted
    return prefix + _SELECTOR_DATA_MARKER + json.dumps(request, separators=(",", ":"), default=str)


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
            transport_prompt = _compact_selector_prompt(prompt)
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=transport_prompt,
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
