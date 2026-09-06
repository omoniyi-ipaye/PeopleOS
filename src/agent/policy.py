"""Policy enforcement for PeopleOS agent and LLM outputs.

Policy lives outside prompts so model behavior cannot silently redefine HR safety
or authority boundaries. The first policy is deliberately narrow: PeopleOS may
analyze workforce systems and recommend systemic interventions, but it must not
recommend punitive or irreversible employment actions about individuals.
"""

import re
from dataclasses import dataclass, field
from typing import List


_PROHIBITED_ACTION_PATTERNS = [
    r"\bterminate(?:d|s|ing|ion)?\b",
    r"\bfire(?:d|s|ing)?\b",
    r"\bdismiss(?:ed|al|es|ing)?\b",
    r"\bdisciplin(?:e|ed|ary|ing)\b",
    r"\bperformance improvement plan\b",
    r"\bput (?:him|her|them|the employee) on (?:a )?pip\b",
    r"\bsalary reduction\b",
    r"\breduce (?:his|her|their|the employee(?:'s)?) salary\b",
    r"\bdemot(?:e|ed|ion|ing)\b",
    r"\bpunitive\b",
]


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


class HRAdvicePolicy:
    """Deterministic policy gate for externally surfaced PeopleOS advice."""

    policy_id = "hr_advice.v1"

    def evaluate_text(self, text: str) -> PolicyDecision:
        reasons: List[str] = []
        for pattern in _PROHIBITED_ACTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reasons.append(
                    "Output recommends or discusses a prohibited punitive employment action."
                )
                break
        return PolicyDecision(allowed=not reasons, reasons=reasons)

    def enforce_text(self, text: str) -> str:
        decision = self.evaluate_text(text)
        if not decision.allowed:
            raise PolicyViolation("; ".join(decision.reasons))
        return text


class PolicyViolation(RuntimeError):
    """Raised when generated advice crosses an enforceable policy boundary."""
