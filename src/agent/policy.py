"""Policy enforcement for PeopleOS agent and LLM outputs.

Policy lives outside prompts so model behavior cannot silently redefine HR safety
or authority boundaries. PeopleOS may analyze workforce systems and describe HR
outcomes, but it must not recommend punitive or irreversible employment actions
about individuals.
"""

import re
from dataclasses import dataclass, field
from typing import List


# Block action-oriented recommendations, not neutral analytical mentions such as
# "termination rate" or "dismissal cases increased". Patterns deliberately look
# for an imperative/recommendation verb close to a consequential employment act,
# or for a direct action verb that explicitly targets employee(s).
_PROHIBITED_ACTION_PATTERNS = [
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise|immediately)\b[^.\n]{0,80}\bterminat(?:e|ing|ion)\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise|immediately)\b[^.\n]{0,80}\bfire\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise|immediately)\b[^.\n]{0,80}\bdismiss\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise|immediately)\b[^.\n]{0,80}\bdisciplin(?:e|ary)\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise)\b[^.\n]{0,80}\bperformance improvement plan\b",
    r"\bput (?:him|her|them|the employee|that employee|those employees) on (?:a )?pip\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise)\b[^.\n]{0,80}\bsalary reduction\b",
    r"\breduce (?:his|her|their|the employee(?:'s)?|that employee(?:'s)?) salary\b",
    r"\b(?:recommend|should|must|need to|consider|propose|suggest|advise)\b[^.\n]{0,80}\bdemot(?:e|ion)\b",
    r"\bfire\b[^.\n]{0,60}\bemploye(?:e|es)\b",
    r"\bterminat(?:e|ing)\b[^.\n]{0,60}\bemploye(?:e|es)\b",
    r"\bdismiss\b[^.\n]{0,60}\bemploye(?:e|es)\b",
    r"\bdisciplin(?:e|ing)\b[^.\n]{0,60}\bemploye(?:e|es)\b",
    r"\bdemot(?:e|ing)\b[^.\n]{0,60}\bemploye(?:e|es)\b",
    r"\bfire (?:him|her|them|the employee|that employee|those employees)\b",
    r"\bterminate (?:him|her|them|the employee|that employee|those employees)\b",
    r"\bdismiss (?:him|her|them|the employee|that employee|those employees)\b",
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
                    "Output recommends a prohibited punitive or irreversible employment action."
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
