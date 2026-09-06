"""Privacy-preserving local audit trail for People Intelligence investigations."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


class AgentAuditLogger:
    """Append minimal operational records without persisting workforce evidence.

    Questions are hashed by default and answers/evidence are never written. This
    provides traceability for tool execution, confidence and policy outcomes while
    minimizing the amount of potentially sensitive user context stored on disk.
    """

    def __init__(self, path: Optional[str] = None):
        configured = path or os.getenv("PEOPLEOS_AGENT_AUDIT_PATH", "data/agent_audit.jsonl")
        self.path = Path(configured)

    @staticmethod
    def question_hash(question: str) -> str:
        return hashlib.sha256(question.encode("utf-8")).hexdigest()

    def record(
        self,
        *,
        request_id: str,
        question: str,
        status: str,
        confidence: float,
        tools_used: Iterable[str],
        tool_results: Iterable[Any],
        model: Optional[str],
        policy_id: str,
        policy_blocked: bool,
        workspace_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        if os.getenv("PEOPLEOS_AGENT_AUDIT_DISABLED", "").lower() in {"1", "true", "yes"}:
            return

        result_statuses = {
            getattr(result, "tool_id", "unknown"): str(getattr(result, "status", "unknown").value if hasattr(getattr(result, "status", None), "value") else getattr(result, "status", "unknown"))
            for result in tool_results
        }
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "question_sha256": self.question_hash(question),
            "status": status,
            "confidence": round(float(confidence), 4),
            "tools_used": list(tools_used),
            "tool_statuses": result_statuses,
            "model": model,
            "policy_id": policy_id,
            "policy_blocked": bool(policy_blocked),
            "workspace_id": workspace_id,
            "dataset_version": dataset_version,
            "actor_id": actor_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
