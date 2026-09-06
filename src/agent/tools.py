"""Typed tool boundary for PeopleOS agent orchestration."""

from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel, Field

from src.agent.evidence import ToolResult


class ToolContext(BaseModel):
    """Execution context supplied to governed PeopleOS tools."""

    request_id: str
    workspace_id: Optional[str] = None
    dataset_version: Optional[str] = None
    actor_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentTool(Protocol):
    """Minimum contract the future orchestrator may invoke.

    Existing analytics engines will be adapted behind this interface rather than
    exposed directly to the LLM.
    """

    tool_id: str
    description: str

    def execute(self, context: ToolContext) -> ToolResult:
        """Execute deterministically or within the tool's declared boundary."""
        ...
