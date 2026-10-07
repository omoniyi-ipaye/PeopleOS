"""Typed tool boundary for PeopleOS agent orchestration."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

from pydantic import BaseModel, Field

from src.agent.evidence import ToolResult


@dataclass(frozen=True)
class AgentToolDescriptor:
    """Machine-readable contract for a tool exposed to the agent.

    ``api_routes`` documents the product surfaces represented by the tool. The
    agent calls the in-process adapter rather than making an HTTP request, so
    the same workspace, snapshot and authorization boundary is retained.
    """

    tool_id: str
    description: str
    access: Literal['read', 'write'] = 'read'
    data_scope: Literal['aggregate', 'schema', 'control_plane'] = 'aggregate'
    engine: Optional[str] = None
    api_routes: Tuple[str, ...] = ()
    availability: Literal['available', 'conditional', 'unavailable'] = 'conditional'
    agent_callable: bool = True

    @property
    def read_only(self) -> bool:
        return self.access == 'read'

    def as_dict(self) -> Dict[str, Any]:
        return {
            'tool_id': self.tool_id,
            'description': self.description,
            'access': self.access,
            'read_only': self.read_only,
            'data_scope': self.data_scope,
            'engine': self.engine,
            'api_routes': list(self.api_routes),
            'availability': self.availability,
            'agent_callable': self.agent_callable,
        }


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
