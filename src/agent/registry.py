"""Allowlisted tool registry for PeopleOS agent execution."""

from typing import Dict, Iterable, List

from src.agent.tools import AgentTool


class ToolRegistry:
    """Registry of explicitly approved tools available to the orchestrator."""

    def __init__(self, tools: Iterable[AgentTool] = ()): 
        self._tools: Dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not getattr(tool, "tool_id", None):
            raise ValueError("Agent tools must define a non-empty tool_id")
        if tool.tool_id in self._tools:
            raise ValueError(f"Duplicate tool_id: {tool.tool_id}")
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> AgentTool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Tool is not registered or allowed: {tool_id}") from exc

    def list_ids(self) -> List[str]:
        return sorted(self._tools.keys())

    def contains(self, tool_id: str) -> bool:
        return tool_id in self._tools
