"""Allowlisted tool registry for PeopleOS agent execution."""

from typing import Dict, Iterable, List

from src.agent.tools import AgentTool, AgentToolDescriptor


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

    @staticmethod
    def _descriptor(tool: AgentTool) -> AgentToolDescriptor:
        descriptor = getattr(tool, 'descriptor', None)
        if callable(descriptor):
            descriptor = descriptor()
        if isinstance(descriptor, AgentToolDescriptor):
            return descriptor
        return AgentToolDescriptor(
            tool_id=tool.tool_id,
            description=getattr(tool, 'description', 'Approved PeopleOS evidence tool.'),
        )

    def list_descriptors(self) -> List[Dict[str, object]]:
        """Return the complete catalog, including conditional/unavailable tools."""
        descriptors: List[Dict[str, object]] = []
        for tool_id in self.list_ids():
            tool = self._tools[tool_id]
            descriptor = self._descriptor(tool).as_dict()
            checker = getattr(tool, 'is_available', None)
            if callable(checker):
                try:
                    descriptor['runtime_available'] = bool(checker())
                except Exception:
                    descriptor['runtime_available'] = False
                reason = getattr(tool, 'availability_reason', None)
                if callable(reason):
                    try:
                        reason = reason()
                    except Exception:
                        reason = None
                if reason:
                    descriptor['availability_reason'] = str(reason)
            descriptors.append(descriptor)
        return descriptors
