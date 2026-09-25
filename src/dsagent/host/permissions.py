"""Host-enforced tool grants, independent of prompts and model profiles.

This controls tool dispatch, not what arbitrary code can do once granted an
execution tool. OS isolation and per-user authorization are separate contracts.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

# A finite compatibility default: adding an environment/plugin tool must never
# silently grant it to every existing persona. Code execution is opt-in.
DEFAULT_TOOLS = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "write_todos",
    "show_chart", "show_table",
})
# These are runtime-provided tools, not necessarily available in every env.
BUILTIN_TOOLS = DEFAULT_TOOLS | {"execute", "run_python", "run_skill_script"}
POLICY_VERSION = 1


def effective_tools(persona) -> frozenset[str]:
    return DEFAULT_TOOLS if persona.tools is None else frozenset(persona.tools)


def tool_name(tool) -> str:
    if isinstance(tool, dict):
        return tool.get("name", "")
    return getattr(tool, "name", getattr(tool, "__name__", ""))


class ToolPolicy(AgentMiddleware):
    """Filter model schemas AND reject forged/replayed calls before dispatch.

    Deep Agents adds filesystem/task tools itself. Filtering only the explicit
    tools passed to its builder would not enforce the persona's contract.
    """

    def __init__(self, allowed: Iterable[str], *, delegates: Iterable[str] = ()):
        self.allowed = frozenset(allowed)
        self.delegates = frozenset(delegates)

    def filter(self, tools):
        return [tool for tool in tools if tool_name(tool) in self.allowed]

    def wrap_model_call(self, request, handler):
        return handler(request.override(tools=self.filter(request.tools)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(tools=self.filter(request.tools)))

    def _denial(self, request):
        call = request.tool_call
        name = call["name"]
        target = call["args"].get("subagent_type")
        if name in self.allowed and (
            name != "task" or isinstance(target, str) and target in self.delegates
        ):
            return None
        return ToolMessage(
            content=f"Permission denied: tool '{name}' or its delegation target is not granted.",
            name=name, tool_call_id=call["id"] or "", status="error",
        )

    def wrap_tool_call(self, request, handler):
        denied = self._denial(request)
        return denied if denied is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        denied = self._denial(request)
        return denied if denied is not None else await handler(request)


def persona_policy(persona, tools) -> ToolPolicy:
    allowed = effective_tools(persona)
    unknown = allowed - BUILTIN_TOOLS - {tool_name(tool) for tool in tools}
    if unknown:
        raise ValueError(f"persona '{persona.name}' requests unavailable tools: {sorted(unknown)}")
    if "task" in allowed:
        raise ValueError("persona delegation is not supported; the host orchestrator owns routing")
    return ToolPolicy(allowed)
