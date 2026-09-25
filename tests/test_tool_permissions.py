"""Exercise denied calls through real Deep Agents graphs, without a paid model."""

import asyncio
from pathlib import Path

import pytest
import yaml
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from dsagent.cartridge import CartridgeError, load_cartridge
from dsagent.host.build import build_orchestrator, build_persona_agent
from dsagent.host.permissions import tool_name
from tests.fakes import ScriptedChatModel, stub_env, tiny_cartridge


class RecordingBackend(LocalShellBackend):
    def __init__(self, workspace):
        super().__init__(root_dir=workspace)
        self.commands = []

    def execute(self, command, *, timeout=None):
        self.commands.append(command)
        return ExecuteResponse(output="test execution", exit_code=0)


def recording_env(spec, ws):
    env = stub_env(spec, ws)
    env.backend = RecordingBackend(ws)
    return env


class ObservedModel(ScriptedChatModel):
    schemas: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.schemas.append([tool_name(t) for t in tools])
        return self


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def invoke(graph, asynchronous):
    payload = {"messages": [{"role": "user", "content": "Run the requested tool."}]}
    if asynchronous:
        return asyncio.run(graph.ainvoke(payload))
    return graph.invoke(payload)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name,args", [
    ("execute", {"command": "touch forbidden"}),
    ("run_python", {"code": "open('forbidden', 'w').write('changed')"}),
    ("run_skill_script", {"skill": "only", "script": "effect.py"}),
    ("write_file", {"file_path": "/forbidden", "content": "changed"}),
    ("task", {"subagent_type": "general-purpose", "description": "write forbidden"}),
])
def test_persona_rejects_runtime_injected_tools(tmp_path, asynchronous, name, args):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.personas["ana"].tools = []
    ws = tmp_path / "ws"
    ws.mkdir()
    env = recording_env(cart.envs["default"], ws)
    model = ObservedModel([call(name, **args), AIMessage(content="finished")])
    graph = build_persona_agent(cart, "ana", env, ws, model=model)
    result = invoke(graph, asynchronous)
    assert not (ws / "forbidden").exists()
    assert not env.backend.commands
    assert all(name not in schema for schema in model.schemas)
    assert any(isinstance(m, ToolMessage) and m.status == "error"
               and "Permission denied" in m.content for m in result["messages"])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_granted_file_tool_executes(tmp_path, asynchronous):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.personas["ana"].tools = ["write_file"]
    ws = tmp_path / "ws"
    ws.mkdir()
    env = recording_env(cart.envs["default"], ws)
    model = ObservedModel([call("write_file", file_path="/allowed", content="yes")])
    graph = build_persona_agent(cart, "ana", env, ws, model=model)
    invoke(graph, asynchronous)
    assert (ws / "allowed").read_text() == "yes"
    assert any("write_file" in schema for schema in model.schemas)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_orchestrator_cannot_delegate_to_implicit_general_purpose(tmp_path, asynchronous):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    ws = tmp_path / "ws"
    ws.mkdir()
    env = recording_env(cart.envs["default"], ws)
    model = ObservedModel([
        call("task", subagent_type="general-purpose", description="execute a shell command"),
        call("execute", command="touch forbidden"), AIMessage(content="finished"),
    ])
    graph = build_orchestrator([cart], env, ws, model=model)
    result = invoke(graph, asynchronous)
    denied = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(denied) == 2
    assert all("Permission denied" in m.content for m in denied)
    assert not env.backend.commands


@pytest.mark.parametrize("asynchronous", [False, True])
def test_named_chat_persona_enforces_its_own_policy(tmp_path, asynchronous):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.personas["ana"].tools = ["write_file"]
    ws = tmp_path / "ws"
    ws.mkdir()
    env = recording_env(cart.envs["default"], ws)
    model = ObservedModel([
        call("task", subagent_type="ana", description="write the allowed file"),
        call("execute", command="touch forbidden"),
        call("write_file", file_path="/allowed", content="yes"),
        AIMessage(content="worker finished"), AIMessage(content="router finished"),
    ])
    graph = build_orchestrator([cart], env, ws, model=model)
    invoke(graph, asynchronous)
    assert (ws / "allowed").read_text() == "yes"
    assert not env.backend.commands
    assert all("execute" not in schema for schema in model.schemas)


def test_new_environment_tool_is_not_implicitly_granted(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    ws = tmp_path / "ws"
    ws.mkdir()
    env = stub_env(cart.envs["default"], ws)
    effects = []

    @tool
    def publish() -> str:
        """Publish an external effect."""
        effects.append("published")
        return "published"

    env.tools.append(publish)
    model = ObservedModel([call("publish"), AIMessage(content="finished")])
    invoke(build_persona_agent(cart, "ana", env, ws, model=model), False)
    assert not effects
    assert all("publish" not in schema for schema in model.schemas)
    # Explicit permission allows a host-provided tool without a core enum edit.
    cart.personas["ana"].tools = ["publish"]
    model = ObservedModel([call("publish"), AIMessage(content="finished")])
    invoke(build_persona_agent(cart, "ana", env, ws, model=model), False)
    assert effects == ["published"]


@pytest.mark.parametrize("declared", [["*"], ["read_file", "read_file"], ["task"], "execute"])
def test_malformed_tool_grants_fail_loading(tmp_path, declared):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    path = cart.root / "cartridge.yaml"
    data = yaml.safe_load(path.read_text())
    data["personas"]["ana"]["tools"] = declared
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(CartridgeError, match="invalid persona configuration"):
        load_cartridge(cart.root)


def test_unknown_tool_grant_fails_before_agent_creation(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.personas["ana"].tools = ["exceute"]
    ws = tmp_path / "ws"
    env = stub_env(cart.envs["default"], ws)
    with pytest.raises(ValueError, match="unavailable tools"):
        build_persona_agent(cart, "ana", env, ws)
    assert not ws.exists()


def test_shipped_domains_declare_different_compute_grants():
    root = Path(__file__).resolve().parents[1] / "cartridges"
    ds = load_cartridge(root / "ds", generate_commands=False)
    marketing = load_cartridge(root / "marketing", generate_commands=False)
    assert all("run_python" in p.tools for p in ds.personas.values())
    assert all(not {"execute", "run_python", "run_skill_script"} & set(p.tools)
               for p in marketing.personas.values())
