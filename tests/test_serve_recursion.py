"""A gated workflow, driven through the served graph, must not end in RUN_ERROR.

Run 003 reached `done` and wrote every artifact, then the stream closed with
`RUN_ERROR` because the orchestrator graph exhausted LangGraph's default
recursion limit of 25 (`docs/runs/eda-to-report-003.md`). The limit is applied
per invocation to the graph the AG-UI bridge streams, so it is the orchestrator's
budget alone — a persona is compiled with `checkpointer=False` and spends its own.

These drive the real serve wiring — `build_orchestrator` with
`CopilotKitMiddleware` and a checkpointer, wrapped in `LangGraphAGUIAgent` — with
a scripted model and a fake persona, so they cost nothing and still exercise the
path that broke.
"""

import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from tests.fakes import ScriptedChatModel, expand

pytest.importorskip("copilotkit", reason="needs the 'ui' extra")
pytest.importorskip("ag_ui_langgraph", reason="needs the 'ui' extra")

# These have to come after the skips: importing them without the `ui` extra is
# the ImportError the skips exist to avoid.
from ag_ui.core import RunAgentInput
from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent
from langgraph.checkpoint.memory import InMemorySaver

from dsagent.host import build_orchestrator
from dsagent.runner import workflow_tools
from dsagent.serve import RECURSION_LIMIT, interrupt_gate

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
WORKFLOW = "eda-to-report"  # four steps, one human gate


class FakeBackend:
    def close(self):
        pass


class FakePersona:
    """Satisfies a step's `produces` without a model."""

    def __init__(self, persona: str, workspace: Path):
        self.persona, self.workspace = persona, workspace

    def invoke(self, payload):
        for rel in expand(payload["messages"][0]["content"]):
            p = self.workspace / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"written by {self.persona}")
        return {"messages": [{"role": "assistant", "content": "done"}]}


def _served(tmp_path, monkeypatch, *, recursion_limit, extra_rounds=0):
    """The serve stack, with a scripted orchestrator that runs the workflow."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    monkeypatch.setattr(
        "dsagent.runner.runner.WorkflowRunner._default_factory",
        staticmethod(lambda cart, persona, env, ws: FakePersona(persona, ws)),
    )

    cart = load_cartridge(DS)
    workspace, runs = tmp_path / "ws", tmp_path / "runs"
    workspace.mkdir()
    runs.mkdir()
    env = Env(spec=cart.envs["default"], workspace=workspace, backend=FakeBackend())

    # `extra_rounds` stands in for an orchestrator that keeps calling tools after
    # the workflow returns — reading artifacts to summarise, which is the shape
    # that spent run 003's budget. A neutral tool, so the rounds cost super-steps
    # without dragging the backend into it.
    from langchain_core.tools import tool as make_tool

    @make_tool
    def probe(n: int) -> str:
        """Stand-in for whatever the orchestrator does after a workflow."""
        return f"looked at {n}"

    replies = [
        AIMessage(content="", tool_calls=[{
            "name": "run_workflow",
            "args": {"name": WORKFLOW, "inputs": {"data_path": "x.csv"}},
            "id": "toolu_TEST",
        }]),
    ]
    for i in range(extra_rounds):
        replies.append(AIMessage(content="", tool_calls=[
            {"name": "probe", "args": {"n": i}, "id": f"toolu_probe{i}"}]))
    replies.append(AIMessage(content="The workflow completed successfully."))

    graph = build_orchestrator(
        [cart], env, workspace,
        model=ScriptedChatModel(replies),
        workflow_tools=[*workflow_tools([cart], runs, ask_human=interrupt_gate), probe],
        middleware=[CopilotKitMiddleware()],
        checkpointer=InMemorySaver(),
    )
    return LangGraphAGUIAgent(
        name="dsagent", graph=graph, config={"recursion_limit": recursion_limit}
    ), runs


def _drive(agent, resume=None):
    """One request through the bridge; returns the AG-UI event type names."""
    payload = {"command": {"resume": resume}} if resume is not None else {}
    run_input = RunAgentInput(
        thread_id="t1", run_id="r1", state={},
        messages=[{"id": "m1", "role": "user", "content": "run it"}],
        tools=[], context=[], forwarded_props=payload,
    )

    async def collect():
        seen = []
        async for event in agent.run(run_input):
            kind = getattr(getattr(event, "type", None), "value", None)
            if kind:
                seen.append(kind)
        return seen

    return asyncio.run(collect())


def test_a_gated_workflow_through_the_served_graph_ends_without_run_error(tmp_path, monkeypatch):
    agent, runs = _served(tmp_path, monkeypatch, recursion_limit=RECURSION_LIMIT)

    first = _drive(agent)
    assert "RUN_ERROR" not in first, first
    assert "RUN_FINISHED" in first

    second = _drive(agent, resume={"decision": "approve"})
    assert "RUN_ERROR" not in second, second
    assert "RUN_FINISHED" in second

    state = next(runs.glob("*/run.json"))
    import json
    run = json.loads(state.read_text())
    assert run["status"] == "done", run["status"]
    assert [s["status"] for s in run["steps"].values()] == ["done"] * 4
    assert run["steps"]["data-gate"]["gate"]["decision"] == "approve"


def test_the_default_limit_survives_an_orchestrator_that_keeps_working(tmp_path, monkeypatch):
    """The failure mode was rounds *after* the workflow, not the workflow itself."""
    agent, _ = _served(tmp_path, monkeypatch, recursion_limit=RECURSION_LIMIT, extra_rounds=20)
    assert "RUN_ERROR" not in _drive(agent)
    assert "RUN_ERROR" not in _drive(agent, resume={"decision": "approve"})


def test_langgraphs_own_default_is_what_broke_run_003(tmp_path, monkeypatch):
    """Pins the diagnosis: at 25, the same run dies once the orchestrator works on.

    The error escapes `agent.run()` rather than arriving as a `RUN_ERROR` frame,
    which is why run 003's browser reported `Error: terminated` — the SSE stream
    ended mid-run and CopilotKit synthesised the error client-side. Asserting the
    raise rather than an event is what actually happened.
    """
    from langgraph.errors import GraphRecursionError

    agent, _ = _served(tmp_path, monkeypatch, recursion_limit=25, extra_rounds=20)
    _drive(agent)
    with pytest.raises(GraphRecursionError, match="Recursion limit of 25"):
        _drive(agent, resume={"decision": "approve"})
