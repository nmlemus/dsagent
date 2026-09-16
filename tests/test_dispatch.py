"""The seam between the runner and the wire.

Three things to pin, and only the first is about our code:

1. every `RunnerEvent` becomes a LangChain custom event with the same name and
   payload, so `docs/ui-slice.md` §3 describes what a browser actually receives;
2. those events survive a real graph — a sync tool under an async
   `astream_events` consumer, which is exactly how `dsagent serve` will run;
3. they arrive *while the tool is still working*. If LangChain buffered them to
   the end of the tool call, the whole design would be pointless for a workflow
   that takes six minutes, and nothing else in the suite would notice.
"""

import asyncio
import time
from pathlib import Path
from typing import ClassVar, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import RunnerEvent, WorkflowRunner, dispatch_runner_event
from dsagent.runner import dispatch as dispatch_module

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class FakeAgent:
    calls: ClassVar[list[str]] = []

    def __init__(self, persona: str, workspace: Path):
        self.persona, self.workspace = persona, workspace

    def invoke(self, payload):
        prompt = payload["messages"][0]["content"]
        FakeAgent.calls.append(self.persona)
        for line in prompt.splitlines():
            if line.startswith("- `") and line.endswith("`"):
                p = self.workspace / line[3:-1]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
        return {"messages": [{"role": "assistant", "content": "done"}]}


class FakeBackend:
    def close(self):
        pass


# --- 1. the adapter --------------------------------------------------------


def test_the_adapter_passes_name_and_value_through_unchanged(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(dispatch_module, "dispatch_custom_event",
                        lambda name, value: seen.append((name, value)))
    payload = {"run_id": "r", "step": "profile", "status": "started"}
    dispatch_runner_event(RunnerEvent(name="dsagent.step", value=payload))
    assert seen == [("dsagent.step", payload)]
    assert seen[0][1] is payload, "the payload is forwarded, not rebuilt"


def test_a_run_dispatches_the_three_event_names_with_their_documented_keys(tmp_path, monkeypatch):
    """`docs/ui-slice.md` §3 is the contract; this is what actually goes out."""
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(dispatch_module, "dispatch_custom_event",
                        lambda name, value: seen.append((name, value)))
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    WorkflowRunner(
        load_cartridge(DS), tmp_path / "run",
        agent_factory=lambda cart, persona, env, ws: FakeAgent(persona, ws),
        log=lambda m: None, on_event=dispatch_runner_event,
    ).run("eda-to-report", {"data_path": "x.csv"})

    by_name: dict[str, list[dict]] = {}
    for name, value in seen:
        by_name.setdefault(name, []).append(value)
    assert set(by_name) == {"dsagent.step", "dsagent.file"}, (
        "this fake cannot stream, so it produces no tool events; see the graph test below"
    )

    step = by_name["dsagent.step"][0]
    assert set(step) >= {
        "run_id", "workflow", "step", "persona", "env", "index", "total",
        "status", "needs", "produces", "error", "ts",
    }
    file_event = by_name["dsagent.file"][0]
    assert set(file_event) >= {"run_id", "step", "path", "kind", "change", "size", "mtime", "ts"}


def test_the_adapter_refuses_to_run_outside_a_run():
    """Recorded, not worked around: it needs a callback manager to dispatch into."""
    with pytest.raises(RuntimeError, match="without a parent run id"):
        dispatch_runner_event(RunnerEvent(name="dsagent.step", value={}))


# --- 2 and 3. through a real graph -----------------------------------------


class S(TypedDict):
    messages: list


def _graph_around(fn, tool_name: str):
    """A minimal graph whose single tool call runs `fn` — the `run_workflow` shape."""
    decorated = tool(fn)
    decorated.name = tool_name

    def plan(state: S):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": tool_name, "args": {}, "id": "call_1"}])]}

    g = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("tools", ToolNode([decorated]))
    g.add_edge(START, "plan")
    g.add_edge("plan", "tools")
    g.add_edge("tools", END)
    return g.compile()


def test_events_from_a_sync_tool_reach_an_async_consumer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    def run_workflow() -> str:
        """Run a workflow."""
        WorkflowRunner(
            load_cartridge(DS), tmp_path / "run",
            agent_factory=lambda cart, persona, env, ws: FakeAgent(persona, ws),
            log=lambda m: None, on_event=dispatch_runner_event,
        ).run("eda-to-report", {"data_path": "x.csv"})
        return "ok"

    graph = _graph_around(run_workflow, "run_workflow")

    async def collect():
        out = []
        async for ev in graph.astream_events({"messages": []}, version="v2"):
            if ev["event"] == "on_custom_event":
                out.append((ev["name"], ev["data"]))
        return out

    events = asyncio.run(collect())
    names = [n for n, _ in events]
    assert "dsagent.step" in names
    assert "dsagent.file" in names
    first = next(v for n, v in events if n == "dsagent.step")
    assert first["status"] == "started"
    assert first["produces"] == ["artifacts/data-profile.md", "artifacts/data-profile.json"]


def test_the_first_event_arrives_before_the_tool_returns():
    """`docs/ui-slice.md` §5: a six-minute tool must not hold its events.

    The sleep is 2 s because the assertion has to be unambiguous — a margin
    smaller than scheduler noise would prove nothing. It is the one slow test in
    the suite, and it is the one that would catch LangChain buffering custom
    events until the tool call completes, which would make the whole event
    design useless for exactly the runs worth watching.
    """
    returned_at: dict[str, float] = {}

    def slow_tool() -> str:
        """Emit, sleep, emit."""
        dispatch_runner_event(RunnerEvent(name="dsagent.step", value={"step": "profile", "status": "started"}))
        time.sleep(2.0)
        dispatch_runner_event(RunnerEvent(name="dsagent.step", value={"step": "profile", "status": "done"}))
        returned_at["t"] = time.monotonic()
        return "ok"

    graph = _graph_around(slow_tool, "slow_tool")

    async def collect():
        started = time.monotonic()
        seen: list[tuple[str, float]] = []
        async for ev in graph.astream_events({"messages": []}, version="v2"):
            if ev["event"] == "on_custom_event":
                seen.append((ev["data"]["status"], time.monotonic()))
        return started, seen

    started, seen = asyncio.run(collect())
    assert [s for s, _ in seen] == ["started", "done"]

    first_at = seen[0][1]
    assert first_at < returned_at["t"], "the first event waited for the tool to return"
    assert first_at - started < 1.0, f"the first event took {first_at - started:.2f}s of a 2s tool"
    assert returned_at["t"] - started >= 2.0, "fixture check: the tool really did take 2s"
