"""The runner's event stream: `dsagent.step` / `dsagent.tool` / `dsagent.file`.

No model, no kernel, no AG-UI — these assert the shape and order of what the
runner hands to `on_event`, which is the only thing `dsagent serve` will need to
turn into AG-UI `CUSTOM` events later.
"""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import GateDecision, RunnerEvent, WorkflowRunner
from tests.fakes import expand

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class StreamingFakeAgent:
    """A fake persona agent that streams like a Deep Agents graph.

    Yields `(mode, chunk)` pairs for `stream_mode=["updates", "values"]`: one
    model-node update carrying tool calls, one tool-node update carrying their
    results, and a final `values` chunk equal to what `invoke()` would return.
    Files are written between the two updates so the runner has something to
    notice mid-stream.
    """

    def __init__(self, persona: str, workspace: Path, skip: set[str] | None = None,
                 strays: list[str] | None = None):
        self.persona, self.workspace, self.skip = persona, workspace, skip or set()
        self.strays = strays or []

    def _produces(self, prompt: str) -> list[str]:
        return expand(prompt)

    def stream(self, payload, stream_mode=None):
        prompt = payload["messages"][0]["content"]
        produces = self._produces(prompt)
        calls = [
            {"name": "write_file", "args": {"file_path": rel, "content": "x" * 900}, "id": f"call_{i}"}
            for i, rel in enumerate(produces)
        ]
        # Deep Agents emits middleware hooks whose update is `None`, not a dict
        # (observed: `updates[PatchToolCallsMiddleware.before_agent] -> None`).
        yield "updates", {"PatchToolCallsMiddleware.before_agent": None}
        ai = _Msg(tool_calls=calls)
        yield "updates", {"model": {"messages": [ai]}}

        for rel in [*produces, *self.strays]:
            if rel in self.skip:
                continue
            p = self.workspace / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"written by {self.persona}")
        results = [_Msg(name="write_file", tool_call_id=c["id"]) for c in calls]
        yield "updates", {"tools": {"messages": results}}

        final = _Msg(content=f"{self.persona} done")
        yield "values", {"messages": [ai, *results, final]}


class _Msg:
    """Minimal stand-in for a LangChain message."""

    def __init__(self, content="", tool_calls=None, name=None, tool_call_id=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.name = name
        self.tool_call_id = tool_call_id
        self.usage_metadata = {}


class FakeBackend:
    def close(self):
        pass


@pytest.fixture
def events_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    def make(skip=None, ask=None, strays=None):
        events: list[RunnerEvent] = []
        c = load_cartridge(DS)
        runner = WorkflowRunner(
            c, tmp_path / "run",
            agent_factory=lambda cart, persona, env, ws: StreamingFakeAgent(persona, ws, skip, strays),
            ask_human=ask or (lambda p: GateDecision.APPROVE),
            log=lambda m: None,
            on_event=events.append,
        )
        return runner, events

    return make


def _of(events, name):
    return [e.value for e in events if e.name == name]


def test_step_events_bracket_every_step_in_order(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    steps = _of(events, "dsagent.step")
    assert [(s["step"], s["status"]) for s in steps] == [
        ("profile", "started"), ("profile", "done"),
        ("data-gate", "started"), ("data-gate", "done"),
        # The gate is announced before anyone is asked and again once answered:
        # a reader who is not the one holding the mouse still sees the run stop,
        # sees what it stopped for, and sees how long it stood there.
        ("data-gate", "awaiting_gate"), ("data-gate", "done"),
        ("analyze", "started"), ("analyze", "done"),
        ("report", "started"), ("report", "done"),
    ]


def test_produces_rides_on_every_step_event(events_runner):
    """The canvas marks deliverables before they exist, so `started` carries it too."""
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    for s in _of(events, "dsagent.step"):
        assert "produces" in s, s
    started = next(s for s in _of(events, "dsagent.step") if s["step"] == "profile")
    assert started["status"] == "started"
    assert started["produces"] == ["artifacts/data-profile.md", "artifacts/data-profile.json"]


def test_step_events_carry_the_run_and_dag_position(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    first = _of(events, "dsagent.step")[0]
    assert first["run_id"] == "run"
    assert first["workflow"] == "eda-to-report"
    assert first["persona"] == "marie"
    assert first["env"] == "default"
    assert (first["index"], first["total"]) == (0, 4)
    assert first["needs"] == []
    gate_step = next(s for s in _of(events, "dsagent.step") if s["step"] == "data-gate")
    assert gate_step["needs"] == ["profile"]


def test_tool_events_are_attributed_and_paired(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    tools = _of(events, "dsagent.tool")
    assert tools, "expected tool events"
    profile_tools = [t for t in tools if t["step"] == "profile"]
    assert {t["phase"] for t in profile_tools} == {"started", "finished"}
    for t in profile_tools:
        assert t["persona"] == "marie"
        assert t["tool"] == "write_file"
        assert t["tool_call_id"].startswith("call_")
    started = [t for t in profile_tools if t["phase"] == "started"]
    finished = [t for t in profile_tools if t["phase"] == "finished"]
    assert [t["tool_call_id"] for t in started] == [t["tool_call_id"] for t in finished]


def test_tool_args_preview_never_carries_file_contents(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    for t in _of(events, "dsagent.tool"):
        blob = repr(t.get("args_preview"))
        assert len(blob) <= 600, blob[:120]
        assert "x" * 300 not in blob


def test_file_events_split_deliverables_from_working_files(events_runner):
    """Run 001 wrote two files nobody meant to deliver; `produces` is the signal."""
    runner, events = events_runner(strays=["artifacts/scratch.csv"])
    runner.run("eda-to-report", {"data_path": "x.csv"})
    by_path = {f["path"]: f for f in _of(events, "dsagent.file")}

    profile_json = by_path["artifacts/data-profile.json"]
    assert profile_json["kind"] == "deliverable"
    assert profile_json["step"] == "profile"
    assert profile_json["change"] == "created"
    assert profile_json["size"] > 0
    assert profile_json["run_id"] == "run"

    assert by_path["artifacts/scratch.csv"]["kind"] == "working"


def test_a_rewritten_file_reports_modified(events_runner):
    runner, events = events_runner(strays=["artifacts/scratch.csv"])
    runner.run("eda-to-report", {"data_path": "x.csv"})
    changes = [f["change"] for f in _of(events, "dsagent.file") if f["path"] == "artifacts/scratch.csv"]
    assert changes[0] == "created"
    assert "modified" in changes[1:], changes


def test_file_events_arrive_before_the_step_finishes(events_runner):
    """A six-minute step must not hold its files until the end."""
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    order = [(e.name, e.value.get("step"), e.value.get("status")) for e in events]
    profile_done = order.index(("dsagent.step", "profile", "done"))
    first_file = next(i for i, o in enumerate(order) if o[0] == "dsagent.file")
    assert first_file < profile_done


def test_failed_step_emits_a_failed_event_with_the_error(events_runner):
    runner, events = events_runner(skip={"artifacts/data-gate.md"})
    runner.run("eda-to-report", {"data_path": "x.csv"})
    failed = [s for s in _of(events, "dsagent.step") if s["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["step"] == "data-gate"
    assert "did not produce" in failed[0]["error"]
    assert not [s for s in _of(events, "dsagent.step") if s["step"] == "analyze"]


def test_rejected_gate_emits_awaiting_gate(events_runner):
    runner, events = events_runner(ask=lambda p: GateDecision.REJECT)
    runner.run("eda-to-report", {"data_path": "x.csv"})
    statuses = [(s["step"], s["status"]) for s in _of(events, "dsagent.step")]
    assert ("data-gate", "awaiting_gate") in statuses
    assert statuses[-1] == ("data-gate", "awaiting_gate")


def test_no_on_event_means_no_cost(events_runner, tmp_path, monkeypatch):
    """The callback is optional; the runner behaves identically without it."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    c = load_cartridge(DS)
    runner = WorkflowRunner(
        c, tmp_path / "quiet",
        agent_factory=lambda cart, persona, env, ws: StreamingFakeAgent(persona, ws),
        log=lambda m: None,
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "done"


def test_dry_run_still_reports_the_dag(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"}, dry_run=True)
    steps = _of(events, "dsagent.step")
    assert [(s["step"], s["status"]) for s in steps][:2] == [("profile", "started"), ("profile", "done")]
    assert _of(events, "dsagent.tool") == []
    assert _of(events, "dsagent.file") == []


def test_stream_contract_holds_against_a_real_deep_agents_graph(tmp_path, monkeypatch):
    """The fake above asserts our reading of the stream; this asserts the stream.

    No network and no model — a stub chat model replays two turns — but the graph
    is the real `create_deep_agent` one, so a Deep Agents upgrade that changes the
    `updates` shape fails here instead of silently emptying the canvas.
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ToolCapableFake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    def factory(cart, persona, env, ws):
        replies = iter([
            AIMessage(content="", tool_calls=[{
                "name": "write_file",
                "args": {"file_path": "artifacts/data-profile.md",
                         "content": "head" + "p" * 5000 + "SECRET-TAIL"},
                "id": "call_1",
            }]),
            AIMessage(content=f"{persona} done"),
        ])
        return create_deep_agent(model=ToolCapableFake(messages=replies), tools=[], system_prompt="t")

    events: list[RunnerEvent] = []
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "real",
        agent_factory=factory, log=lambda m: None, on_event=events.append,
    )
    runner.run("eda-to-report", {"data_path": "x.csv"})

    tools = _of(events, "dsagent.tool")
    assert [t["phase"] for t in tools[:2]] == ["started", "finished"]
    assert tools[0]["tool"] == "write_file"
    assert tools[0]["tool_call_id"] == "call_1"
    assert tools[1]["tool_call_id"] == "call_1"
    # the 5 kB payload never leaves the runner: the head is kept as a label,
    # the length is reported honestly, and the body is gone
    preview = tools[0]["args_preview"]["content"]
    assert preview.startswith("head")
    assert "SECRET-TAIL" not in preview
    assert "(5015 chars)" in preview
    assert len(preview) < 200
    assert tools[0]["args_preview"]["file_path"] == "artifacts/data-profile.md"


def test_a_step_that_dies_before_streaming_claims_no_files(events_runner, monkeypatch):
    """`analyze` blows up between the file snapshot and the stream.

    The files on disk are `profile`'s and `data-gate`'s; none may be reported
    against `analyze`, and the step must still emit `failed` with its error.
    (The runner resets its file baseline at the snapshot rather than inside
    `_drive` so this holds for any raise in between; with this fake the two
    points happen to agree, so this asserts the behaviour, not that line.)
    """
    runner, events = events_runner()
    real = WorkflowRunner._task_message

    def boom(self, wf, step, inputs):
        if step.id == "analyze":
            raise RuntimeError("instructions went missing")
        return real(self, wf, step, inputs)

    monkeypatch.setattr(WorkflowRunner, "_task_message", boom)
    runner.run("eda-to-report", {"data_path": "x.csv"})

    assert [f for f in _of(events, "dsagent.file") if f["step"] == "analyze"] == []
    failed = [s for s in _of(events, "dsagent.step") if s["status"] == "failed"]
    assert [s["step"] for s in failed] == ["analyze"]
    assert "instructions went missing" in failed[0]["error"]


def test_a_broken_consumer_does_not_break_the_run(tmp_path, monkeypatch):
    """A browser disconnecting mid-run is not a reason to lose the run."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    logged: list[str] = []
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "broken",
        agent_factory=lambda cart, persona, env, ws: StreamingFakeAgent(persona, ws),
        log=logged.append,
        on_event=lambda e: (_ for _ in ()).throw(BrokenPipeError("client gone")),
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "done"
    assert any("consumer raised" in m for m in logged)


def test_a_pending_gate_is_announced_before_anyone_is_asked(events_runner):
    """The card is not the only surface that has to know the run stopped.

    A second tab, a reloaded page and a stakeholder on a shared link all read the
    event log rather than the interrupt, so the pause has to be *in* the log — and
    it has to be there while the answer is still missing, not after it arrives.
    """
    seen: list[tuple[str, str]] = []

    def ask(request):
        seen.extend((e.value["step"], e.value["status"]) for e in events if e.name == "dsagent.step")
        return GateDecision.APPROVE

    runner, events = events_runner(ask=ask)
    runner.run("eda-to-report", {"data_path": "x.csv"})
    assert seen[-1] == ("data-gate", "awaiting_gate")

    pending = next(
        s for s in _of(events, "dsagent.step")
        if s["step"] == "data-gate" and s["status"] == "awaiting_gate"
    )
    assert pending["gate"]["kind"] == "human"
    assert pending["gate"]["prompt"].startswith("Data gate report")
    assert pending["gate"]["decision"] is None
    assert pending["gate"]["asked_at"] > 0
    assert pending["produces_matched"]["artifacts/data-gate.md"] == ["artifacts/data-gate.md"]


def test_an_answered_gate_carries_its_decision_and_the_wait(events_runner):
    from dsagent.runner import GateAnswer

    runner, events = events_runner(
        ask=lambda request: GateAnswer(GateDecision.REJECT, note="fog looks wrong")
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})

    last = _of(events, "dsagent.step")[-1]
    assert (last["step"], last["status"]) == ("data-gate", "awaiting_gate")
    assert last["gate"]["decision"] == "reject"
    assert last["gate"]["note"] == "fog looks wrong"
    assert last["gate"]["decided_at"] >= last["gate"]["asked_at"] > 0
    # and the note survives into the run record, which is what a resumed run and
    # a finished run's history are read from
    assert state.steps["data-gate"].gate.note == "fog looks wrong"


def test_an_approved_gate_reports_the_step_back_as_done(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    resolved = [
        s for s in _of(events, "dsagent.step")
        if s["step"] == "data-gate" and (s.get("gate") or {}).get("decision") == "approve"
    ]
    assert len(resolved) == 1
    assert resolved[0]["status"] == "done"


def test_a_step_without_a_gate_carries_no_gate(events_runner):
    runner, events = events_runner()
    runner.run("eda-to-report", {"data_path": "x.csv"})
    for s in _of(events, "dsagent.step"):
        if s["step"] != "data-gate":
            assert s["gate"] is None, s


def test_personas_narrate_into_the_event_stream(tmp_path, monkeypatch):
    """Who said it is known here; on the message stream it is a guess.

    M2.2 attributed narration by time — a message that began inside a step's
    window belonged to that step. The runner is the thing that opened the window,
    so it says so directly, and the note survives a reload and a run nobody
    watched, which a live message stream does not.
    """
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    class Narrating(StreamingFakeAgent):
        def stream(self, payload, stream_mode=None):
            yield "updates", {"model": {"messages": [
                _Msg(content="Reading the eda skill first."),
                # a content block list, as an Anthropic message arrives
                _Msg(content=[{"type": "thinking", "thinking": "hmm"},
                              {"type": "text", "text": "Profiling now."}]),
            ]}}
            yield from super().stream(payload, stream_mode)

    events: list[RunnerEvent] = []
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "narrated",
        agent_factory=lambda cart, persona, env, ws: Narrating(persona, ws),
        log=lambda m: None, on_event=events.append,
    )
    runner.run("eda-to-report", {"data_path": "x.csv"})

    notes = _of(events, "dsagent.note")
    first = [n for n in notes if n["step"] == "profile"]
    assert [n["text"] for n in first] == ["Reading the eda skill first.", "Profiling now."]
    assert {n["persona"] for n in first} == {"marie"}
    assert [n["persona"] for n in notes if n["step"] == "analyze"] == ["noel", "noel"]
    # a note arrives while the step is still working, not at the end
    order = [(e.name, e.value.get("step")) for e in events]
    assert order.index(("dsagent.note", "profile")) < order.index(("dsagent.file", "profile"))


def test_tool_results_are_not_narration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    class Chatty(StreamingFakeAgent):
        def stream(self, payload, stream_mode=None):
            yield "updates", {"tools": {"messages": [
                _Msg(content="a 40 kB tool result nobody wants in a narration log",
                     name="read_file", tool_call_id="call_x"),
            ]}}
            yield from super().stream(payload, stream_mode)

    events: list[RunnerEvent] = []
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "chatty",
        agent_factory=lambda cart, persona, env, ws: Chatty(persona, ws),
        log=lambda m: None, on_event=events.append,
    )
    runner.run("eda-to-report", {"data_path": "x.csv"})
    assert _of(events, "dsagent.note") == []
