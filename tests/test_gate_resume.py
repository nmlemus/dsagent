"""Gate decisions survive re-entry, and re-entry lands in the same run.

Both properties exist for `dsagent serve`, where `run_workflow` is a tool inside
a LangGraph graph: on resume LangGraph re-executes the whole tool from the top
and matches resume values *positionally*. Two consequences, and this file pins
both (`docs/ui-slice.md` §2):

* the sequence of gate calls must not change between entries, or gate *n+1*
  receives gate *n*'s answer;
* the run id must not change between entries, or the re-entry opens an empty run
  and re-pays for every completed step.

`mmm-meridian` is the fixture because it is the cartridge's only workflow with
two human gates — `data-gate` after step 2 and `model-spec` after step 3.
"""

import json
import shlex
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.runner import GateDecision, RunState, StepRecord, WorkflowRunner
from tests.fakes import FakeBackend, expand, stub_env

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
MMM_INPUTS = {"data_source": "csv", "data_path": "d.csv", "kpi": "units"}


class FakeAgent:
    """Writes every `produces` file it is asked for and records the call."""

    calls: ClassVar[list[str]] = []

    def __init__(self, persona: str, workspace: Path):
        self.persona, self.workspace = persona, workspace

    def invoke(self, payload):
        prompt = payload["messages"][0]["content"]
        FakeAgent.calls.append(prompt.split("step `")[1].split("`")[0])
        for rel in expand(prompt):
                p = self.workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                # `fit`'s auto gate reads this file for real, so write something
                # it accepts rather than a placeholder it would reject.
                p.write_text(
                    json.dumps({"rhat_max": 1.01, "divergences": 0, "params": {"roi_m[0]": 1.0}})
                    if rel.endswith("diagnostics.json")
                    else f"written by {self.persona}"
                )
        return {"messages": [{"role": "assistant", "content": "done"}]}


class Answers:
    """A scripted human, recording which prompt got which answer."""

    def __init__(self, *decisions: GateDecision):
        self.left = list(decisions)
        self.asked: list[str] = []
        self.requests: list = []

    def __call__(self, request) -> GateDecision:
        self.asked.append(request.prompt)
        self.requests.append(request)
        return self.left.pop(0) if self.left else GateDecision.APPROVE


@pytest.fixture
def mmm(tmp_path, monkeypatch):
    FakeAgent.calls = []
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )

    def make(answers):
        return WorkflowRunner(
            load_cartridge(DS), tmp_path / "run",
            agent_factory=lambda cart, persona, env, ws: FakeAgent(persona, ws),
            ask_human=answers, log=lambda m: None,
        )

    return make


def _gate_of(state: RunState, step: str):
    return state.steps[step].gate


def test_a_rejected_gate_pauses_the_run_and_is_recorded(mmm):
    a = Answers(GateDecision.REJECT)
    state = mmm(a).run("mmm-meridian", MMM_INPUTS)
    assert state.status == "awaiting_gate"
    assert _gate_of(state, "data-gate").decision == "reject"
    assert _gate_of(state, "model-spec") is None
    assert FakeAgent.calls == ["ingest", "data-gate"]


def test_re_entry_skips_approved_work_but_still_asks_the_decided_gate(mmm):
    """The whole point: the gate call happens again, approved work does not."""
    mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    assert FakeAgent.calls == ["ingest", "data-gate"]

    a = Answers(GateDecision.APPROVE, GateDecision.REJECT)
    state = mmm(a).run("mmm-meridian", MMM_INPUTS, resume=True)

    # `ingest` was approved and is not redone. `data-gate` was sent back, so it
    # is — that is what a rejection means.
    assert FakeAgent.calls == ["ingest", "data-gate", "data-gate", "model-spec"]
    # Gate 1 was asked again — that is what keeps the interrupt sequence stable.
    assert len(a.asked) == 2
    assert "Data gate report ready" in a.asked[0]
    assert "Review priors" in a.asked[1]
    assert state.status == "awaiting_gate"


def test_the_second_gate_records_its_own_answer_not_the_first_gates(mmm):
    """The bug this PR exists for: positional resume feeding gate 2 gate 1's answer."""
    mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    state = mmm(Answers(GateDecision.APPROVE, GateDecision.REJECT)).run(
        "mmm-meridian", MMM_INPUTS, resume=True
    )
    assert _gate_of(state, "data-gate").decision == "approve"
    assert _gate_of(state, "model-spec").decision == "reject"


def test_an_approved_gate_is_asked_again_but_its_answer_is_discarded(mmm):
    """A decided gate keeps its decision no matter what the re-entry answers."""
    mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    mmm(Answers(GateDecision.APPROVE, GateDecision.REJECT)).run(
        "mmm-meridian", MMM_INPUTS, resume=True
    )
    # Gate 1 is approved. Answer REJECT to everything; gate 1 must not flip back.
    a = Answers(GateDecision.REJECT, GateDecision.REJECT)
    state = mmm(a).run("mmm-meridian", MMM_INPUTS, resume=True)
    assert _gate_of(state, "data-gate").decision == "approve"
    assert len(a.asked) == 2, "the approved gate is still asked, for sequence stability"
    assert _gate_of(state, "model-spec").decision == "reject"


def test_step_status_no_longer_carries_gate_state(mmm):
    """`status` is the work; `gate` is the decision. They are different questions.

    A rejected gate makes them disagree in the other direction too: the decision
    is recorded on a step whose work is `pending` again, because saying no is
    what sends the work back.
    """
    state = mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    assert state.steps["data-gate"].status == "pending"   # the work goes round again
    assert state.steps["data-gate"].gate.decision == "reject"
    assert state.status == "awaiting_gate"                 # the run is what is paused


def test_a_gate_record_survives_a_round_trip_through_run_json(mmm, tmp_path):
    mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    reloaded = RunState.load(tmp_path / "run")
    gate = reloaded.steps["data-gate"].gate
    assert gate.decision == "reject"
    assert gate.ts > 0
    assert gate.note == ""


def test_a_completed_run_records_approve_on_its_gate(tmp_path, monkeypatch):
    """`eda-to-report` has one human gate and no auto gate — a clean finish."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "eda",
        agent_factory=lambda cart, persona, env, ws: FakeAgent(persona, ws),
        ask_human=Answers(), log=lambda m: None,
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "done"
    assert state.steps["data-gate"].gate.decision == "approve"
    assert state.steps["profile"].gate is None  # no gate on that step


# --- auto gates ------------------------------------------------------------
#
# `fit` is the cartridge's only auto gate. The first test below reaches it the
# way a real run does, through the DAG; the rest drive `_gate` directly, which is
# the cheaper way to cover the rejection and skip branches.


def test_a_full_run_reaches_the_auto_gate_and_records_its_verdict(mmm):
    """End to end: two human gates approved, then the check script runs for real."""
    state = mmm(Answers()).run("mmm-meridian", MMM_INPUTS)
    assert state.status == "done"
    assert _gate_of(state, "data-gate").decision == "approve"
    assert _gate_of(state, "model-spec").decision == "approve"
    fit = _gate_of(state, "fit")
    assert fit.decision == "approve"
    assert "GATE PASS" in fit.note


def _fit_gate(tmp_path, monkeypatch, diagnostics: dict | None):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )
    cart = load_cartridge(DS)
    runner = WorkflowRunner(cart, tmp_path / "auto", log=lambda m: None)
    wf = cart.workflows["mmm-meridian"]
    step = next(s for s in wf.steps if s.id == "fit")
    if diagnostics is not None:
        art = runner.workspace / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "diagnostics.json").write_text(json.dumps(diagnostics))
    rec = StepRecord(id=step.id, status="done")
    state = RunState(workflow=wf.name, cartridge=cart.name, inputs={}, steps={step.id: rec})
    return runner, wf, step, rec, state


def test_a_passing_auto_gate_records_approve_and_the_scripts_output(tmp_path, monkeypatch):
    runner, wf, step, rec, state = _fit_gate(
        tmp_path, monkeypatch, {"rhat_max": 1.01, "divergences": 0, "params": {"roi_m[0]": 1.0}}
    )
    assert runner._gate(wf, step, rec, state, dry_run=False) is True
    assert rec.gate.decision == "approve"
    assert "GATE PASS" in rec.gate.note


def test_a_failing_auto_gate_records_reject_and_fails_the_step(tmp_path, monkeypatch):
    runner, wf, step, rec, state = _fit_gate(
        tmp_path, monkeypatch, {"rhat_max": 1.4, "divergences": 12, "params": {"roi_m[0]": 1.4}}
    )
    assert runner._gate(wf, step, rec, state, dry_run=False) is False
    assert rec.gate.decision == "reject"
    assert "GATE FAIL" in rec.gate.note
    assert rec.status == "failed"
    assert state.status == "failed"


def test_the_auto_gate_runs_inside_the_step_env(tmp_path, monkeypatch):
    """Not `subprocess` on the host: the check runs where the step ran.

    `fit` in a Docker env writes `artifacts/diagnostics.json` inside a container.
    A check reading that path on the host would be reading a different machine's
    filesystem — and it would read it with the host's interpreter rather than the
    one the env declares. One command, through the backend, for both env kinds.
    """
    runner, wf, step, rec, state = _fit_gate(
        tmp_path, monkeypatch, {"rhat_max": 1.01, "divergences": 0, "params": {}}
    )
    env = runner.env_for(step.env or wf.env)

    assert runner._gate(wf, step, rec, state, dry_run=False) is True

    assert env.backend.commands == [
        f"{shlex.quote(env.python)} .dsagent/gates/mmm-meridian/scripts/check_rhat.py"
    ], "the gate did not go through the env"
    # `fit` declares the Docker env, whose interpreter is the container's. The
    # gate used to hard-code `python3` on the host, which happens to be a
    # different program with different packages.
    assert env.python != sys.executable


def test_the_check_is_copied_into_the_workspace_where_an_env_can_reach_it(tmp_path, monkeypatch):
    """The script ships in the cartridge, which no env can see.

    A container is mounted on the workspace and nothing else, so a path into the
    cartridge tree resolves to nothing inside it. The check is materialized under
    `.dsagent/`, which `_snapshot` skips — so it never shows up as a file the
    step produced.
    """
    runner, wf, step, rec, state = _fit_gate(
        tmp_path, monkeypatch, {"rhat_max": 1.01, "divergences": 0, "params": {}}
    )

    runner._gate(wf, step, rec, state, dry_run=False)

    copied = runner.workspace / ".dsagent" / "gates" / "mmm-meridian" / "scripts" / "check_rhat.py"
    assert copied.is_file()
    assert copied.read_text() == (wf.path / "scripts" / "check_rhat.py").read_text()
    assert str(wf.path) not in runner.env_for(step.env or wf.env).backend.commands[0], (
        "the command names a path in the cartridge, which no container can resolve"
    )
    assert rec.files == [], "a materialized check is harness plumbing, not step output"


def test_a_passed_auto_gate_is_not_re_run(tmp_path, monkeypatch):
    """Unlike a human gate, a check script that already passed is skipped.

    Nothing in an auto gate calls `interrupt()`, so re-running it buys no
    sequence stability, and re-running a convergence check costs minutes. Proven
    by deleting the file the script needs: a second run would fail on it.
    """
    runner, wf, step, rec, state = _fit_gate(
        tmp_path, monkeypatch, {"rhat_max": 1.01, "divergences": 0, "params": {}}
    )
    assert runner._gate(wf, step, rec, state, dry_run=False) is True
    (runner.workspace / "artifacts" / "diagnostics.json").unlink()
    assert runner._gate(wf, step, rec, state, dry_run=False) is True
    assert rec.gate.decision == "approve"


# --- the run id -------------------------------------------------------------


def test_run_id_is_derived_from_the_tool_call_not_the_clock():
    """Re-entry must land in the same run directory, or it re-pays for the run."""
    from dsagent.runner import workflow_run_id

    first = workflow_run_id("eda-to-report", "toolu_01ABC")
    second = workflow_run_id("eda-to-report", "toolu_01ABC")
    assert first == second == "eda-to-report-toolu_01ABC"
    assert workflow_run_id("eda-to-report", "toolu_01XYZ") != first


def test_run_id_falls_back_to_a_timestamp_without_a_tool_call():
    from dsagent.runner import workflow_run_id

    generated = workflow_run_id("eda-to-report")
    assert generated.startswith("eda-to-report-")
    assert generated != "eda-to-report-"


def test_re_entry_reopens_the_same_run_and_keeps_the_finished_work(mmm, tmp_path):
    """The two rules together: same directory, and the decided gate still asked."""
    mmm(Answers(GateDecision.REJECT)).run("mmm-meridian", MMM_INPUTS)
    run_json = tmp_path / "run" / "run.json"
    assert run_json.exists()
    first = json.loads(run_json.read_text())
    assert first["steps"]["ingest"]["status"] == "done"

    a = Answers(GateDecision.APPROVE, GateDecision.REJECT)
    mmm(a).run("mmm-meridian", MMM_INPUTS, resume=True)

    # Same file, not a second run directory beside it.
    assert [d.name for d in tmp_path.iterdir() if d.is_dir()] == ["run"]
    second = json.loads(run_json.read_text())
    assert second["steps"]["ingest"]["started_at"] == first["steps"]["ingest"]["started_at"]
    # `ingest` is not redone; `data-gate` is, because its gate sent it back.
    assert FakeAgent.calls == ["ingest", "data-gate", "data-gate", "model-spec"]


def test_the_gate_request_carries_what_a_gate_card_has_to_show(mmm):
    """A terminal needs only the prompt; a browser needs to know which step waits."""
    a = Answers(GateDecision.REJECT)
    mmm(a).run("mmm-meridian", MMM_INPUTS)
    req = a.requests[0]
    assert req.run_id == "run"
    assert req.workflow == "mmm-meridian"
    assert req.step == "data-gate"
    assert req.persona == "pablo"
    assert req.produces == ["artifacts/data-gate.md"]
    assert "Data gate report ready" in req.prompt


def test_a_persona_never_inherits_the_callers_checkpointer(tmp_path):
    """A persona graph must not resume another persona's conversation.

    A graph compiled without its own checkpointer inherits the caller's when it
    runs inside one, under a namespace derived from the call's *position* in the
    task. `dsagent serve` re-executes `run_workflow` on resume and the runner
    skips finished steps, so step N+1's persona lands in step N's slot — and
    replays step N's finished conversation instead of running.

    Reproduced live: `analyze` failed its `produces` in milliseconds carrying
    `profile`'s telemetry and marie's `skills_read`. This pins the fix at the
    only place that can hold it.
    """
    from typing import TypedDict

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command, interrupt

    from dsagent.envs.base import Env as _Env
    from dsagent.host.build import build_persona_agent

    class ToolCapableFake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    cart = load_cartridge(DS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = _Env(spec=cart.envs["default"], workspace=workspace, backend=FakeBackend(workspace))

    def agent_for(persona: str):
        replies = iter([AIMessage(content=f"I am {persona}")])
        return build_persona_agent(
            cart, persona, env, workspace, model=ToolCapableFake(messages=replies)
        )

    class S(TypedDict):
        messages: list

    done = {"marie": False}
    said: list[tuple[str, str]] = []

    @tool
    def run_workflow() -> str:
        """Two persona steps with a gate between them."""
        if not done["marie"]:
            out = agent_for("marie").invoke({"messages": [{"role": "user", "content": "1"}]})
            said.append(("marie", out["messages"][-1].content))
            done["marie"] = True
        interrupt({"reason": "dsagent.gate"})
        out = agent_for("noel").invoke({"messages": [{"role": "user", "content": "2"}]})
        said.append(("noel", out["messages"][-1].content))
        return "done"

    def plan(state: S):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "run_workflow", "args": {}, "id": "c1"}])]}

    g = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("tools", ToolNode([run_workflow]))
    g.add_edge(START, "plan")
    g.add_edge("plan", "tools")
    g.add_edge("tools", END)
    graph = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    graph.invoke({"messages": []}, config=cfg)
    graph.invoke(Command(resume={"decision": "approve"}), config=cfg)

    assert said == [("marie", "I am marie"), ("noel", "I am noel")], said


def test_the_wait_is_measured_from_when_the_run_stopped(tmp_path, monkeypatch):
    """Under `serve` the first ask never returns — it raises an interrupt.

    The tool re-executes when the answer arrives, so a clock read at the second
    ask measures the resume, not the wait, and every gate in run 003's successor
    reported 0s. The pending record in `run.json` is what remembers the moment the
    run actually stopped; this asserts it is read back rather than overwritten.
    """
    import time

    from dsagent.cartridge import load_cartridge
    from dsagent.runner import GateDecision, WorkflowRunner
    from dsagent.runs import read_state, summarize
    from tests.test_runner_events import StreamingFakeAgent

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )
    ds = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
    run_dir = tmp_path / "waited"

    class Interrupted(Exception):
        """Stands in for LangGraph's own: the first ask does not return."""

    def make(ask):
        return WorkflowRunner(
            load_cartridge(ds), run_dir,
            agent_factory=lambda c, persona, env, ws: StreamingFakeAgent(persona, ws),
            ask_human=ask, log=lambda m: None,
        )

    def refuse_to_return(request):
        raise Interrupted

    with pytest.raises(Interrupted):
        make(refuse_to_return).run("eda-to-report", {"data_path": "x.csv"})

    # the run is parked, and `run.json` remembers when it stopped
    stopped = read_state(run_dir)["gate"]["asked_at"]
    assert read_state(run_dir)["status"] == "awaiting_gate"

    time.sleep(0.2)  # the person reading the gate report
    make(lambda request: GateDecision.APPROVE).run(
        "eda-to-report", {"data_path": "x.csv"}, resume=True
    )

    gate = read_state(run_dir)["steps"]["data-gate"]["gate"]
    assert gate["asked_at"] == pytest.approx(stopped), "the wait must start where the run stopped"
    assert gate["ts"] - gate["asked_at"] >= 0.2
    assert summarize(run_dir).gate_wait >= 0.2


def test_the_second_gate_keeps_its_own_wait_across_a_resume(mmm, tmp_path):
    """Gate 1 is re-asked on re-entry; it must not eat gate 2's pending record.

    Both gates write their pending record to the same `run.json` slot, one at a
    time, because a run stands at one gate at a time. But a re-asked gate 1 used
    to write *and then clear* that slot on its way past — destroying the record
    gate 2 had written before the interrupt, so gate 2's `asked_at` fell back to
    the clock and its wait read as zero. `mmm-meridian` is the fixture because it
    is the only workflow with two human gates.
    """
    import time

    from dsagent.runs import read_state, summarize

    run_dir = tmp_path / "run"

    class Interrupted(Exception):
        """Stands in for LangGraph's own: the first ask does not return."""

    class StopAtGate:
        """Answers the gates already decided; raises at the one still open."""

        def __init__(self, *decided: GateDecision):
            self.left = list(decided)
            self.asked: list[str] = []

        def __call__(self, request):
            self.asked.append(request.step)
            if self.left:
                return self.left.pop(0)
            raise Interrupted

    # first entry: park at gate 1
    with pytest.raises(Interrupted):
        mmm(StopAtGate()).run("mmm-meridian", MMM_INPUTS)
    assert read_state(run_dir)["gate"]["step"] == "data-gate"

    time.sleep(0.2)  # a person reading the data gate
    # the answer arrives: gate 1 approved, and the run walks on to gate 2 and parks
    second = StopAtGate(GateDecision.APPROVE)
    with pytest.raises(Interrupted):
        mmm(second).run("mmm-meridian", MMM_INPUTS, resume=True)

    pending = read_state(run_dir)["gate"]
    assert pending["step"] == "model-spec", "gate 2's pending record must be the one on file"
    assert summarize(run_dir).gate_wait >= 0.2, "gate 1's wait is on the record"

    time.sleep(0.2)  # a person reading the model spec
    # the second answer: gate 1 is re-asked (sequence) and gate 2 is answered
    third = StopAtGate(GateDecision.APPROVE, GateDecision.APPROVE)
    mmm(third).run("mmm-meridian", MMM_INPUTS, resume=True)

    assert third.asked[:2] == ["data-gate", "model-spec"], "the sequence must not change"
    gate2 = read_state(run_dir)["steps"]["model-spec"]["gate"]
    assert gate2["decision"] == "approve"
    assert gate2["ts"] - gate2["asked_at"] >= 0.2, "gate 2 kept its own asked_at"
    assert summarize(run_dir).gate_wait >= 0.4, "both waits count"


def test_a_decided_gate_is_never_announced_again(mmm, tmp_path):
    """A decided gate is asked again for the sequence, not for an audience.

    Once a gate's answer is in the log, no later entry may put a question about
    it back on the stream: a reader rebuilding the screen would show a card for a
    decision that was made minutes ago. An *undecided* gate re-announcing itself
    on the entry that finally answers it is a different thing and is correct —
    under `serve` the first ask raises before it can record anything, so that
    entry really is the run standing there again.
    """
    from dsagent.runs import read_events

    run_dir = tmp_path / "run"

    class Interrupted(Exception):
        pass

    class StopAtGate:
        def __init__(self, *decided):
            self.left = list(decided)

        def __call__(self, request):
            if self.left:
                return self.left.pop(0)
            raise Interrupted

    with pytest.raises(Interrupted):
        mmm(StopAtGate()).run("mmm-meridian", MMM_INPUTS)

    # resume twice; the decided gate is re-asked both times, silently
    second = StopAtGate(GateDecision.APPROVE)
    with pytest.raises(Interrupted):
        mmm(second).run("mmm-meridian", MMM_INPUTS, resume=True)
    mmm(StopAtGate(GateDecision.APPROVE, GateDecision.APPROVE)).run(
        "mmm-meridian", MMM_INPUTS, resume=True
    )

    settled: set[str] = set()
    for event in read_events(run_dir):
        value = event.get("value") or {}
        if event.get("name") != "dsagent.step" or not value.get("gate"):
            continue
        step = value["step"]
        if _is_pending_gate_event(event):
            assert step not in settled, f"{step} was announced again after it was decided"
        elif value["gate"].get("decision"):
            settled.add(step)
    assert settled == {"data-gate", "model-spec", "fit"}, settled


def _is_pending_gate_event(event: dict) -> bool:
    value = event.get("value") or {}
    return (
        event.get("name") == "dsagent.step"
        and value.get("status") == "awaiting_gate"
        and (value.get("gate") or {}).get("decision") is None
    )
