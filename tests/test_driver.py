"""The live driver: a launcher run, owned by the server, through the real graph.

No network and no model — a stub chat model plays the orchestrator and the
personas are the same streaming fake the runner tests use — but the graph is the
real `create_deep_agent` one with the real `run_workflow` tool, so the wiring
this milestone depends on is asserted rather than assumed: the run lands in the
directory the launcher made, it takes its inputs from there rather than from the
model, and its gate can be answered from outside the process that asked.
"""

import json
import time
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from dsagent.cartridge import load_cartridge
from dsagent.driver import GraphDriver, resume_message, start_message
from dsagent.envs.base import Env
from dsagent.host import build_orchestrator
from dsagent.runner import GateDecision, workflow_tools
from dsagent.runs import read_state, summarize
from dsagent.serve import interrupt_gate
from tests.test_runner_events import FakeBackend, StreamingFakeAgent

pytest.importorskip("langgraph.checkpoint.memory")

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class ToolCapableFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def orchestrator_replies(inputs: dict | None = None, turns: int = 4):
    """An orchestrator that answers every request by calling `run_workflow`.

    One call and one summary per turn, with a fresh tool-call id each time: the
    graph is re-entered on every gate answer and on every resume, and a stub that
    runs out of replies fails as "the model raised" rather than as whatever was
    being tested.
    """

    def messages():
        for turn in range(turns):
            yield AIMessage(content="", tool_calls=[{
                "name": "run_workflow",
                "args": {"name": "eda-to-report",
                         "inputs": inputs if inputs is not None else {}},
                "id": f"call_run_{turn}",
            }])
            yield AIMessage(content="The run is where it is.")

    return messages()


@pytest.fixture
def driver(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    monkeypatch.setattr(
        "dsagent.host.build.build_persona_agent",
        lambda cart, persona, env, ws, **kw: StreamingFakeAgent(persona, ws),
    )
    from langgraph.checkpoint.memory import InMemorySaver

    cart = load_cartridge(DS)
    runs_dir = tmp_path / "runs"

    def make(replies=None):
        env = Env(spec=cart.envs["default"], workspace=tmp_path / "ws", backend=FakeBackend())
        graph = build_orchestrator(
            [cart], env, tmp_path / "ws",
            model=ToolCapableFake(messages=replies or orchestrator_replies()),
            workflow_tools=workflow_tools(
                [cart], runs_dir, ask_human=interrupt_gate, log=lambda m: None
            ),
            checkpointer=InMemorySaver(),
        )
        return GraphDriver(graph, [cart], runs_dir, recursion_limit=50)

    return make


def wait_for(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def test_a_created_run_exists_before_anything_runs(driver, tmp_path):
    d = driver()
    run_dir = d.create("eda-to-report-x", {"data_path": "data/seattle.csv"}, "eda-to-report")

    state = read_state(run_dir)
    assert state["status"] == "pending"
    assert list(state["steps"]) == ["profile", "data-gate", "analyze", "report"]
    assert all(s["status"] == "pending" for s in state["steps"].values())
    assert state["inputs"]["data_path"] == "data/seattle.csv"
    assert (run_dir / "workspace").is_dir()
    assert summarize(run_dir).steps_total == 4


def test_the_run_lands_in_the_directory_the_launcher_made(driver, tmp_path):
    """Not in one named after the tool call — the upload is already in this one."""
    d = driver()
    d.create("eda-to-report-mine", {"data_path": "data/seattle.csv"}, "eda-to-report")
    d.start("eda-to-report-mine")

    assert wait_for(lambda: d._interrupted("eda-to-report-mine")), "expected the gate"
    assert d.answer_gate("eda-to-report-mine", GateDecision.APPROVE)
    assert wait_for(lambda: read_state(tmp_path / "runs" / "eda-to-report-mine")["status"] == "done")

    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["eda-to-report-mine"]
    workspace = tmp_path / "runs" / "eda-to-report-mine" / "workspace"
    assert (workspace / "report" / "findings.html").is_file()


def test_the_form_wins_over_whatever_the_model_retypes(driver, tmp_path):
    """The run's inputs are the ones written into its directory, full stop."""
    d = driver(orchestrator_replies({"data_path": "data/something-else.csv", "question": "?"}))
    d.create("eda-to-report-inputs", {"data_path": "data/seattle.csv",
                                      "question": "What happened in 2015?"}, "eda-to-report")
    d.start("eda-to-report-inputs")
    assert wait_for(lambda: d._interrupted("eda-to-report-inputs"))
    d.answer_gate("eda-to-report-inputs", GateDecision.APPROVE)
    assert wait_for(
        lambda: read_state(tmp_path / "runs" / "eda-to-report-inputs")["status"] == "done"
    )

    state = read_state(tmp_path / "runs" / "eda-to-report-inputs")
    assert state["inputs"] == {"data_path": "data/seattle.csv",
                               "question": "What happened in 2015?", "key_column": "None"}


def test_the_gate_is_answered_from_outside_the_thread_that_asked(driver, tmp_path):
    d = driver()
    run_dir = d.create("eda-to-report-gate", {"data_path": "data/seattle.csv"}, "eda-to-report")
    d.start("eda-to-report-gate")

    # the run parks itself, says so in run.json, and stops consuming anything
    assert wait_for(lambda: read_state(run_dir).get("gate"))
    gate = read_state(run_dir)["gate"]
    assert gate["step"] == "data-gate" and gate["persona"] == "marie"
    assert wait_for(lambda: not d.is_running("eda-to-report-gate"))
    assert read_state(run_dir)["status"] == "awaiting_gate"

    assert d.answer_gate("eda-to-report-gate", GateDecision.APPROVE)
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")
    assert read_state(run_dir)["steps"]["data-gate"]["gate"]["decision"] == "approve"
    assert summarize(run_dir).gate_wait > 0


def test_a_rejected_gate_keeps_its_note_and_the_run_resumes(driver, tmp_path):
    d = driver()
    run_dir = d.create("eda-to-report-reject", {"data_path": "data/seattle.csv"}, "eda-to-report")
    d.start("eda-to-report-reject")
    assert wait_for(lambda: read_state(run_dir).get("gate"))
    assert wait_for(lambda: not d.is_running("eda-to-report-reject"))
    assert d.answer_gate("eda-to-report-reject", GateDecision.REJECT, "fog rows look wrong")

    assert wait_for(lambda: read_state(run_dir)["status"] == "awaiting_gate"
                    and not d.is_running("eda-to-report-reject"))
    state = read_state(run_dir)
    assert state["steps"]["data-gate"]["gate"] == {
        **state["steps"]["data-gate"]["gate"],
        "decision": "reject", "note": "fog rows look wrong",
    }
    assert state["steps"]["analyze"]["status"] == "pending"

    d.start("eda-to-report-reject", resume=True)
    assert wait_for(lambda: d._interrupted("eda-to-report-reject"))
    assert d.answer_gate("eda-to-report-reject", GateDecision.APPROVE)
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")
    # the finished steps were not re-run: their telemetry is the first pass's
    assert read_state(run_dir)["steps"]["profile"]["status"] == "done"


def test_answering_a_run_that_is_not_waiting_is_refused(driver):
    d = driver()
    d.create("eda-to-report-idle", {"data_path": "x.csv"}, "eda-to-report")
    assert d.answer_gate("eda-to-report-idle", GateDecision.APPROVE) is False


def test_a_failure_outside_the_runner_still_reaches_the_run(driver, tmp_path):
    """No credentials, a graph that gave up — the screen must not spin forever."""
    d = driver(iter([]))  # the stub model has nothing to say and raises
    run_dir = d.create("eda-to-report-broken", {"data_path": "x.csv"}, "eda-to-report")
    d.start("eda-to-report-broken")

    assert wait_for(lambda: read_state(run_dir)["status"] == "failed")
    assert d.error_for("eda-to-report-broken")
    assert summarize(run_dir).error


def test_the_resume_message_asks_for_the_same_run_not_a_new_one():
    cart = load_cartridge(DS)
    message = resume_message(cart.workflows["eda-to-report"], "eda-to-report-1", {})
    assert "eda-to-report-1" in message
    assert "run_workflow" in message
    assert "Do not start a new run" in message


def test_the_start_message_names_the_workflow_the_run_and_the_inputs():
    cart = load_cartridge(DS)
    message = start_message(
        cart.workflows["eda-to-report"], "eda-to-report-1",
        {"data_path": "data/seattle-weather.csv"},
    )
    assert "run_workflow" in message
    assert "eda-to-report" in message
    assert json.dumps({"data_path": "data/seattle-weather.csv"}) in message
    assert "eda-to-report-1" in message


def test_a_gate_can_be_answered_the_instant_it_is_announced(driver, tmp_path, monkeypatch):
    """The operator who approves immediately must not be told to try again.

    The runner writes the pending gate and emits `awaiting_gate` *before*
    `ask_human` raises the interrupt, so for a moment the run is announced as
    waiting while its thread is still unwinding — LangGraph has to checkpoint and
    the runner has to close its envs, which with a real kernel backend is seconds.

    The window is held open here rather than raced for: a backend that takes half
    a second to close is what a Jupyter kernel is, and with fake agents the real
    window is too short to catch reliably — which is how this arrived, as a test
    that failed once in a cold run and passed four times after.
    """
    import time

    class SlowBackend(FakeBackend):
        def close(self):
            time.sleep(0.5)

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=SlowBackend()),
    )

    d = driver()
    run_dir = d.create("eda-to-report-fast", {"data_path": "data/seattle.csv"}, "eda-to-report")
    d.start("eda-to-report-fast")

    # the moment the run says it is waiting — no pause for the thread to settle
    assert wait_for(lambda: read_state(run_dir).get("gate"))
    assert d.is_running("eda-to-report-fast"), "the window this test exists for is closed"
    assert d.answer_gate("eda-to-report-fast", GateDecision.APPROVE) is True

    assert wait_for(lambda: read_state(run_dir)["status"] == "done")
    assert read_state(run_dir)["steps"]["data-gate"]["gate"]["decision"] == "approve"


def test_answering_a_run_that_is_merely_working_returns_at_once(driver, tmp_path):
    """Waiting is for a parking run, not for every wrong answer.

    A run that is simply working has nothing to park at, so the call returns
    immediately rather than holding the request open for the timeout.
    """
    import time

    d = driver()
    d.create("eda-to-report-busy", {"data_path": "data/seattle.csv"}, "eda-to-report")
    d.start("eda-to-report-busy")

    began = time.time()
    assert d.answer_gate("eda-to-report-busy", GateDecision.APPROVE, timeout=5.0) is False
    assert time.time() - began < 1.0, "a working run is not worth waiting five seconds for"


def test_two_starts_at_once_drive_the_run_only_once(driver, tmp_path):
    """Two clicks on Start are two requests on two threads."""
    import threading

    d = driver()
    d.create("eda-to-report-twice", {"data_path": "data/seattle.csv"}, "eda-to-report")

    ready = threading.Barrier(2)

    def press():
        ready.wait()
        d.start("eda-to-report-twice")

    pressers = [threading.Thread(target=press) for _ in range(2)]
    for t in pressers:
        t.start()
    for t in pressers:
        t.join(10)

    assert wait_for(lambda: read_state(tmp_path / "runs" / "eda-to-report-twice").get("gate"))
    # one thread, one invocation: the second press found the first already running
    assert len([t for t in threading.enumerate() if t.name == "run-eda-to-report-twice"]) <= 1


def test_a_run_the_orchestrator_never_started_is_failed_not_pending(driver, tmp_path):
    """An orchestrator that answers without calling the tool leaves nothing behind.

    The graph is fine, so nothing raises; the run the operator started simply
    never began. Left `pending` it looks like it might still start, for ever.
    """
    from langchain_core.messages import AIMessage

    d = driver(iter([AIMessage(content="I would rather not.")]))
    run_dir = d.create("eda-to-report-ignored", {"data_path": "x.csv"}, "eda-to-report")
    d.start("eda-to-report-ignored")

    assert wait_for(lambda: read_state(run_dir)["status"] == "failed")
    assert "did not start" in summarize(run_dir).error


def test_a_resumed_run_walks_past_a_gate_it_has_already_passed(tmp_path):
    """Retrying a failed step must not stall on a question nobody is asking.

    A resumed run re-raises the `interrupt()` for every gate it has already
    passed — it has to, or the next gate receives the previous one's answer. Each
    of those parks the graph. `state.gate` is `None` throughout, because nobody
    is being asked; the driver answers from the record and the run moves on.

    Found by pressing "Retry from analyze" on a real failed run: it read
    `running`, no thread was behind it, and the step never moved again.
    """
    from dsagent.driver import _standing_decision
    from dsagent.runner import GateDecision, GateRecord, RunState, StepRecord

    approved = RunState(
        workflow="w", cartridge="c", inputs={},
        steps={
            "gate-step": StepRecord(
                id="gate-step", status="done",
                gate=GateRecord(decision="approve", note="", ts=1.0, asked_at=0.0),
            ),
            "next": StepRecord(id="next", status="failed"),
        },
    )
    assert _standing_decision(approved) is GateDecision.APPROVE

    # A rejection leaves its step `pending`; that gate is asked for real, and the
    # driver must not answer it on the person's behalf.
    sent_back = RunState(
        workflow="w", cartridge="c", inputs={},
        steps={
            "gate-step": StepRecord(
                id="gate-step", status="pending",
                gate=GateRecord(decision="reject", note="no", ts=1.0, asked_at=0.0),
            ),
        },
    )
    assert _standing_decision(sent_back) is None
    assert _standing_decision(RunState(workflow="w", cartridge="c", inputs={}, steps={})) is None
