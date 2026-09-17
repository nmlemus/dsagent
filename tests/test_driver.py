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
