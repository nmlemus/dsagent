"""Execution control acceptance tests: real runner, no network or paid model."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dsagent.execution import ExecutionService
from dsagent.execution_storage import RunBusy, locked
from dsagent.runner.runner import GateDecision, RunState
from dsagent.runner.tools import workflow_tools
from tests.fakes import produces_of, stub_env, tiny_cartridge


class Agent:
    def __init__(self, workspace, calls, block=None):
        self.workspace, self.calls, self.block = workspace, calls, block

    def invoke(self, payload):
        paths = produces_of(payload["messages"][0]["content"])
        self.calls.append(paths[0])
        if self.block:
            assert self.block.wait(10), "test failed to release step"
        for path in paths:
            (self.workspace / path).write_text(f"revision {len(self.calls)}")
        return {"messages": []}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("dsagent.runner.runner.make_env", stub_env)
    cart = tiny_cartridge(tmp_path / "cart", [
        {"id": "a", "produces": ["a.txt"], "gate": {"kind": "human"}},
        {"id": "b", "needs": ["a"], "produces": ["b.txt"], "gate": {"kind": "human"}},
        {"id": "c", "needs": ["b"], "produces": ["c.txt"]},
    ])
    calls = []

    def service(**kwargs):
        return ExecutionService([cart], tmp_path / "runs",
                                agent_factory=lambda c, p, e, ws: Agent(ws, calls),
                                isolated=False, **kwargs)

    return cart, service, calls


def idle(service, run="job", timeout=10):
    until = time.monotonic() + timeout
    while service.is_running(run) and time.monotonic() < until:
        time.sleep(.01)
    assert not service.is_running(run), "worker did not park"
    return RunState.load(service._run_dir(run))


def launch(service):
    service.create("job", {"data_path": "data.csv"}, "w")
    assert service.start("job")
    return idle(service)


def answer(service, state, decision=GateDecision.APPROVE, note=""):
    assert service.answer_gate("job", decision, note, gate_id=state.gate["gate_id"])
    return idle(service)


def test_restart_two_gates_skip_completed_steps_and_decisions(setup):
    _, make, calls = setup
    first = make()
    state = launch(first)
    assert state.status == "awaiting_gate" and state.gate["step"] == "a"
    assert state.gate["attempt"] == 1
    restarted = make()
    restarted.recover()
    same = idle(restarted)
    assert same.gate == state.gate
    state = answer(restarted, same)
    assert state.gate["step"] == "b"
    assert calls == ["a.txt", "b.txt"]
    state = answer(make(), state)
    assert state.status == "done"
    assert calls == ["a.txt", "b.txt", "c.txt"]
    assert not restarted.start("job", resume=True)


def test_duplicate_conflicting_and_stale_answers_cannot_approve_next_gate(setup):
    _, make, calls = setup
    service = make()
    first = launch(service)
    second = answer(service, first)
    old_id = first.gate["gate_id"]
    assert service.answer_gate("job", GateDecision.APPROVE, gate_id=old_id)
    assert not service.answer_gate("job", GateDecision.REJECT, gate_id=old_id)
    assert not service.answer_gate("job", GateDecision.APPROVE)
    assert idle(service).gate == second.gate
    assert calls == ["a.txt", "b.txt"]


def test_rejection_is_durable_and_new_attempt_has_new_identity(setup):
    _, make, calls = setup
    service = make()
    first = launch(service)
    rejected = answer(service, first, GateDecision.REJECT, "Needs revision")
    assert rejected.steps["a"].status == "pending"
    assert rejected.steps["a"].gate.note == "Needs revision"
    assert rejected.gate is None
    service = make()
    service.recover()
    assert calls == ["a.txt"]  # rejection does not silently pay for a retry
    assert service.start("job", resume=True)
    revised = idle(service)
    assert revised.gate["attempt"] == 2
    assert revised.gate["gate_id"] != first.gate["gate_id"]
    assert not service.answer_gate("job", GateDecision.APPROVE, gate_id=first.gate["gate_id"])
    assert calls == ["a.txt", "a.txt"]


def test_accepted_answer_survives_crash_before_dispatch(setup, monkeypatch):
    _, make, calls = setup
    service = make()
    state = launch(service)
    monkeypatch.setattr(service, "start", lambda *a, **kw: False)
    assert service.answer_gate("job", GateDecision.APPROVE, gate_id=state.gate["gate_id"])
    assert RunState.load(service._run_dir("job")).gate == state.gate
    restarted = make()
    restarted.recover()
    assert idle(restarted).gate["step"] == "b"
    assert calls == ["a.txt", "b.txt"]


def test_answer_during_environment_cleanup_is_not_lost(setup, monkeypatch):
    _, make, calls = setup
    entered, release = threading.Event(), threading.Event()
    from dsagent.runner.runner import WorkflowRunner

    close = WorkflowRunner.close

    def slow_close(runner):
        close(runner)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(WorkflowRunner, "close", slow_close)
    service = make()
    service.create("job", {"data_path": "data.csv"}, "w")
    service.start("job")
    try:
        assert entered.wait(5)
        gate = RunState.load(service._run_dir("job")).gate
        assert service.answer_gate("job", GateDecision.APPROVE, gate_id=gate["gate_id"])
    finally:
        release.set()
    assert idle(service).gate["step"] == "b"
    assert calls == ["a.txt", "b.txt"]


def test_two_services_cannot_execute_the_same_run(setup):
    _, make, calls = setup
    one, two = make(), make()
    release = threading.Event()
    one.agent_factory = two.agent_factory = lambda c, p, e, ws: Agent(ws, calls, release)
    one.create("job", {"data_path": "data.csv"}, "w")
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda service: service.start("job"), [one, two]))
        assert sorted(results) == [False, True]
        assert one.is_running("job") and two.is_running("job")
        with pytest.raises(RunBusy), two.edit("job"):
            pass
        two.recover()  # another process's live worker is not interrupted
    finally:
        release.set()
    assert idle(one).status == "awaiting_gate"
    assert calls == ["a.txt"]


def test_chat_submission_is_async_project_scoped_and_idempotent(setup, tmp_path):
    cart, _, calls = setup
    from dsagent.projects import ProjectStore
    from dsagent.sessions import session_of

    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    project = store.create("Marketing", cartridge=cart.name).id
    service = ExecutionService([cart], tmp_path / "runs", store=store, isolated=False,
                               agent_factory=lambda c, p, e, ws: Agent(ws, calls))
    tool = next(t for t in workflow_tools(
        [cart], service.runs_dir, ask_human=lambda _: pytest.fail("chat interrupt invoked"),
        execution=service, store=store, session_project=lambda _: project,
    ) if t.name == "run_workflow")
    payload = {"type": "tool_call", "id": "same-call", "name": "run_workflow",
               "args": {"name": "w", "inputs": {"data_path": "data.csv"}}}
    config = {"configurable": {"thread_id": "chat"}}
    result = json.loads(tool.invoke(payload, config=config).content)
    run_id = result["run_id"]
    state = idle(service, run_id)
    assert state.status == "awaiting_gate"
    assert Path(result["run_dir"]).parent == store.runs_dir(project)
    assert session_of(Path(result["run_dir"])) == "chat"
    assert json.loads(tool.invoke(payload, config=config).content)["run_id"] == run_id
    assert calls == ["a.txt"]


def test_launcher_uses_no_orchestrator_and_http_requires_current_gate(setup, tmp_path, monkeypatch):
    cart, _, calls = setup
    from dsagent import serve

    monkeypatch.setattr("dsagent.host.build_orchestrator",
                        lambda *a, **k: pytest.fail("launcher called the chat model"))
    app = serve.build_app([cart], None, tmp_path / "ws", tmp_path / "runs")
    app.state.driver.agent_factory = lambda c, p, e, ws: Agent(ws, calls)
    app.state.driver.isolated = False  # this suite tests the service protocol in-process
    with TestClient(app) as client:
        created = client.post("/runs", json={"workflow": "w", "inputs": {"data_path": "x"}})
        assert created.status_code == 200
        run = created.json()["run_id"]
        assert client.post(f"/runs/{run}/start").json()["started"]
        gate = idle(app.state.driver, run).gate
        assert client.post(f"/runs/{run}/gate", json={"decision": "approve"}).status_code == 400
        body = {"decision": "approve", "gate_id": gate["gate_id"]}
        assert client.post(f"/runs/{run}/gate", json=body).status_code == 200
        assert idle(app.state.driver, run).gate["step"] == "b"
        body["decision"] = "reject"
        assert client.post(f"/runs/{run}/gate", json=body).status_code == 409


def test_os_process_exit_releases_owner_and_never_replays_uncertain_step(setup, tmp_path):
    cart, make, calls = setup
    service = make()
    run_dir = service.create("job", {"data_path": "x"}, "w")
    # Kill the process *inside* a step, after an observable effect, before the
    # runner can save its result or release ownership normally.
    script = """
import os, sys, time
from pathlib import Path
from dsagent.cartridge import load_cartridge
from dsagent.execution import ExecutionService
import dsagent.runner.runner as runner
from tests.fakes import stub_env
runner.make_env = stub_env
class Agent:
    def invoke(self, _):
        Path(sys.argv[2], 'job', 'workspace', 'external-effect').write_text('happened')
        os._exit(17)
service = ExecutionService([load_cartridge(Path(sys.argv[1]))], Path(sys.argv[2]), isolated=False,
                           agent_factory=lambda *args: Agent())
service.start('job')
while True: time.sleep(.1)
"""
    result = subprocess.run([sys.executable, "-c", script, str(cart.root), str(service.runs_dir)],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 17, result.stderr
    assert (run_dir / "workspace/external-effect").read_text() == "happened"
    assert RunState.load(run_dir).steps["a"].status == "running"
    service.recover()
    recovered = idle(service)
    assert recovered.status == "interrupted"
    assert recovered.steps["a"].status == "interrupted"
    assert calls == []
    assert not service.start("job")
    assert service.start("job", resume=True)
    assert idle(service).gate["attempt"] == 2
    assert calls == ["a.txt"]


def test_pending_start_request_survives_failure_to_spawn(setup, monkeypatch):
    _, make, _ = setup
    service = make()
    service.create("job", {"data_path": "x"}, "w")
    with monkeypatch.context() as context:
        def fail(_):
            raise RuntimeError("could not spawn worker")
        context.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError):
            service.start("job")
    restarted = make()
    restarted.recover()
    assert idle(restarted).status == "awaiting_gate"


def test_unstarted_runs_stay_pending_and_active_owner_is_not_recovered(setup):
    _, make, _ = setup
    service = make()
    run_dir = service.create("job", {"data_path": "x"}, "w")
    service.recover()
    assert idle(service).status == "pending"
    with locked(run_dir):
        state = RunState.load(run_dir)
        state.status = "running"
        state.save(run_dir)
        service.recover()
        assert RunState.load(run_dir).status == "running"


def test_owner_lock_is_enforced_between_os_processes(setup):
    _, make, _ = setup
    service = make()
    run_dir = service.create("job", {"data_path": "x"}, "w")
    script = """
import sys
from pathlib import Path
from dsagent.execution_storage import locked, RunBusy
try:
    with locked(Path(sys.argv[1])): pass
except RunBusy:
    sys.exit(23)
"""
    with locked(run_dir):
        result = subprocess.run([sys.executable, "-c", script, str(run_dir)],
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 23, result.stderr
    result = subprocess.run([sys.executable, "-c", script, str(run_dir)],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_process_restart_at_gate_keeps_work_and_accepts_http_decision(setup):
    cart, make, calls = setup
    service = make()
    run_dir = service.create("job", {"data_path": "x"}, "w")
    script = """
import os, sys, time
from pathlib import Path
from dsagent.cartridge import load_cartridge
from dsagent.execution import ExecutionService
from dsagent.runner.runner import RunState
import dsagent.runner.runner as runner
from tests.fakes import stub_env
from tests.test_execution_service import Agent
runner.make_env = stub_env
service = ExecutionService([load_cartridge(Path(sys.argv[1]))], Path(sys.argv[2]), isolated=False,
                           agent_factory=lambda c, p, e, ws: Agent(ws, []))
service.start('job')
while service.is_running('job'): time.sleep(.01)
assert RunState.load(service._run_dir('job')).gate['step'] == 'a'
os._exit(19)
"""
    result = subprocess.run([sys.executable, "-c", script, str(cart.root), str(service.runs_dir)],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 19, result.stderr
    before = (run_dir / "workspace/a.txt").stat().st_mtime_ns
    gate = RunState.load(run_dir).gate
    service.recover()
    from fastapi import FastAPI

    from dsagent.api import add_runs_routes
    app = FastAPI()
    add_runs_routes(app, [cart], service.runs_dir, service)
    with TestClient(app) as client:
        assert client.get("/runs/job").json()["awaiting"] == gate
        response = client.post("/runs/job/gate", json={"decision": "approve",
                                                      "gate_id": gate["gate_id"]})
        assert response.status_code == 200
    assert idle(service).gate["step"] == "b"
    assert (run_dir / "workspace/a.txt").stat().st_mtime_ns == before
    assert calls == ["b.txt"]


def test_administrative_edits_refuse_live_owner(setup):
    cart, make, calls = setup
    from fastapi import FastAPI

    from dsagent.api import add_runs_routes
    service = make()
    release = threading.Event()
    service.agent_factory = lambda c, p, e, ws: Agent(ws, calls, release)
    service.create("job", {"data_path": "x"}, "w")
    app = FastAPI()
    add_runs_routes(app, [cart], service.runs_dir, service)
    try:
        service.start("job")
        with TestClient(app) as client:
            assert client.delete("/runs/job").status_code == 409
            assert client.post("/runs/job/plan").status_code == 409
            assert client.put("/runs/job/data/data_path", content=b"x").status_code == 409
    finally:
        release.set()
    assert idle(service).gate["step"] == "a"


def test_stop_parked_run_invalidates_unanswered_gate(setup):
    _, make, calls = setup
    service = make()
    original = launch(service)
    assert service.stop("job") == {"stopping": False, "status": "stopped"}
    assert not service.answer_gate("job", GateDecision.APPROVE,
                                   gate_id=original.gate["gate_id"])
    assert service.start("job", resume=True)
    resumed = idle(service)
    assert resumed.gate["gate_id"] != original.gate["gate_id"]
    assert calls == ["a.txt"]


def test_stop_in_flight_is_observed_before_gate(setup):
    _, make, calls = setup
    service = make()
    release = threading.Event()
    service.agent_factory = lambda c, p, e, ws: Agent(ws, calls, release)
    service.create("job", {"data_path": "x"}, "w")
    service.start("job")
    try:
        assert service.stop("job")["stopping"]
    finally:
        release.set()
    state = idle(service)
    assert state.status == "stopped"
    assert state.gate is None


def test_legacy_pending_gate_adopts_identity_without_reexecuting_work(setup):
    _, make, calls = setup
    service = make()
    state = launch(service)
    del state.gate["gate_id"]
    del state.gate["attempt"]
    state.save(service._run_dir("job"))
    restarted = make()
    restarted.recover()
    recovered = idle(restarted)
    assert recovered.gate["gate_id"]
    assert recovered.gate["attempt"] == 1
    assert calls == ["a.txt"]
    assert answer(restarted, recovered).gate["step"] == "b"
