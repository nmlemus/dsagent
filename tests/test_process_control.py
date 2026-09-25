"""Actual subprocesses and crash injection, without model calls."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from dsagent.capabilities import CapabilityProvider, builtin_registry, compose_cartridges
from dsagent.execution import ExecutionService
from dsagent.execution_storage import durable_json
from dsagent.processes import (
    TERMINAL,
    LocalProcessProvider,
    cleanup_resources,
    identity,
    kill_group,
    matches,
    read,
    seconds,
)
from dsagent.runner.runner import GateDecision, RunState
from tests.fakes import process_agent, stub_env, tiny_cartridge


def wait_for(check, timeout=15):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = check()
        if result:
            return result
        time.sleep(.05)
    raise AssertionError("timed out waiting for condition")


@pytest.mark.parametrize("value,expected", [(None, 3600), ("2s", 2), ("1.5m", 90), ("2h", 7200)])
def test_duration(value, expected):
    assert seconds(value) == expected


@pytest.mark.parametrize("value", ["0", "-1s", "nan", "forever", "1d"])
def test_invalid_duration(value):
    with pytest.raises(ValueError):
        seconds(value)


def launch(tmp_path, code):
    provider = LocalProcessProvider()
    root = tmp_path / "process"
    provider.start([sys.executable, "-c", code], root, cwd=tmp_path, payload={})
    wait_for(lambda: read(root / "process.json").get("child"))
    return provider, root


def terminal(provider, root):
    return wait_for(lambda: (r if (r := provider.poll(root))["status"] in TERMINAL else None))


def test_cancel_really_terminates_process_and_descendant(tmp_path):
    provider, root = launch(tmp_path, """
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path('child.pid').write_text(str(child.pid))
time.sleep(120)
""")
    try:
        wait_for(lambda: (tmp_path / "child.pid").exists())
        descendant = identity(int((tmp_path / "child.pid").read_text()))
        process = read(root / "process.json")["child"]
        provider.cancel(root)
        result = terminal(provider, root)
        assert result["status"] == "cancelled", result
        assert not matches(process) and not matches(descendant)
    finally:
        provider.reconcile(root)


def test_deadline_terminates_process(tmp_path):
    provider, root = launch(tmp_path, "import time; time.sleep(120)")
    durable_json(root / "deadline.json", {"expires": time.time() + .2, "step": "train"})
    result = terminal(provider, root)
    assert result["status"] == "timed_out"
    assert result["step"] == "train"
    assert not matches(result["child"])


def test_reused_pid_is_never_signalled(monkeypatch):
    token = identity(os.getpid())
    token["created"] -= 100
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("signalled an unrelated process"))
    with pytest.raises(RuntimeError, match="identity changed"):
        kill_group(token)


def test_guardian_death_is_reconciled_before_retry(tmp_path):
    provider, root = launch(tmp_path, "import time; time.sleep(120)")
    process = read(root / "process.json")
    os.kill(process["guardian"]["pid"], signal.SIGKILL)
    result = terminal(provider, root)
    assert result["status"] == "interrupted"
    assert not matches(process["child"])


def test_owner_death_causes_guardian_to_stop_work(tmp_path):
    root = tmp_path / "process"
    script = """
from pathlib import Path
import sys, time
from dsagent.processes import LocalProcessProvider
p=LocalProcessProvider()
p.start([sys.executable,'-c','import time; time.sleep(120)'], Path(sys.argv[1]), cwd=Path.cwd(), payload={})
while True: time.sleep(.1)
"""
    owner = subprocess.Popen([sys.executable, "-c", script, str(root)])
    try:
        wait_for(lambda: (root / "process.json").exists() and read(root / "process.json").get("child"))
        owner.kill()
        owner.wait(timeout=5)
        result = terminal(LocalProcessProvider(), root)
        assert result["status"] == "interrupted"
        assert not matches(result["child"])
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def service_fixture(tmp_path, timeout="5s"):
    cart = tiny_cartridge(tmp_path / "cart", [
        {"id": "work", "produces": ["result.txt"], "timeout": timeout,
         "gate": {"kind": "human"}},
    ])
    from dsagent.capabilities import CapabilityRegistry

    registry = CapabilityRegistry().register(CapabilityProvider(
        id="environment.kernel", version="test", kind="environment", factory=stub_env,
    ))
    service = ExecutionService([cart], tmp_path / "runs", agent_factory=process_agent,
                               capabilities=compose_cartridges([cart], registry))
    run_dir = service.create("job", {"data_path": "x"}, "w")
    return service, run_dir


def test_runner_step_timeout_without_a_model(tmp_path):
    service, run_dir = service_fixture(tmp_path, "0.3s")
    (run_dir / "workspace/long").touch()
    service.start("job")
    wait_for(lambda: not service.is_running("job"))
    state = RunState.load(run_dir)
    assert state.status == "failed", state.driver_error
    assert "deadline" in state.steps["work"].error
    assert service.process_for("job")["status"] == "timed_out"
    assert not (run_dir / "workspace/result.txt").exists()


def test_service_cancel_then_retry_preserves_gate_flow(tmp_path):
    service, run_dir = service_fixture(tmp_path)
    (run_dir / "workspace/long").touch()
    service.start("job")
    wait_for(lambda: (run_dir / "workspace/entered").exists())
    assert service.stop("job")["status"] == "cancelling"
    wait_for(lambda: not service.is_running("job"))
    assert RunState.load(run_dir).status == "stopped"
    assert service.process_for("job")["status"] == "cancelled"
    (run_dir / "workspace/long").unlink()
    service.start("job", resume=True)
    wait_for(lambda: not service.is_running("job"))
    state = RunState.load(run_dir)
    assert state.status == "awaiting_gate", state.driver_error
    modified = (run_dir / "workspace/result.txt").stat().st_mtime_ns
    assert service.answer_gate("job", GateDecision.APPROVE, gate_id=state.gate["gate_id"])
    wait_for(lambda: not service.is_running("job"))
    assert RunState.load(run_dir).status == "done"
    assert (run_dir / "workspace/result.txt").stat().st_mtime_ns == modified


def test_docker_cleanup_requires_label_and_surfaces_unavailable_daemon(tmp_path, monkeypatch):
    durable_json(tmp_path / "resources/docker.json", {"kind": "docker", "name": "owned", "owner": "x"})
    calls = []
    def fake(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps([
            {"Id": "id", "Config": {"Labels": {"dsagent.execution": "wrong"}}}
        ]), "")
    monkeypatch.setattr(subprocess, "run", fake)
    assert "ownership mismatch" in cleanup_resources(tmp_path)[0]
    assert len(calls) == 1
    monkeypatch.setattr(subprocess, "run", lambda args, **kw:
                        subprocess.CompletedProcess(args, 1, "", "daemon unavailable"))
    assert "unverified" in cleanup_resources(tmp_path)[0]


def test_local_process_provider_is_registered():
    provider = builtin_registry().compose(["process.local"]).provider("process.local", kind="process")
    assert isinstance(provider.factory(), LocalProcessProvider)


def test_cancel_closes_a_real_jupyter_kernel(tmp_path):
    provider, root = launch(tmp_path, """
from pathlib import Path
from dsagent.envs.kernel import JupyterKernel
kernel = JupyterKernel(Path.cwd())
Path('kernel.ready').touch()
kernel.run('import time; time.sleep(120)')
""")
    try:
        wait_for(lambda: (tmp_path / "kernel.ready").exists(), timeout=20)
        resources = [read(p) for p in (root / "resources").glob("*.json")]
        assert len(resources) == 1
        kernel = resources[0]["identity"]
        assert matches(kernel)
        provider.cancel(root)
        result = terminal(provider, root)
        assert result["status"] == "cancelled", result
        assert not matches(kernel)
    finally:
        provider.reconcile(root)


def test_cleanup_failure_blocks_restart_until_verified(tmp_path):
    service, run_dir = service_fixture(tmp_path)
    process_root = run_dir / ".execution/processes/unresolved"
    durable_json(process_root / "process.json", {"status": "cleanup_required"})
    durable_json(process_root / "resources/unknown.json", {"kind": "process"})
    from dsagent.execution_storage import RunBusy

    with pytest.raises(RunBusy, match="cleanup is unverified"):
        service.start("job")
    assert RunState.load(run_dir).status == "cleanup_required"
    service.recover()
    assert RunState.load(run_dir).status == "cleanup_required"


def test_shell_background_children_are_cleaned_after_exit(tmp_path):
    provider, root = launch(tmp_path, """
from pathlib import Path
from dsagent.envs.controlled_shell import ControlledShellBackend
import sys
backend = ControlledShellBackend(root_dir=Path.cwd(), inherit_env=True)
result = backend.execute(f'{sys.executable} -c "import time; time.sleep(120)" & echo $! > background.pid')
assert result.exit_code == 0
""")
    result = terminal(provider, root)
    assert result["status"] == "exited" and result["exit_code"] == 0, (root / "process.log").read_text()
    for path in (root / "resources").glob("*.json"):
        assert not matches(read(path)["identity"])


def test_step_model_rejects_bad_deadline_before_execution(tmp_path):
    from dsagent.cartridge.loader import CartridgeError

    with pytest.raises(CartridgeError, match="timeout"):
        tiny_cartridge(tmp_path / "cart", [{"id": "work", "timeout": "never"}])


def test_server_crash_recovery_does_not_replay_step(tmp_path):
    service, run_dir = service_fixture(tmp_path)
    (run_dir / "workspace/long").touch()
    script = """
import sys,time
from pathlib import Path
from tests.test_process_control import service_fixture
service,_ = service_fixture(Path(sys.argv[1]))
service.start('job')
while True: time.sleep(.1)
"""
    owner = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    try:
        wait_for(lambda: (run_dir / "workspace/entered").exists())
        owner.kill()
        owner.wait(timeout=5)
        service.recover()
        state = RunState.load(run_dir)
        assert state.status == "interrupted", state.driver_error
        assert state.steps["work"].attempt == 1
        assert not service.is_running("job")
        assert not (run_dir / "workspace/result.txt").exists()
        assert service.process_for("job")["status"] == "interrupted"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_reconcile_endpoint_does_not_restart_work(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dsagent.api import add_runs_routes

    service, run_dir = service_fixture(tmp_path)
    state = RunState.load(run_dir)
    state.status = "cleanup_required"
    state.save(run_dir)
    app = FastAPI()
    add_runs_routes(app, list(service._by_workflow.values()), service.runs_dir, service)
    with TestClient(app) as client:
        assert client.post("/runs/job/reconcile").json() == {"verified": True, "status": "interrupted"}
    assert not (run_dir / "workspace/entered").exists()
