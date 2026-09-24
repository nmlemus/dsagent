"""Regression cases from the independent review of 7b187d0."""

from pathlib import Path

import pytest

from dsagent.capabilities import (
    CapabilityError,
    CapabilityProvider,
    CapabilityRegistry,
    builtin_registry,
    compose_cartridges,
)
from dsagent.cartridge import load_cartridge
from dsagent.cartridge.models import CapabilitySpec, EnvSpec, Gate
from dsagent.envs.base import SKILLS_DIR, Env, make_env
from dsagent.host import build as host
from dsagent.runner import GateDecision, RunState, WorkflowRunner
from tests.fakes import FakeBackend, tiny_cartridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["runner", "persona", "orchestrator", "tools", "server", "server-env"])
@pytest.mark.parametrize("contract", ["wrong-version", "missing-provider"])
def test_injected_compositions_cannot_bypass_requirements(tmp_path, monkeypatch, entry, contract):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.capabilities = [CapabilitySpec(
        id="script.python" if contract == "wrong-version" else "script.missing",
        version="99" if contract == "wrong-version" else None,
    )]
    composition = builtin_registry().compose(["environment.kernel", "script.python"])
    ws = tmp_path / "ws"
    env = Env(spec=cart.envs["default"], workspace=ws, backend=None, capabilities=composition)
    monkeypatch.setattr(host, "create_deep_agent", lambda **kw: pytest.fail("agent constructed"))
    with pytest.raises(CapabilityError):
        if entry == "runner":
            WorkflowRunner(cart, tmp_path / "run", capabilities=composition)
        elif entry == "persona":
            host.build_persona_agent(cart, "ana", env, ws)
        elif entry == "orchestrator":
            host.build_orchestrator([cart], env, ws)
        elif entry == "tools":
            from dsagent.runner.tools import workflow_tools

            workflow_tools([cart], tmp_path / "runs", capabilities=composition,
                           ask_human=lambda _: GateDecision.APPROVE)
        else:
            pytest.importorskip("fastapi")
            from dsagent.serve import build_app

            options = {"capabilities": composition} if entry == "server" else {}
            build_app([cart], env, ws, tmp_path / "runs", **options)
    assert not ws.exists(), "failed binding materialized skills before validating"


@pytest.mark.parametrize("entry", ["persona", "orchestrator"])
def test_public_two_argument_environment_api_still_runs_ds_scripts(tmp_path, monkeypatch, entry):
    cart = load_cartridge(ROOT / "cartridges/ds", generate_commands=False)
    ws = tmp_path / "ws"
    env = make_env(EnvSpec(name="default"), ws)
    monkeypatch.setattr(host, "create_deep_agent", lambda **kw: kw)
    try:
        if entry == "persona":
            agent = host.build_persona_agent(cart, "marie", env, ws)
        else:
            graph = host.build_orchestrator([cart], env, ws)
            agent = next(a for a in graph["subagents"] if a["name"] == "marie")
        script = ws / SKILLS_DIR / "marie/eda/scripts/probe.py"
        script.write_text('print("compatible")')
        tool = next(t for t in agent["tools"] if t.name == "run_skill_script")
        assert "exit code: 0\ncompatible" in tool.invoke({"skill": "eda", "script": "probe.py"})
    finally:
        env.close()


def test_multicartridge_orchestrator_scopes_executors_per_persona(tmp_path, monkeypatch):
    carts = [load_cartridge(ROOT / "cartridges" / name, generate_commands=False)
             for name in ("ds", "marketing")]
    ws = tmp_path / "ws"
    env = Env(spec=carts[0].envs["default"], workspace=ws, backend=FakeBackend(ws),
              capabilities=compose_cartridges(carts))
    monkeypatch.setattr(host, "create_deep_agent", lambda **kw: kw)
    graph = host.build_orchestrator(carts, env, ws)
    for persona, skill, allowed in [("marie", "eda", True), ("lucia", "briefs", False)]:
        script = ws / SKILLS_DIR / persona / skill / "scripts/probe.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('print("executed")')
        agent = next(a for a in graph["subagents"] if a["name"] == persona)
        tool = next(t for t in agent["tools"] if t.name == "run_skill_script")
        output = tool.invoke({"skill": skill, "script": "probe.py"})
        assert ("exit code: 0" in output) is allowed
    assert len(env.backend.commands) == 1


def test_ambiguous_executor_fails_at_binding_not_during_script(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.capabilities = [CapabilitySpec(id=id) for id in ("script.python", "script.alternative")]
    registry = builtin_registry().register(CapabilityProvider(
        id="script.alternative", kind="script-runner", version="1",
        factory=lambda *args: None, metadata={"extension": ".py"},
    ))
    with pytest.raises(CapabilityError, match="ambiguous"):
        compose_cartridges([cart], registry)


def test_auto_gate_wrong_kind_is_rejected_before_work(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    cart.workflows["w"].steps[0].gate = Gate(
        kind="auto", check="check.py", runner="environment.kernel",
    )
    with pytest.raises(CapabilityError, match="script-runner"):
        WorkflowRunner(cart, tmp_path / "run", agent_factory=lambda *a: pytest.fail("agent built"))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["image", "requirements", "instructions", "skill", "persona", "build", "version"])
def test_resume_rejects_changed_execution_before_touching_state(tmp_path, change):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    build = cart.root / "build"
    build.mkdir()
    (build / "Dockerfile").write_text("FROM example:1")
    cart.envs["default"].build = build
    run_dir = tmp_path / "run"
    WorkflowRunner(cart, run_dir, log=lambda _: None).run("w", {"data_path": "x"}, dry_run=True)
    before = (run_dir / "run.json").read_bytes()
    if change == "image":
        cart.envs["default"].image = "different/runtime:2"
    elif change == "requirements":
        cart.envs["default"].requirements = ["different-package==9"]
    elif change == "instructions":
        wf = cart.workflows["w"]
        (wf.path / wf.steps[0].instructions).write_text("Changed instructions")
    elif change == "skill":
        (cart.skills["only"].path / "SKILL.md").write_text("Changed skill")
    elif change == "persona":
        cart.personas["ana"].system_prompt = "Different worker"
    elif change == "build":
        (build / "Dockerfile").write_text("FROM example:2")
    else:
        cart.version = "2.0.0"
    with pytest.raises(CapabilityError, match="execution configuration or source changed"):
        WorkflowRunner(cart, run_dir, log=lambda _: None).run("w", resume=True, dry_run=True)
    assert (run_dir / "run.json").read_bytes() == before


def test_unchanged_execution_resumes_and_old_runs_adopt_once(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    run_dir = tmp_path / "run"
    runner = WorkflowRunner(cart, run_dir, log=lambda _: None)
    first = runner.run("w", {"data_path": "x"}, dry_run=True)
    fingerprint = first.execution["fingerprint"]
    assert runner.run("w", resume=True, dry_run=True).execution["fingerprint"] == fingerprint
    first.execution = {}
    first.save(run_dir)
    assert runner.run("w", resume=True, dry_run=True).execution["fingerprint"] == fingerprint
    assert "adopting legacy execution snapshot" in (run_dir / "runner.log").read_text()


def test_launcher_records_snapshot_before_the_first_agent_call(tmp_path):
    from dsagent.driver import GraphDriver

    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    driver = GraphDriver(None, [cart], tmp_path / "runs", recursion_limit=10)
    run_dir = driver.create("pending", {"data_path": "x"}, "w")
    before = (run_dir / "run.json").read_bytes()
    cart.personas["ana"].system_prompt = "Changed after launch"
    with pytest.raises(CapabilityError, match="execution configuration or source changed"):
        WorkflowRunner(cart, run_dir).run("w", resume=True, dry_run=True)
    assert (run_dir / "run.json").read_bytes() == before


def test_source_change_during_work_stops_before_gate_or_next_step(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [
        {"id": "first", "gate": {"kind": "human"}},
        {"id": "second", "needs": ["first"]},
    ])
    calls = []

    class Agent:
        def invoke(self, payload):
            calls.append("work")
            (cart.skills["only"].path / "SKILL.md").write_text("changed while working")
            return {"messages": []}

    runner = WorkflowRunner(cart, tmp_path / "run", agent_factory=lambda *a: Agent(),
                            ask_human=lambda _: pytest.fail("gate asked after source changed"),
                            log=lambda _: None)
    state = runner.run("w", {"data_path": "x"})
    assert state.status == "failed"
    assert "source changed during run" in state.steps["first"].error
    assert state.steps["second"].status == "pending"
    assert calls == ["work"]


def test_gate_provider_exception_records_failure_and_closes_all_resources(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work"}])
    wf = cart.workflows["w"]
    wf.steps[0].gate = Gate(kind="auto", check="check.py", runner="script.broken")
    (wf.path / "check.py").write_text("ignored")
    events = []

    def failing_cleanup():
        events.append("cleanup")
        raise RuntimeError("cleanup broke")

    class Backend:
        def close(self):
            events.append("backend")

    def environment(spec, workspace):
        return Env(spec=spec, workspace=workspace, backend=Backend(),
                   closers=[lambda: events.append("remaining"), failing_cleanup])

    def broken_script(*args):
        raise RuntimeError("provider unavailable")

    class Agent:
        def invoke(self, payload):
            return {"messages": []}

    registry = CapabilityRegistry()
    registry.register(CapabilityProvider(id="environment.kernel", kind="environment", version="1", factory=environment))
    registry.register(CapabilityProvider(id="script.broken", kind="script-runner", version="1", factory=broken_script))
    runner = WorkflowRunner(cart, tmp_path / "run", capabilities=compose_cartridges([cart], registry),
                            agent_factory=lambda *a: Agent(), log=lambda _: None)
    state = runner.run("w", {"data_path": "x"})
    assert state.status == "failed"
    assert "provider unavailable" in state.steps["work"].error
    assert RunState.load(tmp_path / "run").status == "failed"
    assert events == ["cleanup", "remaining", "backend"]
    runner.close()
    assert events == ["cleanup", "remaining", "backend"]


def test_changed_docker_identity_prevents_agent_execution_on_resume(tmp_path):
    cart = tiny_cartridge(tmp_path / "cart", [{"id": "work", "gate": {"kind": "human"}}])
    image = {"id": "sha256:first"}
    calls = []

    def environment(spec, workspace):
        return Env(spec=spec, workspace=workspace, backend=FakeBackend(workspace),
                   runtime_identity={"docker_image_id": image["id"]})

    class Agent:
        def invoke(self, payload):
            calls.append("work")
            return {"messages": []}

    registry = CapabilityRegistry().register(CapabilityProvider(
        id="environment.kernel", kind="environment", version="1", factory=environment,
    ))
    runner = WorkflowRunner(cart, tmp_path / "run", capabilities=compose_cartridges([cart], registry),
                            agent_factory=lambda *a: Agent(), ask_human=lambda _: GateDecision.REJECT,
                            log=lambda _: None)
    assert runner.run("w", {"data_path": "x"}).status == "awaiting_gate"
    image["id"] = "sha256:second"
    state = runner.run("w", resume=True)
    assert state.status == "failed"
    assert "runtime identity changed" in state.steps["work"].error
    assert calls == ["work"]
