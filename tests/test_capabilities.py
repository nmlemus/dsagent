"""Capability composition is the executable boundary of a cartridge."""

from pathlib import Path

import pytest

from dsagent.capabilities import (
    CapabilityError,
    CapabilityProvider,
    CapabilityRegistry,
    CapabilityRequirement,
    builtin_registry,
    compose_cartridges,
)
from dsagent.cartridge import load_cartridge
from dsagent.cartridge.models import CapabilitySpec, EnvSpec
from dsagent.envs.base import Env, make_env
from dsagent.host.build import SKILLS_DIR, _skill_script_tool, materialize_skills
from dsagent.runner import WorkflowRunner
from tests.fakes import FakeBackend, tiny_cartridge

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "cartridges" / "ds"


def provider(name: str, *, requires=(), version="1") -> CapabilityProvider:
    return CapabilityProvider(
        id=name, kind="test", version=version, requires=requires,
        factory=lambda: name,
    )


def test_composition_resolves_dependencies_and_has_a_stable_manifest():
    registry = CapabilityRegistry()
    registry.register(provider("store.files"))
    registry.register(provider("tool.search", requires=("store.files",)))

    first = registry.compose(["tool.search"])
    second = registry.compose(["tool.search"])

    assert [item["id"] for item in first.manifest()] == ["store.files", "tool.search"]
    assert first.fingerprint == second.fingerprint
    assert first.provider("tool.search").factory() == "tool.search"
    registry.register(provider("tool.later"))
    assert not first.has("tool.later"), "a boot registry mutation leaked into a frozen run"


def test_composition_rejects_missing_duplicate_wrong_version_and_cycles():
    registry = CapabilityRegistry().register(provider("tool.one"))
    with pytest.raises(CapabilityError, match="already registered"):
        registry.register(provider("tool.one"))
    with pytest.raises(CapabilityError, match="no provider is registered"):
        registry.compose(["tool.missing"])
    with pytest.raises(CapabilityError, match="version 2"):
        registry.compose([CapabilityRequirement("tool.one", "2")])

    cyclic = CapabilityRegistry()
    cyclic.register(provider("tool.left", requires=("tool.right",)))
    cyclic.register(provider("tool.right", requires=("tool.left",)))
    with pytest.raises(CapabilityError, match="dependency cycle"):
        cyclic.compose(["tool.left"])


def test_an_environment_kind_can_be_supplied_without_changing_env_models(tmp_path):
    registry = CapabilityRegistry()

    def remote(spec, workspace):
        return Env(spec=spec, workspace=workspace, backend=FakeBackend(workspace))

    registry.register(CapabilityProvider(
        id="environment.remote-job", kind="environment", version="3", factory=remote
    ))
    composition = registry.compose([CapabilityRequirement("environment.remote-job", "3")])
    spec = EnvSpec(name="batch", kind="remote-job")

    env = make_env(spec, tmp_path / "workspace", composition)

    assert env.spec is spec
    assert env.capabilities is composition
    assert env.workspace.is_dir()


def test_environment_resources_close_in_reverse_acquisition_order(tmp_path):
    events = []

    class Backend:
        def close(self):
            events.append("backend")

    env = Env(
        spec=EnvSpec(name="test"), workspace=tmp_path, backend=Backend(),
        closers=[lambda: events.append("first"), lambda: events.append("second")],
    )

    env.close()

    assert events == ["second", "first", "backend"]


def test_ds_cartridge_declares_its_script_runtime_and_environment():
    cartridge = load_cartridge(DS)
    composition = compose_cartridges([cartridge])

    assert composition.has("environment.kernel")
    assert composition.has("environment.docker")
    assert composition.has("script.python")
    assert not composition.has("script.shell")


def test_skill_scripts_are_dispatched_by_declared_provider(tmp_path):
    cartridge = load_cartridge(DS)
    cartridge.capabilities = [CapabilitySpec(id="script.shell")]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    materialize_skills([cartridge], workspace)
    script = workspace / SKILLS_DIR / "marie" / "eda" / "scripts" / "hello.sh"
    script.write_text("printf 'shell:%s' \"$1\"\n")
    env = Env(
        spec=cartridge.envs["default"], workspace=workspace,
        backend=FakeBackend(workspace),
        capabilities=builtin_registry().compose([
            "environment.kernel", "environment.docker", "script.python", "script.shell",
        ]),
    )

    result = _skill_script_tool(cartridge, "marie", env, workspace).invoke({
        "skill": "eda", "script": "hello.sh", "argv": ["ok"],
    })

    assert "exit code: 0" in result
    assert "shell:ok" in result


def test_a_run_cannot_resume_under_a_different_composition(tmp_path):
    cartridge = tiny_cartridge(tmp_path / "cartridge", [{"id": "one"}])
    run_dir = tmp_path / "run"
    WorkflowRunner(cartridge, run_dir, log=lambda message: None).run(
        "w", {"data_path": "x.csv"}, dry_run=True
    )

    registry = CapabilityRegistry().register(CapabilityProvider(
        id="environment.kernel", kind="environment", version="2", factory=lambda: None,
    ))
    changed = registry.compose(["environment.kernel"])

    with pytest.raises(CapabilityError, match="execution configuration or source changed"):
        WorkflowRunner(
            cartridge, run_dir, log=lambda message: None, capabilities=changed
        ).run("w", {"data_path": "x.csv"}, resume=True, dry_run=True)
