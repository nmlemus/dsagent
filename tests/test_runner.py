"""Runner tests with a fake agent: no model, no kernel, no Docker."""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env
from dsagent.runner import GateDecision, RunState, WorkflowRunner

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class FakeAgent:
    """Writes every `produces` file it is asked for and records the prompt."""

    calls: list[tuple[str, str]] = []

    def __init__(self, persona: str, workspace: Path, skip: set[str] | None = None):
        self.persona, self.workspace, self.skip = persona, workspace, skip or set()

    def invoke(self, payload):
        prompt = payload["messages"][0]["content"]
        FakeAgent.calls.append((self.persona, prompt))
        for line in prompt.splitlines():
            if line.startswith("- `") and line.endswith("`"):
                rel = line[3:-1]
                if rel in self.skip:
                    continue
                p = self.workspace / rel
                p.mkdir(parents=True, exist_ok=True) if False else p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"written by {self.persona}")
        return {"messages": [{"role": "assistant", "content": f"{self.persona} done"}]}


class FakeBackend:
    def close(self):
        pass


@pytest.fixture
def runner_factory(tmp_path, monkeypatch):
    FakeAgent.calls = []
    # no real envs: patch make_env to a stub
    monkeypatch.setattr("dsagent.runner.runner.make_env", lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()))

    def make(skip=None, ask=None):
        c = load_cartridge(DS)
        return WorkflowRunner(
            c, tmp_path / "run",
            agent_factory=lambda cart, persona, env, ws: FakeAgent(persona, ws, skip),
            ask_human=ask or (lambda p: GateDecision.APPROVE),
            log=lambda m: None,
        )

    return make


def test_eda_workflow_runs_end_to_end(runner_factory, tmp_path):
    r = runner_factory()
    state = r.run("eda-to-report", {"data_path": "data/sales.csv"})
    assert state.status == "done"
    assert [c[0] for c in FakeAgent.calls] == ["marie", "marie", "noel", "marie"]
    assert (tmp_path / "run" / "workspace" / "report" / "findings.html").exists()
    # inputs are templated into the step instructions
    assert "`data/sales.csv`" in FakeAgent.calls[0][1]
    assert "What are the main patterns" in FakeAgent.calls[2][1]
    assert RunState.load(tmp_path / "run").steps["report"].status == "done"


def test_missing_required_input_fails_fast(runner_factory):
    with pytest.raises(ValueError, match="missing required inputs: data_path"):
        runner_factory().run("eda-to-report", {})


def test_missing_artifact_fails_step(runner_factory):
    r = runner_factory(skip={"artifacts/data-gate.md"})
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "failed"
    assert "did not produce: artifacts/data-gate.md" in state.steps["data-gate"].error
    assert state.steps["analyze"].status == "pending"


def test_human_gate_pauses_and_resumes(runner_factory, tmp_path):
    decisions = iter([GateDecision.REJECT, GateDecision.APPROVE])
    r = runner_factory(ask=lambda p: next(decisions))
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "awaiting_gate"
    assert state.steps["data-gate"].status == "awaiting_gate"
    assert [c[0] for c in FakeAgent.calls] == ["marie", "marie"]

    state = r.run("eda-to-report", {"data_path": "x.csv"}, resume=True)
    assert state.status == "done"
    # profile and data-gate were not re-run
    assert [c[0] for c in FakeAgent.calls] == ["marie", "marie", "noel", "marie"]


def test_dry_run_touches_no_agent(runner_factory):
    r = runner_factory()
    state = r.run("mmm-meridian", {"data_source": "csv", "data_path": "d.csv", "kpi": "units"}, dry_run=True)
    assert state.status == "done"
    assert FakeAgent.calls == []


def test_input_options_are_enforced(runner_factory):
    with pytest.raises(ValueError, match="must be one of"):
        runner_factory().run("mmm-meridian", {"data_source": "excel", "data_path": "d", "kpi": "k"}, dry_run=True)
