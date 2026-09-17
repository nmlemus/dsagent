"""Runner tests with a fake agent: no model, no kernel, no Docker."""

from pathlib import Path
from typing import ClassVar

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import GateAnswer, GateDecision, RunState, WorkflowRunner
from tests.fakes import paths_for, produces_of

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class FakeAgent:
    """Writes every `produces` file it is asked for and records the prompt."""

    calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, persona: str, workspace: Path, skip: set[str] | None = None):
        self.persona, self.workspace, self.skip = persona, workspace, skip or set()

    def invoke(self, payload):
        prompt = payload["messages"][0]["content"]
        FakeAgent.calls.append((self.persona, prompt))
        for entry in produces_of(prompt):
            if entry in self.skip:
                continue
            for rel in paths_for(entry):
                p = self.workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
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


def test_human_gate_pauses_and_sends_the_step_back(runner_factory, tmp_path):
    """A rejection is not a pause: the step it gated is sent back to be redone.

    What it costs is the point of saying no. `profile` is untouched — it was
    never in question — but `data-gate` returns to `pending`, and resuming runs
    it again with the reviewer's note in its prompt.
    """
    decisions = iter([GateDecision.REJECT, GateDecision.APPROVE])
    r = runner_factory(ask=lambda p: next(decisions))
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "awaiting_gate"
    assert state.steps["data-gate"].status == "pending"
    assert state.steps["data-gate"].gate.decision == "reject"
    assert state.steps["profile"].status == "done"
    assert [c[0] for c in FakeAgent.calls] == ["marie", "marie"]

    state = r.run("eda-to-report", {"data_path": "x.csv"}, resume=True)
    assert state.status == "done"
    # `profile` is not re-run; `data-gate` is, because it was sent back.
    assert [c[0] for c in FakeAgent.calls] == ["marie", "marie", "marie", "noel", "marie"]
    assert state.steps["data-gate"].gate.decision == "approve"


def test_a_step_sent_back_is_told_why(runner_factory):
    """The note is the whole content of a rejection; without it, nothing changes."""
    decisions = iter([GateDecision.REJECT, GateDecision.APPROVE])
    r = runner_factory(ask=lambda p: GateAnswer(next(decisions), "check the fog rows"))
    r.run("eda-to-report", {"data_path": "x.csv"})
    r.run("eda-to-report", {"data_path": "x.csv"}, resume=True)

    redone = [prompt for persona, prompt in FakeAgent.calls if "step `data-gate`" in prompt]
    assert len(redone) == 2
    assert "This work was sent back" not in redone[0]
    assert "check the fog rows" in redone[1]
    assert "sent back" in redone[1]


def test_dry_run_touches_no_agent(runner_factory):
    r = runner_factory()
    state = r.run("mmm-meridian", {"data_source": "csv", "data_path": "d.csv", "kpi": "units"}, dry_run=True)
    assert state.status == "done"
    assert FakeAgent.calls == []


def test_input_options_are_enforced(runner_factory):
    with pytest.raises(ValueError, match="must be one of"):
        runner_factory().run("mmm-meridian", {"data_source": "excel", "data_path": "d", "kpi": "k"}, dry_run=True)
