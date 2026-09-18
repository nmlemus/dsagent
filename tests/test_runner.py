"""Runner tests with a fake agent: no model, no kernel, no Docker."""

from pathlib import Path
from typing import ClassVar

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.runner import GateAnswer, GateDecision, RunState, WorkflowRunner
from tests.fakes import paths_for, produces_of, stub_env

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


@pytest.fixture
def runner_factory(tmp_path, monkeypatch):
    FakeAgent.calls = []
    # no real envs: patch make_env to a stub
    monkeypatch.setattr("dsagent.runner.runner.make_env", stub_env)

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


def test_a_run_can_be_saved_from_two_threads_at_once(tmp_path):
    """`show_chart` records itself from whichever thread the tool call lands on.

    With one shared temp path the runner's own save and a tool's save collide:
    the first `os.replace` consumes `.run.json.tmp` and the second dies with
    `No such file or directory`. That killed `analyze` five minutes into a real
    run, after eleven `run_python` calls, with nothing wrong with the analysis.
    """
    import threading

    from dsagent.runner import RunState, StepRecord

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = RunState(workflow="w", cartridge="c", inputs={},
                     steps={"a": StepRecord(id="a")})
    state.save(run_dir)

    errors: list[Exception] = []
    start = threading.Barrier(8)

    def hammer() -> None:
        start.wait()
        for _ in range(40):
            try:
                state.save(run_dir)
            except Exception as e:  # noqa: BLE001 — the point is that there are none
                errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert RunState.load(run_dir).workflow == "w"
    # And no temp files left behind for the next reader to trip over.
    assert [p.name for p in run_dir.glob(".run.json*")] == []


def test_a_stop_is_honoured_between_steps_and_before_a_gate(tmp_path, monkeypatch):
    """Stop means "do not start the next thing" — including not asking a human.

    The check used to sit *after* the gate, so a stop pressed while a gated step
    was running stopped the run by way of putting a question to somebody first.
    """
    from dsagent.runner import WorkflowRunner
    from dsagent.runner.runner import STOP_FILE
    from tests.fakes import tiny_cartridge

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )
    steps = [
        {"id": "one", "produces": ["a.md"], "gate": {"kind": "human", "prompt": "Go on?"}},
        {"id": "two", "needs": ["one"], "produces": ["b.md"]},
    ]
    run_dir = tmp_path / "run"
    asked: list[str] = []

    class Writer:
        def __init__(self, workspace):
            self.workspace = workspace

        def invoke(self, payload):
            for name in ("a.md", "b.md"):
                if f"`{name}`" in payload["messages"][0]["content"]:
                    (self.workspace / name).write_text("x")
            # Somebody presses Stop while the first step is working — the same
            # way the API does it.
            (run_dir / STOP_FILE).write_text("")
            return {"messages": [{"role": "assistant", "content": "done"}]}

    runner = WorkflowRunner(
        tiny_cartridge(tmp_path / "cartridge", steps), run_dir,
        agent_factory=lambda c, persona, env, ws: Writer(ws),
        ask_human=lambda request: asked.append(request.step) or GateDecision.APPROVE,
        log=lambda m: None,
    )
    state = runner.run("w", {"data_path": "x.csv"})

    assert state.status == "stopped"
    assert asked == [], "a stopped run asked somebody a question on its way out"
    assert state.steps["one"].status == "done"      # the step in flight finished
    assert state.steps["two"].status == "pending"   # and nothing after it began
    # The request is consumed, so resuming is an ordinary resume.
    assert not (run_dir / STOP_FILE).exists()
