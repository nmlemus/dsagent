"""What a re-asked gate owes the person answering it: what changed.

A rejection is the one moment a run holds two versions of the same artifact — the
one that was refused and the one written next — and nothing else in the run keeps
the first. The runner copies it aside at the moment of rejection, and the screen
reads it back to diff against what is on disk now.
"""

from pathlib import Path

import pytest

from dsagent.envs.base import Env
from dsagent.runner import GateDecision, RunnerEvent, WorkflowRunner
from dsagent.runner.runner import GATE_VERSIONS
from tests.fakes import produces_of, tiny_cartridge

STEPS = [
    {"id": "check", "produces": ["artifacts/report.md"],
     "gate": {"kind": "human", "prompt": "Proceed?"}},
    {"id": "after", "needs": ["check"], "produces": ["artifacts/done.md"]},
]


class Writer:
    """Writes its promises, with whatever body the test gives it this pass."""

    def __init__(self, workspace: Path, body: list[str]):
        self.workspace, self.body = workspace, body

    def invoke(self, payload):
        text = self.body[0]
        self.body[:] = self.body[1:] or [text]
        for entry in produces_of(payload["messages"][0]["content"]):
            path = self.workspace / entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return {"messages": [{"role": "assistant", "content": "done"}]}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws,
                             backend=type("B", (), {"close": lambda s: None})()),
    )

    def make(answers: list[GateDecision], bodies: list[str]):
        events: list[RunnerEvent] = []
        cart = tiny_cartridge(tmp_path / "cartridge", STEPS)
        queue = list(answers)
        runner = WorkflowRunner(
            cart, tmp_path / "run",
            agent_factory=lambda c, persona, env, ws: Writer(ws, bodies),
            ask_human=lambda request: queue.pop(0) if queue else GateDecision.APPROVE,
            log=lambda m: None, on_event=events.append,
        )
        return runner, events

    return make


def test_a_rejected_gate_keeps_what_was_refused(run, tmp_path):
    runner, _ = run([GateDecision.REJECT], ["first version", "second version"])
    runner.run("w", {"data_path": "x.csv"})

    kept = tmp_path / "run" / GATE_VERSIONS / "check" / "v1" / "artifacts" / "report.md"
    assert kept.read_text() == "first version"
    # And the workspace still holds the refused one until the run is resumed.
    assert (tmp_path / "run" / "workspace" / "artifacts" / "report.md").read_text() == (
        "first version"
    )


def test_the_gate_event_names_what_it_kept(run):
    runner, events = run([GateDecision.REJECT], ["first version"])
    runner.run("w", {"data_path": "x.csv"})

    rejected = [
        e.value for e in events
        if e.name == "dsagent.step" and (e.value.get("gate") or {}).get("decision") == "reject"
    ]
    assert rejected, "no rejection was announced"
    assert rejected[-1]["gate"]["superseded"] == ["artifacts/report.md"]


def test_resuming_leaves_the_kept_version_alone_and_writes_a_new_one(run, tmp_path):
    runner, _ = run([GateDecision.REJECT], ["first version", "second version"])
    runner.run("w", {"data_path": "x.csv"})
    runner.run("w", {"data_path": "x.csv"}, resume=True)

    kept = tmp_path / "run" / GATE_VERSIONS / "check" / "v1" / "artifacts" / "report.md"
    current = tmp_path / "run" / "workspace" / "artifacts" / "report.md"
    assert kept.read_text() == "first version"
    assert current.read_text() == "second version"


def test_a_gate_sent_back_twice_keeps_both(run, tmp_path):
    runner, _ = run([GateDecision.REJECT], ["first", "second"])
    runner.run("w", {"data_path": "x.csv"})
    runner.ask_human = lambda request: GateDecision.REJECT
    runner.run("w", {"data_path": "x.csv"}, resume=True)

    base = tmp_path / "run" / GATE_VERSIONS / "check"
    assert sorted(p.name for p in base.glob("v*")) == ["v1", "v2"]
    assert (base / "v1" / "artifacts" / "report.md").read_text() == "first"
    assert (base / "v2" / "artifacts" / "report.md").read_text() == "second"


def test_an_approved_gate_keeps_nothing(run, tmp_path):
    """Nothing was superseded, so there is no second version to hold."""
    runner, _ = run([GateDecision.APPROVE], ["only version"])
    state = runner.run("w", {"data_path": "x.csv"})

    assert state.status == "done"
    assert not (tmp_path / "run" / GATE_VERSIONS).exists()


# ---- reading it back over HTTP ---------------------------------------------


def test_the_endpoint_serves_a_kept_version_and_refuses_a_way_out(run, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dsagent.api import add_runs_routes
    from dsagent.cartridge import load_cartridge

    runner, _ = run([GateDecision.REJECT], ["first version", "second version"])
    runner.run("w", {"data_path": "x.csv"})

    app = FastAPI()
    cart = load_cartridge(tmp_path / "cartridge")
    add_runs_routes(app, [cart], tmp_path, driver=_NoDriver())
    client = TestClient(app)

    ok = client.get("/runs/run/gate-version/check/1/artifacts/report.md")
    assert ok.status_code == 200
    assert ok.text == "first version"

    assert client.get("/runs/run/gate-version/check/1/../../run.json").status_code == 404
    assert client.get("/runs/run/gate-version/check/9/artifacts/report.md").status_code == 404
    assert client.get("/runs/nope/gate-version/check/1/artifacts/report.md").status_code == 404


class _NoDriver:
    def is_running(self, run_id: str) -> bool:
        return False

    def error_for(self, run_id: str) -> None:
        return None


def test_a_step_sent_back_that_rewrites_nothing_does_not_pass(run, tmp_path):
    """Existence is not enough on a re-entry: the refused files are still there.

    A persona that reads the note and changes nothing would otherwise satisfy
    the ordinary `produces` check with the very artifact that was refused.
    """
    runner, _ = run([GateDecision.REJECT], ["only version"])
    runner.run("w", {"data_path": "x.csv"})

    class Idle:
        """Writes nothing at all the second time round."""

        def __init__(self, workspace: Path):
            self.workspace = workspace

        def invoke(self, payload):
            return {"messages": [{"role": "assistant", "content": "nothing to do"}]}

    runner.agent_factory = lambda c, persona, env, ws: Idle(ws)
    state = runner.run("w", {"data_path": "x.csv"}, resume=True)

    assert state.status == "failed"
    assert "sent back but rewrote none of" in state.steps["check"].error


def test_rewriting_one_of_several_promises_is_enough(run, tmp_path, monkeypatch):
    """A step may legitimately need to change only one of the files it owes."""
    import time

    from dsagent.envs.base import Env
    from dsagent.runner import WorkflowRunner
    from tests.fakes import tiny_cartridge

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws,
                             backend=type("B", (), {"close": lambda s: None})()),
    )
    steps = [{"id": "check", "produces": ["a.md", "b.json"],
              "gate": {"kind": "human", "prompt": "Proceed?"}}]
    answers = [GateDecision.REJECT]

    class Half:
        """Writes both the first time, then only `a.md`."""

        seen = 0

        def __init__(self, workspace: Path):
            self.workspace = workspace

        def invoke(self, payload):
            Half.seen += 1
            (self.workspace / "a.md").write_text(f"pass {Half.seen}")
            if Half.seen == 1:
                (self.workspace / "b.json").write_text("{}")
            return {"messages": [{"role": "assistant", "content": "done"}]}

    cart = tiny_cartridge(tmp_path / "c2", steps)
    runner = WorkflowRunner(
        cart, tmp_path / "run2",
        agent_factory=lambda c, persona, env, ws: Half(ws),
        ask_human=lambda request: answers.pop(0) if answers else GateDecision.APPROVE,
        log=lambda m: None,
    )
    runner.run("w", {"data_path": "x.csv"})
    time.sleep(0.01)  # mtime resolution, not a race
    state = runner.run("w", {"data_path": "x.csv"}, resume=True)

    assert state.status == "done"
