"""`produces` entries may be glob patterns.

A step that writes a variable number of files — `analyze` produced five figures
in run 001 and four in run 002 — cannot declare them individually, so before
this they went undeclared and the canvas showed them as working files. A pattern
is still a contract: at least one file must match, or the step fails exactly as a
missing literal does.
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import RunnerEvent, WorkflowRunner
from dsagent.runner.runner import is_pattern
from tests.fakes import paths_for, produces_of

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def test_only_magic_characters_make_a_pattern():
    assert is_pattern("artifacts/figures/*.png")
    assert is_pattern("report/findings.?ml")
    assert is_pattern("artifacts/[abc].md")
    assert not is_pattern("artifacts/findings.md")
    assert not is_pattern("report/findings.html")


class FakeAgent:
    """Writes what the prompt asks for, expanding any pattern it is given."""

    calls: ClassVar[list[str]] = []

    def __init__(self, persona: str, workspace: Path, figures: int = 2):
        self.persona, self.workspace, self.figures = persona, workspace, figures

    def invoke(self, payload):
        prompt = payload["messages"][0]["content"]
        FakeAgent.calls.append(self.persona)
        for entry in produces_of(prompt):
            for rel in self._expand(entry):
                p = self.workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"written by {self.persona}")
        return {"messages": [{"role": "assistant", "content": "done"}]}

    def _expand(self, entry: str) -> list[str]:
        """`figures` controls how many files a pattern gets — 0 leaves it unmet."""
        return paths_for(entry, self.figures)


class FakeBackend:
    def close(self):
        pass


@pytest.fixture
def runner(tmp_path, monkeypatch):
    FakeAgent.calls = []
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )

    def make(figures: int = 2):
        events: list[RunnerEvent] = []
        cart = load_cartridge(DS)
        r = WorkflowRunner(
            cart, tmp_path / "run",
            agent_factory=lambda c, persona, env, ws: FakeAgent(persona, ws, figures),
            log=lambda m: None, on_event=events.append,
        )
        return r, events

    return make


def _files(events):
    return {e.value["path"]: e.value for e in events if e.name == "dsagent.file"}


# --- the cartridge declares it ---------------------------------------------


def test_the_analyze_step_declares_its_figures():
    wf = load_cartridge(DS).workflows["eda-to-report"]
    step = next(s for s in wf.steps if s.id == "analyze")
    assert "artifacts/figures/*.png" in step.produces
    assert "artifacts/findings.md" in step.produces


# --- verification -----------------------------------------------------------


def test_a_pattern_is_satisfied_by_any_match(runner):
    r, _ = runner(figures=3)
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "done"
    assert state.steps["analyze"].status == "done"


def test_a_pattern_with_no_match_fails_the_step(runner):
    """A pattern is a contract, not a hint — zero figures is a failed step."""
    r, _ = runner(figures=0)
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert state.status == "failed"
    assert state.steps["analyze"].status == "failed"
    assert "artifacts/figures/*.png" in state.steps["analyze"].error
    assert state.steps["report"].status == "pending"


def test_a_missing_literal_still_fails_the_same_way(runner, tmp_path):
    """Regression: globs did not change how a plain path is checked."""
    r, _ = runner()

    class NoFindings(FakeAgent):
        def _expand(self, entry):
            return [] if entry == "artifacts/findings.md" else super()._expand(entry)

    r.agent_factory = lambda c, persona, env, ws: NoFindings(persona, ws)
    state = r.run("eda-to-report", {"data_path": "x.csv"})
    assert "artifacts/findings.md" in state.steps["analyze"].error


# --- event payloads ---------------------------------------------------------


def test_matched_files_are_deliverables(runner):
    r, events = runner(figures=3)
    r.run("eda-to-report", {"data_path": "x.csv"})
    files = _files(events)

    for i in (1, 2, 3):
        path = f"artifacts/figures/{i:02d}.png"
        assert files[path]["kind"] == "deliverable", path
        assert files[path]["step"] == "analyze"
    assert files["artifacts/findings.md"]["kind"] == "deliverable"


def test_a_file_the_pattern_does_not_cover_stays_working(runner, tmp_path):
    r, events = runner(figures=1)

    class WithStray(FakeAgent):
        def _expand(self, entry):
            out = super()._expand(entry)
            if entry == "artifacts/findings.md":
                out = [*out, "artifacts/figures/notes.txt", "artifacts/scratch.csv"]
            return out

    r.agent_factory = lambda c, persona, env, ws: WithStray(persona, ws, 1)
    r.run("eda-to-report", {"data_path": "x.csv"})
    files = _files(events)

    assert files["artifacts/figures/01.png"]["kind"] == "deliverable"
    # same directory, wrong extension — the pattern is a path pattern, not a folder
    assert files["artifacts/figures/notes.txt"]["kind"] == "working"
    assert files["artifacts/scratch.csv"]["kind"] == "working"


def test_the_step_event_carries_the_pattern_as_declared(runner):
    """`produces` on the event is the contract, unexpanded — `docs/ui-slice.md` §3."""
    r, events = runner(figures=2)
    r.run("eda-to-report", {"data_path": "x.csv"})
    started = next(
        e.value for e in events
        if e.name == "dsagent.step" and e.value["step"] == "analyze"
    )
    assert started["produces"] == ["artifacts/findings.md", "artifacts/figures/*.png"]


def test_the_persona_is_told_a_pattern_is_one_or_more(runner, tmp_path, monkeypatch):
    """A bare `*.png` in the prompt reads like a filename to write."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    seen: list[str] = []

    class Recorder(FakeAgent):
        def invoke(self, payload):
            seen.append(payload["messages"][0]["content"])
            return super().invoke(payload)

    cart = load_cartridge(DS)
    WorkflowRunner(
        cart, tmp_path / "prompt",
        agent_factory=lambda c, persona, env, ws: Recorder(persona, ws),
        log=lambda m: None,
    ).run("eda-to-report", {"data_path": "x.csv"})

    analyze_prompt = next(p for p in seen if "step `analyze`" in p)
    assert "`artifacts/figures/*.png` (one or more matching files)" in analyze_prompt
    assert "`artifacts/findings.md`\n" in analyze_prompt


def test_run_json_records_the_matched_figures(runner, tmp_path):
    r, _ = runner(figures=4)
    r.run("eda-to-report", {"data_path": "x.csv"})
    state = json.loads((tmp_path / "run" / "run.json").read_text())
    written = {f["path"] for f in state["steps"]["analyze"]["files"]}
    assert {f"artifacts/figures/{i:02d}.png" for i in range(1, 5)} <= written
