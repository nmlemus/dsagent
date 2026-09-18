"""Per-step telemetry: what the runner reads back from a step, with a fake agent.

No model, no kernel: the agent is scripted, so every number asserted here is one
the runner derived itself.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.runner import RunState, WorkflowRunner
from dsagent.runner.runner import StepRecord
from tests.fakes import stub_env

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
WORKFLOW = "eda-to-report"
FIRST_STEP = "profile"


def ai(tool_calls=(), usage=None, as_object=False, details=None):
    """One assistant message in the shape LangChain hands back."""
    msg = {"role": "assistant", "content": "ok", "tool_calls": list(tool_calls)}
    if usage is not None:
        msg["usage_metadata"] = dict(usage)
        if details is not None:
            msg["usage_metadata"]["input_token_details"] = details
    return SimpleNamespace(**msg) if as_object else msg


def call(name, **args):
    return {"name": name, "args": args, "id": f"c-{name}"}


class ScriptedAgent:
    """Writes whatever `produces` asks for, then replays a scripted result."""

    def __init__(self, persona, workspace, script):
        self.persona, self.workspace, self.script = persona, workspace, script

    def invoke(self, payload):
        for line in payload["messages"][0]["content"].splitlines():
            if line.startswith("- `") and line.endswith("`"):
                self._write(line[3:-1])
        for rel, mtime in self.script.get("writes", []):
            self._write(rel, mtime)
        return {"messages": self.script.get("messages") or [ai()]}

    def _write(self, rel, mtime=None):
        p = self.workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"written by {self.persona}")
        if mtime is not None:
            os.utime(p, (mtime, mtime))


@pytest.fixture
def run_step(tmp_path, monkeypatch):
    """Run the workflow with a scripted first step; return that step's record."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )

    def go(script, **kwargs):
        scripts = {FIRST_STEP: script}
        runner = WorkflowRunner(
            load_cartridge(DS),
            tmp_path / "run",
            agent_factory=lambda cart, persona, env, ws: ScriptedAgent(
                persona, ws, scripts.pop(FIRST_STEP, {}) if scripts else {}
            ),
            log=lambda m: None,
        )
        state = runner.run(WORKFLOW, {"data_path": "d.csv"}, **kwargs)
        return state, state.steps[FIRST_STEP]

    return go


def test_tool_calls_are_counted_by_name(run_step):
    _, rec = run_step({"messages": [
        ai([call("run_python", code="1"), call("run_python", code="2")]),
        ai([call("write_file", file_path="artifacts/x.md")]),
        ai(),
    ]})
    assert rec.tool_calls == {"run_python": 2, "write_file": 1}


def test_usage_is_summed_over_messages_that_report_it(run_step):
    _, rec = run_step({"messages": [
        ai(usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        ai(usage={"input_tokens": 250, "output_tokens": 35, "total_tokens": 285}),
        ai(),  # a message with no usage_metadata contributes nothing
    ]})
    assert rec.usage == {"input_tokens": 350, "output_tokens": 55}


def test_input_token_details_are_summed_alongside_the_totals(run_step):
    """Cache counts are sub-counts of input_tokens; without them cost is a ceiling."""
    _, rec = run_step({"messages": [
        ai(usage={"input_tokens": 1000, "output_tokens": 50},
           details={"cache_read": 900, "cache_creation": 100}),
        ai(usage={"input_tokens": 2000, "output_tokens": 80},
           details={"cache_read": 1950, "cache_creation": 0}),
    ]})
    assert rec.usage == {
        "input_tokens": 3000,
        "output_tokens": 130,
        "cache_read": 2850,
        "cache_creation": 100,
    }


def test_unknown_detail_keys_are_carried_through(run_step):
    """Whatever the provider reports is summed; the harness knows no key names."""
    _, rec = run_step({"messages": [
        ai(usage={"input_tokens": 10, "output_tokens": 1}, details={"ephemeral_1h_input": 7}),
    ]})
    assert rec.usage["ephemeral_1h_input"] == 7


def test_usage_carries_no_detail_keys_when_none_are_reported(run_step):
    _, rec = run_step({"messages": [ai(usage={"input_tokens": 5, "output_tokens": 2})]})
    assert rec.usage == {"input_tokens": 5, "output_tokens": 2}


def test_usage_is_zero_when_nothing_reports_it(run_step):
    _, rec = run_step({"messages": [ai()]})
    assert rec.usage == {"input_tokens": 0, "output_tokens": 0}


def test_skills_read_collects_skill_manifests_in_order_without_repeats(run_step):
    _, rec = run_step({"messages": [ai([
        call("read_file", file_path="/skills/marie/eda/SKILL.md"),
        call("read_file", file_path="artifacts/data-profile.json"),   # not a skill
        call("read_file", file_path="/skills/marie/eda/SKILL.md"),    # already recorded
        call("read_file", file_path="/skills/marie/reports/SKILL.md"),
        call("execute", command="cat /skills/marie/ml/SKILL.md"),     # not a read_file
    ])]})
    assert rec.skills_read == [
        "/skills/marie/eda/SKILL.md",
        "/skills/marie/reports/SKILL.md",
    ]


def test_message_objects_are_read_the_same_as_dicts(run_step):
    """Real chat models return objects, not dicts; the runner must not care."""
    _, rec = run_step({"messages": [
        ai([call("run_python", code="1")], usage={"input_tokens": 7, "output_tokens": 3}, as_object=True),
    ]})
    assert rec.tool_calls == {"run_python": 1}
    assert rec.usage == {"input_tokens": 7, "output_tokens": 3}


def test_files_lists_workspace_changes_oldest_first(run_step):
    _, rec = run_step({"writes": [
        ("artifacts/data-profile.md", 2_000.0),
        ("artifacts/data-profile.json", 1_000.0),   # written second, older mtime
        ("notes/scratch.txt", 3_000.0),
    ]})
    assert [f["path"] for f in rec.files] == [
        "artifacts/data-profile.json",
        "artifacts/data-profile.md",
        "notes/scratch.txt",
    ]
    assert [f["mtime"] for f in rec.files] == [1_000.0, 2_000.0, 3_000.0]


def test_files_ignores_the_harness_own_skill_copies(run_step):
    _, rec = run_step({"writes": [(".dsagent/skills/marie/eda/SKILL.md", 1_000.0)]})
    assert all(not f["path"].startswith(".dsagent") for f in rec.files)


def test_later_steps_only_report_their_own_files(run_step):
    """A file an earlier step wrote and nobody touched is not this step's work."""
    state, _ = run_step({})
    analyze = state.steps["analyze"]
    assert [f["path"] for f in analyze.files] == ["artifacts/findings.md"]


def test_telemetry_is_recorded_when_the_step_fails_its_produces(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )

    class SilentAgent(ScriptedAgent):
        def invoke(self, payload):  # produces nothing at all
            return {"messages": [ai([call("run_python", code="boom")], usage={"input_tokens": 9, "output_tokens": 1})]}

    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "run",
        agent_factory=lambda cart, persona, env, ws: SilentAgent(persona, ws, {}),
        log=lambda m: None,
    )
    state = runner.run(WORKFLOW, {"data_path": "d.csv"})
    rec = state.steps[FIRST_STEP]
    assert state.status == "failed"
    assert rec.status == "failed"
    assert rec.tool_calls == {"run_python": 1}      # the failed step is the one worth reading
    assert rec.usage == {"input_tokens": 9, "output_tokens": 1}
    assert rec.files == []


def test_telemetry_round_trips_through_run_json(run_step, tmp_path):
    _, rec = run_step({"messages": [
        ai([call("read_file", file_path="/skills/marie/eda/SKILL.md")], usage={"input_tokens": 5, "output_tokens": 2}),
    ], "writes": [("artifacts/data-profile.md", 1_000.0)]})
    reloaded = RunState.load(tmp_path / "run").steps[FIRST_STEP]
    assert reloaded.tool_calls == rec.tool_calls
    assert reloaded.usage == rec.usage
    assert reloaded.skills_read == rec.skills_read
    assert reloaded.files == rec.files
    on_disk = json.loads((tmp_path / "run" / "run.json").read_text())
    assert on_disk["steps"][FIRST_STEP]["skills_read"] == ["/skills/marie/eda/SKILL.md"]


def test_a_run_json_written_before_telemetry_still_loads(tmp_path):
    run_dir = tmp_path / "old"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "workflow": WORKFLOW, "cartridge": "ds", "inputs": {}, "status": "done",
        "steps": {FIRST_STEP: {"id": FIRST_STEP, "status": "done", "output": "", "error": ""}},
    }))
    rec = RunState.load(run_dir).steps[FIRST_STEP]
    assert rec == StepRecord(id=FIRST_STEP, status="done")
    assert (rec.tool_calls, rec.usage, rec.skills_read, rec.files) == ({}, {}, [], [])
