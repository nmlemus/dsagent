"""What a step is allowed to see of the workflow's inputs.

A step sees the inputs its own instructions interpolate, unless it declares
`sees`. Run 001 showed why: every step got every input, so the profiling step
was looking at the analysis question and answered it.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsagent.cartridge import CartridgeError, load_cartridge
from dsagent.cartridge.loader import _load_workflows
from dsagent.cartridge.models import Step
from dsagent.cli import app
from dsagent.runner.runner import _visible_inputs, visible_input_names

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
INPUTS = {"data_path": "data/x.csv", "question": "Why is the sky blue?", "key_column": "date"}


def step(**kwargs) -> Step:
    return Step(id="s", persona="marie", instructions="s.md", **kwargs)


# --- name resolution ---------------------------------------------------------


def test_placeholders_in_the_text_are_what_a_step_sees():
    assert visible_input_names(step(), "Profile `{data_path}` for {key_column}.") == [
        "data_path",
        "key_column",
    ]


def test_repeated_placeholders_are_listed_once_in_order():
    assert visible_input_names(step(), "{b} then {a} then {b}") == ["b", "a"]


def test_prose_mentions_are_not_placeholders():
    """`key_column` in backticks is documentation, not an interpolation."""
    assert visible_input_names(step(), "key uniqueness if a `key_column` was given") == []


def test_explicit_sees_overrides_the_text():
    assert visible_input_names(step(sees=["question"]), "Profile {data_path}") == ["question"]


def test_explicit_empty_sees_hides_everything():
    assert visible_input_names(step(sees=[]), "Profile {data_path}") == []


def test_visible_inputs_keeps_workflow_order_and_drops_the_rest():
    visible = _visible_inputs(step(), "{question} about {data_path}", INPUTS)
    assert list(visible) == ["data_path", "question"]  # workflow order, not text order
    assert "key_column" not in visible


# --- the real cartridge ------------------------------------------------------


def test_the_profile_step_is_not_shown_the_analysis_question():
    wf = load_cartridge(DS).workflows["eda-to-report"]
    profile, analyze = wf.step("profile"), wf.step("analyze")
    assert "question" not in visible_input_names(
        profile, (wf.path / profile.instructions).read_text()
    )
    assert "question" in visible_input_names(
        analyze, (wf.path / analyze.instructions).read_text()
    )


# --- loader validation -------------------------------------------------------


def test_loader_rejects_sees_naming_an_input_the_workflow_does_not_declare(tmp_path):
    wdir = tmp_path / "wf"
    (wdir / "steps").mkdir(parents=True)
    (wdir / "steps" / "one.md").write_text("do the thing")
    (wdir / "workflow.yaml").write_text(
        "name: wf\n"
        "inputs:\n  data_path: { type: path }\n"
        "steps:\n"
        "  - id: one\n    persona: marie\n    instructions: steps/one.md\n"
        "    sees: [data_path, questoin]\n"
    )
    with pytest.raises(CartridgeError, match="`sees` names unknown input\\(s\\): questoin"):
        _load_workflows(tmp_path, ["wf"])


# --- the prompt the persona actually receives --------------------------------


def test_task_message_lists_only_visible_inputs(tmp_path, monkeypatch):
    from dsagent.envs.base import Env
    from dsagent.runner import WorkflowRunner

    class Backend:
        def close(self):
            pass

    prompts: list[tuple[str, str]] = []

    class Recorder:
        def __init__(self, persona, workspace):
            self.persona, self.workspace = persona, workspace

        def invoke(self, payload):
            prompt = payload["messages"][0]["content"]
            prompts.append((self.persona, prompt))
            for line in prompt.splitlines():
                if line.startswith("- `") and line.endswith("`"):
                    p = self.workspace / line[3:-1]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("x")
            return {"messages": [{"role": "assistant", "content": "done"}]}

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=Backend()),
    )
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "run",
        agent_factory=lambda cart, persona, env, ws: Recorder(persona, ws),
        log=lambda m: None,
    )
    question = "Why is the sky blue?"
    runner.run("eda-to-report", {"data_path": "d.csv", "question": question, "key_column": "date"})

    by_step = dict(zip(["profile", "data-gate", "analyze", "report"], [p for _, p in prompts]))
    assert question not in by_step["profile"]
    assert "- question:" not in by_step["profile"]
    assert "- data_path: d.csv" in by_step["profile"]
    assert question in by_step["analyze"]
    assert "- (none)" in by_step["data-gate"]  # it works from artifacts on disk


def test_validate_reports_steps_that_are_shown_nothing():
    result = CliRunner().invoke(app, ["cartridge", "validate", str(DS)])
    assert result.exit_code == 0, result.output
    assert "are shown no" in result.output.replace("\n", " ")
