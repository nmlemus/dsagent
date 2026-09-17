"""`show_chart` / `show_table`: what a persona may emit, and what comes back.

No model and no kernel — the tools are called the way the agent calls them, and
what is asserted is the contract between the persona and the run: a spec that
does not validate never reaches the record, a spec that does is recorded once
with its data left on disk, and a second emission of the same id is a version
rather than a twin.
"""

import json
from pathlib import Path

import pytest

from dsagent.runner.charts import ChartRecord, StepContext, chart_tools, slug, validate_spec

VL = "https://vega.github.io/schema/vega-lite/v5.json"


def good_spec() -> dict:
    return {
        "$schema": VL,
        "width": "container",
        "data": {"name": "table"},
        "mark": {"type": "bar"},
        "encoding": {
            "x": {"field": "weather", "type": "nominal"},
            "y": {"field": "precip_mean", "type": "quantitative"},
        },
    }


@pytest.fixture
def tools(tmp_path):
    """The two tools, plus the list of what they recorded."""
    workspace = tmp_path / "workspace"
    (workspace / "artifacts" / "scratch").mkdir(parents=True)
    (workspace / "artifacts" / "scratch" / "by_category.csv").write_text(
        "weather,n,precip_mean\nrain,641,5.42\nsun,640,0.0\nsnow,26,5.6\n"
    )
    recorded: list[ChartRecord] = []
    here = StepContext(step="analyze", persona="noel", section="Findings")
    show_chart, show_table = chart_tools(
        workspace,
        "run-1",
        on_chart=recorded.append,
        context=lambda: here,
        known=lambda cid: next((r for r in reversed(recorded) if r.chart_id == cid), None),
    )
    return show_chart, show_table, recorded, workspace


REF = "artifacts/scratch/by_category.csv"


def test_a_valid_spec_is_recorded_with_its_data_left_on_disk(tools):
    show_chart, _, recorded, _ = tools
    out = show_chart.invoke(
        {"spec": good_spec(), "data_ref": REF, "title": "Only rain and snow record rain"}
    )

    assert len(recorded) == 1
    rec = recorded[0]
    assert rec.kind == "chart"
    assert rec.version == 1
    assert rec.step == "analyze" and rec.persona == "noel" and rec.section == "Findings"
    # The rows stay in the file: the spec names a table, it does not carry one.
    assert rec.spec["data"] == {"name": "table"}
    assert rec.data_ref == REF
    assert rec.data_url == f"/runs/run-1/preview/{REF}"
    assert rec.rows == 3
    assert rec.columns == ["weather", "n", "precip_mean"]
    # What the persona is told back: enough to check the file, not the file.
    assert "v1" in out and "3 rows" in out
    assert "rain, 641, 5.42" in out


def test_data_the_model_inlined_is_replaced_by_the_reference(tools):
    show_chart, _, recorded, _ = tools
    spec = {**good_spec(), "data": {"values": [{"weather": "rain", "precip_mean": 99.0}]}}
    show_chart.invoke({"spec": spec, "data_ref": REF, "title": "Inlined"})

    assert recorded[0].spec["data"] == {"name": "table"}
    assert "values" not in recorded[0].spec["data"]


def test_an_invalid_spec_comes_back_with_the_validator_message(tools):
    show_chart, _, recorded, _ = tools
    spec = good_spec()
    spec["encoding"]["x"]["type"] = "nominel"
    out = show_chart.invoke({"spec": spec, "data_ref": REF, "title": "Typo"})

    assert out.startswith("error:")
    assert "nominel" in out
    assert "attempt 1 of 3" in out
    # It names the id to repair under, or the retry becomes a second chart.
    assert f'chart_id="{slug("Typo")}"' in out
    assert recorded == []


def test_a_spec_that_will_not_validate_is_sent_to_a_png_after_three_tries(tools):
    show_chart, _, recorded, _ = tools
    spec = good_spec()
    spec["encoding"]["x"]["type"] = "nominel"
    outs = [
        show_chart.invoke({"spec": spec, "data_ref": REF, "title": "Typo", "chart_id": "c1"})
        for _ in range(3)
    ]

    assert "attempt 1 of 3" in outs[0]
    assert "attempt 2 of 3" in outs[1]
    assert "write the figure as a PNG instead" in outs[2]
    assert recorded == []


def test_a_repaired_spec_is_still_version_one(tools):
    show_chart, _, recorded, _ = tools
    broken = good_spec()
    broken["encoding"]["x"]["type"] = "nominel"
    show_chart.invoke({"spec": broken, "data_ref": REF, "title": "T", "chart_id": "c1"})
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "T", "chart_id": "c1"})

    # A repair is not a revision: nothing was ever shown to a reader.
    assert [r.version for r in recorded] == [1]


def test_emitting_the_same_chart_again_is_a_new_version(tools):
    show_chart, _, recorded, _ = tools
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "T", "chart_id": "c1"})
    changed = {**good_spec(), "mark": {"type": "point"}}
    show_chart.invoke({"spec": changed, "data_ref": REF, "title": "T", "chart_id": "c1"})

    assert [r.version for r in recorded] == [1, 2]
    assert recorded[1].spec["mark"] == {"type": "point"}


def test_a_chart_id_defaults_to_its_title_so_a_repair_finds_it(tools):
    show_chart, _, recorded, _ = tools
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "Rain and snow"})
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "Rain and snow"})

    assert [r.chart_id for r in recorded] == ["rain-and-snow", "rain-and-snow"]
    assert [r.version for r in recorded] == [1, 2]


def test_data_outside_the_workspace_is_refused(tools):
    show_chart, _, recorded, workspace = tools
    outside = workspace.parent / "secret.csv"
    outside.write_text("a,b\n1,2\n")

    for ref in ("../secret.csv", str(outside)):
        out = show_chart.invoke({"spec": good_spec(), "data_ref": ref, "title": "T"})
        assert "outside the run workspace" in out
    assert recorded == []


def test_data_that_does_not_exist_says_how_to_make_it(tools):
    show_chart, _, recorded, _ = tools
    out = show_chart.invoke(
        {"spec": good_spec(), "data_ref": "artifacts/scratch/nope.parquet", "title": "T"}
    )
    assert "does not exist" in out and "run_python" in out
    assert recorded == []


def test_a_table_records_its_columns_and_rows(tools):
    _, show_table, recorded, _ = tools
    out = show_table.invoke({"data_ref": REF, "title": "Column profile"})

    assert recorded[0].kind == "table"
    assert recorded[0].spec is None
    assert recorded[0].columns == ["weather", "n", "precip_mean"]
    assert recorded[0].rows == 3
    assert "3 rows" in out


def test_a_table_narrowed_to_columns_it_does_not_have_says_which(tools):
    _, show_table, recorded, _ = tools
    out = show_table.invoke({"data_ref": REF, "title": "T", "columns": ["weather", "nope"]})

    assert "has no column(s) nope" in out
    assert "weather, n, precip_mean" in out
    assert recorded == []


def test_a_table_keeps_the_column_order_it_was_given(tools):
    _, show_table, recorded, _ = tools
    show_table.invoke({"data_ref": REF, "title": "T", "columns": ["precip_mean", "weather"]})

    assert recorded[0].columns == ["precip_mean", "weather"]


def test_a_file_that_is_not_a_table_is_not_a_table(tools):
    _, show_table, recorded, workspace = tools
    (workspace / "artifacts" / "notes.md").write_text("# not a table")
    out = show_table.invoke({"data_ref": "artifacts/notes.md", "title": "T"})

    assert "not a table this server can read" in out
    assert recorded == []


# ---- the validator itself, without the tools around it ----------------------


def test_the_validator_accepts_what_the_mockup_draws():
    """Every spec shape the product promises: hover, pan-zoom, brush, layers."""
    hover = {**good_spec(),
             "params": [{"name": "hover", "select": {"type": "point", "on": "pointerover"}}]}
    zoom = {"$schema": VL, "data": {"name": "table"},
            "params": [{"name": "zoom", "select": "interval", "bind": "scales"}],
            "mark": {"type": "line", "point": True},
            "encoding": {"x": {"field": "month", "type": "temporal"},
                         "y": {"field": "temp_max", "type": "quantitative"}}}
    brush = {"$schema": VL, "data": {"name": "table"},
             "params": [{"name": "brush", "select": {"type": "interval", "encodings": ["x"]}}],
             "mark": {"type": "area", "line": True, "opacity": 0.35},
             "encoding": {"x": {"field": "month", "type": "temporal"},
                          "y": {"field": "precip", "type": "quantitative"}}}
    layered = {"$schema": VL, "data": {"name": "table"}, "layer": [
        {"mark": "errorbar", "encoding": {"x": {"field": "year", "type": "ordinal"},
                                          "y": {"field": "lo", "type": "quantitative"},
                                          "y2": {"field": "hi"}}},
        {"mark": "line", "encoding": {"x": {"field": "year", "type": "ordinal"},
                                      "y": {"field": "tmax", "type": "quantitative"}}}]}

    for spec in (hover, zoom, brush, layered):
        assert validate_spec(spec) == ""


def test_something_that_is_not_a_chart_is_rejected_before_altair_sees_it():
    assert "not a Vega-Lite chart" in validate_spec({"data": {"name": "table"}})
    assert "must be a JSON object" in validate_spec("a bar chart of rain")


def test_the_validator_message_does_not_read_the_whole_schema_back():
    spec = good_spec()
    spec["mark"] = {"type": "barr"}
    message = validate_spec(spec)

    assert "barr" in message
    assert len(message.splitlines()) <= 13


def test_a_title_becomes_a_readable_id():
    assert slug("Seattle warmed every year — +0.74 °C/yr") == "seattle-warmed-every-year-0-74-c-yr"
    assert slug("!!!") == "chart"
    assert len(slug("x" * 200)) == 48


def test_a_chart_record_is_json_before_it_is_an_event():
    rec = ChartRecord(chart_id="c1", kind="chart", title="T", data_ref="a.csv",
                      step="analyze", persona="noel", spec=good_spec())
    event = rec.as_event()

    assert event["chart_id"] == "c1"
    assert event["version"] == 1
    assert event["spec"]["mark"] == {"type": "bar"}


def test_a_run_without_altair_records_the_chart_anyway(tools, monkeypatch):
    """A missing package on the *server* must not cost the run its document."""
    import builtins

    real = builtins.__import__

    def no_altair(name, *args, **kwargs):
        if name == "altair":
            raise ImportError("no altair here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_altair)
    show_chart, _, recorded, _ = tools
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "T"})

    assert len(recorded) == 1


def test_charts_need_no_filesystem_knowledge_of_the_domain():
    """Invariant 1, at the module level: no data-science word lives in here."""
    source = Path(__file__).resolve().parents[1] / "src" / "dsagent" / "runner" / "charts.py"
    text = source.read_text().lower()
    for word in ("pandas", "dataframe", "regression", "correlation", "eda"):
        assert word not in text, f"{word!r} does not belong in the harness"


# ---- and the same thing through a whole run ---------------------------------


def test_a_chart_emitted_in_a_step_reaches_the_log_and_the_run_record(tmp_path, monkeypatch):
    """The path the browser actually reads: `dsagent.chart` in the log, and the
    standing version in `run.json` so a resume knows the chart already exists."""
    import json

    from dsagent.cartridge import load_cartridge
    from dsagent.envs.base import Env
    from dsagent.runner import GateDecision, RunState, WorkflowRunner
    from tests.fakes import paths_for, produces_of

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=type("B", (), {"close": lambda s: None})()),
    )
    ds = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
    run_dir = tmp_path / "run"
    runner = WorkflowRunner(load_cartridge(ds), run_dir, ask_human=lambda p: GateDecision.APPROVE,
                            log=lambda m: None)

    class Agent:
        """Writes its promises, and — in `analyze` — emits a chart like a persona."""

        def __init__(self, persona: str, workspace: Path):
            self.persona, self.workspace = persona, workspace

        def invoke(self, payload):
            prompt = payload["messages"][0]["content"]
            for entry in produces_of(prompt):
                for rel in paths_for(entry):
                    p = self.workspace / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("x")
            if "step `analyze`" in prompt:
                data = self.workspace / "artifacts" / "scratch" / "by_year.csv"
                data.parent.mkdir(parents=True, exist_ok=True)
                data.write_text("year,tmax\n2012,15.3\n2013,16.1\n")
                show_chart = runner.charts()[0]
                show_chart.invoke({"spec": good_spec(), "data_ref": "artifacts/scratch/by_year.csv",
                                   "title": "Seattle warmed", "chart_id": "warming"})
            return {"messages": [{"role": "assistant", "content": "done"}]}

    runner.agent_factory = lambda cart, persona, env, ws: Agent(persona, ws)
    state = runner.run("eda-to-report", {"data_path": "data/x.csv"})

    assert state.status == "done"
    # On the run: one chart, at the version the next resume would build on.
    assert list(state.charts) == ["warming"]
    assert state.charts["warming"]["version"] == 1
    assert state.charts["warming"]["step"] == "analyze"
    assert state.charts["warming"]["section"] == "Findings"
    assert RunState.load(run_dir).charts["warming"]["title"] == "Seattle warmed"

    # In the log: one `dsagent.chart`, between the step's own events.
    lines = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    charts = [e for e in lines if e["name"] == "dsagent.chart"]
    assert len(charts) == 1
    assert charts[0]["value"]["chart_id"] == "warming"
    assert charts[0]["value"]["data_url"].endswith("artifacts/scratch/by_year.csv")
    assert charts[0]["value"]["run_id"] == run_dir.name


def test_every_step_event_carries_the_section_its_workflow_declared(tmp_path, monkeypatch):
    import json

    from dsagent.cartridge import load_cartridge
    from dsagent.envs.base import Env
    from dsagent.runner import GateDecision, WorkflowRunner

    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=type("B", (), {"close": lambda s: None})()),
    )
    ds = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
    run_dir = tmp_path / "run"
    runner = WorkflowRunner(load_cartridge(ds), run_dir, ask_human=lambda p: GateDecision.APPROVE,
                            log=lambda m: None)
    runner.run("eda-to-report", {"data_path": "data/x.csv"}, dry_run=True)

    lines = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    sections = {e["value"]["step"]: e["value"]["section"]
                for e in lines if e["name"] == "dsagent.step"}
    assert sections == {"profile": "The data", "data-gate": "Data quality",
                        "analyze": "Findings", "report": "Report"}


# ---- the second half of validation: does it actually draw? ------------------


def test_a_spec_the_schema_accepts_but_vega_refuses_is_caught():
    """Run 1 of M2.6 emitted this shape twice, and both charts drew nothing.

    A selection `param` at the top level of a *layered* spec is copied into every
    layer by Vega-Lite, and Vega then refuses the duplicate signal. The schema
    has no opinion about it — which is why validating against the schema alone
    was not enough, and why every spec is now drawn once, headless, before it is
    recorded.
    """
    pytest.importorskip("vl_convert")
    layered = {
        "$schema": VL,
        "data": {"name": "table"},
        "params": [{"name": "zoom", "select": {"type": "interval", "encodings": ["x"]},
                    "bind": "scales"}],
        "layer": [
            {"mark": "line", "encoding": {"x": {"field": "month", "type": "temporal"},
                                          "y": {"field": "temp", "type": "quantitative"}}},
            {"mark": "line", "encoding": {"x": {"field": "month", "type": "temporal"},
                                          "y": {"field": "fit", "type": "quantitative"}}},
        ],
    }
    message = validate_spec(layered)

    assert "Duplicate signal name" in message
    # Which signal duplicates depends on how the selection is bound; that it
    # names one is what lets the persona find it.
    assert "zoom_" in message
    # The persona is told what is wrong, not shown a JavaScript stack trace
    # through a library it cannot edit.
    assert "    at " not in message
    assert len(message.splitlines()) <= 4


def test_the_same_chart_with_the_param_in_a_layer_is_fine():
    pytest.importorskip("vl_convert")
    layered = {
        "$schema": VL,
        "data": {"name": "table"},
        "layer": [
            {
                "params": [{"name": "zoom", "select": {"type": "interval", "encodings": ["x"]},
                            "bind": "scales"}],
                "mark": "line",
                "encoding": {"x": {"field": "month", "type": "temporal"},
                             "y": {"field": "temp", "type": "quantitative"}},
            },
            {"mark": "line", "encoding": {"x": {"field": "month", "type": "temporal"},
                                          "y": {"field": "fit", "type": "quantitative"}}},
        ],
    }
    assert validate_spec(layered) == ""


def test_the_smoke_test_compiles_against_the_version_the_browser_renders():
    """Server and client must speak one grammar, or validation is theatre."""
    from dsagent.runner.charts import VEGA_LITE

    package = Path(__file__).resolve().parents[1] / "ui" / "package.json"
    pinned = json.loads(package.read_text())["dependencies"]["vega-lite"]

    assert VEGA_LITE.startswith("v5"), VEGA_LITE
    assert pinned.startswith(("^5", "5")), pinned


def test_the_smoke_test_is_not_a_reason_to_reach_the_network():
    """A persona's spec may name a URL; validating it must not fetch one."""
    pytest.importorskip("vl_convert")
    spec = {
        "$schema": VL,
        "data": {"url": "https://example.invalid/data.json"},
        "mark": "bar",
        "encoding": {"x": {"field": "a", "type": "nominal"},
                     "y": {"field": "b", "type": "quantitative"}},
    }
    # The rewrite to `{"name": "table"}` happens in the tool; the validator is
    # handed whatever it is given, and still may not go out.
    message = validate_spec(spec)
    assert "example.invalid" not in message or "not allowed" in message.lower()


# ---- amending a chart from the conversation --------------------------------


def test_a_chart_can_be_revised_from_outside_the_run(tmp_path):
    """The round trip the milestone is about: ask for a change, get a version.

    The orchestrator is not inside a run — no workspace, no step, no event log —
    so the tool it is given takes the run by name and writes into it. Emitted
    under the same `chart_id`, the answer *replaces* the chart on the page.
    """
    from dsagent.runner import RunState, StepRecord
    from dsagent.runner.charts import amend_tools

    run_dir = tmp_path / "eda-1"
    (run_dir / "workspace" / "artifacts").mkdir(parents=True)
    (run_dir / "workspace" / "artifacts" / "by_cat.csv").write_text(
        "weather,precip\nrain,5.42\nsun,0.0\n"
    )
    state = RunState(workflow="w", cartridge="c", inputs={}, status="done",
                     steps={"analyze": StepRecord(id="analyze", status="done")})
    state.charts = {"wet": {
        "chart_id": "wet", "kind": "chart", "title": "Wet days", "data_ref": "artifacts/by_cat.csv",
        "data_url": "/runs/eda-1/preview/artifacts/by_cat.csv", "step": "analyze",
        "persona": "noel", "section": "Findings", "spec": good_spec(),
        "columns": ["weather", "precip"], "rows": 2, "version": 1, "ts": 0.0,
    }}
    state.save(run_dir)

    _, amend = amend_tools(tmp_path)
    changed = {**good_spec(), "mark": {"type": "point"}}
    out = amend.invoke({
        "run_id": "eda-1", "chart_id": "wet", "spec": changed,
        "data_ref": "artifacts/by_cat.csv", "title": "Wet days, as points",
    })

    assert "v2" in out
    after = RunState.load(run_dir).charts["wet"]
    assert after["version"] == 2
    assert after["spec"]["mark"] == {"type": "point"}
    assert after["title"] == "Wet days, as points"
    # And the screen hears about it the same way it hears about everything else.
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["name"] for e in events] == ["dsagent.chart"]
    assert events[0]["value"]["version"] == 2
    assert events[0]["value"]["run_id"] == "eda-1"


def test_amending_a_run_that_is_not_there_says_so(tmp_path):
    from dsagent.runner.charts import amend_tools

    _, amend = amend_tools(tmp_path)
    for run_id in ("nope", "../escape", ""):
        out = amend.invoke({"run_id": run_id, "chart_id": "c", "spec": good_spec(),
                            "data_ref": "a.csv", "title": "T"})
        assert "unknown run" in out


def test_an_amendment_is_validated_like_any_other_chart(tmp_path):
    """Same code, same rules: a spec that will not draw is refused here too."""
    pytest.importorskip("vl_convert")
    from dsagent.runner import RunState
    from dsagent.runner.charts import amend_tools

    run_dir = tmp_path / "eda-1"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "t.csv").write_text("a,b\n1,2\n")
    RunState(workflow="w", cartridge="c", inputs={}, status="done", steps={}).save(run_dir)

    _, amend = amend_tools(tmp_path)
    broken = good_spec()
    broken["encoding"]["x"]["type"] = "nominel"
    out = amend.invoke({"run_id": "eda-1", "chart_id": "c", "spec": broken,
                        "data_ref": "t.csv", "title": "T"})

    assert "error:" in out
    assert RunState.load(run_dir).charts == {}


def test_a_chart_s_spec_can_be_read_back_from_outside_the_run(tmp_path):
    """A spec lives on the run, not in its workspace, so no file tool reaches it.

    Without this the orchestrator was told to "read the current spec" from a
    file it cannot open, and spent a minute failing to find it before answering
    nothing at all.
    """
    from dsagent.runner import RunState
    from dsagent.runner.charts import amend_tools

    run_dir = tmp_path / "r"
    (run_dir / "workspace").mkdir(parents=True)
    state = RunState(workflow="w", cartridge="c", inputs={}, steps={})
    state.charts = {"wet": {
        "chart_id": "wet", "kind": "chart", "title": "Wet days", "data_ref": "a.csv",
        "data_url": "", "step": "analyze", "persona": "noel", "section": "",
        "spec": good_spec(), "columns": [], "rows": 2, "version": 1, "ts": 0.0,
    }}
    state.save(run_dir)

    read, _ = amend_tools(tmp_path)

    listed = read.invoke({"run_id": "r"})
    assert "wet" in listed and "Wet days" in listed and "a.csv" in listed

    one = json.loads(read.invoke({"run_id": "r", "chart_id": "wet"}))
    assert one["spec"]["mark"] == {"type": "bar"}
    assert one["version"] == 1

    assert "no chart" in read.invoke({"run_id": "r", "chart_id": "nope"})
    assert "unknown run" in read.invoke({"run_id": "nope"})


def test_a_revision_stays_where_the_chart_it_replaces_was(tools):
    """Otherwise an amended chart vanishes off the page instead of changing on it.

    The document groups cards by the step that emitted them. A chart amended
    from the conversation is emitted by the orchestrator, which is not a step —
    so a revision inherits the original's step, section and persona. It is the
    same chart; only its version moved.
    """
    show_chart, _, recorded, workspace = tools
    show_chart.invoke({"spec": good_spec(), "data_ref": REF, "title": "T", "chart_id": "c1"})

    elsewhere: list[ChartRecord] = []
    amender, _ = chart_tools(
        workspace, "run-1",
        on_chart=elsewhere.append,
        context=lambda: StepContext(step="chat", persona="orchestrator"),
        known=lambda cid: next((r for r in reversed(recorded) if r.chart_id == cid), None),
    )
    amender.invoke({"spec": {**good_spec(), "mark": {"type": "point"}}, "data_ref": REF,
                    "title": "T, as points", "chart_id": "c1"})

    assert elsewhere[0].version == 2
    assert elsewhere[0].step == "analyze"
    assert elsewhere[0].persona == "noel"
    assert elsewhere[0].section == "Findings"
    assert elsewhere[0].title == "T, as points"
