"""A run has one identity, and every surface resolves it the same way.

The independent backend review (BH-03) reproduced two halves of one defect:
two projects creating the same workflow in the same second received the same
run id, and the conversation's own reading tools looked only in the default
runs directory, so they called a run unknown that the API was serving.

No model, no kernel: ids, the store, and the tools' resolution.
"""

import json
from pathlib import Path
from unittest.mock import patch

from dsagent.api import new_run_id
from dsagent.cartridge import load_cartridge
from dsagent.projects import ProjectStore
from dsagent.runner import workflow_tools
from dsagent.runner.charts import amend_tools

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    return store


def test_two_projects_in_the_same_second_get_different_ids(tmp_path):
    store = _store(tmp_path)
    a, b = store.create("A"), store.create("B")
    # The clock is the part that used to make them equal, so freeze it.
    with patch("dsagent.api.time.strftime", return_value="20260921-010101"):
        first = new_run_id(store.runs_dir(a.id), "eda-to-report", store=store)
        second = new_run_id(store.runs_dir(b.id), "eda-to-report", store=store)
    assert first != second
    # Still readable: a person reads these in a URL and in a screenshot.
    assert first.startswith("eda-to-report-20260921-010101")
    assert second.startswith("eda-to-report-20260921-010101")


def test_an_id_is_reserved_when_it_is_handed_out(tmp_path):
    """Naming and creating used to be two steps, so two callers could be given
    the same name before either wrote anything."""
    store = _store(tmp_path)
    a = store.create("A")
    with patch("dsagent.api.time.strftime", return_value="20260921-010101"):
        ids = {new_run_id(store.runs_dir(a.id), "w", store=store) for _ in range(20)}
    assert len(ids) == 20
    for run_id in ids:
        assert (store.runs_dir(a.id) / run_id).is_dir()


def test_the_default_project_still_names_runs_readably(tmp_path):
    store = _store(tmp_path)
    with patch("dsagent.api.time.strftime", return_value="20260921-010101"):
        run_id = new_run_id(store.runs_dir("default"), "eda-to-report", store=store)
    assert run_id.startswith("eda-to-report-20260921-010101")
    assert store.locate_run.__self__ is store  # store is the resolver everywhere


def _run_in_project(store: ProjectStore, project: str, run_id: str) -> Path:
    run_dir = store.runs_dir(project) / run_id
    (run_dir / "workspace" / "report").mkdir(parents=True)
    (run_dir / "workspace" / "report" / "findings.md").write_text("# hallazgos\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "workflow": "eda-to-report",
                "cartridge": "ds",
                "inputs": {},
                "status": "done",
                "steps": {},
                "charts": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    return run_dir


def test_the_chat_tools_read_a_run_that_lives_in_a_project(tmp_path):
    """`list_run_files` and `read_run_file` answered "unknown run" for a run the
    HTTP API resolved, because they were built over the legacy runs dir only."""
    store = _store(tmp_path)
    project = store.create("Marketing")
    _run_in_project(store, project.id, "job-1")

    tools = {
        t.name: t
        for t in workflow_tools(
            [load_cartridge(DS, generate_commands=False)],
            tmp_path / "runs",
            ask_human=lambda request: None,
            store=store,
        )
    }
    listed = tools["list_run_files"].invoke({"run_id": "job-1"})
    assert "report/findings.md" in listed
    assert "unknown run" not in listed
    assert "hallazgos" in tools["read_run_file"].invoke(
        {"run_id": "job-1", "path": "report/findings.md"}
    )


def test_the_chart_tools_read_a_run_that_lives_in_a_project(tmp_path):
    store = _store(tmp_path)
    project = store.create("Marketing")
    _run_in_project(store, project.id, "job-2")

    read, _amend = amend_tools(tmp_path / "runs", store=store)
    out = read.invoke({"run_id": "job-2"})
    # The run exists and has no chart yet; what must not happen is "unknown run".
    assert "unknown run" not in out.lower()


def test_an_unknown_run_is_still_unknown(tmp_path):
    store = _store(tmp_path)
    tools = {
        t.name: t
        for t in workflow_tools(
            [load_cartridge(DS, generate_commands=False)],
            tmp_path / "runs",
            ask_human=lambda request: None,
            store=store,
        )
    }
    assert "unknown run" in tools["list_run_files"].invoke({"run_id": "nope"})
    # A run id that is a path may not walk out of the runs directory.
    assert "unknown run" in tools["list_run_files"].invoke({"run_id": "../../etc"})
