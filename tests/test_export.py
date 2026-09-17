"""The export: one file, charts alive, nothing fetched from anywhere."""

import json
from pathlib import Path

import pytest

from dsagent.export import export_html, markdown_to_html, snapshot
from dsagent.runner.charts import VEGA_LITE

VL = "https://vega.github.io/schema/vega-lite/v5.json"


@pytest.fixture
def run(tmp_path):
    workspace = tmp_path / "run" / "workspace"
    (workspace / "artifacts" / "scratch").mkdir(parents=True)
    (workspace / "artifacts" / "findings.md").write_text(
        "# Findings\n\nRain and snow are the only wet categories.\n"
    )
    (workspace / "artifacts" / "scratch" / "by_cat.csv").write_text(
        "weather,precip\nrain,5.42\nsun,0.0\n"
    )
    state = {
        "workflow": "eda-to-report",
        "inputs": {"data_path": "data/w.csv", "question": "What is wet?"},
        "steps": {
            "analyze": {
                "id": "analyze",
                "cost_usd": 0.33,
                "files": [{"path": "artifacts/findings.md"}],
            }
        },
        "charts": {
            "wet": {
                "chart_id": "wet", "kind": "chart", "title": "Only rain and snow",
                "data_ref": "artifacts/scratch/by_cat.csv", "step": "analyze",
                "persona": "noel", "version": 1,
                "spec": {"$schema": VL, "data": {"name": "table"}, "mark": "bar",
                         "encoding": {"x": {"field": "weather", "type": "nominal"},
                                      "y": {"field": "precip", "type": "quantitative"}}},
            }
        },
    }
    return tmp_path / "run", state


def test_the_export_is_one_file_that_needs_nothing_else(run):
    pytest.importorskip("vl_convert")
    run_dir, state = run

    document = export_html(run_dir, state, vl_version=VEGA_LITE)

    # The libraries travel with it.
    assert "window.vegaEmbed" in document
    # So do the rows: an export has no server to fetch them from.
    assert '"weather": "rain"' in document or '"weather":"rain"' in document
    # And nothing points at anybody else's server.
    assert "cdn.jsdelivr" not in document
    assert "https://cdn." not in document
    assert "unpkg" not in document


def test_the_export_says_what_the_personas_said(run):
    pytest.importorskip("vl_convert")
    run_dir, state = run

    document = export_html(run_dir, state, vl_version=VEGA_LITE)

    assert "Rain and snow are the only wet categories." in document
    assert "Only rain and snow" in document          # the chart's own title
    assert "artifacts/scratch/by_cat.csv" in document  # and where its rows came from
    assert "What is wet?" in document                 # the question that was asked


def test_a_chart_whose_data_is_gone_is_not_a_crash(run):
    pytest.importorskip("vl_convert")
    run_dir, state = run
    (run_dir / "workspace" / "artifacts" / "scratch" / "by_cat.csv").unlink()

    document = export_html(run_dir, state, vl_version=VEGA_LITE)

    assert "0 rows travelling with this file" in document


def test_rows_are_read_as_objects_not_arrays(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,x\n2,y\n")
    assert snapshot(tmp_path, "t.csv") == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_a_missing_table_is_no_rows_rather_than_an_exception(tmp_path):
    assert snapshot(tmp_path, "nope.parquet") == []


def test_markdown_without_a_renderer_is_still_readable(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_markdown(name, *args, **kwargs):
        if name == "markdown":
            raise ImportError("not here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_markdown)
    out = markdown_to_html("# Title\n\n<script>alert(1)</script>")

    assert out.startswith("<pre>")
    assert "&lt;script&gt;" in out


def test_the_export_endpoint_serves_it(tmp_path):
    """And a run that does not exist is a 404, not a traceback."""
    pytest.importorskip("vl_convert")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dsagent.api import add_runs_routes
    from dsagent.cartridge import load_cartridge

    (tmp_path / "run" / "workspace").mkdir(parents=True)
    (tmp_path / "run" / "run.json").write_text(json.dumps({
        "workflow": "eda-to-report", "inputs": {}, "steps": {}, "charts": {},
    }))

    app = FastAPI()
    ds = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
    add_runs_routes(app, [load_cartridge(ds)], tmp_path, driver=_NoDriver())
    client = TestClient(app)

    ok = client.get("/runs/run/export.html")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/html")
    assert client.get("/runs/nope/export.html").status_code == 404


class _NoDriver:
    def is_running(self, run_id: str) -> bool:
        return False

    def error_for(self, run_id: str) -> None:
        return None
