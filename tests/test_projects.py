"""The project layer: a `default` that adopts the legacy runs dir in place, new
projects that namespace, and the `/projects` routes. No model, kernel or Docker."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dsagent.projects import DEFAULT_ID, ProjectStore

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "replay" / "run-eda-003"


def store_at(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "projects", tmp_path / "runs")


def test_default_adopts_the_legacy_runs_dir_in_place(tmp_path):
    store = store_at(tmp_path)
    store.ensure_default()
    # default keeps the old flat path; nothing is moved.
    assert store.runs_dir(DEFAULT_ID) == tmp_path / "runs"
    listed = store.list()
    assert listed[0].id == DEFAULT_ID


def test_a_new_project_namespaces_and_persists(tmp_path):
    store = store_at(tmp_path)
    proj = store.create("Titanic survival", goal="which attrs drive survival")
    assert proj.id == "titanic-survival"
    assert (store.dir(proj.id) / "project.json").is_file()
    assert store.data_dir(proj.id).is_dir()
    assert store.runs_dir(proj.id) == store.dir(proj.id) / "runs"
    # readable back
    got = store.get(proj.id)
    assert got is not None and got.name == "Titanic survival" and got.goal.startswith("which")


def test_slug_collisions_get_a_suffix(tmp_path):
    store = store_at(tmp_path)
    a = store.create("My Project")
    b = store.create("My Project")
    assert a.id == "my-project"
    assert b.id == "my-project-2"


def test_default_id_is_reserved(tmp_path):
    store = store_at(tmp_path)
    proj = store.create("default")
    assert proj.id != DEFAULT_ID


# ---- routes ---------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsagent.api_projects import add_projects_routes


def app_for(tmp_path: Path) -> tuple[FastAPI, ProjectStore]:
    store = store_at(tmp_path)
    store.ensure_default()
    app = FastAPI()
    add_projects_routes(app, store)
    return app, store


def test_list_and_create_over_http(tmp_path):
    app, _ = app_for(tmp_path)
    client = TestClient(app)

    first = client.get("/projects").json()["projects"]
    assert [p["id"] for p in first] == [DEFAULT_ID]

    created = client.post("/projects", json={"name": "Marketing Q3"}).json()
    assert created["id"] == "marketing-q3"
    assert created["runs"] == 0

    ids = [p["id"] for p in client.get("/projects").json()["projects"]]
    assert DEFAULT_ID in ids and "marketing-q3" in ids


def test_creating_without_a_name_is_a_400(tmp_path):
    app, _ = app_for(tmp_path)
    client = TestClient(app)
    assert client.post("/projects", json={}).status_code == 400


def test_runs_of_an_unknown_project_is_404(tmp_path):
    app, _ = app_for(tmp_path)
    client = TestClient(app)
    assert client.get("/projects/nope/runs").status_code == 404


def test_default_project_lists_the_runs_in_its_adopted_dir(tmp_path):
    app, store = app_for(tmp_path)
    # a real run, dropped into the adopted (default) runs dir.
    dest = store.runs_dir(DEFAULT_ID) / "eda-to-report-000001"
    shutil.copytree(FIXTURE_RUN, dest)

    client = TestClient(app)
    runs = client.get(f"/projects/{DEFAULT_ID}/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["workflow"] == "eda-to-report"


# ---- running inside a project (Phase 1.5) ---------------------------------

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def test_a_run_created_in_a_project_lands_and_resolves_there(tmp_path):
    from dsagent.cartridge import load_cartridge
    from dsagent.driver import GraphDriver

    store = store_at(tmp_path)
    store.ensure_default()
    proj = store.create("Titanic")
    cart = load_cartridge(DS)
    # No model: create() only writes pending state; the graph is never invoked.
    driver = GraphDriver(None, [cart], store.runs_dir(DEFAULT_ID), recursion_limit=1, store=store)

    inputs = {"question": "q", "data_path": "data/x.csv", "key_column": "k"}
    run_dir = driver.create("eda-p-001", inputs, "eda-to-report", project=proj.id)

    # It lives under the project, not the default dir…
    assert run_dir == store.runs_dir(proj.id) / "eda-p-001"
    assert (run_dir / "run.json").is_file()
    assert not (store.runs_dir(DEFAULT_ID) / "eda-p-001").exists()
    # …and is found again by id, across projects.
    assert store.locate_run("eda-p-001") == run_dir
    assert driver._run_dir("eda-p-001") == run_dir
    assert store.project_of(run_dir) == proj.id
