"""Project datasets: upload once, seeded into each new run. No model."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsagent.datasets import (
    delete_dataset,
    list_datasets,
    safe_name,
    seed_run,
    store_dataset,
)
from dsagent.projects import DEFAULT_ID, ProjectStore


def test_safe_name():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("weird name!.csv") == "weird_name_.csv"
    assert safe_name("") == "file"
    assert safe_name(".hidden") == "hidden"


def test_store_list_delete(tmp_path):
    d = tmp_path / "data"
    assert list_datasets(d) == []
    store_dataset(d, "sales.csv", b"a,b\n1,2\n")
    store_dataset(d, "x.parquet", b"\x00\x01")
    names = {x["name"] for x in list_datasets(d)}
    assert names == {"sales.csv", "x.parquet"}
    assert delete_dataset(d, "sales.csv") is True
    assert {x["name"] for x in list_datasets(d)} == {"x.parquet"}
    assert delete_dataset(d, "missing.csv") is False


def test_seed_copies_project_data_into_a_run(tmp_path):
    project_data = tmp_path / "proj" / "data"
    store_dataset(project_data, "titanic.csv", b"Survived\n1\n")
    workspace = tmp_path / "run" / "workspace"
    copied = seed_run(project_data, workspace)
    assert copied == ["titanic.csv"]
    assert (workspace / "data" / "titanic.csv").read_bytes() == b"Survived\n1\n"
    # a project with no data seeds nothing, quietly
    assert seed_run(tmp_path / "empty", workspace) == []


# ---- routes ---------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsagent.api_projects import add_projects_routes


def app_for(tmp_path: Path) -> tuple[FastAPI, ProjectStore]:
    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    app = FastAPI()
    add_projects_routes(app, store)
    return app, store


def test_dataset_upload_list_delete_over_http(tmp_path):
    app, _ = app_for(tmp_path)
    client = TestClient(app)

    assert client.get(f"/projects/{DEFAULT_ID}/data").json()["datasets"] == []

    up = client.put(f"/projects/{DEFAULT_ID}/data/sales.csv", content=b"a,b\n1,2\n")
    assert up.status_code == 200 and up.json()["name"] == "sales.csv"

    listed = client.get(f"/projects/{DEFAULT_ID}/data").json()["datasets"]
    assert [d["name"] for d in listed] == ["sales.csv"]

    assert client.delete(f"/projects/{DEFAULT_ID}/data/sales.csv").status_code == 200
    assert client.get(f"/projects/{DEFAULT_ID}/data").json()["datasets"] == []


def test_dataset_upload_to_unknown_project_is_404(tmp_path):
    app, _ = app_for(tmp_path)
    client = TestClient(app)
    assert client.put("/projects/nope/data/x.csv", content=b"x").status_code == 404
