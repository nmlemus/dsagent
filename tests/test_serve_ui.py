"""The static SPA mount: it serves the built UI at `/` and never shadows an API
route. No model, kernel or Docker — just `_mount_ui` on a bare FastAPI app."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsagent.serve import _mount_ui, _ui_dist, UI_DIST_ENV


def _build_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>DSAgent</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ui')", encoding="utf-8")
    return dist


def _app_with_api() -> FastAPI:
    app = FastAPI()

    @app.get("/cartridges")
    def cartridges() -> dict:
        return {"cartridges": [], "workflows": []}

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict:
        return {"run_id": run_id}

    return app


def test_serves_spa_and_assets(tmp_path, monkeypatch):
    monkeypatch.setenv(UI_DIST_ENV, str(_build_dist(tmp_path)))
    app = _app_with_api()
    _mount_ui(app)
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "DSAgent" in root.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text


def test_api_routes_are_not_shadowed(tmp_path, monkeypatch):
    monkeypatch.setenv(UI_DIST_ENV, str(_build_dist(tmp_path)))
    app = _app_with_api()
    _mount_ui(app)
    client = TestClient(app)

    # The mount is at "/", but the API routes were registered first and win.
    assert client.get("/cartridges").json() == {"cartridges": [], "workflows": []}
    assert client.get("/runs/eda-123").json() == {"run_id": "eda-123"}


def test_unknown_path_is_404_not_index(tmp_path, monkeypatch):
    monkeypatch.setenv(UI_DIST_ENV, str(_build_dist(tmp_path)))
    app = _app_with_api()
    _mount_ui(app)
    client = TestClient(app)
    assert client.get("/does-not-exist.js").status_code == 404


def test_no_dist_means_no_mount(tmp_path, monkeypatch):
    # Point the override at an empty dir: nothing to serve, nothing mounted.
    monkeypatch.setenv(UI_DIST_ENV, str(tmp_path / "nope"))
    assert _ui_dist() is None
    app = _app_with_api()
    _mount_ui(app)
    client = TestClient(app)
    assert client.get("/").status_code == 404
    # API still works with no UI built.
    assert client.get("/cartridges").status_code == 200
