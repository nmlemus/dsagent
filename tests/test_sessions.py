"""Sessions: the run↔session link is a marker on the run, surfaced on the run
summary and filterable. No model, kernel or Docker."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dsagent.projects import DEFAULT_ID, ProjectStore
from dsagent.runs import summarize
from dsagent.sessions import mark_session, session_of

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "replay" / "run-eda-003"
FIXTURE = FIXTURE_RUN
DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def test_marker_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert session_of(run_dir) == ""
    mark_session(run_dir, "thread-abc")
    assert session_of(run_dir) == "thread-abc"
    # empty session id writes nothing
    mark_session(tmp_path / "other", "")
    assert session_of(tmp_path / "other") == ""


def test_summary_carries_the_session(tmp_path):
    run_dir = tmp_path / "eda-1"
    shutil.copytree(FIXTURE_RUN, run_dir)
    assert summarize(run_dir).session == ""
    mark_session(run_dir, "thread-xyz")
    assert summarize(run_dir).session == "thread-xyz"


# ---- filter over HTTP -----------------------------------------------------

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsagent.api_projects import add_projects_routes


def test_project_runs_filter_by_session(tmp_path):
    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    a = store.runs_dir(DEFAULT_ID) / "run-a"
    b = store.runs_dir(DEFAULT_ID) / "run-b"
    shutil.copytree(FIXTURE_RUN, a)
    shutil.copytree(FIXTURE_RUN, b)
    mark_session(a, "sess-1")
    mark_session(b, "sess-2")

    app = FastAPI()
    add_projects_routes(app, store)
    client = TestClient(app)

    everything = client.get(f"/projects/{DEFAULT_ID}/runs").json()["runs"]
    assert len(everything) == 2

    just_one = client.get(f"/projects/{DEFAULT_ID}/runs?session=sess-1").json()["runs"]
    assert [r["run_id"] for r in just_one] == ["run-a"]
    assert just_one[0]["session"] == "sess-1"

    none = client.get(f"/projects/{DEFAULT_ID}/runs?session=nope").json()["runs"]
    assert none == []


# ---- session→project registry + run-dir resolution (Phase 3.5) -------------


def test_session_store_roundtrip(tmp_path):
    from dsagent.sessions import SessionStore

    reg = SessionStore(tmp_path / "sessions")
    assert reg.project_of("s1") == "default"  # unknown → default
    reg.save("s1", project="titanic", title="hi", messages=[{"role": "user"}], updated_at=1.0)
    assert reg.project_of("s1") == "titanic"
    assert reg.get("s1")["messages"] == [{"role": "user"}]
    assert [s["id"] for s in reg.list_for("titanic")] == ["s1"]
    assert reg.list_for("other") == []
    assert reg.delete("s1") is True
    assert reg.get("s1") is None


def test_a_save_that_cannot_be_written_is_not_reported_as_saved(tmp_path, monkeypatch):
    """`save` swallowed OSError and the endpoint answered 200, so a disk fault
    looked like a successful save and the next read returned 404 (BH-04)."""
    from dsagent.sessions import SessionStore

    store = SessionStore(tmp_path / "sessions")

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_text", explode)
    with pytest.raises(OSError):
        store.save("s", project="default", title="t", messages=["hola"], updated_at=1.0)


def test_an_older_save_does_not_overwrite_a_newer_one(tmp_path):
    """Two tabs, or a retried request arriving late. The stored `updated_at` was
    written but never compared, so the late save won and took the history with
    it: messages were emptied and the project changed underneath the chat."""
    from dsagent.sessions import SessionStore, StaleSession

    store = SessionStore(tmp_path / "sessions")
    store.save("s", project="a", title="Nuevo", messages=["nuevo"], updated_at=20)
    with pytest.raises(StaleSession):
        store.save("s", project="a", title="Viejo", messages=[], updated_at=10)

    kept = store.get("s")
    assert kept["messages"] == ["nuevo"]
    assert kept["updated_at"] == 20
    # The same timestamp is the same client saving again, which is allowed.
    store.save("s", project="a", title="Nuevo", messages=["nuevo", "otro"], updated_at=20)
    assert store.get("s")["messages"] == ["nuevo", "otro"]


def test_a_session_does_not_change_project_underneath_itself(tmp_path):
    """A chat belongs to one project: its runs, its files and its workspace are
    that project's. A save naming another project is a mistake, not a move."""
    from dsagent.sessions import SessionStore, WrongProject

    store = SessionStore(tmp_path / "sessions")
    store.save("s", project="a", title="t", messages=["x"], updated_at=1)
    with pytest.raises(WrongProject):
        store.save("s", project="b", title="t", messages=["x"], updated_at=2)
    assert store.get("s")["project"] == "a"


@pytest.mark.usefixtures()
def test_the_http_surface_reports_both_conflicts(tmp_path):
    pytest.importorskip("fastapi", reason="needs the 'ui' extra")
    from fastapi.testclient import TestClient

    from dsagent import serve
    from dsagent.cartridge import load_cartridge

    """What the browser sees: a stale save and a project change are refused with
    a status the client can act on, and the stored chat is untouched."""
    app = serve.build_app(
        [load_cartridge(DS)], None, tmp_path / "ws", tmp_path / "runs",
        replay=FIXTURE, replay_speed=2000.0,
    )
    client = TestClient(app)
    body = {"project": "default", "title": "t", "messages": ["uno"], "updated_at": 20}
    assert client.put("/sessions/s", json=body).status_code == 200

    stale = client.put("/sessions/s", json={**body, "messages": [], "updated_at": 10})
    assert stale.status_code == 409
    assert client.get("/sessions/s").json()["messages"] == ["uno"]

    pid = client.post("/projects", json={"name": "Otro"}).json()["id"]
    moved = client.put("/sessions/s", json={**body, "project": pid, "updated_at": 30})
    assert moved.status_code == 409
    assert client.get("/sessions/s").json()["project"] == "default"


def test_resolve_run_dir_chat_goes_to_the_sessions_project(tmp_path):
    from dsagent.runner.tools import resolve_run_dir, workflow_run_id
    from dsagent.sessions import SessionStore

    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    proj = store.create("Titanic")
    reg = SessionStore(tmp_path / "projects" / "sessions")
    reg.save("thread-1", project=proj.id, title="t", messages=[], updated_at=0.0)

    # chat path: no RUN_ID_KEY, thread maps to the project
    rd = resolve_run_dir(
        None, "thread-1", "eda-to-report", "tc-1",
        store.runs_dir(DEFAULT_ID), store, reg.project_of,
    )
    assert rd == store.runs_dir(proj.id) / workflow_run_id("eda-to-report", "tc-1")

    # an unregistered chat falls back to default
    rd_def = resolve_run_dir(
        None, "unknown-thread", "eda-to-report", "tc-2",
        store.runs_dir(DEFAULT_ID), store, reg.project_of,
    )
    assert rd_def == store.runs_dir(DEFAULT_ID) / workflow_run_id("eda-to-report", "tc-2")


def test_resolve_run_dir_launcher_locates_the_precreated_run(tmp_path):
    from dsagent.runner.tools import resolve_run_dir

    store = ProjectStore(tmp_path / "projects", tmp_path / "runs")
    store.ensure_default()
    proj = store.create("P")
    # a run pre-created in the project (launcher path)
    pre = store.runs_dir(proj.id) / "eda-pre"
    shutil.copytree(FIXTURE_RUN, pre)

    rd = resolve_run_dir(
        "eda-pre", "run:eda-pre", "eda-to-report", "tc",
        store.runs_dir(DEFAULT_ID), store, None,
    )
    assert rd == pre  # located in its project, not recomputed under default
