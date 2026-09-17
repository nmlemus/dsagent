"""The runs API: what the home screen, the launcher and the run screen read.

Driven with `TestClient` against a **replayed** run, which is how the whole UI is
developed — no model, no kernel, and the same routes a live server serves. The
live driver's own wiring is asserted in `tests/test_driver.py`.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.runner import GateDecision

pytest.importorskip("fastapi", reason="needs the 'ui' extra")

from fastapi.testclient import TestClient

from dsagent import serve

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
FIXTURE = Path(__file__).resolve().parents[1] / "ui" / "fixtures" / "run-eda-003"
CSV = Path(__file__).resolve().parents[1] / "tests" / "data" / "seattle-weather.csv"


@pytest.fixture
def app(tmp_path):
    return serve.build_app(
        [load_cartridge(DS)], None, tmp_path / "ws", tmp_path / "runs",
        replay=FIXTURE, replay_speed=2000.0,
    )


@pytest.fixture
def client(app):
    return TestClient(app)


def _approve_when_asked(app, run_id, timeout=20.0):
    """Answer the gate from outside the request that is reading the stream.

    `TestClient` serves one request at a time on one portal, so a POST issued
    while its own stream is open deadlocks the test rather than the server. The
    HTTP gate has its own test; this one is about the stream.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if app.state.driver.answer_gate(run_id, GateDecision.APPROVE):
            return
        time.sleep(0.02)


def wait_for(predicate, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def start_a_run(client, inputs=None, upload=True):
    r = client.post("/runs", json={
        "workflow": "eda-to-report",
        "inputs": inputs or {"question": "What is going on?"},
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    if upload:
        put = client.put(f"/runs/{run_id}/data/data_path",
                         params={"filename": "seattle-weather.csv"}, content=CSV.read_bytes())
        assert put.status_code == 200, put.text
    assert client.post(f"/runs/{run_id}/start").json()["started"] is True
    return run_id


# --- the launcher's source of truth -----------------------------------------


def test_cartridges_describes_every_workflow_the_launcher_can_offer(client):
    body = client.get("/cartridges").json()
    assert [c["name"] for c in body["cartridges"]] == ["ds"]
    eda = next(w for w in body["workflows"] if w["name"] == "eda-to-report")

    assert eda["inputs"]["data_path"] == {
        "type": "path", "required": True, "default": None, "options": None,
    }
    # `question` has a default, so the form can prefill it and not demand it;
    # `key_column` is optional. §7.2 is exactly this rendered.
    assert eda["inputs"]["question"]["required"] is False
    assert eda["inputs"]["question"]["default"].startswith("What are the main patterns")
    assert eda["inputs"]["key_column"]["required"] is False

    assert eda["personas"] == ["marie", "noel"]
    assert [s["id"] for s in eda["steps"]] == ["profile", "data-gate", "analyze", "report"]
    gated = [s for s in eda["steps"] if s["gate"]]
    assert [s["id"] for s in gated] == ["data-gate"]
    assert gated[0]["gate"]["kind"] == "human"
    assert eda["steps"][2]["produces"] == ["artifacts/findings.md"]
    assert [s["section"] for s in eda["steps"]] == [
        "The data", "Data quality", "Findings", "Report",
    ]


def test_an_unknown_workflow_is_refused_rather_than_started(client):
    r = client.post("/runs", json={"workflow": "nope", "inputs": {}})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_bad_inputs_are_refused(client):
    assert client.post("/runs", json={"workflow": "eda-to-report", "inputs": "nope"}).status_code == 400
    assert client.post("/runs", content=b"not json").status_code == 400


# --- creating and starting ---------------------------------------------------


def test_an_upload_lands_in_the_run_and_fills_its_input(client, tmp_path):
    """The operator drags a CSV in; nobody types `data_path=`."""
    run_id = client.post("/runs", json={
        "workflow": "eda-to-report", "inputs": {"question": "q"},
    }).json()["run_id"]
    assert run_id.startswith("eda-to-report-")

    put = client.put(f"/runs/{run_id}/data/data_path",
                     params={"filename": "seattle weather (1).csv"}, content=b"a,b\n1,2\n")
    assert put.json() == {"input": "data_path", "value": "data/seattle-weather-1-.csv", "bytes": 8}
    landed = tmp_path / "runs" / run_id / "workspace" / "data" / "seattle-weather-1-.csv"
    assert landed.read_bytes() == b"a,b\n1,2\n"

    # the run carries the value the steps will read, and is listed from now on
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "pending"
    assert body["inputs"]["data_path"] == "data/seattle-weather-1-.csv"
    assert body["inputs"]["question"] == "q"
    assert run_id in [s["run_id"] for s in client.get("/runs").json()["runs"]]


def test_an_upload_cannot_escape_the_run_directory(client, tmp_path):
    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    put = client.put(f"/runs/{run_id}/data/data_path",
                     params={"filename": "../../../etc/passwd"}, content=b"x")
    assert put.json()["value"] == "data/passwd"
    assert (tmp_path / "runs" / run_id / "workspace" / "data" / "passwd").is_file()


def test_an_upload_names_an_input_the_workflow_declares(client):
    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    assert client.put(f"/runs/{run_id}/data/nonsense", content=b"x").status_code == 400
    assert client.put(f"/runs/{run_id}/data/data_path", content=b"").status_code == 400


def test_a_started_run_does_not_accept_a_different_dataset(client):
    """A run's inputs are the ones it began with, file included."""
    run_id = start_a_run(client)
    assert client.put(f"/runs/{run_id}/data/data_path", content=b"x,y\n1,2\n").status_code == 409


def test_two_runs_started_in_the_same_second_get_different_directories(client):
    a = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    b = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    assert a != b


# --- following a run ---------------------------------------------------------


def test_a_run_reports_its_progress_while_it_happens(client):
    run_id = start_a_run(client)
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["steps_done"] >= 1)

    body = client.get(f"/runs/{run_id}").json()
    assert body["workflow"] == "eda-to-report"
    assert body["steps_total"] == 4
    assert body["steps"]["profile"]["status"] == "done"
    assert body["workflow_shape"]["personas"] == ["marie", "noel"]
    assert body["live"] is True


def test_the_event_log_is_replayable_from_any_point(client):
    run_id = start_a_run(client)
    assert wait_for(lambda: client.get(f"/runs/{run_id}/events.json").json()["events"])

    events = client.get(f"/runs/{run_id}/events.json").json()["events"]
    assert [e["index"] for e in events] == list(range(1, len(events) + 1))
    first = events[0]
    assert first["name"] == "dsagent.step"
    assert first["value"]["run_id"] == run_id

    # a reader that has seen `cursor` is handed everything after it and nothing
    # before — the run keeps writing while we ask, which is the point of a cursor
    cursor = len(events) - 1
    tail = client.get(f"/runs/{run_id}/events.json", params={"after": cursor}).json()["events"]
    assert [e["index"] for e in tail][: 2] == [cursor + 1, cursor + 2][: len(tail)]
    assert tail[0] == events[cursor]


def test_the_stream_hands_over_the_whole_run_and_then_follows_it(client, app):
    """One endpoint does the restore and the follow — §4.2's entire point.

    The gate is answered through the driver rather than over HTTP: `TestClient`
    serves one request at a time on one portal, so a POST issued while its own
    stream is open deadlocks the test, not the server. The HTTP gate has its own
    test below.
    """
    run_id = start_a_run(client)
    approver = threading.Thread(target=_approve_when_asked, args=(app, run_id), daemon=True)
    approver.start()

    seen, ended = [], False
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if "name" not in event:
                ended = True  # the end frame: the run is over and so is the stream
                break
            seen.append(event)

    assert ended, "the stream must close itself when the run finishes"
    names = [e["name"] for e in seen]
    assert names[0] == "dsagent.step"
    assert "dsagent.file" in names and "dsagent.note" in names
    # the whole run came down one connection: four steps, both ends of the gate
    steps = [e["value"] for e in seen if e["name"] == "dsagent.step"]
    assert len({s["step"] for s in steps}) == 4
    assert any(s["status"] == "awaiting_gate" and s["gate"]["decision"] is None for s in steps)
    assert any((s["gate"] or {}).get("decision") == "approve" for s in steps)
    assert client.get(f"/runs/{run_id}").json()["status"] == "done"


def test_a_second_reader_starting_late_gets_the_same_run(client):
    run_id = start_a_run(client)
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json().get("awaiting"))
    client.post(f"/runs/{run_id}/gate", json={"decision": "approve"})
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["status"] == "done", timeout=20)

    # nobody was attached while it ran, and the whole thing is still there
    events = client.get(f"/runs/{run_id}/events.json").json()["events"]
    assert len({e["value"]["step"] for e in events if e["name"] == "dsagent.step"}) == 4
    assert client.get(f"/runs/{run_id}/log").text  # M2.2.1 item 2


# --- gates -------------------------------------------------------------------


def test_the_gate_is_answered_against_the_run_not_against_a_stream(client):
    run_id = start_a_run(client)
    gate = wait_for(lambda: client.get(f"/runs/{run_id}").json().get("awaiting"))
    assert gate["step"] == "data-gate"
    assert gate["prompt"].startswith("Data gate report")
    assert gate["produces"] == ["artifacts/data-gate.md"]
    # the file the gate is about is already servable
    assert client.get(f"/runs/{run_id}/files/artifacts/data-gate.md").status_code == 200

    assert client.post(f"/runs/{run_id}/gate", json={"decision": "approve"}).json() == {
        "accepted": True, "decision": "approve",
    }
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["status"] == "done", timeout=20)
    assert client.get(f"/runs/{run_id}").json()["gate_wait"] > 0


def test_a_rejection_keeps_its_note_and_the_run_can_be_resumed(client):
    run_id = start_a_run(client)
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json().get("awaiting"))
    client.post(f"/runs/{run_id}/gate", json={"decision": "reject", "note": "fog rows look wrong"})

    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["status"] == "awaiting_gate")
    body = client.get(f"/runs/{run_id}").json()
    assert body["steps"]["data-gate"]["gate"]["note"] == "fog rows look wrong"
    assert body["steps"]["analyze"]["status"] == "pending"

    assert client.post(f"/runs/{run_id}/start").json() == {"started": True, "resumed": True}
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json().get("awaiting"), timeout=20)
    client.post(f"/runs/{run_id}/gate", json={"decision": "approve"})
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["status"] == "done", timeout=20)


def test_answering_a_gate_that_is_not_waiting_is_a_conflict(client):
    run_id = start_a_run(client)
    assert client.post(f"/runs/{run_id}/gate", json={"decision": "approve"}).status_code == 409


def test_an_unknown_run_is_a_404_everywhere(client):
    for path in ("/runs/nope", "/runs/nope/events.json", "/runs/nope/log", "/runs/nope/start"):
        method = client.post if path.endswith("start") else client.get
        assert method(path).status_code == 404, path
    assert client.get("/runs/..%2F..%2Fetc/events.json").status_code == 404


def test_an_empty_install_lists_no_runs(client):
    assert client.get("/runs").json() == {"runs": []}


# --- the canvas's server side -------------------------------------------------


def test_a_run_downloads_as_one_zip_of_its_deliverables(client):
    import io
    import zipfile

    run_id = start_a_run(client)
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json().get("awaiting"))
    client.post(f"/runs/{run_id}/gate", json={"decision": "approve"})
    assert wait_for(lambda: client.get(f"/runs/{run_id}").json()["status"] == "done", timeout=20)

    r = client.get(f"/runs/{run_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert run_id in r.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        names = sorted(n.split("/", 1)[1] for n in archive.namelist())
        assert names == [
            "artifacts/data-gate.md",
            "artifacts/data-profile.json",
            "artifacts/data-profile.md",
            "artifacts/figures/fig1_precip_by_category.png",
            "artifacts/figures/fig2_temp_by_category.png",
            "artifacts/figures/fig3_temp_trend.png",
            "artifacts/figures/fig4_precip_trend.png",
            "artifacts/findings.md",
            "report/findings.html",
            "report/findings.md",
        ]
        # the report is the deliverable, and it is whole
        report = archive.read(f"{run_id}/report/findings.html")
        assert report.startswith(b"<!doctype html>") and len(report) > 200_000

    # …and the dataset the operator uploaded is *not* in it: a deliverable is what
    # the run produced, not what it was given
    assert not any("seattle" in n for n in names)

    everything = client.get(f"/runs/{run_id}/download", params={"everything": True})
    with zipfile.ZipFile(io.BytesIO(everything.content)) as archive:
        assert any("data/seattle-weather.csv" in n for n in archive.namelist())


def test_downloading_a_run_that_has_produced_nothing_is_a_404(client):
    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    assert client.get(f"/runs/{run_id}/download").status_code == 404


def test_parquet_is_previewed_server_side(client, tmp_path):
    """No browser reads parquet; the M2.2 canvas could only offer a link (§4.4)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="pandas needs an engine to write parquet")

    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    frame = pd.DataFrame({
        "date": ["2012-01-01", "2012-01-02", "2012-01-03"],
        "precipitation": [0.0, 10.9, float("nan")],
        "weather": ["drizzle", "rain", "rain"],
    })
    target = tmp_path / "runs" / run_id / "workspace" / "data" / "weather.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)

    body = client.get(f"/runs/{run_id}/preview/data/weather.parquet").json()
    assert body["columns"] == ["date", "precipitation", "weather"]
    assert body["rows"][0] == ["2012-01-01", 0.0, "drizzle"]
    # NaN is not JSON, and a quality preview is exactly where it turns up
    assert body["rows"][2][1] is None
    assert (body["total_rows"], body["shown_rows"]) == (3, 3)

    assert client.get(f"/runs/{run_id}/preview/data/weather.parquet", params={"rows": 1}).json()[
        "shown_rows"
    ] == 1


def test_previewing_something_that_is_not_a_table_says_so(client, tmp_path):
    pytest.importorskip("pandas")
    run_id = start_a_run(client, upload=False)
    assert wait_for(lambda: (tmp_path / "runs" / run_id / "workspace" / "artifacts").is_dir())
    assert wait_for(
        lambda: (tmp_path / "runs" / run_id / "workspace" / "artifacts" / "data-gate.md").is_file(),
        timeout=20,
    )
    r = client.get(f"/runs/{run_id}/preview/artifacts/data-gate.md")
    assert r.status_code == 415
    assert "no preview for .md" in r.json()["detail"]
    assert client.get(f"/runs/{run_id}/preview/nope.parquet").status_code == 404


def test_delimited_text_is_previewed_without_any_dependency(client, tmp_path):
    """The preview dispatches on the extension; only parquet needs pandas.

    Quoted fields are the reason this is worth doing server-side at all: the
    browser's own split-on-comma reads `"Seattle, WA"` as two columns, and `csv`
    does not.
    """
    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    target = tmp_path / "runs" / run_id / "workspace" / "data" / "places.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('city,note\n"Seattle, WA",wet\nBend,dry\n', encoding="utf-8")

    body = client.get(f"/runs/{run_id}/preview/data/places.csv").json()
    assert body["columns"] == ["city", "note"]
    assert body["rows"][0] == ["Seattle, WA", "wet"]
    assert (body["total_rows"], body["shown_rows"]) == (2, 2)
    assert client.get(f"/runs/{run_id}/preview/data/places.csv", params={"rows": 1}).json()[
        "shown_rows"
    ] == 1


def test_an_interactive_artifact_is_served_under_its_own_prefix(client):
    """`/runs-x` is where the canvas points a frame that may run scripts (§4.5)."""
    run_id = start_a_run(client)
    assert wait_for(
        lambda: client.get(f"/runs/{run_id}/files/artifacts/data-gate.md").status_code == 200,
        timeout=20,
    )
    plain = client.get(f"/runs/{run_id}/files/artifacts/data-gate.md")
    scripted = client.get(f"/runs-x/{run_id}/files/artifacts/data-gate.md")

    assert scripted.status_code == 200
    assert scripted.content == plain.content
    assert "sandbox allow-scripts" in scripted.headers["content-security-policy"]
    assert "content-security-policy" not in plain.headers
    # and the same path rules apply: nothing escapes the workspace
    assert client.get(f"/runs-x/{run_id}/files/../../run.json").status_code == 404


def test_a_run_that_never_started_can_be_cleaned_up(client, tmp_path):
    """The launcher creates the directory first; a failure after that leaves one."""
    run_id = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()["run_id"]
    assert (tmp_path / "runs" / run_id).is_dir()

    assert client.delete(f"/runs/{run_id}").json() == {"deleted": run_id}
    assert not (tmp_path / "runs" / run_id).exists()
    assert client.get(f"/runs/{run_id}").status_code == 404


def test_a_run_that_happened_is_not_deletable(client):
    """A run directory is the record of what happened; this API does not erase it."""
    run_id = start_a_run(client)
    r = client.delete(f"/runs/{run_id}")
    assert r.status_code == 409
    assert "not deletable" in r.json()["detail"]
    assert client.get(f"/runs/{run_id}").status_code == 200


# ---- the plan, and stopping -------------------------------------------------


def test_a_plan_describes_the_run_without_running_any_of_it(client):
    """Everything in it is derivable, so nothing in it can be hallucinated."""
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]
    client.put(
        f"/runs/{run_id}/data/data_path?filename=weather.csv",
        content=b"date,temp,weather\n2012-01-01,12.8,drizzle\n2012-01-02,10.6,rain\n",
    )

    plan = client.post(f"/runs/{run_id}/plan").json()

    assert plan["workflow"] == "eda-to-report"
    assert plan["personas"] == ["marie", "noel"]
    assert plan["gates"] == ["data-gate"]
    assert [s["id"] for s in plan["steps"]] == ["profile", "data-gate", "analyze", "report"]
    # The file's shape, read where it landed.
    assert plan["profile"]["rows"] == 2
    assert [c["name"] for c in plan["profile"]["columns"]] == ["date", "temp", "weather"]
    assert [c["type"] for c in plan["profile"]["columns"]] == ["text", "number", "text"]
    assert len(plan["profile"]["preview"]) == 2
    # And nothing ran.
    assert client.get(f"/runs/{run_id}").json()["status"] == "pending"


def test_the_plan_guesses_a_key_from_the_column_that_has_no_repeats():
    """The harness does not know what a key is; it knows which column repeats."""
    from dsagent.api import _guesses

    shape = {"inputs": {
        "data_path": {"type": "path"},
        "question": {"type": "string"},
        "key_column": {"type": "string"},
    }}
    profile = {
        "rows": 3,
        "columns": [
            {"name": "date", "type": "text", "distinct": 3, "empty": 0},
            {"name": "weather", "type": "text", "distinct": 2, "empty": 0},
        ],
    }

    guesses = _guesses(shape, {"data_path": "data/w.csv"}, profile)

    assert guesses["key_column"]["value"] == "date"
    assert "unique" in guesses["key_column"]["why"]
    # `weather` repeats, so it is not offered — and `question` is not a column.
    assert "weather" not in str(guesses)
    assert "question" not in guesses


def test_a_value_the_operator_already_gave_is_not_guessed_over():
    from dsagent.api import _guesses

    shape = {"inputs": {"key_column": {"type": "string"}}}
    profile = {"rows": 2, "columns": [{"name": "id", "distinct": 2, "empty": 0}]}

    assert _guesses(shape, {"key_column": "date"}, profile) == {}
    # …but the ds cartridge's literal "None" default means *unfilled*.
    assert _guesses(shape, {"key_column": "None"}, profile)["key_column"]["value"] == "id"


def test_a_plan_with_nothing_to_compare_says_so_rather_than_inventing_a_price(client):
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    plan = client.post(f"/runs/{created['run_id']}/plan").json()

    assert plan["estimate"] == {"runs": 0, "cost_usd": None, "seconds": None}


def test_the_plan_is_kept_on_the_run(client):
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]
    client.post(f"/runs/{run_id}/plan")

    # It is the first entry in the run's audit trail: what was offered.
    assert client.get(f"/runs/{run_id}").json()["plan"]["workflow"] == "eda-to-report"


def test_stopping_a_run_nobody_is_driving_marks_it_stopped(client):
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]

    answer = client.post(f"/runs/{run_id}/stop").json()

    assert answer == {"stopping": False, "status": "stopped"}
    assert client.get(f"/runs/{run_id}").json()["status"] == "stopped"


def test_stopping_a_finished_run_changes_nothing(client, tmp_path):
    runs_dir = tmp_path / "runs"
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]
    state = json.loads((runs_dir / run_id / "run.json").read_text())
    state["status"] = "done"
    (runs_dir / run_id / "run.json").write_text(json.dumps(state))

    assert client.post(f"/runs/{run_id}/stop").json() == {"stopping": False, "status": "done"}
    assert client.get(f"/runs/{run_id}").json()["status"] == "done"


def test_the_plan_carries_the_operator_s_edits_back_to_the_run(client):
    """A plan you can change but whose changes the run never sees is a lie."""
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]

    client.post(f"/runs/{run_id}/plan", json={"inputs": {"key_column": "date"}})

    assert client.get(f"/runs/{run_id}").json()["inputs"]["key_column"] == "date"


def test_a_run_that_has_started_keeps_the_inputs_it_began_with(client, tmp_path):
    created = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}}).json()
    run_id = created["run_id"]
    runs_dir = tmp_path / "runs"
    state = json.loads((runs_dir / run_id / "run.json").read_text())
    state["status"] = "running"
    (runs_dir / run_id / "run.json").write_text(json.dumps(state))

    refused = client.post(f"/runs/{run_id}/plan", json={"inputs": {"key_column": "date"}})

    assert refused.status_code == 409
    assert "already started" in refused.json()["detail"]
