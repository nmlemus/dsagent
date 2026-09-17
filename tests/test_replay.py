"""`dsagent serve --replay`: a recorded run played back, without a model.

The fixture under `ui/fixtures/run-eda-003` is run 003's own record (see
`tools/make_replay_fixture.py`), so these also assert that a replayed run is
indistinguishable, to every reader downstream, from the run it was made from: the
same event log, the same run.json shape, the same files on disk.
"""

import time
from pathlib import Path

import pytest

from dsagent.replay import Replay
from dsagent.runner import GateDecision
from dsagent.runs import read_events, read_state, summarize

FIXTURE = Path(__file__).resolve().parents[1] / "ui" / "fixtures" / "run-eda-003"
FAST = 2000.0


@pytest.fixture
def replay(tmp_path):
    return Replay(FIXTURE, tmp_path / "runs", speed=FAST)


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_fixture_is_the_run_it_was_made_from(replay):
    assert replay.workflow == "eda-to-report"
    names = {e["name"] for e in replay.events}
    assert names == {"dsagent.step", "dsagent.tool", "dsagent.file", "dsagent.note"}
    assert len(replay.events) > 100


def test_a_replayed_run_walks_the_whole_workflow(replay, tmp_path):
    run_dir = replay.create("replayed", {})
    replay.start("replayed")
    assert wait_for(lambda: replay.answer_gate("replayed", GateDecision.APPROVE)), "no gate arrived"
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")

    s = summarize(run_dir)
    assert (s.steps_done, s.steps_total) == (4, 4)
    assert s.workflow == "eda-to-report"
    # the numbers are the recorded run's, not zeros
    assert s.usage["input_tokens"] > 500_000
    assert s.gate_wait > 0

    events = read_events(run_dir)
    assert [e["value"]["run_id"] for e in events] == ["replayed"] * len(events)
    steps = [e["value"] for e in events if e["name"] == "dsagent.step"]
    assert [(v["step"], v["status"]) for v in steps][:2] == [
        ("profile", "started"), ("profile", "done"),
    ]
    assert steps[-1]["step"] == "report" and steps[-1]["status"] == "done"


def test_files_land_in_the_workspace_as_their_events_fire(replay):
    run_dir = replay.create("files", {})
    replay.start("files")
    assert wait_for(lambda: replay.answer_gate("files", GateDecision.APPROVE))
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")

    workspace = run_dir / "workspace"
    assert (workspace / "report" / "findings.html").is_file()
    assert (workspace / "artifacts" / "figures").is_dir()
    assert len(list((workspace / "artifacts" / "figures").glob("*.png"))) == 4
    # and the harness's own materialized skills are not part of a fixture
    assert not (workspace / ".dsagent").exists()


def test_the_run_stands_at_the_gate_until_someone_answers(replay):
    run_dir = replay.create("gated", {})
    replay.start("gated")
    assert wait_for(lambda: read_state(run_dir).get("gate"))

    gate = read_state(run_dir)["gate"]
    assert gate["step"] == "data-gate"
    assert gate["produces"] == ["artifacts/data-gate.md"]
    assert (run_dir / "workspace" / "artifacts" / "data-gate.md").is_file()

    # the recorded wait is not replayed: nothing moves while we do not answer
    settled = len(read_events(run_dir))
    time.sleep(0.4)
    assert len(read_events(run_dir)) == settled
    assert read_state(run_dir)["status"] == "awaiting_gate"

    assert replay.answer_gate("gated", GateDecision.APPROVE)
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")
    assert read_state(run_dir)["gate"] is None


def test_a_rejection_stops_the_run_and_keeps_the_note(replay):
    run_dir = replay.create("rejected", {})
    replay.start("rejected")
    assert wait_for(lambda: read_state(run_dir).get("gate"))
    assert replay.answer_gate("rejected", GateDecision.REJECT, "fog looks wrong")

    assert wait_for(lambda: not replay.is_running("rejected"))
    state = read_state(run_dir)
    assert state["status"] == "awaiting_gate"
    assert state["steps"]["data-gate"]["gate"]["note"] == "fog looks wrong"
    assert state["steps"]["analyze"]["status"] == "pending"


def test_resuming_a_rejected_run_reopens_the_gate_and_finishes(replay):
    run_dir = replay.create("resumed", {})
    replay.start("resumed")
    assert wait_for(lambda: read_state(run_dir).get("gate"))
    replay.answer_gate("resumed", GateDecision.REJECT, "check the fog rows")
    assert wait_for(lambda: not replay.is_running("resumed"))
    before = read_events(run_dir)

    replay.start("resumed", resume=True)
    assert wait_for(lambda: replay.answer_gate("resumed", GateDecision.APPROVE), timeout=10)
    assert wait_for(lambda: read_state(run_dir)["status"] == "done")

    after = read_events(run_dir)[len(before):]
    # the finished steps are not walked again; the gate is
    assert not [e for e in after if e["value"].get("step") == "profile"]
    replayed = (e["value"]["step"] for e in after if e["name"] == "dsagent.step")
    assert next(replayed) == "data-gate"
    assert {e["value"]["step"] for e in after if e["name"] == "dsagent.file"} == {"analyze", "report"}
    # and the rejection stays in the history
    rejections = [
        e for e in read_events(run_dir)
        if (e["value"].get("gate") or {}).get("decision") == "reject"
    ]
    assert rejections and rejections[0]["value"]["gate"]["note"] == "check the fog rows"


def test_answering_a_gate_nobody_is_holding_is_a_no(replay):
    replay.create("idle", {})
    assert replay.answer_gate("idle", GateDecision.APPROVE) is False


def test_inputs_from_the_launcher_win_over_the_recording(replay):
    run_dir = replay.create("mine", {"question": "Why is it always raining?"})
    state = read_state(run_dir)
    assert state["inputs"]["question"] == "Why is it always raining?"
    assert state["inputs"]["data_path"] == "data/seattle-weather.csv"
    assert state["status"] == "pending"
