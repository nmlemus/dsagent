"""The run directory as a readable record: `events.jsonl`, `runner.log`, the list.

No model, no kernel, no HTTP. These pin what `dsagent serve` reads back and what
makes a reload, a second tab and a CLI-started run show the same screen
(`docs/ui-product.md` §4.2).
"""

import json
from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import EVENT_LOG, RUN_LOG, GateAnswer, GateDecision, WorkflowRunner
from dsagent.runs import (
    deliverables,
    event_count,
    is_live,
    list_runs,
    read_events,
    read_log,
    summarize,
)
from tests.test_runner_events import FakeBackend, StreamingFakeAgent

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        lambda spec, ws: Env(spec=spec, workspace=ws, backend=FakeBackend()),
    )
    return tmp_path / "runs"


def drive(runs_dir: Path, run_id: str, *, ask=None, skip=None, resume=False):
    runner = WorkflowRunner(
        load_cartridge(DS), runs_dir / run_id,
        agent_factory=lambda cart, persona, env, ws: StreamingFakeAgent(persona, ws, skip),
        ask_human=ask or (lambda request: GateDecision.APPROVE),
        log=lambda m: None,
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"}, resume=resume)
    return runner.run_dir, state


def test_a_run_writes_its_own_event_log(runs_dir):
    run_dir, _ = drive(runs_dir, "r1")
    lines = (run_dir / EVENT_LOG).read_text().splitlines()
    assert lines
    names = [json.loads(line)["name"] for line in lines]
    assert set(names) == {"dsagent.step", "dsagent.tool", "dsagent.file"}
    # every line is a whole event, carrying the run it belongs to
    for line in lines:
        assert json.loads(line)["value"]["run_id"] == "r1"


def test_the_log_is_written_with_no_consumer_attached(runs_dir):
    """`dsagent serve` passes no logger and used to leave nothing behind (M2.2.1, item 2)."""
    run_dir, _ = drive(runs_dir, "quiet")
    assert event_count(run_dir) > 0
    assert "[gate] human" in read_log(run_dir)
    assert (run_dir / RUN_LOG).is_file()


def test_events_are_read_back_with_a_resumable_cursor(runs_dir):
    run_dir, _ = drive(runs_dir, "r1")
    everything = read_events(run_dir)
    assert [e["index"] for e in everything] == list(range(1, len(everything) + 1))

    half = len(everything) // 2
    rest = read_events(run_dir, after=half)
    assert rest == everything[half:]
    assert read_events(run_dir, after=len(everything)) == []


def test_a_half_written_line_is_not_an_error(runs_dir):
    """The file is being appended to while it is read; the tail can be partial."""
    run_dir, _ = drive(runs_dir, "r1")
    before = len(read_events(run_dir))
    with (run_dir / EVENT_LOG).open("a") as fh:
        fh.write('{"name": "dsagent.step", "value": {"step"')
    assert len(read_events(run_dir)) == before


def test_a_finished_run_summarises_itself(runs_dir):
    run_dir, _ = drive(runs_dir, "r1")
    s = summarize(run_dir)
    assert s.run_id == "r1"
    assert s.workflow == "eda-to-report"
    assert s.status == "done"
    assert (s.steps_done, s.steps_total) == (4, 4)
    assert s.inputs["data_path"] == "x.csv"
    assert s.started_at and s.finished_at and s.duration > 0
    assert not is_live(run_dir)


def test_the_summary_counts_what_the_run_waited_for_a_human(runs_dir):
    """Run 003 spent 37 s at its gate and no surface showed it (M2.2.1, item 5)."""
    import time

    def slow_yes(request):
        time.sleep(0.05)
        return GateDecision.APPROVE

    run_dir, _ = drive(runs_dir, "slow", ask=slow_yes)
    assert summarize(run_dir).gate_wait >= 0.05


def test_a_rejected_run_reads_as_awaiting_gate_with_its_note(runs_dir):
    run_dir, _ = drive(
        runs_dir, "r2", ask=lambda request: GateAnswer(GateDecision.REJECT, "fog looks wrong")
    )
    s = summarize(run_dir)
    assert s.status == "awaiting_gate"
    assert s.steps_done == 2
    assert is_live(run_dir), "a paused run is not finished"
    assert s.awaiting is None, "nobody is being asked until it is resumed"
    from dsagent.runs import read_state

    assert read_state(run_dir)["steps"]["data-gate"]["gate"]["note"] == "fog looks wrong"


def test_a_pending_gate_survives_the_process_that_was_asking(runs_dir):
    """Killed at a gate, the run directory still says which gate and when.

    This is what lets a restarted server offer the same question rather than
    losing it with the interrupt that was holding it (§7.11).
    """
    from dsagent.runs import read_state

    seen = {}

    def ask_and_look(request):
        seen.update(read_state(runs_dir / "r3"))
        return GateDecision.APPROVE

    drive(runs_dir, "r3", ask=ask_and_look)
    assert seen["status"] == "awaiting_gate"
    assert seen["gate"]["step"] == "data-gate"
    assert seen["gate"]["persona"] == "marie"
    assert seen["gate"]["produces"] == ["artifacts/data-gate.md"]
    assert seen["gate"]["asked_at"] > 0
    # and it is gone once answered
    assert read_state(runs_dir / "r3")["gate"] is None


def test_a_failed_run_surfaces_the_reason(runs_dir):
    run_dir, _ = drive(runs_dir, "bad", skip={"artifacts/data-gate.md"})
    s = summarize(run_dir)
    assert s.status == "failed"
    assert "did not produce" in s.error


def test_runs_are_listed_newest_first_and_a_broken_one_is_skipped(runs_dir):
    drive(runs_dir, "old")
    drive(runs_dir, "new")
    (runs_dir / "junk").mkdir()
    (runs_dir / "junk" / "run.json").write_text("{not json")
    (runs_dir / "not-a-run").mkdir()

    listed = list_runs(runs_dir)
    assert [s.run_id for s in listed] == ["new", "old"]
    assert list_runs(runs_dir, limit=1)[0].run_id == "new"


def test_deliverables_are_the_declared_files_in_the_order_they_landed(runs_dir):
    run_dir, _ = drive(runs_dir, "r1")
    assert deliverables(run_dir) == [
        "artifacts/data-profile.md",
        "artifacts/data-profile.json",
        "artifacts/data-gate.md",
        "artifacts/findings.md",
        "artifacts/figures/01.png",
        "artifacts/figures/02.png",
        "report/findings.md",
        "report/findings.html",
    ]


def test_resuming_appends_to_the_same_log(runs_dir):
    run_dir, _ = drive(runs_dir, "r4", ask=lambda request: GateDecision.REJECT)
    paused = event_count(run_dir)
    drive(runs_dir, "r4", resume=True)
    assert event_count(run_dir) > paused
    assert summarize(run_dir).status == "done"
    # the pause is still in the record: a resumed run keeps its history
    rejections = [
        e for e in read_events(run_dir)
        if (e["value"].get("gate") or {}).get("decision") == "reject"
    ]
    assert rejections


def test_the_wait_is_a_total_across_every_answer(runs_dir):
    """A gate sent back and then approved waited twice, and both count.

    A step keeps only its standing decision, so summing the step records reports
    the second wait and forgets the first — and the first is the one where
    somebody read the report and said no.
    """
    import time

    def slow_reject(request):
        time.sleep(0.15)
        return GateAnswer(GateDecision.REJECT, "look again")

    def slow_approve(request):
        time.sleep(0.1)
        return GateDecision.APPROVE

    run_dir, _ = drive(runs_dir, "twice", ask=slow_reject)
    assert summarize(run_dir).gate_wait >= 0.15

    drive(runs_dir, "twice", ask=slow_approve, resume=True)
    s = summarize(run_dir)
    assert s.status == "done"
    assert s.gate_wait >= 0.25, f"both waits must count, got {s.gate_wait}"
