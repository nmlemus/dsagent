"""Replay a recorded run into a fresh run directory, faster than it happened.

`dsagent serve --replay <fixture>` runs the whole product without a model: the
launcher starts a run, the events arrive spread over time the way they did,
files appear as they appeared, the gate stops and waits for a real answer, and
the canvas has real artifacts to render. Six minutes of run at 10× is forty
seconds, and it costs nothing, so a screen can be built and looked at fifty times
in an afternoon.

What it is not: a simulator. It plays back one recording — `tools/make_replay_fixture.py`
makes one from a real run directory — and everything it writes is that run's, with
new timestamps and a new run id. The gate is the one thing not replayed: the
recorded wait is discarded and the run stands there until someone answers, which
is the behaviour the screen has to get right.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from dsagent.models import default_model
from dsagent.runner.runner import EVENT_LOG, GateDecision, GateRecord, RunState, StepRecord

DEFAULT_SPEED = 10.0
EVENTS = "events.jsonl"


class ReplayError(RuntimeError):
    pass


class Replay:
    """A fixture, and the runs played back from it.

    One instance lives for the life of the server. Each `start()` drives one run
    on its own thread; a thread does the waiting because the recording is a list
    of timestamps and because the runner it stands in for is synchronous too.
    """

    def __init__(self, fixture: Path, runs_dir: Path, *, speed: float = DEFAULT_SPEED,
                 prices: Any = None) -> None:
        self.fixture = fixture
        self.runs_dir = runs_dir
        self.speed = max(speed, 0.01)
        self.prices = prices
        events_path = fixture / EVENTS
        if not events_path.is_file():
            raise ReplayError(f"{fixture} has no {EVENTS} — build one with tools/make_replay_fixture.py")
        self.events: list[dict[str, Any]] = [
            json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.source = json.loads((fixture / "run.json").read_text(encoding="utf-8"))
        self._gates: dict[str, _PendingGate] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ---- what the server asks of it ----------------------------------------

    @property
    def workflow(self) -> str:
        return self.source.get("workflow", "")

    def create(self, run_id: str, inputs: dict[str, Any], workflow: str = "") -> Path:
        """A run directory in `pending`, with the recorded inputs filled in.

        Whatever the launcher uploaded stays where it put it; the replay only
        adds the files the recording says the run wrote, and the recorded
        `data/` file is one of them.
        """
        if workflow and workflow != self.workflow:
            raise ReplayError(
                f"this fixture replays '{self.workflow}'; nothing else can be started in replay mode"
            )
        run_dir = self.runs_dir / run_id
        (run_dir / "workspace").mkdir(parents=True, exist_ok=True)
        state = RunState(
            workflow=self.workflow,
            cartridge=self.source.get("cartridge", ""),
            inputs={**self.source.get("inputs", {}), **inputs},
            status="pending",
            steps={k: StepRecord(id=k) for k in self.source.get("steps", {})},
        )
        state.save(run_dir)
        return run_dir

    def start(self, run_id: str, *, resume: bool = False) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return
        thread = threading.Thread(target=self._play, args=(run_id, resume), daemon=True,
                                  name=f"replay-{run_id}")
        self._threads[run_id] = thread
        thread.start()

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "") -> bool:
        """Hand an answer to a run standing at a gate. False if none is waiting."""
        pending = self._gates.get(run_id)
        if pending is None:
            return False
        pending.answer(decision, note)
        return True

    def is_running(self, run_id: str) -> bool:
        thread = self._threads.get(run_id)
        return bool(thread and thread.is_alive())

    # ---- the playback -------------------------------------------------------

    def _play(self, run_id: str, resume: bool = False) -> None:
        run_dir = self.runs_dir / run_id
        state = RunState.load(run_dir)
        done = {k for k, rec in state.steps.items() if rec.status == "done"} if resume else set()
        state.status = "running"
        state.gate = None
        state.save(run_dir)

        # A resumed run does not redo finished work — it walks straight back to
        # the gate it stopped at and asks again, which is what the runner does
        # with `resume=True` and what makes re-entering cheap.
        pending = [e for e in self.events if e["value"].get("step") not in done
                   or _is_pending_gate(e)]
        if not pending:
            pending = self.events

        origin = pending[0]["value"]["ts"]
        started = time.time()
        try:
            for event in pending:
                value = event["value"]
                _sleep_until(started + (value["ts"] - origin) / self.speed)
                if _is_pending_gate(event):
                    started += self._hold_at_gate(run_id, run_dir, state, event)
                    continue
                if event["name"] == "dsagent.step" and (value.get("gate") or {}).get("decision"):
                    continue  # the recorded answer; this run's own was just written
                self._apply(run_dir, state, event, run_id)
            state.status = self.source.get("status", "done")
            state.save(run_dir)
        except _Rejected:
            pass  # the run is paused at its gate, exactly where a real one stops
        finally:
            self._threads.pop(run_id, None)

    def _hold_at_gate(self, run_id: str, run_dir: Path, state: RunState, event: dict) -> float:
        """Emit the pending gate, then wait for a person. Returns the wait.

        The wait is added to the playback's clock rather than eaten out of it:
        the rest of the recording still plays at its own pace once the answer
        arrives, instead of fast-forwarding to catch up with a schedule that was
        set before anybody thought about it.
        """
        value = event["value"]
        asked_at = time.time()
        gate = {**value["gate"], "asked_at": asked_at}
        self._write(run_dir, run_id, event["name"], {**value, "gate": gate})
        state.status = "awaiting_gate"
        state.gate = {
            "run_id": run_id, "workflow": state.workflow, "step": value["step"],
            "persona": value["persona"], "produces": list(value["produces"]),
            "prompt": gate.get("prompt", ""), "kind": "human", "asked_at": asked_at,
        }
        state.save(run_dir)

        pending = _PendingGate()
        self._gates[run_id] = pending
        decision, note = pending.wait()
        self._gates.pop(run_id, None)
        decided_at = time.time()

        rec = state.steps[value["step"]]
        rec.gate = GateRecord(decision=decision.value, note=note, ts=decided_at, asked_at=asked_at)
        state.gate = None
        state.status = "running" if decision is GateDecision.APPROVE else "awaiting_gate"
        state.save(run_dir)
        self._write(run_dir, run_id, "dsagent.step", {
            **value,
            "status": "done" if decision is GateDecision.APPROVE else "awaiting_gate",
            "gate": {**gate, "decision": decision.value, "note": note, "decided_at": decided_at},
        })
        if decision is not GateDecision.APPROVE:
            # A rejected replay stops exactly where a rejected run stops.
            raise _Rejected()
        return decided_at - asked_at

    def _apply(self, run_dir: Path, state: RunState, event: dict, run_id: str) -> None:
        value = event["value"]
        name = event["name"]
        if name == "dsagent.file":
            self._copy_file(run_dir, value["path"])
        if name == "dsagent.step":
            self._record_step(state, value)
            state.save(run_dir)
        self._write(run_dir, run_id, name, value)

    def _record_step(self, state: RunState, value: dict) -> None:
        rec = state.steps.setdefault(value["step"], StepRecord(id=value["step"]))
        source = self.source["steps"].get(value["step"], {})
        if value["status"] == "started":
            rec.status, rec.started_at = "running", time.time()
            return
        if value["status"] in ("done", "failed"):
            rec.status = value["status"]
            rec.finished_at = time.time()
            # Telemetry is the recording's: replaying a run that cost $0.47 and
            # showing $0 would make every number on the screen a lie.
            rec.tool_calls = dict(source.get("tool_calls") or {})
            rec.usage = dict(source.get("usage") or {})
            rec.skills_read = list(source.get("skills_read") or [])
            rec.output = source.get("output", "")
            rec.model = source.get("model") or default_model()
            # Priced from the recording's own tokens rather than copied: the
            # recording predates the field, and a screen showing "—" where a real
            # run shows a cost would be a screen developed against a lie.
            if self.prices is not None:
                rec.cost_usd = self.prices.cost(rec.model, rec.usage)

    def _copy_file(self, run_dir: Path, rel: str) -> None:
        src = self.fixture / "workspace" / rel
        if not src.is_file():
            return
        dest = run_dir / "workspace" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def _write(self, run_dir: Path, run_id: str, name: str, value: dict[str, Any]) -> None:
        """Append to the run's own event log — the same file a real run writes.

        Everything downstream (the SSE endpoint, the reload path, the home
        screen) reads that file and nothing else, so a replayed run and a real
        one are the same run to every reader.
        """
        line = json.dumps({"name": name, "value": {**value, "run_id": run_id, "ts": time.time()}})
        with (run_dir / EVENT_LOG).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _is_pending_gate(event: dict[str, Any]) -> bool:
    """A recorded gate that was asked and not yet answered, in the recording."""
    value = event["value"]
    return (
        event["name"] == "dsagent.step"
        and value.get("status") == "awaiting_gate"
        and (value.get("gate") or {}).get("decision") is None
    )


class _Rejected(Exception):
    """A replayed gate was rejected; the playback stops where the run would."""


class _PendingGate:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._decision = GateDecision.REJECT
        self._note = ""

    def answer(self, decision: GateDecision, note: str) -> None:
        self._decision, self._note = decision, note
        self._event.set()

    def wait(self) -> tuple[GateDecision, str]:
        self._event.wait()
        return self._decision, self._note


def _sleep_until(due: float) -> None:
    remaining = due - time.time()
    if remaining > 0:
        time.sleep(remaining)


def replay_state(fixture: Path) -> dict[str, Any]:
    """The recorded run's own `run.json`, for tests and for `--replay` diagnostics."""
    return json.loads((fixture / "run.json").read_text(encoding="utf-8"))


__all__ = ["DEFAULT_SPEED", "Replay", "ReplayError", "replay_state"]
