"""Reading runs off disk — the list, one run, and its event log.

A run directory is the unit of truth: `run.json` is its state, `events.jsonl`
everything that happened in order, `runner.log` the narration, `workspace/` the
files. Nothing here knows which front end produced any of it, which is the point
— a run started from the CLI, from the chat or from the launcher reads the same,
and so does one whose process is long gone.

No FastAPI and no `[ui]` extra: `dsagent serve` puts HTTP in front of this, and
the tests read it directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dsagent.runner.runner import EVENT_LOG, RUN_LOG

RUN_JSON = "run.json"
WORKSPACE = "workspace"

TERMINAL = ("done", "failed")
"""Run states that will not change again without someone asking for it."""


@dataclass
class RunSummary:
    """One row of the home screen, and the header of the run screen.

    Everything here is derived from the run directory alone. `status` is the
    run's, not a step's: `awaiting_gate` means the run is stopped at a gate,
    whether the process is still standing there waiting for an answer or died
    while it waited.
    """

    run_id: str
    workflow: str
    cartridge: str
    status: str
    inputs: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    duration: float | None = None
    """Wall time of the run so far — to `finished_at`, or to now while it runs."""
    steps_done: int = 0
    steps_total: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    gate_wait: float = 0.0
    """Seconds this run spent waiting for a human, summed over its gates."""
    awaiting: dict[str, Any] | None = None
    """The gate being asked right now, if any — step, prompt, produces, asked_at."""
    live: bool = False
    """Whether a process is actually standing behind this run at the moment.

    `run.json` records what the run was doing, not whether anyone is still doing
    it: a run whose server was killed mid-step reads `running` forever. The
    driver owns the threads, so the API fills this in — a spinner that never
    stops is worse than "this run was interrupted"."""
    error: str | None = None

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def run_dirs(runs_dir: Path) -> list[Path]:
    """Every directory under `runs_dir` that looks like a run."""
    if not runs_dir.is_dir():
        return []
    return [d for d in runs_dir.iterdir() if d.is_dir() and (d / RUN_JSON).is_file()]


def list_runs(runs_dir: Path, *, limit: int | None = None) -> list[RunSummary]:
    """Every readable run, newest first.

    A run directory whose `run.json` is unreadable — half-written, or from a
    future schema — is skipped rather than fatal: one bad run must not cost the
    operator the list of the good ones.
    """
    summaries = []
    for d in run_dirs(runs_dir):
        try:
            summaries.append(summarize(d))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    summaries.sort(key=lambda s: s.started_at or 0, reverse=True)
    return summaries[:limit] if limit else summaries


def read_state(run_dir: Path) -> dict[str, Any]:
    """`run.json` as written, no interpretation."""
    return json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))


def summarize(run_dir: Path, state: dict[str, Any] | None = None) -> RunSummary:
    state = state if state is not None else read_state(run_dir)
    steps: dict[str, dict[str, Any]] = state.get("steps") or {}
    starts = [s["started_at"] for s in steps.values() if s.get("started_at")]
    ends = [s["finished_at"] for s in steps.values() if s.get("finished_at")]
    status = state.get("status", "pending")
    started_at = min(starts) if starts else None
    finished_at = max(ends) if ends and status in TERMINAL else None

    usage: dict[str, int] = {}
    for s in steps.values():
        for k, v in (s.get("usage") or {}).items():
            usage[k] = usage.get(k, 0) + int(v or 0)

    gate_wait = 0.0
    for s in steps.values():
        gate = s.get("gate") or {}
        if gate.get("ts") and gate.get("asked_at"):
            gate_wait += max(0.0, gate["ts"] - gate["asked_at"])

    error = next((s.get("error") for s in steps.values() if s.get("error")), None)
    return RunSummary(
        run_id=run_dir.name,
        workflow=state.get("workflow", ""),
        cartridge=state.get("cartridge", ""),
        status=status,
        inputs=state.get("inputs") or {},
        started_at=started_at,
        finished_at=finished_at,
        duration=_duration(started_at, finished_at, status),
        steps_done=sum(1 for s in steps.values() if s.get("status") == "done"),
        steps_total=len(steps),
        usage=usage,
        gate_wait=gate_wait,
        awaiting=state.get("gate"),
        error=error,
    )


def _duration(started_at: float | None, finished_at: float | None, status: str) -> float | None:
    if started_at is None:
        return None
    if finished_at is not None:
        return max(0.0, finished_at - started_at)
    # Still going (or abandoned mid-step): the honest number is "so far".
    return max(0.0, time.time() - started_at)


def read_events(run_dir: Path, after: int = 0) -> list[dict[str, Any]]:
    """Events `after` onwards, each tagged with its 1-based index.

    The index is the cursor a reader resumes from, and it is the line number
    rather than a timestamp so that two events in the same millisecond cannot
    collapse into one. A half-written last line — the run is appending to this
    file as we read it — is dropped, not raised on; the next read gets it whole.
    """
    path = run_dir / EVENT_LOG
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if i <= after or not line.endswith("\n"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event["index"] = i
            out.append(event)
    return out


def event_count(run_dir: Path) -> int:
    path = run_dir / EVENT_LOG
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.endswith("\n"))


def read_log(run_dir: Path) -> str:
    path = run_dir / RUN_LOG
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def is_live(run_dir: Path, state: dict[str, Any] | None = None) -> bool:
    """Whether a reader should keep waiting for more events.

    True while the run's own state says it has not finished. A run whose process
    died mid-step stays "live" by this test, which is the safe direction: the
    stream stays open, the screen keeps showing what it has, and nothing claims a
    result that was never produced.
    """
    try:
        state = state if state is not None else read_state(run_dir)
    except (OSError, ValueError):
        return False
    return state.get("status") not in TERMINAL


def deliverables(run_dir: Path) -> list[str]:
    """Workspace-relative paths this run declared and delivered, in order.

    Read from the file events rather than from the workspace, because the
    declaration is what separates a deliverable from a scratch file and only the
    events carry it.
    """
    out: list[str] = []
    for event in read_events(run_dir):
        value = event.get("value") or {}
        if event.get("name") == "dsagent.file" and value.get("kind") == "deliverable":
            path = value.get("path")
            if path and path not in out:
                out.append(path)
    return out
