"""Workflow runner: executes a declared DAG, one persona agent per step.

The runner never plans. It walks `workflow.ordered_steps()`, builds (or reuses)
the env each step declares, invokes the step's persona with the step
instructions, verifies `produces` on disk (entries may be globs), and handles
gates:

* ``human`` — calls `ask_human(prompt)`; on "no" the run stops in state
  ``awaiting_gate`` and can be resumed later with `--resume`.
* ``auto``  — runs the check script inside the step's env; non-zero exit fails
  the step.

State lives in ``<run_dir>/run.json`` so a run survives restarts. Each step
also records what it actually did — tool calls by name, token usage, skills
read and workspace files touched — read back from the agent result and from the
workspace itself, never from anything provider-specific.

While a step runs, the runner also *streams* what it sees to an optional
``on_event`` callback as `RunnerEvent`s: ``dsagent.step`` at each status change,
``dsagent.tool`` as the persona calls tools, and ``dsagent.file`` as workspace
files appear. Personas are executed with ``.stream()`` rather than ``.invoke()``
precisely so those land while the step is still working — a step can run for
minutes, and a consumer that only hears from it at the end is useless. Nothing
here knows about AG-UI or HTTP; `docs/ui-slice.md` describes the adapter that
turns these into `CUSTOM` events.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dsagent.cartridge.models import Cartridge, Step, Workflow
from dsagent.envs.base import Env, make_env
from dsagent.models import default_model
from dsagent.runner.charts import (
    CHART_EVENT,
    ChartRecord,
    StepContext,
    chart_tools,
)


class GateDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


STEP_EVENT = "dsagent.step"
TOOL_EVENT = "dsagent.tool"
FILE_EVENT = "dsagent.file"
NOTE_EVENT = "dsagent.note"
"""A persona saying something while it works, attributed to its step.

M2.2 read persona narration off the AG-UI message stream and attributed it by
time — a message that started inside a step's window was that step's — because
nothing on the wire says who is talking (`docs/ui-slice.md` §4). The runner does
know, so it says so: attribution by construction rather than by clock, and it
survives a reload, a second tab and a run nobody was watching, none of which a
message stream does.
"""

GATE_VERSIONS = "gate-versions"
"""Where a rejected gate's artifacts are kept, inside the run directory.

Deliberately *outside* `workspace/`: the workspace is what the run produced and
what its zip contains, and a copy kept so a person can see what changed is
neither. Read back through `GET /runs/{id}/gate-version/...`.
"""

EVENT_LOG = "events.jsonl"
"""Every `RunnerEvent` of a run, one JSON object per line, in the run directory.

The live `on_event` callback reaches whoever is attached *now*; this file is what
a reader gets who attached late, reloaded the page, opened a second tab, or is
looking at a run the CLI started. Reconstructing a screen from it is what makes
those three the same screen — `docs/ui-product.md` §4.2.
"""

RUN_LOG = "runner.log"
"""The `log()` lines of a run, in the run directory.

Written by the runner rather than by a front end, because otherwise it depends on
which front end drove the run: `dsagent serve` passes `log=lambda m: None`, so a
browser-driven run left nothing to read afterwards while a CLI run did — backwards,
and M2.2.1's second item.
"""

ARG_PREVIEW_CHARS = 120
"""Per-argument cap in a `dsagent.tool` preview. A `write_file` call carries the
whole file in its args; the event stream is for watching a run, not for shipping
its contents."""


@dataclass
class RunnerEvent:
    """Something a consumer can render while the run is still going.

    `name` is one of the three `dsagent.*` names; `value` is a JSON-safe dict.
    The schemas are in `docs/ui-slice.md` §3 and are the contract `dsagent serve`
    will hand to AG-UI unchanged.
    """

    name: str
    value: dict[str, Any]


@dataclass
class GateRequest:
    """What the runner tells a human gate, so an answer can be asked for.

    `ask_human` used to receive the prompt alone, which is all a terminal needs.
    A gate card in a browser needs to say which step of which run is waiting and
    what it produced, and under `dsagent serve` this becomes the `interrupt()`
    payload — see `docs/ui-slice.md` §3.
    """

    run_id: str
    workflow: str
    step: str
    persona: str
    produces: list[str]
    prompt: str


@dataclass
class GateAnswer:
    """A gate decision with the words that came with it.

    `ask_human` may return a bare `GateDecision` — that is all a y/n terminal
    prompt has — or this, which is what a gate card sends: a rejection is only
    actionable if the note saying *why* survives into `run.json` and onto the
    step's history in the browser.
    """

    decision: GateDecision
    note: str = ""


def as_answer(value: GateDecision | GateAnswer) -> GateAnswer:
    """Normalise whatever `ask_human` returned."""
    return value if isinstance(value, GateAnswer) else GateAnswer(decision=value)


@dataclass
class GateRecord:
    """What a human (or a check script) decided about a step, and when.

    Separate from `StepRecord.status` because they answer different questions:
    `status` is whether the step's *work* finished, `gate` is whether anyone
    agreed to go on. Conflating them is what made a re-entered run skip a decided
    gate along with its step — see `_gate`.

    `asked_at` is when the run stopped and `ts` when someone answered, so
    `ts - asked_at` is the time a run spent waiting for a person. Run 003 spent
    37 seconds there and no surface showed it, which is exactly the number needed
    to judge whether a gate earns its cost (M2.2.1, item 5).
    """

    decision: str  # GateDecision value: "approve" | "reject"
    note: str = ""
    ts: float = 0.0
    asked_at: float = 0.0


@dataclass
class StepRecord:
    id: str
    status: str = "pending"  # pending | running | done | failed — the work, not the gate
    gate: GateRecord | None = None
    """The decision on this step's gate, once one has been made. `None` while
    undecided, and re-set on every entry until it reads `approve`."""
    started_at: float | None = None
    finished_at: float | None = None
    output: str = ""
    error: str = ""
    tool_calls: dict[str, int] = field(default_factory=dict)
    """How many times the step called each tool, by tool name."""
    usage: dict[str, int] = field(default_factory=dict)
    """`input_tokens` / `output_tokens`, summed over the step's AI messages.

    Whatever `input_token_details` the provider reports (`cache_read`,
    `cache_creation`, ...) is summed alongside them, flattened, and present only
    when reported. Those are sub-counts of `input_tokens`, not additions to it.
    """
    skills_read: list[str] = field(default_factory=list)
    """Paths of SKILL.md files the step read, in first-read order."""
    files: list[dict[str, Any]] = field(default_factory=list)
    """`{path, mtime}` for workspace files the step created or modified, oldest first."""
    model: str = ""
    """The model this step actually ran on — the persona's, or the default."""
    cost_usd: float | None = None
    """What the step's tokens cost, by `prices.yaml`. `None` when the model has no
    rate there: an unknown price is not zero."""


@dataclass
class RunState:
    workflow: str
    cartridge: str
    inputs: dict[str, Any]
    status: str = "pending"  # pending | running | done | failed | awaiting_gate
    steps: dict[str, StepRecord] = field(default_factory=dict)
    gate_wait: float = 0.0
    """Seconds this run has spent waiting for a person, accumulated.

    A total rather than something derived from the step records, because a step
    keeps only its *standing* decision: a gate sent back and later approved would
    otherwise report the second wait and forget the first, and the first is the
    one where somebody read the report and said no."""
    charts: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Every chart and table the run has emitted, by `chart_id`, latest version.

    Kept on the run rather than only in the event log because a resumed run has
    to know that `precipitation-by-category` already exists at version 1 — that
    is what makes a persona's correction a *second version of that chart* rather
    than a second chart with the same title. The log keeps every version; this
    keeps the standing one.
    """
    stop_requested: bool = False
    """Somebody pressed stop. Honoured between steps, and cleared when it is.

    Between steps, not inside one: a step is a persona holding a kernel and half
    a file, and killing it there leaves a workspace nothing can describe. So the
    step in flight finishes and the run stops before the next one — which is
    also the only point at which "stopping never costs more than what already
    ran" is a promise the runner can keep.
    """
    plan: dict[str, Any] | None = None
    """The proposal this run was started from, if it was started from one.

    Kept on the run because it is the *audit trail's* first entry: what was
    offered, what it was expected to cost, and — once the run has happened —
    what it actually did. A plan the operator edited is the edited one.
    """
    gate: dict[str, Any] | None = None
    """The gate being asked right now: step, persona, prompt, produces, asked_at.

    `run.json` is the only thing that outlives the process, so a gate that exists
    solely as a blocked `ask_human` call is invisible the moment the server is
    restarted — and a run waiting for a person is exactly the run most likely to
    still be waiting when that happens. Cleared as soon as an answer arrives; a
    *rejected* gate leaves `status` at `awaiting_gate` with nothing pending,
    because nobody is being asked until the run is resumed.
    """

    @classmethod
    def load(cls, run_dir: Path) -> RunState:
        d = json.loads((run_dir / "run.json").read_text())
        d["steps"] = {k: _step_record(v) for k, v in d["steps"].items()}
        return cls(**d)

    def save(self, run_dir: Path) -> None:
        """Write `run.json`, atomically, from any thread.

        A run is written while it is being read: the runs API reads this file on
        every list and every poll, and a plain overwrite has a window where the
        reader gets a truncated — or empty — file. Same-directory temp plus
        `os.replace`, which is atomic on POSIX and on Windows.

        The temp name carries the thread id because a run is also written from
        **two** threads: the runner's, and whichever one a tool call lands on —
        `show_chart` records itself the moment it is called. With one shared temp
        path the two collide, the first `os.replace` consumes it, and the second
        dies with `No such file or directory: .run.json.tmp`. That killed
        `analyze` five minutes into a real run, after eleven `run_python` calls,
        with nothing wrong with the analysis at all.
        """
        path = run_dir / "run.json"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _step_record(d: dict[str, Any]) -> StepRecord:
    """`asdict` flattens `GateRecord`; reading it back has to rebuild it."""
    gate = d.get("gate")
    return StepRecord(**{**d, "gate": GateRecord(**gate) if gate else None})


AgentFactory = Callable[[Cartridge, str, Env, Path], Any]
"""(cartridge, persona, env, workspace) -> a compiled agent.

It must have ``.invoke({"messages": [...]})``. If it also has ``.stream()`` the
runner uses that instead, with ``stream_mode=["updates", "values"]``: the
``updates`` chunks are what make `dsagent.tool` and `dsagent.file` arrive while
the step is still running, and the last ``values`` chunk is the same final state
``.invoke()`` would have returned. An agent without ``.stream()`` still runs and
still reports its files — just all at once, when the step ends.
"""


class WorkflowRunner:
    def __init__(
        self,
        cartridge: Cartridge,
        run_dir: Path,
        *,
        agent_factory: AgentFactory | None = None,
        ask_human: Callable[[GateRequest], GateDecision | GateAnswer] | None = None,
        log: Callable[[str], None] = print,
        on_event: Callable[[RunnerEvent], None] | None = None,
        prices: Any = None,
    ) -> None:
        self.cartridge = cartridge
        self.run_dir = run_dir
        self.workspace = run_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agent_factory = agent_factory or self._default_factory
        self.ask_human = ask_human or (lambda request: GateDecision.APPROVE)
        self._log = log
        self.on_event = on_event
        self.run_id = run_dir.name
        self.event_log = run_dir / EVENT_LOG
        self.run_log = run_dir / RUN_LOG
        self.prices = prices
        self._envs: dict[str, Env] = {}
        self._reported: dict[str, float] = {}
        """Workspace mtimes already announced via `dsagent.file` for the running step."""
        self._here = StepContext()
        """Which step the chart tools should attribute an emission to. One runner
        drives every step of a workflow through one set of tools, so the answer
        has to be read when a tool is called, not when it is built."""
        self._state: RunState | None = None
        """The run being driven, so a chart emitted mid-step is saved with it."""

    # ---- events -------------------------------------------------------------

    def log(self, message: str) -> None:
        """One line of run narration: to `runner.log`, then to the front end.

        The file first, because that is the record; a front end that raises (a
        closed terminal, a disconnected browser) must not cost the run its log.
        """
        self._append(self.run_log, f"{time.strftime('%H:%M:%S')} {message}\n")
        self._log(message)

    def emit(self, name: str, value: dict[str, Any]) -> None:
        """Record one event, then hand it to the consumer if there is one.

        A consumer that raises must not take the run down with it: a browser
        disconnecting mid-run is not a reason to lose four minutes of work. The
        event is written to `events.jsonl` either way — a run nobody is watching
        still has to be readable afterwards.
        """
        event = RunnerEvent(name=name, value={"run_id": self.run_id, **value, "ts": time.time()})
        self._append(self.event_log, json.dumps({"name": event.name, "value": event.value}) + "\n")
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception as e:  # noqa: BLE001 — a broken consumer is not a broken run
            self._log(f"[event] consumer raised on {name}: {e}")

    def _append(self, path: Path, line: str) -> None:
        """Append one line, and never fail the run over it.

        Opened per line rather than held open: a run directory is written by
        whoever drives it — CLI, server, a resumed second process — and a handle
        held across a gate that lasts minutes is a handle held across a restart.
        """
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as e:
            self._log(f"[run] could not write {path.name}: {e}")

    def _emit_step(
        self, wf: Workflow, step: Step, status: str, error: str = "",
        gate: dict[str, Any] | None = None,
    ) -> None:
        order = [s.id for s in wf.ordered_steps()]
        self.emit(
            STEP_EVENT,
            {
                "workflow": wf.name,
                "step": step.id,
                "persona": step.persona,
                "env": step.env or wf.env,
                "index": order.index(step.id),
                "total": len(order),
                "section": step.section or "",
                "status": status,
                "needs": list(step.needs),
                "produces": list(step.produces),
                "produces_matched": self._produces_matched(step, status),
                "gate": gate,
                "error": error or None,
            },
        )

    def _produces_matched(self, step: Step, status: str) -> dict[str, list[str]]:
        """Each `produces` entry mapped to the real files it names, right now.

        `produces` is the promise and stays as declared; this is what has
        actually landed. A consumer needs both: before a step runs there is
        nothing to link to but there is something to show, and once a pattern is
        involved the promise is not a path at all — `artifacts/figures/*.png`
        cannot be a link or a tick until it is expanded.

        Keyed by entry rather than flattened so a tick is per promise: with a
        flat list, a reader cannot tell which pattern a matched file came from,
        and with two patterns it cannot tell whether both were satisfied.

        Empty lists on `started` — the step has not written anything yet, and
        anything matching then belongs to an earlier step. Computed for every
        other status, `failed` included, because what a failed step *did* manage
        to write is exactly what its reader wants to see.
        """
        if status == "started":
            return {entry: [] for entry in step.produces}
        return {entry: self.matched(entry) for entry in step.produces}

    def matched(self, entry: str) -> list[str]:
        """Workspace-relative files a `produces` entry names right now.

        Empty when nothing matches, which is what both the verification and the
        deliverable label key off. Real glob semantics via `Path.glob`, rather
        than `fnmatch` or `PurePath.match`: those two would let
        `figures/*.png` claim `artifacts/figures/x.png`, because one ignores the
        separator and the other matches from the right.
        """
        if not is_pattern(entry):
            return [entry] if (self.workspace / entry).is_file() else []
        return sorted(
            f.relative_to(self.workspace).as_posix()
            for f in self.workspace.glob(entry)
            if f.is_file()
        )

    def _emit_files(self, step: Step, before: dict[str, float], after: dict[str, float]) -> None:
        deliverables = {p for entry in step.produces for p in self.matched(entry)}
        for f in _changed_files(before, after):
            path = f["path"]
            try:
                size = (self.workspace / path).stat().st_size
            except OSError:
                # Written and removed again between two snapshots. Report it —
                # the step did touch it — but do not let a vanished scratch file
                # raise inside the `finally` that is reporting a real failure.
                size = 0
            self.emit(
                FILE_EVENT,
                {
                    "step": step.id,
                    "path": path,
                    "kind": "deliverable" if path in deliverables else "working",
                    "change": "modified" if path in before else "created",
                    "size": size,
                    "mtime": f["mtime"],
                },
            )

    # ---- envs ---------------------------------------------------------------

    def charts(self) -> list[Any]:
        """The chart tools, bound to this run.

        Handed to every persona, in every env: emitting a chart is not something
        a kernel does, it is something the *run* records, and a persona working
        in a Docker env has the same report to write into.
        """
        return chart_tools(
            self.workspace,
            self.run_id,
            on_chart=self._on_chart,
            context=lambda: self._here,
            known=self._known_chart,
        )

    def _known_chart(self, chart_id: str) -> ChartRecord | None:
        stored = (self._state.charts if self._state else {}).get(chart_id)
        return ChartRecord(**stored) if stored else None

    def _on_chart(self, rec: ChartRecord) -> None:
        """Record one emission: on the run, then in the log, then to the screen.

        Order matters on a resume. `run.json` is what a re-entering runner reads
        to decide whether this is version 1 or version 2; the event log is what
        the document is rebuilt from; the live consumer is a browser that may not
        be there. Writing the run first means an interrupted emission is still
        counted.
        """
        if self._state is not None:
            self._state.charts[rec.chart_id] = asdict(rec)
            self._state.save(self.run_dir)
        self.emit(CHART_EVENT, rec.as_event())

    def env_for(self, name: str) -> Env:
        if name not in self._envs:
            spec = self.cartridge.envs[name]
            self.log(f"[env] provisioning '{name}' ({spec.kind})")
            self._envs[name] = make_env(spec, self.workspace)
        return self._envs[name]

    def close(self) -> None:
        for e in self._envs.values():
            e.close()
        self._envs.clear()

    def _default_factory(self, cartridge: Cartridge, persona: str, env: Env, workspace: Path):
        from dsagent.host.build import build_persona_agent

        return build_persona_agent(cartridge, persona, env, workspace,
                                   extra_tools=self.charts())

    # ---- run ----------------------------------------------------------------

    def run(self, workflow_name: str, inputs: dict[str, Any] | None = None, *, resume: bool = False, dry_run: bool = False) -> RunState:
        wf = self.cartridge.workflows[workflow_name]
        state = (
            RunState.load(self.run_dir)
            if resume and (self.run_dir / "run.json").exists()
            else None
        )
        # A run keeps the inputs it started with, and they are resolved *after*
        # the recorded ones are merged in — a re-entering caller is the
        # accident-prone half. `--resume` without the original `-i`, or a
        # re-executed tool call where a model retypes them, would otherwise fail
        # validation for an input the run directory has recorded all along. Under
        # `dsagent serve` this is what keeps the launcher's form, rather than the
        # model, authoritative about what the run is running on.
        inputs = self._resolve_inputs(wf, {**(inputs or {}), **(state.inputs if state else {})})
        if state is not None:
            state.inputs = inputs
        else:
            state = RunState(workflow=wf.name, cartridge=self.cartridge.name, inputs=inputs,
                             steps={s.id: StepRecord(id=s.id) for s in wf.steps})
        state.status = "running"
        self._state = state
        state.save(self.run_dir)

        try:
            for step in wf.ordered_steps():
                if self._stopped(state):
                    return state
                rec = state.steps[step.id]
                # The work is skipped once it is done; the gate never is. A gate
                # is a `interrupt()` call under `dsagent serve`, and LangGraph
                # matches resume values by position, so a gate that disappears
                # from one entry to the next hands its answer to the next gate.
                if rec.status != "done":
                    self._run_step(wf, step, rec, state, inputs, dry_run)
                    if rec.status == "failed":
                        state.status = "failed"
                        state.save(self.run_dir)
                        return state
                if step.gate and not self._gate(wf, step, rec, state, dry_run):
                    return state
                if self._stopped(state):
                    return state
            state.status = "done"
            state.save(self.run_dir)
            return state
        finally:
            if not dry_run:
                self.close()

    def _stopped(self, state: RunState) -> bool:
        """Whether somebody asked this run to stop. Read from disk, not memory.

        The request arrives over HTTP, in another thread, and `run.json` is the
        only thing both sides agree on — the same reason a gate lives there.
        """
        try:
            asked = RunState.load(self.run_dir).stop_requested
        except (OSError, ValueError, TypeError):
            return False
        if not asked:
            return False
        state.stop_requested = False
        state.status = "stopped"
        state.save(self.run_dir)
        self.log("[run] stopped between steps, as asked")
        return True

    def _resolve_inputs(self, wf: Workflow, given: dict[str, Any]) -> dict[str, Any]:
        out = dict(given)
        missing = []
        for k, spec in wf.inputs.items():
            if k not in out:
                if spec.default is not None:
                    out[k] = spec.default
                elif spec.required:
                    missing.append(k)
            if spec.options and k in out and out[k] not in spec.options:
                raise ValueError(f"input '{k}' must be one of {spec.options}, got {out[k]!r}")
        if missing:
            raise ValueError(f"workflow '{wf.name}' missing required inputs: {', '.join(missing)}")
        return out

    def _task_message(self, wf: Workflow, step: Step, inputs: dict[str, Any],
                      sent_back: GateRecord | None = None) -> str:
        raw = (wf.path / step.instructions).read_text(encoding="utf-8")
        visible = _visible_inputs(step, raw, inputs)
        instructions = _fill(raw, visible)
        produces = "\n".join(
            f"- `{p}` (one or more matching files)" if is_pattern(p) else f"- `{p}`"
            for p in step.produces
        ) or "- (nothing mandatory)"
        inputs_md = "\n".join(f"- {k}: {v}" for k, v in visible.items()) or "- (none)"
        review = ""
        if sent_back is not None:
            note = sent_back.note.strip() or "(no note was given)"
            review = (
                f"## This work was sent back\n"
                f"A person reviewed what you produced here and did not approve it. "
                f"They said:\n\n> {note}\n\n"
                f"Read what is already on disk, address what they raised, and rewrite the "
                f"files below. Do not start from scratch and do not argue the point; if you "
                f"disagree, say so in the file and change what you can.\n\n"
            )
        return (
            f"# Workflow `{wf.name}` — step `{step.id}`\n\n"
            f"## Inputs\n{inputs_md}\n\n"
            f"{review}"
            f"## Instructions\n{instructions}\n\n"
            f"## You must produce these files (workspace-relative)\n{produces}\n\n"
            f"Finish with a short summary of what you did and any concerns for the next step."
        )

    def _run_step(self, wf: Workflow, step: Step, rec: StepRecord, state: RunState, inputs: dict[str, Any], dry_run: bool) -> None:
        env_name = step.env or wf.env
        self._here = StepContext(step=step.id, persona=step.persona, section=step.section or "")
        rec.status, rec.started_at = "running", time.time()
        state.save(self.run_dir)
        # The step event carries this and more; `log` keeps the lines it owns
        # alone (env provisioning, gates, failures).
        self._emit_step(wf, step, "started")
        if dry_run:
            rec.status, rec.finished_at, rec.output = "done", time.time(), "(dry run)"
            state.save(self.run_dir)
            self._emit_step(wf, step, "done")
            return
        before: dict[str, float] | None = None
        try:
            env = self.env_for(env_name)
            agent = self.agent_factory(self.cartridge, step.persona, env, self.workspace)
            before = _snapshot(self.workspace)
            # Set here, not in `_drive`: everything between this line and the
            # stream is a chance to raise, and the `finally` below diffs against
            # `_reported`. A stale baseline would blame this step for the last
            # step's files.
            self._reported = dict(before)
            sent_back = rec.gate if rec.gate and rec.gate.decision == "reject" else None
            result = self._drive(agent, self._task_message(wf, step, inputs, sent_back), step)
            rec.output = _last_text(result)
            # Before the produces check: a step that failed is the one worth reading.
            _record_telemetry(rec, result)
            self._record_cost(rec, step)
            missing = [entry for entry in step.produces if not self.matched(entry)]
            if missing:
                raise RuntimeError(f"step '{step.id}' did not produce: {', '.join(missing)}")
            rec.status = "done"
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            rec.status, rec.error = "failed", str(e)
            self.log(f"[step] {step.id} FAILED: {e}")
        finally:
            if before is not None:
                after = _snapshot(self.workspace)
                rec.files = _changed_files(before, after)
                # Anything the stream did not already report — everything, for an
                # agent that cannot stream; usually nothing for one that can.
                self._emit_files(step, self._reported, after)
            rec.finished_at = time.time()
            state.save(self.run_dir)
            self._emit_step(wf, step, rec.status, rec.error)

    def _record_cost(self, rec: StepRecord, step: Step) -> None:
        """Price the step's tokens, if this run was given a price list.

        The model is the persona's own or the default — the same resolution
        `build_persona_agent` makes — so a persona that opted into a stronger one
        is priced as what it actually ran on.
        """
        persona = self.cartridge.personas.get(step.persona)
        rec.model = (persona.model if persona and persona.model else default_model())
        if self.prices is not None:
            rec.cost_usd = self.prices.cost(rec.model, rec.usage)

    def _drive(self, agent: Any, message: str, step: Step) -> Any:
        """Run the persona, forwarding what it does as it does it.

        Returns the same final state `.invoke()` would have: the last `values`
        chunk. `self._reported` tracks the workspace as already-announced, so the
        `finally` in `_run_step` can emit whatever the stream missed without
        repeating what it did not.
        """
        payload = {"messages": [{"role": "user", "content": message}]}
        if not hasattr(agent, "stream"):
            return agent.invoke(payload)

        result: Any = None
        seen_calls: set[str] = set()
        seen_notes: set[str] = set()
        for mode, chunk in agent.stream(payload, stream_mode=["updates", "values"]):
            if mode == "values":
                result = chunk
                continue
            for node_update in (chunk or {}).values():
                messages = node_update.get("messages") or [] if isinstance(node_update, dict) else []
                self._emit_tools(step, messages, seen_calls)
                self._emit_notes(step, messages, seen_notes)
            after = _snapshot(self.workspace)
            self._emit_files(step, self._reported, after)
            self._reported = after
        return result if result is not None else {"messages": []}

    def _emit_tools(self, step: Step, messages: list[Any], seen_calls: set[str]) -> None:
        """One `dsagent.tool` per call as it is requested, one more as it returns."""
        for msg in messages:
            for call in _attr(msg, "tool_calls") or []:
                name, call_id = _attr(call, "name"), _attr(call, "id")
                if not name or call_id in seen_calls:
                    continue
                seen_calls.add(call_id)
                self.emit(TOOL_EVENT, {
                    "step": step.id, "persona": step.persona, "tool": name,
                    "tool_call_id": call_id, "phase": "started",
                    "args_preview": _preview(_attr(call, "args") or {}),
                })
            call_id = _attr(msg, "tool_call_id")
            if call_id:
                self.emit(TOOL_EVENT, {
                    "step": step.id, "persona": step.persona,
                    "tool": _attr(msg, "name") or "", "tool_call_id": call_id,
                    "phase": "finished", "args_preview": None,
                })

    def _emit_notes(self, step: Step, messages: list[Any], seen: set[str]) -> None:
        """One `dsagent.note` per thing the persona says, as it says it.

        A tool result is not narration and neither is an empty content block —
        only what the persona wrote for a reader. Deduped by message id because
        the same message reappears in later `updates` chunks; a message with no
        id at all is keyed by its text, which is the best available and costs one
        duplicate at worst.
        """
        for msg in messages:
            if _attr(msg, "tool_call_id") is not None:
                continue
            text = _text_of(_attr(msg, "content"))
            if not text.strip():
                continue
            key = str(_attr(msg, "id") or text)
            if key in seen:
                continue
            seen.add(key)
            self.emit(NOTE_EVENT, {"step": step.id, "persona": step.persona, "text": text})

    def _gate(self, wf: Workflow, step: Step, rec: StepRecord, state: RunState, dry_run: bool) -> bool:
        """Decide whether the run may go past this step. True = carry on.

        An **auto** gate is a script, and re-running a convergence check that
        already passed costs minutes for nothing, so it is skipped once approved.

        A **human** gate is asked on every entry, decided or not, and the answer
        is thrown away when `rec.gate` already reads `approve`. That looks
        wasteful and is the whole point of this method: under `dsagent serve`
        `ask_human` becomes a LangGraph `interrupt()`, LangGraph re-executes
        `run_workflow` from the top on resume and matches resume values
        positionally, and `Interrupt.id` comes from the call's position rather
        than its payload. Drop the call for gate 1 on re-entry and gate 2 silently
        receives gate 1's answer. Reproduced on langgraph 1.2.11; see
        `docs/ui-slice.md` §2.
        """
        gate = step.gate
        assert gate is not None
        decided = rec.gate.decision if rec.gate else None
        if gate.kind == "auto":
            return self._auto_gate(wf, step, rec, state, dry_run, decided)

        prompt = gate.prompt or f"Step '{step.id}' finished. Continue?"
        self.log(f"[gate] human: {prompt}")
        request = GateRequest(
            run_id=self.run_id, workflow=wf.name, step=step.id, persona=step.persona,
            produces=list(step.produces), prompt=prompt,
        )
        standing = decided == GateDecision.APPROVE.value
        if standing:
            # This gate is already answered and is only being re-asked to keep the
            # `interrupt()` sequence stable (see above). It must therefore touch
            # *nothing*: not `state.gate`, which belongs to whichever later gate
            # is actually waiting — gate 1 clearing it on re-entry is what left
            # gate 2 without an `asked_at` and reporting a 0 s wait — not the
            # accumulated wait, and not the event log, where a re-announced
            # "awaiting" for a decided gate is a question nobody is being asked.
            if not dry_run:
                self.ask_human(request)  # the call is the point; the answer is discarded
            self._emit_step(wf, step, rec.status, gate=_gate_event(gate.kind, prompt, rec.gate))
            return True

        # When the run stopped, not when it was resumed. Under `dsagent serve` the
        # first pass never returns from `ask_human` — it raises a LangGraph
        # interrupt — and the whole tool re-executes when the answer arrives, so
        # a clock read here would measure the resume rather than the wait, and
        # every gate would report 0s. The pending record written below is what
        # remembers the real moment; on re-entry it is read back, and it is keyed
        # by step so one gate never reads another's.
        pending = state.gate if isinstance(state.gate, dict) else None
        asked_at = (
            float(pending["asked_at"])
            if pending and pending.get("step") == step.id and pending.get("asked_at")
            else time.time()
        )
        # Announced *before* asking, so a reader who is not the one being asked —
        # a second tab, a reloaded page, a stakeholder on the link — sees the run
        # stop and sees what it stopped for. Recorded in `run.json` for the same
        # reason one step further out: a process that dies here must not take the
        # question with it.
        state.status = "awaiting_gate"
        state.gate = {**asdict(request), "kind": "human", "asked_at": asked_at}
        state.save(self.run_dir)
        self._emit_step(wf, step, "awaiting_gate",
                        gate={"kind": "human", "prompt": prompt, "asked_at": asked_at,
                              "decision": None, "note": "", "decided_at": None})
        answer = (
            GateAnswer(GateDecision.APPROVE) if dry_run else as_answer(self.ask_human(request))
        )
        decided_at = time.time()
        state.status, state.gate = "running", None
        state.gate_wait = round(state.gate_wait + max(0.0, decided_at - asked_at), 3)
        rec.gate = GateRecord(decision=answer.decision.value, note=answer.note,
                              ts=decided_at, asked_at=asked_at)
        state.save(self.run_dir)
        if answer.decision is GateDecision.APPROVE:
            self.log(f"[gate] approved after {decided_at - asked_at:.0f}s")
            self._emit_step(wf, step, rec.status, gate=_gate_event(gate.kind, prompt, rec.gate))
            return True
        state.status = "awaiting_gate"
        superseded = self._keep_version(step)
        # **A rejection sends the step back.** It used to leave the step `done`
        # and simply re-ask on resume, which made "send back with a note" a
        # pause: the persona never saw the note, nothing was rewritten, and the
        # only way forward was to approve the same artifact you had just
        # refused. A gate that cannot change anything is not a gate.
        #
        # The cost is stated rather than hidden: resuming re-runs the step, and
        # re-running a step costs what the step costs. That is the trade a person
        # makes when they say no.
        rec.status = "pending"
        state.save(self.run_dir)
        self._emit_step(wf, step, "awaiting_gate",
                        gate={**_gate_event(gate.kind, prompt, rec.gate),
                              "superseded": superseded})
        self.log(f"[gate] run paused at '{step.id}'. Resume with --resume once approved.")
        return False

    def _keep_version(self, step: Step) -> list[str]:
        """Copy what the reader was looking at when they sent this gate back.

        A rejection is the one moment a run has two versions of the same
        artifact: the one that was refused, and the one the persona writes next.
        Nothing else in the run keeps the first, so without this the question
        "what did they change?" can only be answered by reading the new file and
        remembering the old one. Returns the paths kept, for the gate event.
        """
        kept: list[str] = []
        # Numbered by how many versions are already kept, so a gate sent back
        # twice keeps both and neither overwrites the other.
        base = self.run_dir / GATE_VERSIONS / step.id
        version = len(list(base.glob("v*"))) + 1 if base.is_dir() else 1
        root = base / f"v{version}"
        for entry in step.produces:
            for rel in self.matched(entry):
                source = self.workspace / rel
                if not source.is_file():
                    continue
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(source, dest)
                except OSError as e:  # a copy that fails is not a run that fails
                    self._log(f"[gate] could not keep {rel}: {e}")
                    continue
                kept.append(rel)
        return kept

    def _auto_gate(self, wf: Workflow, step: Step, rec: StepRecord, state: RunState,
                   dry_run: bool, decided: str | None) -> bool:
        gate = step.gate
        assert gate is not None and gate.check is not None
        if decided == GateDecision.APPROVE.value or dry_run:
            return True
        self.log(f"[gate] auto check {gate.check} after {step.id}")
        env = self.env_for(step.env or wf.env)  # noqa: F841  # M2.3: auto-gate runs inside the env
        script = wf.path / gate.check
        asked_at = time.time()
        r = subprocess.run(
            ["python3", str(script)], cwd=self.workspace, capture_output=True, text=True, check=False
        )
        output = f"{r.stdout}{r.stderr}".strip()
        prompt = f"auto check {gate.check}"
        if r.returncode != 0:
            rec.gate = GateRecord(decision=GateDecision.REJECT.value, note=output[-500:],
                                  ts=time.time(), asked_at=asked_at)
            rec.status, rec.error = "failed", f"auto gate failed:\n{output}"
            state.status = "failed"
            state.save(self.run_dir)
            self._emit_step(wf, step, "failed", rec.error, gate=_gate_event("auto", prompt, rec.gate))
            return False
        rec.gate = GateRecord(decision=GateDecision.APPROVE.value, note=output[-500:],
                              ts=time.time(), asked_at=asked_at)
        state.save(self.run_dir)
        self._emit_step(wf, step, rec.status, gate=_gate_event("auto", prompt, rec.gate))
        return True


def _gate_event(kind: str, prompt: str, record: GateRecord | None) -> dict[str, Any]:
    """A decided gate, as it rides on `dsagent.step`.

    The same shape as the pending form the runner emits before asking, with the
    answer filled in — so a consumer keeps one reducer and a step row shows
    "waiting" and then "approved after 37 s" without special-casing either.
    """
    if record is None:
        return {"kind": kind, "prompt": prompt, "asked_at": None,
                "decision": None, "note": "", "decided_at": None}
    return {
        "kind": kind,
        "prompt": prompt,
        "asked_at": record.asked_at or None,
        "decision": record.decision,
        "note": record.note,
        "decided_at": record.ts or None,
    }


GLOB_CHARS = "*?["
"""What makes a `produces` entry a pattern rather than a path."""


def is_pattern(entry: str) -> bool:
    """True when a `produces` entry is a glob rather than a literal path.

    A step that writes a variable number of files cannot name them: `analyze`
    produced five figures in run 001 and four in run 002. `artifacts/figures/*.png`
    declares them as a group. It is still a contract — at least one file must
    match, or the step failed just as surely as if a named file were missing.
    """
    return any(c in entry for c in GLOB_CHARS)


SKILL_MANIFEST = "/SKILL.md"
"""A skill is a directory with this file; reading it is how a persona loads one."""


def _attr(obj: Any, key: str) -> Any:
    """Messages are objects from a chat model and plain dicts from a fake agent."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return result.get("messages") or []
    return _attr(result, "messages") or []


def _record_telemetry(rec: StepRecord, result: Any) -> None:
    """Read back what the step did from its own messages.

    Only LangChain's standard message surface is touched — `tool_calls` and
    `usage_metadata` — so this stays true for any provider, and a message that
    carries neither simply contributes nothing.
    """
    usage = {"input_tokens": 0, "output_tokens": 0}
    details: dict[str, int] = {}
    for msg in _messages(result):
        for call in _attr(msg, "tool_calls") or []:
            name = _attr(call, "name")
            if not name:
                continue
            rec.tool_calls[name] = rec.tool_calls.get(name, 0) + 1
            _record_skill_reads(rec, name, _attr(call, "args") or {})
        meta = _attr(msg, "usage_metadata") or {}
        for field_name in ("input_tokens", "output_tokens"):
            usage[field_name] += int(meta.get(field_name, 0) or 0)
        for name, count in (meta.get("input_token_details") or {}).items():
            details[name] = details.get(name, 0) + int(count or 0)
    rec.usage = {**usage, **details}


def _record_skill_reads(rec: StepRecord, tool: str, args: dict[str, Any]) -> None:
    if tool != "read_file":
        return
    for value in args.values():
        if isinstance(value, str) and SKILL_MANIFEST in value and value not in rec.skills_read:
            rec.skills_read.append(value)


def _preview(args: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments, minus the payload.

    `write_file` carries the whole file in `content`; a `dsagent.tool` event is
    for telling a reader *that* a file is being written, not for shipping it.
    Long strings are cut to `ARG_PREVIEW_CHARS` with the full length noted.
    """
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > ARG_PREVIEW_CHARS:
            out[k] = f"{v[:ARG_PREVIEW_CHARS]}… ({len(v)} chars)"
        elif isinstance(v, str | int | float | bool | type(None)):
            out[k] = v
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def _snapshot(workspace: Path) -> dict[str, float]:
    """mtime per workspace file, skipping the harness's own materialized skills."""
    out: dict[str, float] = {}
    for p in workspace.rglob("*"):
        rel = p.relative_to(workspace)
        if not p.is_file() or rel.parts[0] == ".dsagent":
            continue
        out[rel.as_posix()] = p.stat().st_mtime
    return out


def _changed_files(before: dict[str, float], after: dict[str, float]) -> list[dict[str, Any]]:
    changed = [{"path": p, "mtime": m} for p, m in after.items() if before.get(p) != m]
    return sorted(changed, key=lambda f: (f["mtime"], f["path"]))


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def visible_input_names(step: Step, instructions: str) -> list[str]:
    """Input names this step is allowed to see, without needing their values.

    A step sees the inputs its own instruction text interpolates, unless it
    declares `sees` explicitly. Showing every step every input is how a persona
    learns the finish line and runs ahead of its own job.
    """
    if step.sees is not None:
        return list(step.sees)
    seen: list[str] = []
    for name in _PLACEHOLDER.findall(instructions):
        if name not in seen:
            seen.append(name)
    return seen


def _visible_inputs(step: Step, instructions: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """The inputs this step is shown, in workflow declaration order."""
    names = visible_input_names(step, instructions)
    return {k: v for k, v in inputs.items() if k in names}


def _fill(text: str, inputs: dict[str, Any]) -> str:
    """Substitute `{input_name}` in step instructions. Everything else is literal.

    Substitution goes through `_PLACEHOLDER`, the same pattern
    `visible_input_names` uses to decide what a step sees — the two must agree,
    or a step is shown an input it cannot interpolate, or interpolates one it was
    never shown.

    This used `str.format_map`, which disagreed: format syntax reads
    `{"rhat_max": float}` as a field with a format spec and raises, so a step
    whose markdown documents a JSON artifact could not run at all
    (`mmm-meridian`'s `fit`). Step instructions are prose written for a persona,
    and prose contains braces — JSON, dict literals, CSS, f-string examples. They
    are all literal here; only a bare `{name}` naming a known input is replaced.
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in inputs:
            return match.group(0)
        value = inputs[name]
        return "None" if value is None else str(value)

    return _PLACEHOLDER.sub(substitute, text)


def _text_of(content: Any) -> str:
    """The readable text of a message's content, blocks or plain string.

    A content list mixes text with whatever else the provider sends (thinking,
    tool-use blocks, images); only `text` is narration.
    """
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


def _last_text(result: Any) -> str:
    try:
        msg = result["messages"][-1]
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            return "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return str(content)
    # Any message shape we do not recognise degrades to str().
    except Exception:  # noqa: BLE001  # pragma: no cover
        return str(result)
