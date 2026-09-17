"""The runs API — the HTTP surface the product is built on.

Everything the UI needs about a run that is not the chat: the catalogue of
workflows, the list of runs, creating one (with the file the operator dropped on
the form), starting it, following its event log, and answering its gate.

A run belongs to the **server**, not to the browser tab that started it. That one
decision is what §4.2 is asking for: reload the page, open a second tab, or come
back after a restart, and the screen is rebuilt from `events.jsonl` and keeps
following. It is also why the gate is answered against the run (`POST
/runs/{id}/gate`) rather than against whichever stream happens to hold an
interrupt.

**One run is not like the others: a run started from the chat.** Its interrupt
lives in the chat graph's saver, which is in memory (`dsagent.serve.build_app`
explains why), and `POST /runs/{id}/gate` resumes the *driver's* graph. So such a
run advertises `awaiting_gate` in `run.json` and answers 409 here; it has to be
approved from the chat that started it, and a server restart loses the question
entirely. Runs started from the launcher — every run the product's screens
create — do not have this problem. Closing it means building the AG-UI agent
inside the server's lifespan so it can hold an async SQLite saver.

Nothing here knows any domain. The launcher's form is generated from whatever
inputs a workflow declares, so a cartridge with other inputs renders without a
line changing (invariant 1, applied to a screen).

Imported only from `dsagent.serve.build_app`, which already requires the `[ui]`
extra — so FastAPI is a module-level import here, and route annotations resolve.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from dsagent.cartridge.models import Cartridge
from dsagent.export import CHART_CONFIG, export_html
from dsagent.runner.charts import VEGA_LITE
from dsagent.runner.runner import GATE_VERSIONS, STOP_FILE
from dsagent.runs import (
    deliverables,
    is_live,
    list_runs,
    read_events,
    read_log,
    read_state,
    summarize,
)
from dsagent.serve import decision_of, resolve_run_file
from dsagent.tabular import READERS, TableUnreadable, read_table

POLL_SECONDS = 0.25
"""How often the SSE stream looks for new lines in a run's event log.

A quarter of a second is below what a person reads as lag and costs one `stat`
per stream per tick. The alternative — a filesystem watcher — buys nothing here:
the writer and the reader are the same process in every mode we ship.
"""

HEARTBEAT_SECONDS = 15.0
"""A comment line while nothing is happening, so a proxy does not close the stream
while a step is thinking. A step can go minutes between events and still be
working; a closed connection in the middle of that is a screen that stops."""

PREVIEW_ROWS = 200
PREVIEW_MAX_ROWS = 2000
PROFILE_ROWS = 500
"""How much of an uploaded file the plan reads to describe its shape.

Enough for "is this column unique" to mean something and cheap enough to answer
while the operator is still looking at the drop zone. The row *count* is the
file's real one; the column descriptions are of this head, and the screen says
so."""
PREVIEW_PROFILE_ROWS = 20
EMPTY_VALUES = {"", "none", "null", "nil"}
"""Declared defaults that mean "nothing".

A workflow may give an optional input a default of the literal string "None" —
YAML has no other way to say "unset" for a string — and a screen offering a guess
has to read that as unfilled rather than as a value somebody chose."""
UNIQUE_COLUMN = "unique_column"
"""The one kind of guess the harness can make: a column with no repeats and
nothing missing. What that is *for* is the cartridge's business; it asks for it
per input with `guess:` in `workflow.yaml`."""
"""How much of a table is a preview. 200 is what `docs/ui-product.md` §4.4 asks
for; the ceiling is there because the parameter arrives from a URL."""


def workflow_shape(cartridge: Cartridge, name: str) -> dict[str, Any]:
    """One workflow as the launcher needs it: inputs, personas, steps, gates.

    Everything comes off the loaded cartridge, so a second cartridge with other
    personas and other input types renders without a line changing here — which
    is invariant 1 applied to a screen.
    """
    wf = cartridge.workflows[name]
    steps = wf.ordered_steps()
    return {
        "name": wf.name,
        "cartridge": cartridge.name,
        "description": wf.description,
        "env": wf.env,
        "title_input": wf.title_input,
        "data_input": wf.data_input,
        "inputs": {
            key: {
                "type": spec.type,
                "required": spec.required and spec.default is None,
                "default": spec.default,
                "options": spec.options,
                "guess": spec.guess,
            }
            for key, spec in wf.inputs.items()
        },
        "personas": list(dict.fromkeys(s.persona for s in steps)),
        "steps": [
            {
                "id": s.id,
                "persona": s.persona,
                "env": s.env or wf.env,
                "needs": list(s.needs),
                "produces": list(s.produces),
                "section": s.section or "",
                "gate": {"kind": s.gate.kind, "prompt": s.gate.prompt} if s.gate else None,
            }
            for s in steps
        ],
    }


def new_run_id(runs_dir: Path, workflow: str) -> str:
    """A readable, unique directory name for a run the launcher is creating.

    Readable because it is in the URL of every screen and in every screenshot;
    unique by second, with a counter for the operator who starts two runs inside
    one. (A run started from the chat is still named after its tool call — there
    is no directory to name before the model calls the tool.)
    """
    base = f"{workflow}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not (runs_dir / base).exists():
        return base
    n = 2
    while (runs_dir / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def add_runs_routes(
    app: FastAPI, cartridges: list[Cartridge], runs_dir: Path, driver: Any
) -> None:
    by_workflow = {w: c for c in cartridges for w in c.workflows}

    def run_dir_of(run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise HTTPException(status_code=404, detail="unknown run")
        run_dir = runs_dir / run_id
        if not (run_dir / "run.json").is_file():
            raise HTTPException(status_code=404, detail="unknown run")
        return run_dir

    @app.get("/cartridges")
    def cartridges_route() -> dict[str, Any]:
        """What this server can run — the launcher's entire source of truth."""
        return {
            "cartridges": [
                {"name": c.name, "version": c.version, "description": c.description}
                for c in cartridges
            ],
            "workflows": [workflow_shape(c, name) for name, c in by_workflow.items()],
            "replay": getattr(driver, "workflow", None) if _is_replay(driver) else None,
        }

    @app.get("/runs")
    def runs_route(limit: int = 50) -> dict[str, Any]:
        return {"runs": [_with_live(driver, s).dict() for s in list_runs(runs_dir, limit=limit)]}

    @app.get("/runs/{run_id}")
    def run_route(run_id: str) -> dict[str, Any]:
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        summary = _with_live(driver, summarize(run_dir, state))
        workflow = state.get("workflow", "")
        cartridge = by_workflow.get(workflow)
        return {
            **summary.dict(),
            "steps": state.get("steps", {}),
            "workflow_shape": workflow_shape(cartridge, workflow) if cartridge else None,
            "plan": state.get("plan"),
            "driver_error": getattr(driver, "error_for", lambda _: None)(run_id),
        }

    @app.post("/runs")
    async def create_run(request: Request) -> dict[str, Any]:
        """Create a run directory. JSON: `{workflow, inputs}`.

        The run exists from this moment — listed, openable, `pending` — so the
        file the operator dropped has somewhere to land and the browser has a URL
        before anything starts. `PUT /runs/{id}/data/{input}` puts the file there;
        `POST /runs/{id}/start` sets it going.
        """
        body = await _json_body(request)
        workflow = str(body.get("workflow") or "")
        if workflow not in by_workflow:
            raise HTTPException(status_code=400, detail=f"unknown workflow {workflow!r}")
        inputs = body.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=400, detail="inputs must be a JSON object")

        run_id = new_run_id(runs_dir, workflow)
        try:
            driver.create(run_id, inputs, workflow)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"run_id": run_id, "inputs": inputs}

    @app.delete("/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, Any]:
        """Remove a run that never started.

        The launcher creates a run directory before it uploads or starts
        anything, so a failure in between leaves one behind: `pending`, empty,
        and for ever at the top of the list. This is how it cleans up after
        itself.

        A run that has started is not deletable here. Its directory is the record
        of what happened — the point of the whole event log — and a button that
        can erase that is not a button this API offers.
        """
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        if state.get("status") != "pending":
            raise HTTPException(
                status_code=409, detail="this run has started; its record is not deletable"
            )
        if driver.is_running(run_id):
            raise HTTPException(status_code=409, detail="this run is being driven right now")
        shutil.rmtree(run_dir)
        return {"deleted": run_id}

    @app.put("/runs/{run_id}/data/{name}")
    async def upload_input(run_id: str, name: str, request: Request,
                           filename: str = "") -> dict[str, Any]:
        """Put a file in the run's `data/` and point one input at it.

        The body is the file, raw — not a multipart part. Multipart would mean a
        new dependency (`python-multipart`) for a form with exactly one file in
        it, and this is what a drag-and-drop already has in hand: `fetch(url,
        {method: "PUT", body: file})`. `name` is the input the file fills, so a
        workflow declaring two datasets needs no new convention, and the operator
        never types `data_path=` — §7.2.

        Refused once the run has started: a run's inputs are the ones it began
        with, and the file under them is part of that.
        """
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        if state.get("status") != "pending":
            raise HTTPException(status_code=409, detail="this run has already started")
        workflow = by_workflow.get(state.get("workflow", ""))
        declared = workflow.workflows[state["workflow"]].inputs if workflow else {}
        if name not in declared:
            raise HTTPException(status_code=400, detail=f"{state.get('workflow')} has no input {name!r}")

        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        value = _store_upload(run_dir, filename or name, payload)

        state["inputs"] = {**(state.get("inputs") or {}), name: value}
        _rewrite_state(run_dir, state)
        return {"input": name, "value": value, "bytes": len(payload)}

    @app.post("/runs/{run_id}/start")
    def start_run(run_id: str) -> dict[str, Any]:
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        if driver.is_running(run_id):
            return {"started": False, "reason": "already running"}
        resume = state.get("status") not in ("pending",)
        driver.start(run_id, resume=resume)
        return {"started": True, "resumed": resume}

    @app.post("/runs/{run_id}/plan")
    async def plan_run(run_id: str, request: Request) -> dict[str, Any]:
        """What this run will do, before it does any of it.

        The plan-first gate the research puts at the top of what a new entrant
        can own (part C: plan-as-artifact with one Start gate). It executes
        nothing and costs nothing: every part of it is derivable — the steps and
        their gates from the workflow the cartridge declares, the shape of the
        data from the file already uploaded into this run, and the money from
        what past runs of the same workflow actually cost.

        It is deliberately not a model call. A proposal a person is about to
        approve should not itself be a thing that can hallucinate, and there is
        nothing here for a model to add that the cartridge and the file do not
        already say. The one judgement in it — which column is the key — is
        offered as a *guess*, named as one, and editable before Start.

        Stored on the run, because it is the first entry in its audit trail:
        what was offered, and what it was expected to cost.

        A body of `{"inputs": {...}}` is how the operator's edits get there. The
        plan is editable by design — the guess is a guess — and a plan you can
        change but whose changes the run never sees is a form that lies. Refused
        once the run has started: a run keeps the inputs it began with.
        """
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        edited = (await _json_body(request)).get("inputs") if await request.body() else None
        if edited:
            if state.get("status") != "pending":
                raise HTTPException(status_code=409, detail="this run has already started")
            if not isinstance(edited, dict):
                raise HTTPException(status_code=400, detail="inputs must be a JSON object")
            state["inputs"] = {**(state.get("inputs") or {}), **edited}
            _rewrite_state(run_dir, state)
        workflow = state.get("workflow", "")
        cartridge = by_workflow.get(workflow)
        if cartridge is None:
            raise HTTPException(status_code=400, detail=f"unknown workflow {workflow!r}")

        shape = workflow_shape(cartridge, workflow)
        inputs = dict(state.get("inputs") or {})
        profile = _profile_inputs(runs_dir, run_id, inputs)
        plan = {
            "workflow": workflow,
            "description": shape["description"],
            "inputs": inputs,
            "steps": shape["steps"],
            "personas": shape["personas"],
            "gates": [s["id"] for s in shape["steps"] if s["gate"]],
            "profile": profile,
            "guesses": _guesses(shape, inputs, profile),
            "estimate": _estimate(runs_dir, workflow),
        }
        state["plan"] = plan
        _rewrite_state(run_dir, state)
        return plan

    @app.post("/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        """Ask a run to stop. It stops between steps, and says so.

        Not a kill: a step is a persona holding a kernel and half a written
        file, and ending it there leaves a workspace nothing can describe. The
        step in flight finishes and the run stops before the next one — which is
        also what makes "stopping never costs more than what already ran" true
        rather than approximately true (§1.7).

        A run nobody is driving is simply marked stopped.
        """
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        if state.get("status") in ("done", "failed", "stopped"):
            return {"stopping": False, "status": state.get("status")}
        if driver.is_running(run_id):
            # A file, not a field in `run.json`: the runner rewrites that file
            # at the end of every step and would erase a flag it does not own.
            (run_dir / STOP_FILE).write_text("", encoding="utf-8")
            return {"stopping": True, "status": state.get("status")}
        state["status"] = "stopped"
        state["gate"] = None
        _rewrite_state(run_dir, state)
        return {"stopping": False, "status": "stopped"}

    @app.post("/runs/{run_id}/gate")
    async def answer_gate(run_id: str, request: Request) -> dict[str, Any]:
        """Answer the gate this run is standing at.

        The gate is answered against the *run*, not against whichever browser
        stream happens to be holding an interrupt — which is what lets a second
        tab, a reloaded page, or a browser opened after a server restart answer
        it (§7.6, §7.11).
        """
        run_dir_of(run_id)
        body = await _json_body(request)
        decision = decision_of(body)
        note = str(body.get("note") or "") if isinstance(body, dict) else ""
        accepted = driver.answer_gate(run_id, decision, note)
        if not accepted:
            raise HTTPException(status_code=409, detail="this run is not waiting at a gate")
        return {"accepted": True, "decision": decision.value}

    @app.get("/runs/{run_id}/events.json")
    def events_json(run_id: str, after: int = 0) -> dict[str, Any]:
        return {"events": read_events(run_dir_of(run_id), after=after)}

    @app.get("/runs/{run_id}/events")
    async def events_stream(run_id: str, request: Request, after: int = 0):
        """The run's event log as SSE, from `after` onwards, then live.

        One endpoint does both the restore and the follow: a page that connects
        at `after=0` is handed the whole run and then keeps receiving. That is
        what makes a reload, a second tab and a CLI-started run the same screen
        (§4.2) — none of them has to have been present at the start.
        """
        run_dir = run_dir_of(run_id)
        # A browser reconnecting an `EventSource` sends the last id it saw. Honour
        # it, or a dropped connection replays the whole run into a screen that
        # already has it.
        resumed = request.headers.get("last-event-id")
        start_at = after or (int(resumed) if (resumed or "").isdigit() else 0)

        async def stream():
            cursor = start_at
            quiet = time.time()
            while True:
                if await request.is_disconnected():
                    return
                events = read_events(run_dir, after=cursor)
                for event in events:
                    cursor = event["index"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if events:
                    quiet = time.time()
                elif not is_live(run_dir) and not driver.is_running(run_id):
                    yield f"event: end\ndata: {json.dumps({'index': cursor})}\n\n"
                    return
                elif time.time() - quiet > HEARTBEAT_SECONDS:
                    quiet = time.time()
                    yield ": still here\n\n"
                await asyncio.sleep(POLL_SECONDS)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            # `no-transform` and an explicit `identity` encoding are aimed at
            # whatever sits between this and the browser. The Next dev/prod proxy
            # gzips what it forwards, and a gzip stream buffers: the browser got
            # ten bytes of gzip header and then nothing, while `curl` — which
            # asks for no compression — saw every event. The stream is the whole
            # product; it may not be compressed.
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
        })

    @app.get("/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(run_id: str) -> str:
        return read_log(run_dir_of(run_id)) or "(this run has written no log)"

    @app.get("/runs/{run_id}/preview/{path:path}")
    def preview(run_id: str, path: str, rows: int = PREVIEW_ROWS) -> dict[str, Any]:
        """The first rows of a tabular file, as JSON, for a browser that cannot read it.

        Dispatched on the file's extension rather than assuming one format: this
        endpoint belongs to the harness, and the harness does not get to decide
        that "a table" means parquet. Parquet is the case that needed building —
        no browser reads it — and delimited text is here because reading it costs
        a stdlib import and saves the browser from parsing a 50 MB file it was
        handed whole.

        A format nobody here can read, or a reader that is not installed, is a
        415: the client can still download the file, and a guess would be worse.
        """
        run_dir = run_dir_of(run_id)
        target = resolve_run_file(runs_dir, run_id, path)
        if target is None:
            raise HTTPException(status_code=404, detail="not found")

        wanted = max(1, min(rows, PREVIEW_MAX_ROWS))
        if target.suffix.lower() not in READERS:
            raise HTTPException(
                status_code=415,
                detail=f"no preview for {target.suffix or 'a file with no extension'}",
            )
        try:
            columns, table, total = read_table(target, wanted)
        except TableUnreadable as e:
            raise HTTPException(status_code=415, detail=str(e)) from e

        return {
            "path": path,
            "run_id": run_dir.name,
            "columns": columns,
            "rows": table,
            "total_rows": total,
            "shown_rows": len(table),
        }

    @app.get("/runs/{run_id}/gate-version/{step}/{version}/{path:path}",
             response_class=PlainTextResponse)
    def gate_version(run_id: str, step: str, version: int, path: str) -> str:
        """An artifact as it was when a gate was sent back.

        The one thing a re-asked gate owes the person answering it: not "here is
        the report again", but "here is what changed since you refused it". The
        runner keeps a copy at the moment of rejection — outside the workspace,
        because it is not something the run produced — and this is how the screen
        reads it back to diff against what is there now.
        """
        run_dir = run_dir_of(run_id)
        root = (run_dir / GATE_VERSIONS).resolve()
        try:
            target = (root / step / f"v{version}" / path).resolve()
        except OSError:
            raise HTTPException(status_code=404, detail="not found") from None
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/chart-theme")
    def chart_theme() -> dict[str, Any]:
        """The Vega config every chart is drawn with, here and in an export.

        One copy, served: the screen and the exported file must look the same,
        and two hand-kept palettes drift the first time one of them is edited.
        """
        return CHART_CONFIG

    @app.get("/runs/{run_id}/export.html")
    def export_report(run_id: str) -> Response:
        """The run as one file somebody else can open, charts still alive.

        Self-contained: the Vega bundle, every spec, and a snapshot of the rows
        each chart draws. It works from a `file://` URL with no network and no
        server behind it, which is the difference between sending someone a
        report and sending them a screenshot.

        **Served as a download, and sandboxed if anything renders it anyway.**
        Every word in it was written by a persona, and a persona's words come
        from a model that read the operator's data; the export escapes and strips
        on the way in, but it must not *also* be a script running on this API's
        own origin. So: `Content-Disposition: attachment`, and a CSP that would
        give it an opaque origin with no same-origin access if a browser ever
        showed it inline.
        """
        run_dir = run_dir_of(run_id)
        state = read_state(run_dir)
        cartridge = by_workflow.get(state.get("workflow", ""))
        workflow = cartridge.workflows[state["workflow"]] if cartridge else None
        try:
            document = export_html(
                run_dir, state, vl_version=VEGA_LITE,
                title_input=workflow.title_input if workflow else None,
            )
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="this server has no vl-convert, so it cannot bundle an interactive report",
            ) from None
        return Response(
            content=document,
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="{run_dir.name}.html"',
                "Content-Security-Policy": "sandbox allow-scripts allow-downloads",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/runs/{run_id}/download")
    def download_all(run_id: str, everything: bool = False):
        """Every deliverable this run declared, as one zip.

        Deliverables by default and the whole workspace on request: what a
        stakeholder wants is the report and the figures, not the scratch files,
        and the run already knows which is which — the file events carry `kind`.
        """
        run_dir = run_dir_of(run_id)
        workspace = run_dir / "workspace"
        wanted = _archive_members(run_dir, workspace, everything=everything)
        if not wanted:
            raise HTTPException(status_code=404, detail="this run has produced nothing yet")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in wanted:
                archive.write(workspace / rel, arcname=f"{run_dir.name}/{rel}")
        payload = buffer.getvalue()
        return Response(
            content=payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{run_dir.name}.zip"',
                "Content-Length": str(len(payload)),
            },
        )


def _profile_inputs(runs_dir: Path, run_id: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """The shape of whatever table this run was given, read where it landed.

    Domain-agnostic on purpose: rows, columns, how many distinct values each
    holds and how many are empty, plus a few rows to look at. What those columns
    *mean* is the cartridge's business — the harness reports the shape of a file
    and nothing about its subject.

    Local in the sense that matters: the file is read by the server the operator
    is already running, and nothing about it leaves this process.
    """
    for name, value in inputs.items():
        if not isinstance(value, str):
            continue
        target = resolve_run_file(runs_dir, run_id, value)
        if target is None or target.suffix.lower() not in READERS:
            continue
        try:
            columns, rows, total = read_table(target, PROFILE_ROWS)
        except TableUnreadable:
            continue
        return {
            "input": name,
            "path": value,
            "bytes": target.stat().st_size,
            "rows": total,
            "columns": [_column(columns[i], [r[i] for r in rows]) for i in range(len(columns))],
            "preview": rows[:PREVIEW_PROFILE_ROWS],
            "preview_columns": columns,
        }
    return None


def _column(name: str, values: list[Any]) -> dict[str, Any]:
    """One column, described by what is in it rather than by what it is called."""
    present = [v for v in values if v not in (None, "")]
    numeric = bool(present) and all(_numberish(v) for v in present)
    return {
        "name": name,
        "type": "number" if numeric else "text",
        "distinct": len({str(v) for v in present}),
        "empty": len(values) - len(present),
        "sample": str(present[0]) if present else "",
    }


def _numberish(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _guesses(shape: dict[str, Any], inputs: dict[str, Any], profile: dict[str, Any] | None
             ) -> dict[str, dict[str, str]]:
    """Values the operator has not given, offered as guesses and labelled as such.

    Only one kind of guess is made, and only from the file's own shape: a column
    that is unique across every row it was shown, offered for an input whose
    workflow asked for exactly that with `guess: unique_column`. The harness does
    not know what a "key" is and never reads the *name* of an input to decide —
    a harness that looks for something ending in "column" has learned a
    cartridge's naming convention, which is invariant 1 going quietly.
    """
    if not profile:
        return {}
    unique = [
        c["name"] for c in profile["columns"]
        if c["distinct"] and c["distinct"] == min(profile["rows"], PROFILE_ROWS) and not c["empty"]
    ]
    out: dict[str, dict[str, str]] = {}
    for key, spec in shape["inputs"].items():
        given = str(inputs.get(key) or "").strip()
        if given and given.lower() not in EMPTY_VALUES:
            continue
        if spec.get("guess") == UNIQUE_COLUMN and unique:
            out[key] = {
                "value": unique[0],
                "why": f"unique across all {profile['rows']} rows, with nothing missing",
            }
    return out


def _estimate(runs_dir: Path, workflow: str) -> dict[str, Any]:
    """What this has cost before. The only honest source there is.

    Not a price list and not a guess from token counts: the same workflow, on
    this machine, finished. When it has never finished, the estimate says so —
    an invented number is how a product earns "cheap until it isn't".
    """
    past = [
        s for s in list_runs(runs_dir, limit=50)
        if s.workflow == workflow and s.status == "done" and s.cost_usd
    ]
    if not past:
        return {"runs": 0, "cost_usd": None, "seconds": None}
    return {
        "runs": len(past),
        "cost_usd": round(sum(s.cost_usd or 0 for s in past) / len(past), 3),
        "seconds": round(sum(s.duration or 0 for s in past) / len(past)),
    }


def _archive_members(run_dir: Path, workspace: Path, *, everything: bool) -> list[str]:
    """What goes into a run's zip, workspace-relative and in a stable order."""
    if everything:
        return sorted(
            f.relative_to(workspace).as_posix()
            for f in workspace.rglob("*")
            if f.is_file() and ".dsagent" not in f.relative_to(workspace).parts
        )
    return [rel for rel in deliverables(run_dir) if (workspace / rel).is_file()]


def _is_replay(driver: Any) -> bool:
    return type(driver).__name__ == "Replay"


def _with_live(driver: Any, summary):
    """Mark a run whose process is actually standing behind it.

    `run.json` cannot tell "running" from "was running when the machine died",
    so the driver — which owns the threads — is asked. A run the server is not
    driving and has not finished is stale, and saying so is better than a spinner
    that never stops.
    """
    summary.live = driver.is_running(summary.run_id)
    return summary


def _store_upload(run_dir: Path, filename: str, payload: bytes) -> str:
    """Write one uploaded file into the run's `data/`, return its input value.

    The name is reduced to its last path segment and to characters that cannot
    mean anything to a filesystem: it arrives from a browser, and a run directory
    is not the place to find out what a creative filename does.
    """
    safe = _safe_name(filename)
    dest = run_dir / "workspace" / "data" / safe
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return f"data/{safe}"


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="expected a JSON object") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")
    return body


def _rewrite_state(run_dir: Path, state: dict[str, Any]) -> None:
    """Write `run.json` back atomically, the way the runner writes it.

    Including the per-thread temp name: this process writes the file from a
    request handler while the run's own threads write it too, and a shared temp
    path is a crash waiting for the two to coincide.
    """
    import os
    import threading

    path = run_dir / "run.json"
    tmp = path.with_name(f".run.json.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _safe_name(filename: str) -> str:
    """A filename from a browser, reduced to something a run directory can hold.

    Last path segment only, and nothing but letters, digits and `-_.`: a run
    directory is not the place to discover what a creative filename does. Runs of
    substitutions collapse, so a name full of spaces and brackets arrives as
    something a person can still read in a file list.
    """
    name = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"-+", "-", "".join(
        c if c.isalnum() or c in "-_." else "-" for c in name
    )).strip("-.")
    return cleaned or "upload"


