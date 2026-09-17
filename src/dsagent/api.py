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

Nothing here knows any domain. The launcher's form is generated from whatever
inputs a workflow declares, so a cartridge with other inputs renders without a
line changing (invariant 1, applied to a screen).

Imported only from `dsagent.serve.build_app`, which already requires the `[ui]`
extra — so FastAPI is a module-level import here, and route annotations resolve.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from dsagent.cartridge.models import Cartridge
from dsagent.runs import is_live, list_runs, read_events, read_log, read_state, summarize
from dsagent.serve import decision_of

POLL_SECONDS = 0.25
"""How often the SSE stream looks for new lines in a run's event log.

A quarter of a second is below what a person reads as lag and costs one `stat`
per stream per tick. The alternative — a filesystem watcher — buys nothing here:
the writer and the reader are the same process in every mode we ship.
"""

HEARTBEAT_SECONDS = 15.0
"""A comment line while nothing is happening, so a proxy does not close the stream
during `analyze`'s three silent minutes."""


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
        "inputs": {
            key: {
                "type": spec.type,
                "required": spec.required and spec.default is None,
                "default": spec.default,
                "options": spec.options,
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

        async def stream():
            cursor = after
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.get("/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(run_id: str) -> str:
        return read_log(run_dir_of(run_id)) or "(this run has written no log)"


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
    """Write `run.json` back atomically, the way the runner writes it."""
    import os

    path = run_dir / "run.json"
    tmp = path.with_name(".run.json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _safe_name(filename: str) -> str:
    """A filename from a browser, reduced to something a run directory can hold.

    Last path segment only, and nothing but letters, digits and `-_.`: a run
    directory is not the place to discover what a creative filename does. Runs of
    substitutions collapse, because `seattle weather (1).csv` should read as
    `seattle-weather-1-.csv`'s tidier cousin in a file list someone is looking at.
    """
    name = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"-+", "-", "".join(
        c if c.isalnum() or c in "-_." else "-" for c in name
    )).strip("-.")
    return cleaned or "upload"


