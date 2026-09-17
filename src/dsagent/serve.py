"""`dsagent serve` — the orchestrator over AG-UI, and the runs API the UI is built on.

Three surfaces on one FastAPI app:

* **`POST /agent`** (and `GET /agent/health`), registered by
  `ag_ui_langgraph.add_langgraph_fastapi_endpoint`. The chat: the orchestrator's
  own messages and tool calls over AG-UI SSE, and a gate as an interrupt.
* **The runs API** — `GET /cartridges`, `GET|POST /runs`, `GET /runs/{id}`,
  `POST /runs/{id}/start`, `POST /runs/{id}/gate`, `GET /runs/{id}/events` (SSE)
  and `/events.json`, `GET /runs/{id}/log`. A run belongs to the server, not to
  the tab that started it, so a reload, a second tab and a CLI-started run all
  show the same thing (`docs/ui-product.md` §4.1, §4.2).
* **`GET /runs/{run_id}/files/{path}`**, a run's workspace, for the canvas.

Everything domain-specific still comes from cartridges; this module only knows
about workflows in the abstract — the launcher's form is generated from whatever
inputs a workflow declares, and nothing here has heard of a dataset.

The `[ui]` extra is required: `pip install -e ".[ui,anthropic]"`.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from dsagent.cartridge.models import Cartridge
from dsagent.envs.base import Env
from dsagent.runner import (
    GateAnswer,
    GateDecision,
    GateRequest,
    RunnerEvent,
    dispatch_runner_event,
    workflow_tools,
)

RECURSION_LIMIT = 150
"""Super-steps one orchestrator turn may take before LangGraph calls it a loop.

LangGraph's default is 25, and it is applied **per invocation** to the graph the
bridge streams — the orchestrator's. Measured on this graph: a turn costs about
three super-steps of fixed overhead plus two per model↔tool round, and
`CopilotKitMiddleware` adds an `after_model` node to each round that a CLI graph
does not have. So 25 buys roughly ten rounds, and an orchestrator that starts a
workflow and then reads a few artifacts to summarise can spend them — which is
what ended run 003 in `RUN_ERROR` *after* the workflow had finished and written
everything (`docs/runs/eda-to-report-003.md`).

150 is ~70 rounds: six times the longest turn observed, and still low enough that
a model genuinely looping stops instead of running forever.

Persona graphs are not affected. They are compiled with `checkpointer=False`
(`build_persona_agent`), which detaches them from the caller, so a step making
twenty tool calls spends its own budget rather than the orchestrator's — verified,
and the reason `analyze` survived run 003 while the orchestrator did not.
"""

GATE_REASON = "dsagent.gate"
"""`reason` on the interrupt payload. `ag_ui_langgraph.interrupts` copies it onto
the AG-UI interrupt, and the frontend's `useInterrupt({enabled: ...})` filters on
it — see `docs/ui-slice.md` §3 and §4."""

GATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["decision"],
    "properties": {
        "decision": {"enum": [GateDecision.APPROVE.value, GateDecision.REJECT.value]},
        "note": {"type": "string"},
    },
}
"""What a valid answer looks like. The bridge copies `response_schema` onto the
AG-UI interrupt, so the gate card can render itself from the payload."""


def gate_payload(request: GateRequest) -> dict[str, Any]:
    """The `interrupt()` value for one human gate.

    `reason`, `message` and `response_schema` are the keys
    `lg_interrupt_to_agui` lifts onto the AG-UI interrupt; everything else
    survives under `metadata.langgraph.raw`, which is where the gate card reads
    the step it is about.
    """
    return {
        "reason": GATE_REASON,
        "message": request.prompt,
        "response_schema": GATE_RESPONSE_SCHEMA,
        "run_id": request.run_id,
        "workflow": request.workflow,
        "step": request.step,
        "persona": request.persona,
        "produces": list(request.produces),
    }


def decision_of(answer: Any) -> GateDecision:
    """Read a resumed answer back as a decision.

    A single resolved resume arrives as `Command(resume=payload)`, so this gets
    exactly the object the gate card sent — `{"decision": "approve"}` by the
    schema above. A bare string is accepted too, because that is what a client
    resuming by hand is likely to send. Anything else is a rejection: refusing to
    guess is the safe direction for a gate.
    """
    if isinstance(answer, GateDecision):
        return answer
    value = answer.get("decision") if isinstance(answer, dict) else answer
    if isinstance(value, str) and value.strip().lower() == GateDecision.APPROVE.value:
        return GateDecision.APPROVE
    return GateDecision.REJECT


def interrupt_gate(request: GateRequest) -> GateAnswer:
    """`ask_human` for serve mode: stop the graph and wait for an answer.

    Imported lazily so the module stays importable without a graph in scope.
    The runner calls this for every gate it reaches, decided or not — that is
    what keeps the interrupt sequence stable across the re-execution a resume
    causes; see `docs/ui-slice.md` §2 and `WorkflowRunner._gate`.

    The note comes back with the decision. A rejection whose reason is dropped
    here is a rejection nobody can act on when the run is resumed tomorrow.
    """
    from langgraph.types import interrupt

    answer = interrupt(gate_payload(request))
    note = answer.get("note") if isinstance(answer, dict) else ""
    return GateAnswer(decision=decision_of(answer), note=str(note or ""))


def build_app(
    cartridges: list[Cartridge],
    env: Env | None,
    workspace: Path,
    runs_dir: Path,
    *,
    model: str | None = None,
    path: str = "/agent",
    seed: Path | None = None,
    recursion_limit: int = RECURSION_LIMIT,
    replay: Path | None = None,
    replay_speed: float = 10.0,
    memory_checkpointer: bool = False,
    prices: Any = None,
):
    """The FastAPI app: the orchestrator at `path`, the runs API under `/runs`.

    `replay` swaps the thing that drives runs for a recording (`dsagent.replay`)
    and leaves every other route identical, which is the point: a screen built
    against a replayed run is built against the same API. No model is loaded and
    no agent endpoint is mounted in that mode — there is nothing behind it.
    """
    from fastapi import FastAPI

    app = FastAPI(title="DSAgent")

    if replay is not None:
        from dsagent.replay import Replay

        driver: Any = Replay(replay, runs_dir, speed=replay_speed, prices=prices)
    else:
        from ag_ui_langgraph import add_langgraph_fastapi_endpoint
        from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent

        from dsagent.driver import GraphDriver
        from dsagent.host import build_orchestrator

        graph = build_orchestrator(
            cartridges, env, workspace, model=model,
            workflow_tools=workflow_tools(
                cartridges, runs_dir,
                ask_human=interrupt_gate, on_event=dispatch_runner_event, seed=seed,
                prices=prices,
            ),
            middleware=[CopilotKitMiddleware()],
            checkpointer=checkpointer_for(workspace, memory=memory_checkpointer),
        )
        add_langgraph_fastapi_endpoint(
            app,
            # `thread_id` comes off each `RunAgentInput` and the bridge puts it into
            # `config["configurable"]`, so one browser tab is one resumable thread.
            # The bridge merges this `config` into what it hands `astream_events`.
            LangGraphAGUIAgent(
                name="dsagent", graph=graph, description="DSAgent orchestrator",
                config={"recursion_limit": recursion_limit},
            ),
            path=path,
        )
        driver = GraphDriver(graph, cartridges, runs_dir, recursion_limit=recursion_limit)

    from dsagent.api import add_runs_routes

    app.state.driver = driver
    app.state.runs_dir = runs_dir
    add_runs_routes(app, cartridges, runs_dir, driver)
    _add_files_route(app, runs_dir)
    return app


CHECKPOINTS = "checkpoints.sqlite"
"""Where interrupts wait, under the serve workspace's `.dsagent/`."""


def checkpointer_for(workspace: Path, *, memory: bool = False):
    """Where an interrupted graph waits for its answer.

    SQLite, because a gate can be answered after a restart (§4.3, §7.11): the
    interrupt lives in the checkpoint, and an in-memory saver loses it with the
    process — which is precisely the run most likely to still be waiting when the
    process goes away.

    `memory=True` keeps the old behaviour for a throwaway session, and a missing
    `langgraph-checkpoint-sqlite` falls back to it rather than refusing to serve:
    losing resumability is worse than nothing, but not as bad as no server.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    if memory:
        return InMemorySaver()
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return InMemorySaver()

    path = workspace / ".dsagent" / CHECKPOINTS
    path.parent.mkdir(parents=True, exist_ok=True)
    # `check_same_thread=False` because the runs are driven on their own threads
    # (`dsagent.driver`) while the request that answered the gate returns on
    # another; SQLite's own locking is what keeps those honest.
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection)


def resolve_run_file(runs_dir: Path, run_id: str, rel: str) -> Path | None:
    """The file `rel` names inside a run's workspace, or None if it may not be served.

    Refusals, in order: a run id that is not a single path segment; a path that
    escapes the workspace once resolved (`..`, an absolute path, a symlink out);
    anything under `.dsagent/`, which is the harness's own materialized skills and
    not the run's output; and anything that is not a regular file.
    """
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        return None
    workspace = (runs_dir / run_id / "workspace").resolve()
    try:
        target = (workspace / rel).resolve()
    except OSError:
        return None
    if not target.is_relative_to(workspace) or target == workspace:
        return None
    if ".dsagent" in target.relative_to(workspace).parts:
        return None
    return target if target.is_file() else None


SCRIPTED_PREFIX = "/runs-x"
"""Second route prefix for artifacts the canvas frames with `allow-scripts`.

`docs/ui-product.md` §4.5: a Plotly dashboard is useless without scripts, and an
iframe that has both `allow-scripts` and `allow-same-origin` is not sandboxed at
all. So the canvas frames an interactive artifact with `allow-scripts` *only*,
which gives the frame an opaque origin — and this prefix exists so that choice is
visible in the URL rather than hidden in an attribute, and so the response can
carry a matching `Content-Security-Policy` of its own.

The two prefixes serve the same bytes with the same path rules. What differs is
what the page around them permits.
"""

SCRIPTED_CSP = "sandbox allow-scripts; base-uri 'none'; form-action 'none'"
"""What the served document may do, said by the server as well as by the frame.

A run's artifacts are written by a model. Belt and braces is the right posture:
the header sandboxes the document even if a future canvas forgets the attribute,
and denies it a base URI and anywhere to post a form.
"""


def _add_files_route(app, runs_dir: Path) -> None:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    def serve_file(run_id: str, path: str, headers: dict[str, str] | None = None):
        target = resolve_run_file(runs_dir, run_id, path)
        if target is None:
            raise HTTPException(status_code=404, detail="not found")
        media_type, _ = mimetypes.guess_type(target.name)
        return FileResponse(
            target,
            media_type=media_type or "application/octet-stream",
            headers=headers,
        )

    @app.get("/runs/{run_id}/files/{path:path}")
    def run_file(run_id: str, path: str):
        """Serve one file from a run's workspace, for the canvas to render."""
        return serve_file(run_id, path)

    @app.get(SCRIPTED_PREFIX + "/{run_id}/files/{path:path}")
    def run_file_scripted(run_id: str, path: str):
        """The same file, for a frame that is allowed to run its scripts."""
        return serve_file(run_id, path, headers={"Content-Security-Policy": SCRIPTED_CSP})


__all__ = [
    "GATE_REASON",
    "GATE_RESPONSE_SCHEMA",
    "SCRIPTED_PREFIX",
    "RunnerEvent",
    "build_app",
    "decision_of",
    "gate_payload",
    "interrupt_gate",
    "resolve_run_file",
]
