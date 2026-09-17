"""`dsagent serve` — the orchestrator over AG-UI, plus the run workspace over HTTP.

Two surfaces on one FastAPI app:

* **`POST /agent`** (and `GET /agent/health`), registered by
  `ag_ui_langgraph.add_langgraph_fastapi_endpoint`. It streams AG-UI events over
  SSE: the orchestrator's own messages and tool calls, the runner's
  `dsagent.step` / `dsagent.tool` / `dsagent.file` as `CUSTOM` events, and a
  human gate as an interrupt.
* **`GET /runs/{run_id}/files/{path}`**, which serves a run's workspace so the
  canvas can render the artifacts those file events announce.

Everything domain-specific still comes from cartridges; this module only knows
about workflows in the abstract. See `docs/ui-slice.md`.

The `[ui]` extra is required: `pip install -e ".[ui,anthropic]"`.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from dsagent.cartridge.models import Cartridge
from dsagent.envs.base import Env
from dsagent.runner import (
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


def interrupt_gate(request: GateRequest) -> GateDecision:
    """`ask_human` for serve mode: stop the graph and wait for the browser.

    Imported lazily so the module stays importable without a graph in scope.
    The runner calls this for every gate it reaches, decided or not — that is
    what keeps the interrupt sequence stable across the re-execution a resume
    causes; see `docs/ui-slice.md` §2 and `WorkflowRunner._gate`.
    """
    from langgraph.types import interrupt

    return decision_of(interrupt(gate_payload(request)))


def build_app(
    cartridges: list[Cartridge],
    env: Env,
    workspace: Path,
    runs_dir: Path,
    *,
    model: str | None = None,
    path: str = "/agent",
    seed: Path | None = None,
    recursion_limit: int = RECURSION_LIMIT,
):
    """The FastAPI app: the orchestrator at `path`, the run files under `/runs`."""
    from ag_ui_langgraph import add_langgraph_fastapi_endpoint
    from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent
    from fastapi import FastAPI
    from langgraph.checkpoint.memory import InMemorySaver

    from dsagent.host import build_orchestrator

    graph = build_orchestrator(
        cartridges, env, workspace, model=model,
        workflow_tools=workflow_tools(
            cartridges, runs_dir,
            ask_human=interrupt_gate, on_event=dispatch_runner_event, seed=seed,
        ),
        middleware=[CopilotKitMiddleware()],
        # Interrupts need a checkpointer to resume from. In-memory for the slice:
        # one `dsagent serve` process, and the thread only has to outlive the gate
        # answer — `run.json` is what survives a restart. SQLite is the follow-up.
        checkpointer=InMemorySaver(),
    )
    app = FastAPI(title="DSAgent")
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
    _add_files_route(app, runs_dir)
    return app


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


def _add_files_route(app, runs_dir: Path) -> None:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    @app.get("/runs/{run_id}/files/{path:path}")
    def run_file(run_id: str, path: str):
        """Serve one file from a run's workspace, for the canvas to render."""
        target = resolve_run_file(runs_dir, run_id, path)
        if target is None:
            raise HTTPException(status_code=404, detail="not found")
        media_type, _ = mimetypes.guess_type(target.name)
        return FileResponse(target, media_type=media_type or "application/octet-stream")


__all__ = [
    "GATE_REASON",
    "GATE_RESPONSE_SCHEMA",
    "RunnerEvent",
    "build_app",
    "decision_of",
    "gate_payload",
    "interrupt_gate",
    "resolve_run_file",
]
