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


def validate_serve_environments(cartridges: list[Cartridge], *, allow_host_compute: bool) -> None:
    """Refuse host execution and request-time image builds unless opted in."""
    if allow_host_compute:
        return
    kernels = [
        f"{cartridge.name}.{name}"
        for cartridge in cartridges
        for name, spec in cartridge.envs.items()
        if spec.kind == "kernel"
    ]
    builds = [
        f"{cartridge.name}.{name}"
        for cartridge in cartridges
        for name, spec in cartridge.envs.items()
        if spec.kind == "docker" and not spec.image
    ]
    if kernels or builds:
        details = []
        if kernels:
            details.append("host kernels: " + ", ".join(kernels))
        if builds:
            details.append("request-time image builds: " + ", ".join(builds))
        raise ValueError(
            "serve refuses " + "; ".join(details)
            + ". Use prebuilt docker/filesystem providers, or --insecure-local for trusted development only."
        )


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
    capabilities: Any = None,
    allow_host_compute: bool = False,
):
    """The FastAPI app: the orchestrator at `path`, the runs API under `/runs`.

    `replay` swaps the thing that drives runs for a recording (`dsagent.replay`)
    and leaves every other route identical, which is the point: a screen built
    against a replayed run is built against the same API. No model is loaded and
    no agent endpoint is mounted in that mode — there is nothing behind it.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from dsagent.capabilities import compose_cartridges

    if replay is None:
        validate_serve_environments(cartridges, allow_host_compute=allow_host_compute)

    source = capabilities if capabilities is not None else (env.capabilities if env else None)
    capabilities = compose_cartridges(cartridges, source)
    if env is not None:
        env.capabilities = capabilities

    @asynccontextmanager
    async def lifespan(app: Any):
        if hasattr(app.state.driver, "recover"):
            app.state.driver.recover()
        yield
        # Each non-default project's chat holds a live env (a kernel/container);
        # release them on shutdown. The default env is the caller's to close.
        for pid, chat_env in getattr(app.state, "chat_envs", {}).items():
            if pid == "default":
                continue
            try:
                chat_env.close()
            except Exception:  # noqa: BLE001, S110 — shutdown is best-effort
                pass

    app = FastAPI(title="DSAgent", lifespan=lifespan)

    from dsagent.projects import DEFAULT_ID, ProjectStore
    from dsagent.sessions import SessionStore

    # Projects sit beside the runs dir; `default` adopts today's runs in place,
    # so an existing deployment keeps working and only new projects namespace.
    projects_root = runs_dir.parent / "projects"
    store = ProjectStore(projects_root, runs_dir)
    store.ensure_default()
    # Chats live on the server (durable, shared, project-mapped): a run the
    # orchestrator starts in a chat lands in that chat's project.
    sessions = SessionStore(projects_root / "sessions")

    # A chat's files (a persona's report, its charts) are written to the project's
    # chat workspace. `default` uses the legacy serve workspace in place; every
    # other project namespaces under its own dir. Defined here so both the chat
    # orchestrator and the artifacts route below agree on one location.
    def project_workspace(project_id: str) -> Path:
        if project_id == DEFAULT_ID:
            return workspace
        return store.dir(project_id) / "workspace"

    if replay is not None:
        from dsagent.replay import Replay

        driver: Any = Replay(replay, runs_dir, speed=replay_speed, prices=prices)
    else:
        from ag_ui.core.types import RunAgentInput
        from ag_ui.encoder import EventEncoder
        from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent
        from fastapi import Request
        from fastapi.responses import StreamingResponse

        from dsagent.datasets import seed_run
        from dsagent.envs.base import make_env
        from dsagent.execution import ExecutionService
        from dsagent.host import build_orchestrator

        driver = ExecutionService(cartridges, runs_dir, store=store,
                                  capabilities=capabilities, seed=seed, prices=prices)

        def orchestrator(
            chat_env: Env, ws: Path, checkpointer: Any, carts: list[Cartridge] = cartridges
        ):
            tools = workflow_tools(
                carts, runs_dir,
                ask_human=interrupt_gate, on_event=dispatch_runner_event, seed=seed,
                prices=prices, store=store, session_project=sessions.project_of,
                capabilities=chat_env.capabilities or capabilities,
                execution=driver,
            )
            return build_orchestrator(
                carts, chat_env, ws, model=model, workflow_tools=tools,
                middleware=[CopilotKitMiddleware()], checkpointer=checkpointer,
            )

        # A project bound to one cartridge gets that cartridge's team and
        # workflows in its chat — the roster the orchestrator introduces and the
        # workflows `list_workflows` offers are the project's domain, not the
        # union of everything the server loaded. Unbound projects see it all.
        def cartridges_for(project_id: str) -> list[Cartridge]:
            proj = store.get(project_id)
            bound = proj.cartridge if proj is not None else ""
            return [c for c in cartridges if c.name == bound] or cartridges

        # One chat orchestrator per project, so a chat's filesystem — its
        # datasets and the files a persona writes — is the project's, not a
        # global scratch dir every project shares (that leak is what made the
        # orchestrator list one project's datasets in another). Invariant 1
        # holds: a "project workspace" is a generic container. `default` keeps
        # the legacy serve workspace in place (mirrors ProjectStore adopting the
        # legacy runs dir); every other project gets its own workspace and env,
        # built on first use and cached.
        #
        # Single source of truth for a chat's transcript: the SessionStore.
        #
        # The chat checkpoint is deliberately in-memory (ephemeral). It is not a
        # second durable copy of the conversation — it is per-process execution
        # state for the live turn (what an `interrupt()`/`ask_human` resumes
        # against). On restart it is gone and rebuilt from the client's next
        # `RunAgentInput.messages`, which the client loaded from the SessionStore;
        # message ids are stable across that round-trip (SessionStore →
        # fromAgUiMessages → runtime → toAgUiMessages → SessionStore), so
        # LangGraph's `add_messages` dedups against them and the checkpoint always
        # converges to the stored transcript. It can never durably diverge from it.
        #
        # Workflow jobs and gates belong to ExecutionService, including jobs
        # submitted from chat. Its journal survives a restart independently of
        # this ephemeral graph. A lost chat turn does not lose its submitted job.
        chat_agents: dict[str, Any] = {}
        chat_envs: dict[str, Env] = {}

        def chat_agent_for(project_id: str) -> Any:
            if project_id not in chat_agents:
                ws = project_workspace(project_id)
                ws.mkdir(parents=True, exist_ok=True)
                carts = cartridges_for(project_id)
                env_spec = carts[0].envs["default"]
                chat_capabilities = compose_cartridges(carts, capabilities)
                chat_env = (
                    env if project_id == DEFAULT_ID
                    else make_env(env_spec, ws, chat_capabilities)
                )
                chat_envs[project_id] = chat_env
                chat_agents[project_id] = LangGraphAGUIAgent(
                    name="dsagent",
                    graph=orchestrator(chat_env, ws, checkpointer_for(ws, memory=True), carts),
                    description="DSAgent orchestrator",
                    config={"recursion_limit": recursion_limit},
                )
            # Keep the chat's `data/` current with the project's datasets, so a
            # dataset uploaded mid-session shows up on the next turn.
            seed_run(store.data_dir(project_id), project_workspace(project_id))
            return chat_agents[project_id]

        # FastAPI resolves a route's annotations against the *module* globals,
        # and `from __future__ import annotations` makes them strings — so these
        # function-local imports must be published there, or FastAPI can't see
        # `RunAgentInput` as the body and treats both params as query fields.
        globals()["RunAgentInput"] = RunAgentInput
        globals()["Request"] = Request

        # The body of `ag_ui_langgraph.add_langgraph_fastapi_endpoint`, with the
        # single fixed agent replaced by a per-project lookup keyed on the
        # thread id: the bridge can serve only one graph per path, and we need
        # one per project. `thread_id` comes off each `RunAgentInput`; an
        # unregistered thread maps to `default`.
        @app.post(path)
        async def agent_endpoint(input_data: RunAgentInput, request: Request):
            project_id = sessions.project_of(input_data.thread_id or "")
            request_agent = chat_agent_for(project_id).clone()
            encoder = EventEncoder(accept=request.headers.get("accept"))

            async def event_generator():
                async for event in request_agent.run(input_data):
                    yield encoder.encode(event)

            return StreamingResponse(event_generator(), media_type=encoder.get_content_type())

        @app.get(f"{path.rstrip('/')}/health")
        def agent_health() -> dict[str, Any]:
            return {"status": "ok", "agent": {"name": "dsagent"}}

        # The lifespan shutdown closes these (the per-project kernels/containers).
        app.state.chat_envs = chat_envs

    from dsagent.api import add_runs_routes
    from dsagent.api_projects import add_projects_routes

    app.state.driver = driver
    app.state.runs_dir = runs_dir
    app.state.projects = store
    app.state.sessions = sessions
    add_runs_routes(app, cartridges, runs_dir, driver, store)
    add_projects_routes(app, store, sessions)
    _add_files_route(app, runs_dir, store)
    _add_project_files_route(app, project_workspace, store)
    _mount_ui(app)
    return app


UI_DIST_ENV = "DSAGENT_UI_DIST"
"""Override for where the built SPA lives; defaults to `<repo>/ui/dist`."""


def _ui_dist() -> Path | None:
    """The built SPA directory, or None when the UI has not been built.

    A bare checkout, or an API-only deployment, simply has no `ui/dist`; the
    server then runs exactly as before with nothing mounted at `/`.
    """
    import os

    override = os.environ.get(UI_DIST_ENV)
    candidate = (
        Path(override)
        if override
        # <repo>/src/dsagent/serve.py -> <repo>/ui/dist
        else Path(__file__).resolve().parents[2] / "ui" / "dist"
    )
    return candidate if (candidate / "index.html").is_file() else None


def _mount_ui(app) -> None:
    """Serve the built SPA at `/`, without shadowing any API route.

    Registered last: `/agent`, `/runs…`, `/cartridges` and the files endpoint
    are all added before this, and Starlette matches routes in registration
    order, so a mount at `/` only ever catches what no API route claimed — the
    SPA's `index.html` and its assets. The UI keeps its own state in memory (no
    client-side URL routes), so there is nothing here to collide with `/runs/{id}`.
    """
    dist = _ui_dist()
    if dist is None:
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")


CHECKPOINTS = "checkpoints.sqlite"
"""Where interrupts wait, under the serve workspace's `.dsagent/`."""


def checkpointer_for(workspace: Path, *, memory: bool = False):
    """Where an interrupted graph waits for its answer.

    SQLite, because a gate can be answered after a restart (§4.3, §7.11): the
    interrupt lives in the checkpoint, and an in-memory saver loses it with the
    process — which is precisely the run most likely to still be waiting when the
    process goes away.

LangGraph's SQLite savers each implement one half of the protocol: `SqliteSaver`
    raises `NotImplementedError: does not support async methods` under
    `astream_events`, and `AsyncSqliteSaver` is the mirror image under a plain
    `.invoke()` — and cannot be constructed outside a running event loop at all.
    `InMemorySaver` implements both, which is why the mismatch stayed hidden until
    a real chat message reached a real server. `build_app` therefore gives the
    sync driver the file and the async bridge memory; see the note there.

    `memory=True` keeps the old behaviour for a throwaway session, and a missing
    `langgraph-checkpoint-sqlite` falls back to it rather than refusing to serve:
    losing resumability is worse than nothing, but not as bad as no server.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    if memory:
        return InMemorySaver()

    path = workspace / ".dsagent" / CHECKPOINTS
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # `check_same_thread=False` because the runs are driven on their own
        # threads (`dsagent.driver`) while the request that answered the gate
        # returns on another; SQLite's own locking is what keeps those honest.
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    except ImportError:
        return InMemorySaver()


def resolve_run_file(runs_dir: Path, run_id: str, rel: str, store: Any = None) -> Path | None:
    """The file `rel` names inside a run's workspace, or None if it may not be served.

    Refusals, in order: a run id that is not a single path segment; a path that
    escapes the workspace once resolved (`..`, an absolute path, a symlink out);
    anything under `.dsagent/`, which is the harness's own materialized skills and
    not the run's output; and anything that is not a regular file.

    With a `store`, the run is located across projects; without one it is the
    single-dir behaviour it always had.
    """
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        return None
    # Locate across projects when a store is wired; fall back to the default dir
    # so a run whose `run.json` is not (yet) written still serves its files.
    base = None
    if store is not None:
        base = store.locate_run(run_id)
    if base is None:
        base = runs_dir / run_id
    workspace = (base / "workspace").resolve()
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


def _add_files_route(app, runs_dir: Path, store: Any = None) -> None:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    def serve_file(run_id: str, path: str, headers: dict[str, str] | None = None):
        target = resolve_run_file(runs_dir, run_id, path, store)
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


ARTIFACT_SKIP_DIRS = frozenset({".dsagent", "data", "work", "model", "large_tool_results"})
"""Top-level workspace dirs that are seeds or harness scratch, not a chat's output.
`data/` is the project's seeded datasets; the rest is materialized skills and
tooling. What is left — `report/`, `artifacts/`, charts, top-level files — is what
a persona actually produced for the user to see."""


def resolve_project_file(project_workspace: Path, rel: str) -> Path | None:
    """A file inside a project's chat workspace, or None if it may not be served.

    Same rules as `resolve_run_file`: the path may not escape the workspace, may
    not reach into `.dsagent/`, and must be a regular file.
    """
    ws = project_workspace.resolve()
    try:
        target = (ws / rel).resolve()
    except OSError:
        return None
    if not target.is_relative_to(ws) or target == ws:
        return None
    if ".dsagent" in target.relative_to(ws).parts:
        return None
    return target if target.is_file() else None


def list_project_artifacts(project_workspace: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    """The files a chat wrote under the project workspace — reports and charts —
    skipping seeds and harness scratch. Relative paths, newest first."""
    ws = project_workspace
    if not ws.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in ws.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
        if rel.parts and rel.parts[0] in ARTIFACT_SKIP_DIRS:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"path": str(rel), "bytes": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda a: a["mtime"], reverse=True)
    return out[:limit]


def _add_project_files_route(app, project_workspace, store: Any = None) -> None:
    """Serve, list and preview a project's chat-workspace artifacts, so files a
    persona writes in a chat (a report, its charts, a table) are reachable — the
    runs files route only ever covered files produced by a *run*.

    `/preview` mirrors the runs one deliberately: a chat's artifact is read in
    the same viewer as a run's, so a CSV a persona wrote in a conversation draws
    as a table instead of downloading as raw text.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from dsagent.api import PREVIEW_MAX_ROWS, PREVIEW_ROWS
    from dsagent.tabular import READERS, TableUnreadable, read_table

    def _project_or_404(project_id: str) -> Path:
        if store is not None and store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        return project_workspace(project_id)

    @app.get("/projects/{project_id}/artifacts")
    def project_artifacts(project_id: str):
        return {"artifacts": list_project_artifacts(project_workspace(project_id))}

    @app.get("/projects/{project_id}/files/{path:path}")
    def project_file(project_id: str, path: str):
        target = resolve_project_file(project_workspace(project_id), path)
        if target is None:
            raise HTTPException(status_code=404, detail="not found")
        media_type, _ = mimetypes.guess_type(target.name)
        return FileResponse(target, media_type=media_type or "application/octet-stream")

    @app.get("/projects/{project_id}/preview/{path:path}")
    def project_preview(project_id: str, path: str, rows: int = PREVIEW_ROWS) -> dict[str, Any]:
        """The first rows of a tabular artifact, as JSON. 415 for anything no
        reader here understands — the client can still open the file."""
        target = resolve_project_file(_project_or_404(project_id), path)
        if target is None:
            raise HTTPException(status_code=404, detail="not found")
        if target.suffix.lower() not in READERS:
            raise HTTPException(
                status_code=415,
                detail=f"no preview for {target.suffix or 'a file with no extension'}",
            )
        try:
            columns, table, total = read_table(target, max(1, min(rows, PREVIEW_MAX_ROWS)))
        except TableUnreadable as e:
            raise HTTPException(status_code=415, detail=str(e)) from e
        return {
            "path": path,
            "project": project_id,
            "columns": columns,
            "rows": table,
            "total_rows": total,
            "truncated": total > len(table),
        }


__all__ = [
    "GATE_REASON",
    "GATE_RESPONSE_SCHEMA",
    "SCRIPTED_PREFIX",
    "RunnerEvent",
    "build_app",
    "decision_of",
    "gate_payload",
    "interrupt_gate",
    "list_project_artifacts",
    "resolve_project_file",
    "resolve_run_file",
]
