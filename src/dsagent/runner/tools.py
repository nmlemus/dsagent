"""The `run_workflow` / `list_workflows` tools, built once for both front ends.

`dsagent chat` and `dsagent serve` differ in exactly two ways — how a gate is
asked, and where events go — so they pass those in rather than each keeping a
copy of the tool. Keeping one copy is the point: the run-id rule below is easy to
get wrong in a way nothing notices until a resumed run silently re-pays for work
it already did.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import InjectedToolCallId, tool

from dsagent.capabilities import CapabilitySet
from dsagent.capabilities.builtin import requirements_for_cartridges
from dsagent.cartridge.models import Cartridge
from dsagent.datasets import seed_run
from dsagent.runner.charts import amend_tools
from dsagent.runner.runner import GateDecision, GateRequest, RunnerEvent, WorkflowRunner
from dsagent.sessions import mark_session

RUN_ID_KEY = "dsagent_run_id"
"""`configurable` key naming the run directory one invocation must use.

The launcher creates a run directory *before* the orchestrator is asked to run
anything — that is where the uploaded dataset lands and what the browser has a
URL for — so the id cannot be minted inside the tool that run. It travels in the
graph config, which the server controls and replays unchanged on a resume, rather
than through the model, which would make the name of a directory a thing a model
could get wrong.

Read with `ensure_config()`, **not** with a `config: RunnableConfig` parameter.
This module has `from __future__ import annotations`, so that parameter's
annotation is the *string* `"RunnableConfig | None"`, LangChain does not
recognise it as the config injection, and the tool is handed `None` — silently,
with the run then landing in a directory named after the tool call and the
launcher's uploaded dataset nowhere near it. Probed both ways on
langchain-core 1.6.3.
"""


def workflow_run_id(workflow: str, tool_call_id: str = "") -> str:
    """The run directory name for one `run_workflow` invocation.

    Derived from the tool call, never from the clock. Under `dsagent serve` the
    gate is a LangGraph `interrupt()`, and resuming re-executes `run_workflow`
    from the top: a timestamped id would mint a fresh, empty run directory on
    every re-entry, so `resume` would find no `run.json`, every step would read
    as `pending`, and the run would redo — and re-pay for — work it had already
    done. `tool_call_id` is stable across the original call and the re-entry,
    because it belongs to the `AIMessage` the checkpoint replays (verified
    against a `create_deep_agent` graph).

    The timestamp remains the fallback for a caller that reaches this without a
    tool call — a direct programmatic call, where there is nothing to re-enter.
    `docs/ui-slice.md` §2 names `thread_id` plus a counter in graph state as the
    other option; it is strictly more machinery for the same guarantee, so it
    stays unbuilt until something needs it.
    """
    suffix = tool_call_id or time.strftime("%Y%m%d-%H%M%S")
    return f"{workflow}-{suffix}"


def resolve_run_dir(
    given: str | None,
    thread_id: str | None,
    name: str,
    tool_call_id: str,
    runs_dir: Path,
    store: Any = None,
    session_project: Callable[[str], str] | None = None,
) -> Path:
    """Where a workflow run's directory goes.

    Launcher path (`given` = `RUN_ID_KEY`): the run was pre-created in its own
    project, so locate it by id rather than recompute a path. Chat path (no
    `given`): the orchestrator started it, so put it in the chat session's
    project — resolved from the AG-UI `thread_id` via the session registry.
    """
    if given:
        located = store.locate_run(given) if store is not None else None
        return located or (runs_dir / given)
    project = session_project(thread_id) if (session_project and thread_id) else "default"
    base = store.runs_dir(project) if store is not None else runs_dir
    return base / workflow_run_id(name, tool_call_id)


def workflow_tools(
    cartridges: list[Cartridge],
    runs_dir: Path,
    *,
    ask_human: Callable[[GateRequest], GateDecision],
    on_event: Callable[[RunnerEvent], None] | None = None,
    log: Callable[[str], None] = lambda m: None,
    seed: Path | None = None,
    prices: Any = None,
    store: Any = None,
    session_project: Callable[[str], str] | None = None,
    capabilities: CapabilitySet | None = None,
) -> list[Any]:
    """The workflow and run-reading tools, bound to one front end's gate and events.

    `list_workflows` and `run_workflow` start work; `list_run_files` and
    `read_run_file` are how the orchestrator answers a question about a run that
    has already happened ("which finding should I be most careful with?"). Its own
    file tools are rooted in the chat workspace, and a run lives in its own
    directory, so without these it can only repeat what the tool result said.

    `seed` is a directory copied into a *new* run's workspace before the first
    step. It exists because a run directory is named after the tool call, so
    nothing can put a dataset in it beforehand — the CLI can seed by hand
    (`--run-id`), a browser cannot. The copy is structural: the harness knows
    nothing about what is in there, so `<seed>/data/x.csv` lands at
    `<workspace>/data/x.csv` and the workflow's `data_path` reads `data/x.csv`.
    A resumed run is never re-seeded; its workspace is already the run's.
    """
    by_wf = {w: c for c in cartridges for w in c.workflows}

    @tool
    def list_workflows() -> str:
        """List the workflows this team can run, with their declared inputs."""
        return "\n".join(
            f"- {name}: {c.workflows[name].description} "
            f"(inputs: {', '.join(c.workflows[name].inputs) or 'none'})"
            for name, c in by_wf.items()
        )

    @tool
    def run_workflow(
        name: str,
        inputs: dict | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Run a cartridge workflow end to end. `inputs` is a dict matching the workflow's declared inputs."""
        if name not in by_wf:
            return f"unknown workflow {name}; use list_workflows"
        configurable = ensure_config().get("configurable") or {}
        given = configurable.get(RUN_ID_KEY)
        thread_id = configurable.get("thread_id")
        run_dir = resolve_run_dir(
            given, thread_id, name, tool_call_id, runs_dir, store, session_project
        )
        resume = (run_dir / "run.json").exists()
        run_capabilities = (
            capabilities.restrict(requirements_for_cartridges([by_wf[name]]))
            if capabilities is not None else None
        )
        runner = WorkflowRunner(
            by_wf[name], run_dir, ask_human=ask_human, log=log, on_event=on_event,
            prices=prices, capabilities=run_capabilities,
        )
        if seed is not None and not resume:
            if not seed.is_dir():
                return f"seed directory {seed} does not exist"
            shutil.copytree(seed, runner.workspace, dirs_exist_ok=True)
            log(f"[seed] copied {seed} into {runner.workspace}")
        # Seed the run with the project's datasets so `data_path` resolves — the
        # same reuse the launcher gets, for a run the orchestrator starts.
        if not resume and store is not None:
            names = seed_run(store.data_dir(store.project_of(run_dir)), runner.workspace)
            if names:
                log(f"[seed] project data {names} into {runner.workspace}")
        state = runner.run(name, inputs or {}, resume=resume)
        # Link the run to the chat that started it: the AG-UI thread id is the
        # session id. Best-effort — a run started outside a chat has no thread.
        if thread_id:
            mark_session(run_dir, str(thread_id))
        # `status` is the work and no longer says "paused", so report the gate
        # decisions alongside it — otherwise a paused run reads as all-done.
        return json.dumps({
            "run_dir": str(run_dir), "status": state.status,
            "steps": {k: v.status for k, v in state.steps.items()},
            "gates": {k: v.gate.decision for k, v in state.steps.items() if v.gate},
        })

    @tool
    def list_run_files(run_id: str) -> str:
        """List the files a run produced, deliverables first. `run_id` names a run directory."""
        workspace = _run_workspace(runs_dir, run_id, store)
        if workspace is None:
            return f"unknown run {run_id}"
        declared = set(_deliverables(workspace.parent))
        files = sorted(
            f.relative_to(workspace).as_posix()
            for f in workspace.rglob("*")
            if f.is_file() and ".dsagent" not in f.relative_to(workspace).parts
        )
        if not files:
            return f"run {run_id} has written nothing yet"
        return "\n".join(
            f"- {p}" + (" (deliverable)" if p in declared else "") for p in files
        )

    @tool
    def read_run_file(run_id: str, path: str, max_chars: int = 20000) -> str:
        """Read a text file from a run's workspace. Use `list_run_files` first to see what there is."""
        target = _resolve_in_workspace(runs_dir, run_id, path, store)
        if target is None:
            return f"no such file in run {run_id}: {path}"
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return f"{path} is not readable as text ({target.stat().st_size} bytes)"
        if len(text) > max_chars:
            return text[:max_chars] + f"\n… truncated at {max_chars} of {len(text)} chars"
        return text

    return [list_workflows, run_workflow, list_run_files, read_run_file,
            *amend_tools(runs_dir, store=store)]


def _deliverables(run_dir: Path) -> list[str]:
    from dsagent.runs import deliverables

    return deliverables(run_dir)


def _run_workspace(runs_dir: Path, run_id: str, store: Any = None) -> Path | None:
    """One run's workspace, if `run_id` names a run and nothing else.

    A run id arrives from outside — from a URL at the files endpoint, from a
    model here — and one that is a path rather than a name walks out of the runs
    directory entirely. `locate_run_dir` is where that rule lives now, shared
    with the HTTP routes so a run in a project reads the same from both.
    """
    from dsagent.runs import locate_run_dir

    run_dir = locate_run_dir(runs_dir, run_id, store)
    if run_dir is None:
        return None
    workspace = (run_dir / "workspace").resolve()
    return workspace if workspace.is_dir() else None


def _resolve_in_workspace(
    runs_dir: Path, run_id: str, rel: str, store: Any = None
) -> Path | None:
    """A run-workspace path that is really inside that workspace, or None.

    The same rule the files endpoint applies, for the same reason: a run id and a
    path both arrive from outside — there, from a URL; here, from a model — and
    neither may be allowed to walk out of the run it names.
    """
    workspace = _run_workspace(runs_dir, run_id, store)
    if workspace is None:
        return None
    try:
        target = (workspace / rel).resolve()
    except OSError:
        return None
    if not target.is_relative_to(workspace) or ".dsagent" in target.relative_to(workspace).parts:
        return None
    return target if target.is_file() else None
