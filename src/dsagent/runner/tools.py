"""The `run_workflow` / `list_workflows` tools, built once for both front ends.

`dsagent chat` and `dsagent serve` differ in exactly two ways — how a gate is
asked, and where events go — so they pass those in rather than each keeping a
copy of the tool. Keeping one copy is the point: the run-id rule below is easy to
get wrong in a way nothing notices until a resumed run silently re-pays for work
it already did.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool

from dsagent.cartridge.models import Cartridge
from dsagent.runner.runner import GateDecision, GateRequest, RunnerEvent, WorkflowRunner


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


def workflow_tools(
    cartridges: list[Cartridge],
    runs_dir: Path,
    *,
    ask_human: Callable[[GateRequest], GateDecision],
    on_event: Callable[[RunnerEvent], None] | None = None,
    log: Callable[[str], None] = lambda m: None,
) -> list[Any]:
    """`[list_workflows, run_workflow]`, bound to one front end's gate and events."""
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
        run_dir = runs_dir / workflow_run_id(name, tool_call_id)
        state = WorkflowRunner(
            by_wf[name], run_dir, ask_human=ask_human, log=log, on_event=on_event,
        ).run(name, inputs or {}, resume=(run_dir / "run.json").exists())
        # `status` is the work and no longer says "paused", so report the gate
        # decisions alongside it — otherwise a paused run reads as all-done.
        return json.dumps({
            "run_dir": str(run_dir), "status": state.status,
            "steps": {k: v.status for k, v in state.steps.items()},
            "gates": {k: v.gate.decision for k, v in state.steps.items() if v.gate},
        })

    return [list_workflows, run_workflow]
