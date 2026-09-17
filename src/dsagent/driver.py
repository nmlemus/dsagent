"""Who owns a run while it is happening.

A run started from the launcher is driven by the **server**, on its own thread,
not by whichever browser tab happened to press Start. That is what makes a reload
harmless, a second tab possible, and a run outlive the page that started it —
`docs/ui-product.md` §2.3, §4.2, §7.6. The browser follows the run's
`events.jsonl` over SSE and answers its gates over HTTP; it is a viewer.

It still goes through the orchestrator, because that is the path the chat takes
and because the run then exists in a thread the orchestrator can be asked about
afterwards (§7.9). The only thing the server decides for it is *which directory*
the run writes to — `RUN_ID_KEY` in the graph config — since the launcher created
that directory, put the uploaded dataset in it, and handed its id to the browser
before any model saw the request.

`Replay` (`dsagent.replay`) implements the same three methods with no model at
all, which is why every screen can be built without one.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from dsagent.cartridge.models import Cartridge, Workflow
from dsagent.runner.runner import GateDecision, RunState, StepRecord
from dsagent.runner.tools import RUN_ID_KEY

THREAD_PREFIX = "run:"
"""Graph thread for a run. The chat on the run screen uses a thread of its own:
a run and a question asked while it works would otherwise be two concurrent
writers on one checkpoint."""


class Driver(Protocol):
    """What the runs API needs from whatever is actually running the run."""

    def create(self, run_id: str, inputs: dict[str, Any]) -> Path: ...

    def start(self, run_id: str, *, resume: bool = False) -> None: ...

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "") -> bool: ...

    def is_running(self, run_id: str) -> bool: ...


def pending_state(workflow: Workflow, cartridge: Cartridge, inputs: dict[str, Any]) -> RunState:
    """A run that exists but has not started.

    Written before the orchestrator is asked for anything, so the run is in the
    list the moment it is created and the uploaded file has somewhere to land.
    The runner picks it up as a resume — every step `pending`, so nothing is
    skipped — and reads its inputs back out of it rather than from the model.
    """
    return RunState(
        workflow=workflow.name,
        cartridge=cartridge.name,
        inputs=inputs,
        status="pending",
        steps={s.id: StepRecord(id=s.id) for s in workflow.ordered_steps()},
    )


def start_message(workflow: Workflow, run_id: str, inputs: dict[str, Any]) -> str:
    """What the server says to the orchestrator to start a launcher run.

    Deliberately not a conversation: the operator already chose the workflow and
    filled the form, so the only decision left is which tool to call. The inputs
    are repeated here because the model has to pass them, and they are recorded
    in the run directory as well because the runner reads them from there — the
    message is the request, the directory is the contract.
    """
    return (
        f"Start the `{workflow.name}` workflow now by calling `run_workflow` with "
        f"name=\"{workflow.name}\" and inputs={json.dumps(inputs)}.\n\n"
        f"This run is `{run_id}`; its directory and its inputs are already prepared. "
        f"Call the tool once, do not ask any questions first, and when it returns "
        f"summarise what the run produced in two or three sentences."
    )


def resume_message(workflow: Workflow, run_id: str, inputs: dict[str, Any]) -> str:
    """What the server says to continue a run that stopped at a rejected gate.

    The same tool call: `run_workflow` reopens the run directory, skips the steps
    already done and stops at the gate again. Saying so explicitly is what keeps
    the model from deciding that a run it has already "finished" needs a fresh
    one — which would be a second directory, a second dataset copy and a second
    bill.
    """
    return (
        f"Continue run `{run_id}`. Call `run_workflow` again with "
        f"name=\"{workflow.name}\" and inputs={json.dumps(inputs)} — it reopens the same "
        f"run, skips the steps that are already done, and stops at the gate that is "
        f"waiting. Do not start a new run, and do not ask questions first."
    )


class GraphDriver:
    """Runs the orchestrator graph in a background thread, one thread per run."""

    def __init__(
        self,
        graph: Any,
        cartridges: list[Cartridge],
        runs_dir: Path,
        *,
        recursion_limit: int,
        log: Callable[[str], None] = lambda m: None,
    ) -> None:
        self.graph = graph
        self.runs_dir = runs_dir
        self.recursion_limit = recursion_limit
        self.log = log
        self._by_workflow = {w: c for c in cartridges for w in c.workflows}
        self._threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}

    # ---- the Driver surface -------------------------------------------------

    def create(self, run_id: str, inputs: dict[str, Any], workflow: str = "") -> Path:
        cartridge = self._by_workflow[workflow]
        run_dir = self.runs_dir / run_id
        (run_dir / "workspace").mkdir(parents=True, exist_ok=True)
        pending_state(cartridge.workflows[workflow], cartridge, inputs).save(run_dir)
        return run_dir

    def start(self, run_id: str, *, resume: bool = False) -> None:
        if self.is_running(run_id):
            return
        state = RunState.load(self.runs_dir / run_id)
        cartridge = self._by_workflow[state.workflow]
        say = resume_message if resume else start_message
        message = say(cartridge.workflows[state.workflow], run_id, state.inputs)
        self._spawn(run_id, {"messages": [{"role": "user", "content": message}]})

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "") -> bool:
        """Resume the run's graph with the answer. False if it is not at a gate.

        The interrupt lives in the checkpoint rather than in this process's
        memory, which is the whole reason the checkpointer exists: a run that was
        waiting when the server was killed is still waiting when it comes back,
        and this is the call that answers it (§7.11).
        """
        if self.is_running(run_id):
            return False
        if not self._interrupted(run_id):
            return False
        from langgraph.types import Command

        self._spawn(run_id, Command(resume={"decision": decision.value, "note": note}))
        return True

    def is_running(self, run_id: str) -> bool:
        thread = self._threads.get(run_id)
        return bool(thread and thread.is_alive())

    def error_for(self, run_id: str) -> str | None:
        return self._errors.get(run_id)

    # ---- internals ----------------------------------------------------------

    def config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": f"{THREAD_PREFIX}{run_id}", RUN_ID_KEY: run_id},
            "recursion_limit": self.recursion_limit,
        }

    def _interrupted(self, run_id: str) -> bool:
        """Whether the run's thread is parked on an interrupt."""
        try:
            snapshot = self.graph.get_state(self.config(run_id))
        except Exception as e:  # noqa: BLE001 — an unknown thread is simply not waiting
            self.log(f"[driver] no state for {run_id}: {e}")
            return False
        if getattr(snapshot, "interrupts", None):
            return True
        return any(getattr(t, "interrupts", None) for t in getattr(snapshot, "tasks", ()) or ())

    def _spawn(self, run_id: str, payload: Any) -> None:
        thread = threading.Thread(
            target=self._invoke, args=(run_id, payload), daemon=True, name=f"run-{run_id}"
        )
        self._threads[run_id] = thread
        self._errors.pop(run_id, None)
        thread.start()

    def _invoke(self, run_id: str, payload: Any) -> None:
        try:
            self.graph.invoke(payload, config=self.config(run_id))
        except Exception as e:  # noqa: BLE001 — recorded on the run, not raised into a thread
            self._errors[run_id] = f"{type(e).__name__}: {e}"
            self.log(f"[driver] run {run_id} failed: {e}")
            self._mark_failed(run_id, str(e))
        finally:
            self._threads.pop(run_id, None)

    def _mark_failed(self, run_id: str, error: str) -> None:
        """A failure *outside* the runner still has to reach the run's record.

        The runner writes its own failures; this is for everything around it —
        no model credentials, a graph that hit its recursion limit, the
        orchestrator refusing to call the tool. Without this the screen would
        show a run that is simply never going to move again.
        """
        run_dir = self.runs_dir / run_id
        try:
            state = RunState.load(run_dir)
        except (OSError, ValueError):
            return
        if state.status in ("done", "failed"):
            return
        state.status = "failed"
        state.gate = None
        for rec in state.steps.values():
            if rec.status in ("pending", "running") and not rec.error:
                rec.status, rec.error = "failed", error
                rec.finished_at = time.time()
                break
        state.save(run_dir)
