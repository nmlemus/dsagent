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

PARK_SECONDS = 5.0
"""How long an answer waits for a parking run before giving up on it.

The gap between "the gate is announced" and "the thread has stopped" is a
checkpoint write plus the runner closing its envs. Five seconds covers a kernel
env shutting down; past that, the run is not parking, it is working."""


class Driver(Protocol):
    """What the runs API needs from whatever is actually running the run."""

    def create(self, run_id: str, inputs: dict[str, Any]) -> Path: ...

    def start(self, run_id: str, *, resume: bool = False) -> None: ...

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "") -> bool: ...

    def is_running(self, run_id: str) -> bool: ...


def _standing_decision(state: RunState) -> GateDecision | None:
    """The decision a re-raised gate is waiting to be told about.

    The first step whose work is done and whose gate is already answered: that is
    the one the runner is walking back through. Approvals only — a rejection
    leaves its step `pending`, and the runner asks that gate for real.
    """
    for rec in state.steps.values():
        if rec.status == "done" and rec.gate and rec.gate.decision == GateDecision.APPROVE.value:
            return GateDecision.APPROVE
    return None


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
        self._starting = threading.Lock()

    # ---- the Driver surface -------------------------------------------------

    def create(self, run_id: str, inputs: dict[str, Any], workflow: str = "") -> Path:
        cartridge = self._by_workflow[workflow]
        run_dir = self.runs_dir / run_id
        (run_dir / "workspace").mkdir(parents=True, exist_ok=True)
        pending_state(cartridge.workflows[workflow], cartridge, inputs).save(run_dir)
        return run_dir

    def start(self, run_id: str, *, resume: bool = False) -> None:
        """Drive the run. `resume` re-enters one that stopped.

        A resumed run walks back through every gate it has already passed, and
        under `dsagent serve` each of those is an `interrupt()` that has to be
        raised again — dropping it would hand the next gate the previous one's
        answer (`docs/ui-slice.md` §2). So the graph parks on a question nobody
        is being asked, and `_walk_past_decided_gates` is what answers it: from
        the run's own record, which is where the decision has been all along.

        Without it a retry stalls silently. The run reads `running`, no thread is
        behind it, and the screen shows a step that will never move — which is
        what "Retry from analyze" did the first time it was pressed on a real
        failed run.
        """
        # Held across the check *and* the spawn: two clicks on Start, or a retry
        # behind a slow response, are two requests on two threads, and without
        # the lock both pass `is_running` and both invoke the same checkpoint
        # thread.
        with self._starting:
            if self.is_running(run_id):
                return
            state = RunState.load(self.runs_dir / run_id)
            cartridge = self._by_workflow[state.workflow]
            say = resume_message if resume else start_message
            message = say(cartridge.workflows[state.workflow], run_id, state.inputs)
            self._spawn(run_id, {"messages": [{"role": "user", "content": message}]})
        if resume:
            threading.Thread(
                target=self._walk_past_decided_gates, args=(run_id,), daemon=True,
                name=f"gates-{run_id}",
            ).start()

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "",
                    timeout: float = PARK_SECONDS) -> bool:
        """Resume the run's graph with the answer. False if it is not at a gate.

        The interrupt lives in the checkpoint rather than in this process's
        memory, which is the whole reason the checkpointer exists: a run that was
        waiting when the server was killed is still waiting when it comes back,
        and this is the call that answers it (§7.11).

        **It waits for the run to finish parking.** The runner writes the pending
        gate to `run.json` and emits `awaiting_gate` *before* `ask_human` raises
        the interrupt, so between the browser seeing the question and the thread
        actually stopping there is a window — LangGraph still has to checkpoint,
        and the runner's `finally` still has to close the envs, which with a real
        kernel backend is seconds rather than microseconds. An operator who
        approves quickly lands in it. Refusing them with a 409 because they were
        fast is the wrong answer; waiting a few seconds for the thread they are
        answering is the right one.
        """
        if not self._awaiting(run_id):
            return False
        thread = self._threads.get(run_id)
        if thread and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                return False  # genuinely still working, not merely parking
        if not self._interrupted(run_id):
            return False
        from langgraph.types import Command

        with self._starting:
            if self.is_running(run_id):
                return False
            self._spawn(run_id, Command(resume={"decision": decision.value, "note": note}))
        return True

    def _walk_past_decided_gates(self, run_id: str, rounds: int = 12) -> None:
        """Answer the interrupts a resumed run raises for gates already decided.

        The run's record is the authority: `state.gate` holds the question a
        person is actually being asked, and it is `None` while the runner is
        merely re-raising a decision it already has. So an interrupt with no
        pending gate behind it is answered here, with the decision on the record,
        and the run carries on to wherever it was really going.

        Bounded, because this is a loop that talks to a graph: a workflow has a
        handful of gates, and a dozen rounds is more than any of them needs.
        """
        for _ in range(rounds):
            time.sleep(PARK_SECONDS / 5)
            if self.is_running(run_id) or not self._interrupted(run_id):
                continue
            try:
                state = RunState.load(self.runs_dir / run_id)
            except (OSError, ValueError, TypeError):
                return
            if state.gate or state.status in ("done", "failed", "stopped"):
                return  # a person is being asked, or there is nothing to ask about
            decided = _standing_decision(state)
            if decided is None:
                return
            self.log(f"[driver] {run_id}: walking past a gate already {decided.value}d")
            from langgraph.types import Command

            with self._starting:
                if self.is_running(run_id):
                    continue
                self._spawn(run_id, Command(resume={"decision": decided.value, "note": ""}))

    def _awaiting(self, run_id: str) -> bool:
        """Whether the run itself says it is standing at a gate.

        Asked before waiting on the thread, so answering a run that is simply
        working returns at once instead of holding the request for `timeout`.
        """
        try:
            state = RunState.load(self.runs_dir / run_id)
        except (OSError, ValueError, TypeError):
            return False
        return bool(state.gate) or state.status == "awaiting_gate"

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
        else:
            # The orchestrator answered without ever calling `run_workflow`: it
            # asked a question, or decided the request was not one. Nothing is
            # wrong with the *graph*, but the run the operator started never
            # began, and a run sitting at `pending` for ever with no reason given
            # is the worst of both — it looks like it might still start.
            self._mark_failed(
                run_id,
                "The orchestrator did not start this workflow. Nothing ran; "
                "try again, or start it from the chat.",
                only_if_pending=True,
            )
        finally:
            self._threads.pop(run_id, None)

    def _mark_failed(self, run_id: str, error: str, *, only_if_pending: bool = False) -> None:
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
        if only_if_pending and state.status != "pending":
            return  # it ran; whatever stopped it has already said so
        state.status = "failed"
        state.gate = None
        for rec in state.steps.values():
            if rec.status in ("pending", "running") and not rec.error:
                rec.status, rec.error = "failed", error
                rec.finished_at = time.time()
                break
        state.save(run_dir)
