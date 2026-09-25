"""A deterministic, local execution service shared by HTTP and chat.

The graph executes personas; it does not own jobs or human approvals. See
docs/execution-service.md for persistence, crash semantics and deployment scope.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from dsagent.capabilities import (
    CapabilityError,
    CapabilitySet,
    builtin_registry,
    compose_cartridges,
)
from dsagent.cartridge.models import Cartridge
from dsagent.datasets import seed_run
from dsagent.driver import pending_state
from dsagent.execution_storage import RunBusy, acquire, durable_json, locked
from dsagent.processes import TERMINAL, read, reference
from dsagent.projects import ProjectStore
from dsagent.runner.runner import (
    STOP_FILE,
    AgentFactory,
    GateAnswer,
    GateDecision,
    GateRequest,
    RunState,
    WorkflowRunner,
)
from dsagent.runs import locate_run_dir, run_dirs
from dsagent.sessions import mark_session


class GatePending(Exception):
    """A durable question is published; the worker may release its resources."""


class ExecutionService:
    requires_gate_id = True

    def __init__(self, cartridges: list[Cartridge], runs_dir: Path, *,
                 store: ProjectStore | None = None, capabilities: CapabilitySet | None = None,
                 seed: Path | None = None, prices: Any = None,
                 log: Callable[[str], None] = lambda m: None,
                 agent_factory: AgentFactory | None = None, isolated: bool = True,
                 process_provider=None):
        self.capabilities = compose_cartridges(cartridges, capabilities)
        self._by_workflow = {w: c for c in cartridges for w in c.workflows}
        self.runs_dir = runs_dir
        self.store, self.seed, self.prices = store, seed, prices
        self.log, self.agent_factory = log, agent_factory
        self.isolated = isolated
        self.process_spec = process_provider or builtin_registry().compose(
            ["process.local"]).provider("process.local", kind="process")
        if self.process_spec.kind != "process":
            raise CapabilityError("execution host requires a process provider")
        self.processes = self.process_spec.factory()

    def _process_roots(self, run_dir: Path):
        return sorted((run_dir / ".execution/processes").glob("*/process.json"))

    def process_for(self, run_id: str) -> dict | None:
        paths = self._process_roots(self._run_dir(run_id))
        if not paths:
            return None
        return read(max(paths, key=lambda p: p.stat().st_mtime))

    def _reconcile_processes(self, run_dir: Path) -> None:
        for path in self._process_roots(run_dir):
            record = read(path)
            if record["status"] not in TERMINAL or record["status"] == "cleanup_required":
                result = self.processes.reconcile(path.parent)
                if result["status"] == "cleanup_required":
                    raise RunBusy("Previous compute cleanup is unverified: " +
                                  "; ".join(result.get("cleanup_errors", [])))

    def _run_dir(self, run_id: str) -> Path:
        found = locate_run_dir(self.runs_dir, run_id, self.store)
        if found is None or not (found / "run.json").is_file():
            raise ValueError(f"unknown run {run_id}")
        return found

    def create(self, run_id: str, inputs: dict[str, Any], workflow: str = "",
               project: str = "default") -> Path:
        if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
            raise ValueError("invalid run id")
        cartridge = self._by_workflow[workflow]
        if self.store is not None:
            proj = self.store.get(project)
            if proj is None or (proj.cartridge and proj.cartridge != cartridge.name):
                raise ValueError("workflow does not belong to this project")
        base = self.store.runs_dir(project) if self.store else self.runs_dir
        run_dir = base / run_id
        with locked(run_dir):
            if (run_dir / "run.json").exists():
                state = RunState.load(run_dir)
                if state.workflow != workflow or state.inputs != inputs:
                    raise ValueError("run id already belongs to a different request")
                return run_dir
            (run_dir / "workspace").mkdir(parents=True, exist_ok=True)
            if self.seed is not None:
                if not self.seed.is_dir():
                    raise ValueError(f"seed directory {self.seed} does not exist")
                shutil.copytree(self.seed, run_dir / "workspace", dirs_exist_ok=True)
            if self.store is not None:
                seed_run(self.store.data_dir(project), run_dir / "workspace")
            pending_state(cartridge.workflows[workflow], cartridge, inputs,
                          self.capabilities).save(run_dir)
        return run_dir

    def submit(self, run_id: str, workflow: str, inputs: dict, *, project="default",
               session="") -> RunState:
        """Idempotent chat submission. Replayed tool calls never retry failed work."""
        try:
            run_dir = self._run_dir(run_id)
        except ValueError:
            run_dir = self.create(run_id, inputs, workflow, project)
        state = RunState.load(run_dir)
        if state.workflow != workflow or (self.store and self.store.project_of(run_dir) != project):
            raise ValueError("run id already belongs to a different request")
        if session:
            mark_session(run_dir, session)
        if state.status == "pending":
            self.start(run_id)
        return RunState.load(run_dir)

    def start(self, run_id: str, *, resume: bool = False) -> bool:
        run_dir = self._run_dir(run_id)
        try:
            owner = acquire(run_dir)
        except RunBusy:
            return False
        try:
            state = RunState.load(run_dir)
            if self.isolated:
                executor = {"id": self.process_spec.id, "version": self.process_spec.version}
                if state.process_executor and state.process_executor != executor:
                    raise ValueError("process executor changed; restore the recorded provider")
                try:
                    self._reconcile_processes(run_dir)
                except RunBusy as exc:
                    state.status, state.driver_error = "cleanup_required", str(exc)
                    state.save(run_dir)
                    raise
                state.process_executor = executor
            if state.status == "done":
                owner.close()
                return False
            if state.gate and not self._decision(run_dir, state.gate):
                owner.close()
                return False
            if state.status != "pending" and not resume:
                owner.close()
                return False
            # This is the durable acceptance point for a new job. Recovery only
            # re-dispatches pending jobs, never uncertain running steps.
            durable_json(run_dir / ".execution/request.json", {"requested_at": time.time()})
            state.driver_error = ""
            state.save(run_dir)
            thread = threading.Thread(target=self._work, args=(run_dir, owner),
                                      name=f"job-{run_id}", daemon=True)
            thread.start()
        except BaseException:
            owner.close()
            raise
        return True

    @contextmanager
    def edit(self, run_id: str):
        """Administrative writers must not overwrite a worker's snapshot."""
        with locked(self._run_dir(run_id)):
            yield

    def is_running(self, run_id: str) -> bool:
        try:
            with self.edit(run_id):
                return False
        except RunBusy:
            return True
        except ValueError:
            return False

    def error_for(self, run_id: str) -> str | None:
        return RunState.load(self._run_dir(run_id)).driver_error or None

    def reconcile(self, run_id: str) -> dict:
        """Verify cleanup without starting or replaying any domain work."""
        run_dir = self._run_dir(run_id)
        with self.edit(run_id):
            self._reconcile_processes(run_dir)
            state = RunState.load(run_dir)
            if state.status == "cleanup_required":
                state.status = "interrupted"
                state.driver_error = "Cleanup verified. Review partial outputs before retrying."
                for rec in state.steps.values():
                    if rec.status == "running":
                        rec.status = "interrupted"
                state.save(run_dir)
            return {"verified": True, "status": state.status}

    def stop(self, run_id: str) -> dict:
        run_dir = self._run_dir(run_id)
        with locked(run_dir, "control", blocking=True):
            state = RunState.load(run_dir)
            if state.status in ("done", "failed", "stopped", "interrupted"):
                return {"stopping": False, "status": state.status}
            try:
                owner = acquire(run_dir)
            except RunBusy:
                durable_json(run_dir / STOP_FILE, {"requested_at": time.time()})
                return {"stopping": True, "status": "cancelling" if self.isolated else state.status}
            try:
                state.status, state.gate = "stopped", None
                state.save(run_dir)
                return {"stopping": False, "status": state.status}
            finally:
                owner.close()

    def _decision(self, run_dir: Path, gate: dict) -> dict | None:
        gate_id = gate.get("gate_id", "")
        try:
            UUID(gate_id)
        except (ValueError, TypeError, AttributeError):
            return None
        path = run_dir / ".execution/decisions" / f"{gate_id}.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (record["run_id"], record["step"], record["attempt"]) != (
            run_dir.name, gate["step"], gate["attempt"]
        ):
            raise ValueError("gate decision identity mismatch")
        return record

    def answer_gate(self, run_id: str, decision: GateDecision, note: str = "", *,
                    gate_id: str = "") -> bool:
        run_dir = self._run_dir(run_id)
        try:
            UUID(gate_id)
        except (ValueError, TypeError, AttributeError):
            return False
        path = run_dir / ".execution/decisions" / f"{gate_id}.json"
        with locked(run_dir, "control", blocking=True):
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["decision"] != decision.value or saved["note"] != note:
                    return False
                # Same response can be retried after a lost HTTP acknowledgement.
            else:
                state = RunState.load(run_dir)
                gate = state.gate
                if state.status != "awaiting_gate" or not gate or gate.get("gate_id") != gate_id:
                    return False
                durable_json(path, {"run_id": run_id, "gate_id": gate_id,
                                    "step": gate["step"], "attempt": gate["attempt"],
                                    "decision": decision.value, "note": note,
                                    "decided_at": time.time()})
        # If still parking, the owner sees the decision before releasing its
        # lock. If already parked, we obtain ownership here. Neither can miss it.
        state = RunState.load(run_dir)
        if state.gate and state.gate.get("gate_id") == gate_id:
            self.start(run_id, resume=True)
        return True

    def _ask(self, run_dir: Path, request: GateRequest) -> GateAnswer:
        decision = self._decision(run_dir, {"gate_id": request.gate_id,
                                           "step": request.step, "attempt": request.attempt})
        if decision is None:
            raise GatePending()
        return GateAnswer(GateDecision(decision["decision"]), decision["note"],
                          decided_at=decision["decided_at"])

    def _work(self, run_dir: Path, owner) -> None:
        try:
            while True:
                state = RunState.load(run_dir)
                if self.isolated:
                    self._process_run(run_dir, state)
                    runner = None
                else:
                    runner = self._inline_run(run_dir, state)
                with locked(run_dir, "control", blocking=True):
                    state = RunState.load(run_dir)
                    if state.status not in ("done", "failed", "cleanup_required") and (run_dir / STOP_FILE).exists():
                        if runner is not None:
                            runner._stopped(state)
                        else:
                            state.status, state.gate = "stopped", None
                            (run_dir / STOP_FILE).unlink(missing_ok=True)
                            state.save(run_dir)
                    if state.status == "awaiting_gate" and state.gate and self._decision(
                        run_dir, state.gate
                    ):
                        continue
                    owner.close()
                    return
        except Exception as exc:  # noqa: BLE001 — failures must survive the worker
            self.log(f"[execution] {run_dir.name}: {exc}")
            state = RunState.load(run_dir)
            state.driver_error = f"{type(exc).__name__}: {exc}"
            state.status = "failed"
            state.save(run_dir)
        finally:
            owner.close()

    def _inline_run(self, run_dir, state):
        cartridge = self._by_workflow[state.workflow]
        runner = WorkflowRunner(
            cartridge, run_dir, capabilities=self.capabilities,
            ask_human=lambda req: self._ask(run_dir, req),
            replay_decided_gates=False, prices=self.prices, log=self.log,
            agent_factory=self.agent_factory,
        )
        try:
            runner.run(state.workflow, resume=True)
        except GatePending:
            pass
        return runner

    def _process_run(self, run_dir: Path, state: RunState) -> None:
        cartridge = self._by_workflow[state.workflow]
        scoped = compose_cartridges([cartridge], self.capabilities)
        payload = {
            "run_dir": str(run_dir.resolve()), "cartridge": cartridge.model_dump(mode="json"),
            "providers": [{**p, "factory": reference(scoped.provider(p["id"]).factory)}
                          for p in scoped.manifest()],
            "agent_factory": reference(self.agent_factory) if self.agent_factory else None,
            "rates": {k: asdict(v) for k, v in self.prices.rates.items()} if self.prices else {},
        }
        root = (run_dir / ".execution/processes" / uuid4().hex).resolve()
        self.processes.start([sys.executable, "-m", "dsagent.execution_worker", str(root)],
                             root, cwd=Path.cwd(), payload=payload)
        try:
            while True:
                if (run_dir / STOP_FILE).exists():
                    self.processes.cancel(root)
                process = self.processes.poll(root)
                if process["status"] in TERMINAL:
                    break
                time.sleep(.1)
        except Exception:  # a supervisor failure must also terminate its work
            process = self.processes.reconcile(root)
            if process["status"] != "cleanup_required":
                raise
        state = RunState.load(run_dir)
        status = process["status"]
        if status == "exited" and process.get("exit_code") == 0:
            return
        if status == "exited" and state.status == "failed":
            return  # retain the worker's precise configuration/agent error
        message = {
            "cancelled": "Execution cancelled; partial outputs may remain.",
            "timed_out": "Step exceeded its execution deadline.",
            "cleanup_required": "Compute cleanup could not be verified. Retry is blocked.",
        }.get(status, "Worker interrupted before publishing a complete result.")
        state.status = {"cancelled": "stopped", "timed_out": "failed",
                        "cleanup_required": "cleanup_required"}.get(status, "interrupted")
        state.driver_error, state.gate = message, None
        for rec in state.steps.values():
            if rec.status == "running" or rec.id == process.get("step"):
                rec.status = "failed" if status == "timed_out" else "interrupted"
                rec.error, rec.finished_at = message, time.time()
        if status == "cancelled":
            (run_dir / STOP_FILE).unlink(missing_ok=True)
        state.save(run_dir)

    def recover(self) -> None:
        """Reconcile without replaying uncertain work; call at server startup."""
        bases = [self.runs_dir] if self.store is None else [
            self.store.runs_dir(p.id) for p in self.store.list()
        ]
        for base in bases:
            for run_dir in run_dirs(base):
                dispatch = False
                try:
                    with locked(run_dir):
                        state = RunState.load(run_dir)
                        try:
                            self._reconcile_processes(run_dir)
                        except (RunBusy, RuntimeError) as exc:
                            state.status, state.driver_error = "cleanup_required", str(exc)
                            state.save(run_dir)
                            continue
                        if state.status in ("running", "cleanup_required"):
                            state.status = "interrupted"
                            state.driver_error = (
                                "Execution interrupted. Review the incomplete step before retrying; "
                                "external effects may already have occurred."
                            )
                            for rec in state.steps.values():
                                if rec.status == "running":
                                    rec.status = "interrupted"
                            state.save(run_dir)
                        elif state.status == "awaiting_gate" and state.gate:
                            # Adopt old questions without consulting graph checkpoints.
                            if not state.gate.get("gate_id"):
                                state.gate["gate_id"] = uuid4().hex
                                state.gate["attempt"] = state.steps[state.gate["step"]].attempt
                                state.save(run_dir)
                            dispatch = bool(self._decision(run_dir, state.gate))
                        elif state.status == "pending":
                            dispatch = (run_dir / ".execution/request.json").exists()
                    if dispatch:
                        self.start(run_dir.name, resume=True)
                except RunBusy:
                    continue
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.log(f"[execution] cannot recover {run_dir.name}: {exc}")
