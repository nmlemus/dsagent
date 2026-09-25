"""Importable workflow worker launched by the process provider."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dsagent.capabilities import CapabilityProvider, CapabilitySet
from dsagent.cartridge.models import Cartridge
from dsagent.execution import GatePending
from dsagent.pricing import Prices, Rate
from dsagent.processes import PROCESS_DIR_ENV, read, resolve
from dsagent.runner.runner import GateAnswer, GateDecision, RunState, WorkflowRunner


def main(root: Path):
    payload = read(root / "request.json")["payload"]
    run_dir = Path(payload["run_dir"])
    providers = {p["id"]: CapabilityProvider(**{**p, "factory": resolve(p["factory"])})
                 for p in payload["providers"]}

    def ask(request):
        path = run_dir / ".execution/decisions" / f"{request.gate_id}.json"
        if not path.exists():
            raise GatePending()
        decision = read(path)
        if (decision["run_id"], decision["step"], decision["attempt"]) != (
            run_dir.name, request.step, request.attempt
        ):
            raise ValueError("gate identity mismatch")
        return GateAnswer(GateDecision(decision["decision"]), decision["note"], decision["decided_at"])

    state = RunState.load(run_dir)
    runner = WorkflowRunner(
        Cartridge.model_validate(payload["cartridge"]), run_dir,
        capabilities=CapabilitySet(providers), ask_human=ask, replay_decided_gates=False,
        prices=Prices({k: Rate(**v) for k, v in payload["rates"].items()}),
        agent_factory=resolve(payload["agent_factory"]) if payload["agent_factory"] else None,
        process_dir=Path(os.environ[PROCESS_DIR_ENV]),
    )
    try:
        runner.run(state.workflow, resume=True)
    except GatePending:
        pass
    except Exception as exc:
        state = RunState.load(run_dir)
        state.status, state.driver_error = "failed", f"{type(exc).__name__}: {exc}"
        state.save(run_dir)
        raise


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        main(root)
    except Exception as exc:
        # Construction/import failures happen before the runner's own handler.
        run_dir = Path(read(root / "request.json")["payload"]["run_dir"])
        state = RunState.load(run_dir)
        state.status, state.driver_error = "failed", f"{type(exc).__name__}: {exc}"
        state.save(run_dir)
        raise
