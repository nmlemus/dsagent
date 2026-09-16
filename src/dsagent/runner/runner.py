"""Workflow runner: executes a declared DAG, one persona agent per step.

The runner never plans. It walks `workflow.ordered_steps()`, builds (or reuses)
the env each step declares, invokes the step's persona with the step
instructions, verifies `produces` on disk, and handles gates:

* ``human`` — calls `ask_human(prompt)`; on "no" the run stops in state
  ``awaiting_gate`` and can be resumed later with `--resume`.
* ``auto``  — runs the check script inside the step's env; non-zero exit fails
  the step.

State lives in ``<run_dir>/run.json`` so a run survives restarts. Each step
also records what it actually did — tool calls by name, token usage, skills
read and workspace files touched — read back from the agent result and from the
workspace itself, never from anything provider-specific.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dsagent.cartridge.models import Cartridge, Step, Workflow
from dsagent.envs.base import Env, make_env


class GateDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class StepRecord:
    id: str
    status: str = "pending"  # pending | running | done | failed | awaiting_gate
    started_at: float | None = None
    finished_at: float | None = None
    output: str = ""
    error: str = ""
    tool_calls: dict[str, int] = field(default_factory=dict)
    """How many times the step called each tool, by tool name."""
    usage: dict[str, int] = field(default_factory=dict)
    """`input_tokens` / `output_tokens`, summed over the step's AI messages."""
    skills_read: list[str] = field(default_factory=list)
    """Paths of SKILL.md files the step read, in first-read order."""
    files: list[dict[str, Any]] = field(default_factory=list)
    """`{path, mtime}` for workspace files the step created or modified, oldest first."""


@dataclass
class RunState:
    workflow: str
    cartridge: str
    inputs: dict[str, Any]
    status: str = "pending"  # pending | running | done | failed | awaiting_gate
    steps: dict[str, StepRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: Path) -> RunState:
        d = json.loads((run_dir / "run.json").read_text())
        d["steps"] = {k: StepRecord(**v) for k, v in d["steps"].items()}
        return cls(**d)

    def save(self, run_dir: Path) -> None:
        d = asdict(self)
        (run_dir / "run.json").write_text(json.dumps(d, indent=2))


AgentFactory = Callable[[Cartridge, str, Env, Path], Any]
"""(cartridge, persona, env, workspace) -> object with .invoke({"messages": [...]})"""


class WorkflowRunner:
    def __init__(
        self,
        cartridge: Cartridge,
        run_dir: Path,
        *,
        agent_factory: AgentFactory | None = None,
        ask_human: Callable[[str], GateDecision] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cartridge = cartridge
        self.run_dir = run_dir
        self.workspace = run_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agent_factory = agent_factory or self._default_factory
        self.ask_human = ask_human or (lambda prompt: GateDecision.APPROVE)
        self.log = log
        self._envs: dict[str, Env] = {}

    # ---- envs ---------------------------------------------------------------

    def env_for(self, name: str) -> Env:
        if name not in self._envs:
            spec = self.cartridge.envs[name]
            self.log(f"[env] provisioning '{name}' ({spec.kind})")
            self._envs[name] = make_env(spec, self.workspace)
        return self._envs[name]

    def close(self) -> None:
        for e in self._envs.values():
            e.close()
        self._envs.clear()

    @staticmethod
    def _default_factory(cartridge: Cartridge, persona: str, env: Env, workspace: Path):
        from dsagent.host.build import build_persona_agent

        return build_persona_agent(cartridge, persona, env, workspace)

    # ---- run ----------------------------------------------------------------

    def run(self, workflow_name: str, inputs: dict[str, Any] | None = None, *, resume: bool = False, dry_run: bool = False) -> RunState:
        wf = self.cartridge.workflows[workflow_name]
        inputs = self._resolve_inputs(wf, inputs or {})
        if resume and (self.run_dir / "run.json").exists():
            state = RunState.load(self.run_dir)
        else:
            state = RunState(workflow=wf.name, cartridge=self.cartridge.name, inputs=inputs,
                             steps={s.id: StepRecord(id=s.id) for s in wf.steps})
        state.status = "running"
        state.save(self.run_dir)

        try:
            for step in wf.ordered_steps():
                rec = state.steps[step.id]
                if rec.status == "done":
                    continue
                if rec.status == "awaiting_gate":
                    if not self._gate(wf, step, rec, state, dry_run):
                        return state
                    continue
                self._run_step(wf, step, rec, state, inputs, dry_run)
                if rec.status == "failed":
                    state.status = "failed"
                    state.save(self.run_dir)
                    return state
                if step.gate and not self._gate(wf, step, rec, state, dry_run):
                    return state
            state.status = "done"
            state.save(self.run_dir)
            return state
        finally:
            if not dry_run:
                self.close()

    def _resolve_inputs(self, wf: Workflow, given: dict[str, Any]) -> dict[str, Any]:
        out = dict(given)
        missing = []
        for k, spec in wf.inputs.items():
            if k not in out:
                if spec.default is not None:
                    out[k] = spec.default
                elif spec.required:
                    missing.append(k)
            if spec.options and k in out and out[k] not in spec.options:
                raise ValueError(f"input '{k}' must be one of {spec.options}, got {out[k]!r}")
        if missing:
            raise ValueError(f"workflow '{wf.name}' missing required inputs: {', '.join(missing)}")
        return out

    def _task_message(self, wf: Workflow, step: Step, inputs: dict[str, Any]) -> str:
        instructions = _fill(
            (wf.path / step.instructions).read_text(encoding="utf-8"), inputs
        )
        produces = "\n".join(f"- `{p}`" for p in step.produces) or "- (nothing mandatory)"
        inputs_md = "\n".join(f"- {k}: {v}" for k, v in inputs.items()) or "- (none)"
        return (
            f"# Workflow `{wf.name}` — step `{step.id}`\n\n"
            f"## Inputs\n{inputs_md}\n\n"
            f"## Instructions\n{instructions}\n\n"
            f"## You must produce these files (workspace-relative)\n{produces}\n\n"
            f"Finish with a short summary of what you did and any concerns for the next step."
        )

    def _run_step(self, wf: Workflow, step: Step, rec: StepRecord, state: RunState, inputs: dict[str, Any], dry_run: bool) -> None:
        env_name = step.env or wf.env
        rec.status, rec.started_at = "running", time.time()
        state.save(self.run_dir)
        self.log(f"[step] {step.id} → {step.persona} (env={env_name})")
        if dry_run:
            rec.status, rec.finished_at, rec.output = "done", time.time(), "(dry run)"
            state.save(self.run_dir)
            return
        before: dict[str, float] | None = None
        try:
            env = self.env_for(env_name)
            agent = self.agent_factory(self.cartridge, step.persona, env, self.workspace)
            before = _snapshot(self.workspace)
            result = agent.invoke({"messages": [{"role": "user", "content": self._task_message(wf, step, inputs)}]})
            rec.output = _last_text(result)
            # Before the produces check: a step that failed is the one worth reading.
            _record_telemetry(rec, result)
            missing = [p for p in step.produces if not (self.workspace / p).exists()]
            if missing:
                raise RuntimeError(f"step '{step.id}' did not produce: {', '.join(missing)}")
            rec.status = "done"
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            rec.status, rec.error = "failed", str(e)
            self.log(f"[step] {step.id} FAILED: {e}")
        finally:
            if before is not None:
                rec.files = _changed_files(before, _snapshot(self.workspace))
            rec.finished_at = time.time()
            state.save(self.run_dir)

    def _gate(self, wf: Workflow, step: Step, rec: StepRecord, state: RunState, dry_run: bool) -> bool:
        gate = step.gate
        assert gate is not None
        if gate.kind == "auto":
            self.log(f"[gate] auto check {gate.check} after {step.id}")
            if dry_run:
                return True
            env = self.env_for(step.env or wf.env)
            script = wf.path / gate.check
            r = subprocess.run(
                ["python3", str(script)], cwd=self.workspace, capture_output=True, text=True, check=False
            )
            if r.returncode != 0:
                rec.status, rec.error = "failed", f"auto gate failed:\n{r.stdout}{r.stderr}"
                state.status = "failed"
                state.save(self.run_dir)
                return False
            return True
        prompt = gate.prompt or f"Step '{step.id}' finished. Continue?"
        self.log(f"[gate] human: {prompt}")
        decision = GateDecision.APPROVE if dry_run else self.ask_human(prompt)
        if decision is GateDecision.APPROVE:
            rec.status = "done"
            state.save(self.run_dir)
            return True
        rec.status = state.status = "awaiting_gate"
        state.save(self.run_dir)
        self.log(f"[gate] run paused at '{step.id}'. Resume with --resume once approved.")
        return False


SKILL_MANIFEST = "/SKILL.md"
"""A skill is a directory with this file; reading it is how a persona loads one."""


def _attr(obj: Any, key: str) -> Any:
    """Messages are objects from a chat model and plain dicts from a fake agent."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return result.get("messages") or []
    return _attr(result, "messages") or []


def _record_telemetry(rec: StepRecord, result: Any) -> None:
    """Read back what the step did from its own messages.

    Only LangChain's standard message surface is touched — `tool_calls` and
    `usage_metadata` — so this stays true for any provider, and a message that
    carries neither simply contributes nothing.
    """
    usage = {"input_tokens": 0, "output_tokens": 0}
    for msg in _messages(result):
        for call in _attr(msg, "tool_calls") or []:
            name = _attr(call, "name")
            if not name:
                continue
            rec.tool_calls[name] = rec.tool_calls.get(name, 0) + 1
            _record_skill_reads(rec, name, _attr(call, "args") or {})
        for field_name in usage:
            usage[field_name] += int((_attr(msg, "usage_metadata") or {}).get(field_name, 0) or 0)
    rec.usage = usage


def _record_skill_reads(rec: StepRecord, tool: str, args: dict[str, Any]) -> None:
    if tool != "read_file":
        return
    for value in args.values():
        if isinstance(value, str) and SKILL_MANIFEST in value and value not in rec.skills_read:
            rec.skills_read.append(value)


def _snapshot(workspace: Path) -> dict[str, float]:
    """mtime per workspace file, skipping the harness's own materialized skills."""
    out: dict[str, float] = {}
    for p in workspace.rglob("*"):
        rel = p.relative_to(workspace)
        if not p.is_file() or rel.parts[0] == ".dsagent":
            continue
        out[rel.as_posix()] = p.stat().st_mtime
    return out


def _changed_files(before: dict[str, float], after: dict[str, float]) -> list[dict[str, Any]]:
    changed = [{"path": p, "mtime": m} for p, m in after.items() if before.get(p) != m]
    return sorted(changed, key=lambda f: (f["mtime"], f["path"]))


class _Defaults(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill(text: str, inputs: dict[str, Any]) -> str:
    """Substitute `{input_name}` in step instructions; unknown names are left as-is."""
    return text.format_map(_Defaults({k: ("None" if v is None else v) for k, v in inputs.items()}))


def _last_text(result: Any) -> str:
    try:
        msg = result["messages"][-1]
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            return "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return str(content)
    # Any message shape we do not recognise degrades to str().
    except Exception:  # noqa: BLE001  # pragma: no cover
        return str(result)
