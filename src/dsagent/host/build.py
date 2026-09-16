"""Wire cartridges into Deep Agents.

Two entry points:

* `build_persona_agent` — one Deep Agent for one persona, with exactly the skills
  the cartridge matrix grants it. The workflow runner uses this per step.
* `build_orchestrator` — the chat-mode agent: a small system prompt generated
  from the roster, every persona as a subagent, the traversal skills, and the
  `list_workflows` / `run_workflow` tools.

Skill scoping is done on disk: for each persona we materialize
``<workspace>/.dsagent/skills/<persona>/<skill>`` (a copy of the cartridge skill
dir) and hand Deep Agents that directory, so nothing outside the matrix is even
visible to the persona.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend

from dsagent.cartridge.models import Cartridge, Persona
from dsagent.envs.base import Env

DEFAULT_MODEL = os.environ.get("DSAGENT_MODEL", "anthropic:claude-sonnet-5")
SKILLS_MOUNT = "/skills/"


def materialize_skills(cartridges: list[Cartridge], workspace: Path) -> Path:
    """Copy each persona's granted skills into a per-persona folder. Idempotent."""
    root = workspace / ".dsagent" / "skills"
    if root.exists():
        shutil.rmtree(root)
    for c in cartridges:
        for p in c.personas.values():
            dest = root / p.name
            dest.mkdir(parents=True, exist_ok=True)
            for s in c.skills_for(p.name):
                shutil.copytree(s.path, dest / s.name, dirs_exist_ok=True)
        # orchestrator gets the traversal skills + the auto-generated command skills
        orch = root / "_orchestrator"
        orch.mkdir(parents=True, exist_ok=True)
        for s in c.traversal_skills():
            shutil.copytree(s.path, orch / s.name, dirs_exist_ok=True)
        for wf in c.workflows.values():
            cmd = c.root / "skills" / f"{c.name}-{wf.name}"
            if cmd.exists():
                shutil.copytree(cmd, orch / cmd.name, dirs_exist_ok=True)
    return root


def _backend(env: Env, skills_root: Path) -> CompositeBackend:
    return CompositeBackend(
        default=env.backend,
        routes={SKILLS_MOUNT: FilesystemBackend(root_dir=skills_root, virtual_mode=True)},
    )


def _persona_prompt(c: Cartridge, p: Persona) -> str:
    peers = ", ".join(f"{q.name} ({q.role})" for q in c.personas.values() if q.name != p.name)
    return (
        f"{p.system_prompt}\n\n"
        f"## Context\n"
        f"You are **{p.name}**, {p.role}, part of the '{c.name}' team. Peers: {peers or 'none'}.\n"
        f"The run workspace is the filesystem root; write every artifact under it using the exact "
        f"paths you are asked for. Read a skill's SKILL.md before using it."
    )


def build_persona_agent(
    cartridge: Cartridge,
    persona: str,
    env: Env,
    workspace: Path,
    *,
    model: str | None = None,
    extra_tools: list[Callable[..., Any]] | None = None,
):
    p = cartridge.personas[persona]
    skills_root = workspace / ".dsagent" / "skills"
    if not (skills_root / persona).exists():
        materialize_skills([cartridge], workspace)
    return create_deep_agent(
        model=model or p.model or DEFAULT_MODEL,
        system_prompt=_persona_prompt(cartridge, p),
        tools=[*env.tools, *(extra_tools or [])],
        skills=[f"{SKILLS_MOUNT}{persona}/"],
        backend=_backend(env, skills_root),
        name=persona,
    )


def _roster(cartridges: list[Cartridge]) -> str:
    lines = []
    for c in cartridges:
        lines.append(f"### Cartridge `{c.name}` — {c.description}")
        for p in c.personas.values():
            wfs = f" · can start: {', '.join(p.workflows)}" if p.workflows else ""
            lines.append(f"- **{p.name}** — {p.role}. Skills: {', '.join(p.skills)}{wfs}")
        if c.workflows:
            lines.append("Workflows: " + ", ".join(f"`{w.name}` ({w.description})" for w in c.workflows.values()))
    return "\n".join(lines)


ORCHESTRATOR_PROMPT = """You are the DSAgent orchestrator. You do not do domain work yourself: you route
requests to the right persona with the `task` tool, or start a workflow with `run_workflow`.

Routing rules:
- If the user names a persona ("Ana, ...") delegate to that persona.
- If the user names a workflow or a `/<cartridge>-<workflow>` command, call `run_workflow`.
- If a request clearly matches a workflow a persona owns (e.g. "build an MMM"), call
  `run_workflow` — the same thing that persona would do.
- Otherwise pick the persona whose role fits and delegate.
- Never run a workflow that no persona in the roster owns unless the user explicitly asks.

## Roster
{roster}
"""


def build_orchestrator(
    cartridges: list[Cartridge],
    env: Env,
    workspace: Path,
    *,
    model: str | None = None,
    workflow_tools: list[Callable[..., Any]] | None = None,
):
    skills_root = materialize_skills(cartridges, workspace)
    backend = _backend(env, skills_root)
    subagents = []
    for c in cartridges:
        for p in c.personas.values():
            subagents.append(
                {
                    "name": p.name,
                    "description": f"{p.role}. {p.description}",
                    "system_prompt": _persona_prompt(c, p),
                    "skills": [f"{SKILLS_MOUNT}{p.name}/"],
                    "tools": list(env.tools),
                    **({"model": p.model} if p.model else {}),
                }
            )
    return create_deep_agent(
        model=model or DEFAULT_MODEL,
        system_prompt=ORCHESTRATOR_PROMPT.format(roster=_roster(cartridges)),
        tools=[*env.tools, *(workflow_tools or [])],
        subagents=subagents,
        skills=[f"{SKILLS_MOUNT}_orchestrator/"],
        backend=backend,
        name="orchestrator",
    )
