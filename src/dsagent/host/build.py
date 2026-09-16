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
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_core.tools import tool

from dsagent.cartridge.models import Cartridge, Persona
from dsagent.envs.base import Env

DEFAULT_MODEL = os.environ.get("DSAGENT_MODEL", "anthropic:claude-sonnet-5")
SKILLS_MOUNT = "/skills/"
"""Where skills are mounted for the file tools. A virtual path: nothing running
inside the env can resolve it — that is what `run_skill_script` is for."""
SKILLS_DIR = Path(".dsagent") / "skills"
"""Where skills are materialized on disk, relative to the workspace."""


def materialize_skills(cartridges: list[Cartridge], workspace: Path) -> Path:
    """Copy each persona's granted skills into a per-persona folder. Idempotent."""
    root = workspace / SKILLS_DIR
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


def _skill_script_tool(cartridge: Cartridge, persona: str, env: Env, workspace: Path):
    """A persona's only way to run the scripts that ship with its skills.

    Skills are mounted at a virtual path for the file tools, which nothing
    running inside the env can resolve, and their real location is an internal
    detail of skill materialization. Resolving it here also enforces the matrix:
    asking for a skill the persona was not granted is an error, not a path that
    happens not to exist.
    """
    granted = sorted(s.name for s in cartridge.skills_for(persona))

    @tool
    def run_skill_script(skill: str, script: str, argv: list[str] | None = None) -> str:
        """Run a script that ships with one of your skills.

        `skill` is the skill's name, `script` the file name inside its `scripts/`
        directory (for example "profile.py"), and `argv` the command-line
        arguments to pass it. The script runs with this environment's Python,
        from the workspace root, so paths you pass in `argv` are
        workspace-relative.

        Always run a skill's script this way — its files are not reachable by
        path from `execute` or `run_python`.

        Returns the exit code followed by the script's output.
        """
        if skill not in granted:
            return (
                f"error: '{skill}' is not one of your skills. "
                f"You have: {', '.join(granted) or 'none'}."
            )
        scripts = workspace / SKILLS_DIR / persona / skill / "scripts"
        target = scripts / script
        if not target.is_file():
            available = sorted(f.name for f in scripts.glob("*")) if scripts.is_dir() else []
            return (
                f"error: skill '{skill}' has no script '{script}'. "
                f"Available: {', '.join(available) or 'none'}."
            )
        command = [env.python, str(target.relative_to(workspace)), *(argv or [])]
        result = env.backend.execute(" ".join(shlex.quote(a) for a in command))
        return f"exit code: {result.exit_code}\n{result.output}"

    return run_skill_script


def _persona_prompt(c: Cartridge, p: Persona) -> str:
    peers = ", ".join(f"{q.name} ({q.role})" for q in c.personas.values() if q.name != p.name)
    return (
        f"{p.system_prompt}\n\n"
        f"## Context\n"
        f"You are **{p.name}**, {p.role}, part of the '{c.name}' team. Peers: {peers or 'none'}.\n"
        f"The run workspace is the filesystem root; write every artifact under it using the exact "
        f"paths you are asked for. Read a skill's SKILL.md before using it, and run a skill's "
        f"scripts with `run_skill_script` — never by path, they are not reachable from the shell."
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
    skills_root = workspace / SKILLS_DIR
    if not (skills_root / persona).exists():
        materialize_skills([cartridge], workspace)
    return create_deep_agent(
        model=model or p.model or DEFAULT_MODEL,
        system_prompt=_persona_prompt(cartridge, p),
        tools=[*env.tools, _skill_script_tool(cartridge, persona, env, workspace),
               *(extra_tools or [])],
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
                    "tools": [*env.tools, _skill_script_tool(c, p.name, env, workspace)],
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
