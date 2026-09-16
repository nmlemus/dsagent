"""Load and validate cartridges.

Reads, in order:
  1. ``.claude-plugin/plugin.json``   (name/version/description, optional)
  2. ``cartridge.yaml``               (persona<->skill matrix, workflows, envs)
  3. ``agents/*.md``                  (personas; frontmatter + body = system prompt)
  4. ``skills/*/SKILL.md``            (Agent Skills standard)
  5. ``workflows/*/workflow.yaml``

Then validates that the matrix in ``cartridge.yaml`` and the ``skills:`` list in
each agent's frontmatter agree, so a cartridge opened in Claude Code behaves the
same as in DSAgent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from dsagent.cartridge.models import (
    Cartridge,
    EnvSpec,
    Persona,
    Skill,
    Workflow,
)
from dsagent.requirements import distribution_name

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class CartridgeError(Exception):
    """Raised for structural problems in a cartridge (missing files, matrix conflicts)."""


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, m.group(2)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_plugin_json(root: Path) -> dict[str, Any]:
    p = root / ".claude-plugin" / "plugin.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_skills(root: Path, matrix: dict[str, Any]) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        return skills
    for d in sorted(skills_dir.iterdir()):
        md = d / "SKILL.md"
        if not d.is_dir() or not md.exists():
            continue
        meta, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
        name = meta.get("name", d.name)
        if name != d.name:
            raise CartridgeError(f"skill {md}: frontmatter name '{name}' != directory '{d.name}'")
        entry = matrix.get(name, {}) or {}
        scope = entry.get("scope", "all")
        skills[name] = Skill(
            name=name, description=meta.get("description", ""), path=d, scope=scope
        )
    for name in matrix:
        if name not in skills and not name.startswith("ds-"):
            raise CartridgeError(f"cartridge.yaml lists skill '{name}' but skills/{name}/SKILL.md is missing")
    return skills


def _load_personas(root: Path, matrix: dict[str, Any]) -> dict[str, Persona]:
    personas: dict[str, Persona] = {}
    agents_dir = root / "agents"
    if not agents_dir.is_dir():
        return personas
    for md in sorted(agents_dir.glob("*.md")):
        meta, body = parse_frontmatter(md.read_text(encoding="utf-8"))
        name = meta.get("name", md.stem)
        entry = matrix.get(name, {}) or {}
        personas[name] = Persona(
            name=name,
            role=entry.get("role", ""),
            description=meta.get("description", ""),
            model=meta.get("model"),
            system_prompt=body.strip(),
            skills=list(meta.get("skills", [])),
            workflows=list(entry.get("workflows", [])),
            tools=meta.get("tools"),
            path=md,
        )
    for name in matrix:
        if name not in personas:
            raise CartridgeError(f"cartridge.yaml lists persona '{name}' but agents/{name}.md is missing")
    return personas


def _load_workflows(root: Path, declared: list[str]) -> dict[str, Workflow]:
    workflows: dict[str, Workflow] = {}
    for rel in declared:
        wdir = root / rel
        wf_yaml = wdir / "workflow.yaml"
        if not wf_yaml.exists():
            raise CartridgeError(f"workflow '{rel}' declared but {wf_yaml} is missing")
        data = _read_yaml(wf_yaml)
        data["path"] = wdir
        wf = Workflow(**data)
        for s in wf.steps:
            if not (wdir / s.instructions).exists():
                raise CartridgeError(f"workflow {wf.name}: step {s.id} instructions not found: {s.instructions}")
            if s.gate and s.gate.kind == "auto" and not (wdir / s.gate.check).exists():
                raise CartridgeError(f"workflow {wf.name}: step {s.id} gate check not found: {s.gate.check}")
        wf.ordered_steps()  # raises on cycles
        workflows[wf.name] = wf
    return workflows


def _load_envs(root: Path, declared: dict[str, Any]) -> dict[str, EnvSpec]:
    envs: dict[str, EnvSpec] = {"default": EnvSpec(name="default", kind="kernel")}
    for name, spec in (declared or {}).items():
        spec = dict(spec or {})
        _validate_requirements(name, spec.get("requirements"))
        if "build" in spec:
            build = root / spec["build"]
            if not (build / "Dockerfile").exists():
                raise CartridgeError(f"env '{name}': {build}/Dockerfile not found")
            spec["build"] = build
        envs[name] = EnvSpec(name=name, **spec)
    return envs


def _validate_requirements(env_name: str, declared: Any) -> None:
    """Requirements must be a list of pip requirement strings; nothing else is read."""
    if declared is None:
        return
    if not isinstance(declared, list):
        raise CartridgeError(
            f"env '{env_name}': requirements must be a list, got {type(declared).__name__}"
        )
    for r in declared:
        if not isinstance(r, str):
            raise CartridgeError(f"env '{env_name}': requirement must be a string, got {r!r}")
        try:
            distribution_name(r)
        except ValueError as e:
            raise CartridgeError(f"env '{env_name}': {e}") from e


def _validate_matrix(c: Cartridge) -> None:
    """The persona frontmatter and cartridge.yaml must tell the same story."""
    problems: list[str] = []
    for p in c.personas.values():
        for s in p.skills:
            if s not in c.skills:
                problems.append(f"persona '{p.name}' lists unknown skill '{s}'")
                continue
            scope = c.skills[s].scope
            if scope != "all" and p.name not in scope:
                problems.append(f"persona '{p.name}' claims skill '{s}' but cartridge.yaml scope is {scope}")
        for s in c.skills.values():
            if s.scope == "all" and s.name not in p.skills:
                problems.append(f"persona '{p.name}' is missing traversal skill '{s.name}' in its frontmatter")
            if s.scope != "all" and p.name in s.scope and s.name not in p.skills:
                problems.append(f"cartridge.yaml grants '{s.name}' to '{p.name}' but agents/{p.name}.md does not list it")
        for w in p.workflows:
            if w not in c.workflows:
                problems.append(f"persona '{p.name}' lists unknown workflow '{w}'")
    for w in c.workflows.values():
        for st in w.steps:
            if st.persona not in c.personas:
                problems.append(f"workflow '{w.name}' step '{st.id}' uses unknown persona '{st.persona}'")
            env = st.env or w.env
            if env not in c.envs:
                problems.append(f"workflow '{w.name}' step '{st.id}' uses unknown env '{env}'")
    if problems:
        raise CartridgeError("cartridge '%s' is inconsistent:\n  - %s" % (c.name, "\n  - ".join(problems)))


def generate_command_skills(c: Cartridge) -> list[Path]:
    """Write ``skills/ds-<workflow>/SKILL.md`` for each workflow.

    These are the ``/ds-<workflow>`` entry points. Claude Code sees them as
    ordinary command skills; DSAgent's orchestrator sees them as the trigger for
    ``run_workflow``. Regenerated on every load so they never drift from
    ``workflow.yaml``.
    """
    written: list[Path] = []
    for wf in c.workflows.values():
        d = c.root / "skills" / f"{c.name}-{wf.name}"
        d.mkdir(parents=True, exist_ok=True)
        inputs = "\n".join(
            f"- `{k}` ({v.type}{', options: ' + ', '.join(v.options) if v.options else ''})"
            for k, v in wf.inputs.items()
        ) or "- (none)"
        steps = "\n".join(
            f"{i + 1}. **{s.id}** — {s.persona}" + (f" · gate: {s.gate.kind}" if s.gate else "")
            for i, s in enumerate(wf.ordered_steps())
        )
        owners = [p.name for p in c.personas.values() if wf.name in p.workflows]
        body = f"""---
name: {c.name}-{wf.name}
description: Run the "{wf.name}" workflow. {wf.description}
---
<!-- AUTO-GENERATED by dsagent from workflows/{wf.path.name}/workflow.yaml — do not edit. -->

# /{c.name}-{wf.name}

{wf.description}

**Inputs**

{inputs}

**Steps**

{steps}

**How to run**

In DSAgent, call the `run_workflow` tool with `name="{wf.name}"` and the inputs above.
Personas allowed to start it: {', '.join(owners) or '(orchestrator only)'}.

In Claude Code (no DSAgent backend), execute the steps in order yourself, delegating
each one to the named persona subagent and reading its instructions from
`workflows/{wf.path.name}/steps/`. Stop at every human gate and ask the user.
"""
        (d / "SKILL.md").write_text(body, encoding="utf-8")
        written.append(d / "SKILL.md")
    return written


def load_cartridge(root: str | Path, *, generate_commands: bool = True) -> Cartridge:
    root = Path(root).resolve()
    if not root.is_dir():
        raise CartridgeError(f"cartridge directory not found: {root}")
    plugin = _load_plugin_json(root)
    cy_path = root / "cartridge.yaml"
    if not cy_path.exists():
        raise CartridgeError(f"{root}: cartridge.yaml is missing (a plain Claude Code plugin is not a cartridge)")
    cy = _read_yaml(cy_path)

    skills = _load_skills(root, cy.get("skills", {}) or {})
    # drop auto-generated command skills from the matrix view; they are regenerated below
    name = cy.get("name") or plugin.get("name") or root.name
    skills = {k: v for k, v in skills.items() if not k.startswith(f"{name}-")}
    personas = _load_personas(root, cy.get("personas", {}) or {})
    workflows = _load_workflows(root, cy.get("workflows", []) or [])
    envs = _load_envs(root, cy.get("envs", {}) or {})

    c = Cartridge(
        name=name,
        version=cy.get("version") or plugin.get("version", "0.0.0"),
        description=cy.get("description") or plugin.get("description", ""),
        root=root,
        personas=personas,
        skills=skills,
        workflows=workflows,
        envs=envs,
    )
    _validate_matrix(c)
    if generate_commands:
        generate_command_skills(c)
    return c


def load_cartridges(roots: list[str | Path]) -> list[Cartridge]:
    """Load several cartridges; a persona defined twice is an error (no silent override)."""
    loaded: list[Cartridge] = []
    seen: dict[str, str] = {}
    for r in roots:
        c = load_cartridge(r)
        for p in c.personas:
            if p in seen:
                raise CartridgeError(f"persona '{p}' defined in both '{seen[p]}' and '{c.name}'")
            seen[p] = c.name
        loaded.append(c)
    return loaded
