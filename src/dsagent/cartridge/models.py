"""Pydantic models for a cartridge.

A cartridge directory is simultaneously a Claude Code plugin (``.claude-plugin/``,
``agents/``, ``skills/``) and a DSAgent cartridge (``cartridge.yaml``,
``workflows/``, ``envs/``). These models describe the merged view the harness
works with after loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Skill(BaseModel):
    name: str
    description: str = ""
    path: Path
    """Directory containing SKILL.md."""
    scope: list[str] | Literal["all"] = "all"
    """Personas allowed to load this skill; "all" makes it traversal."""


class Persona(BaseModel):
    name: str
    role: str = ""
    description: str = ""
    model: str | None = None
    system_prompt: str
    skills: list[str] = Field(default_factory=list)
    workflows: list[str] = Field(default_factory=list)
    tools: list[str] | None = None
    path: Path


class Gate(BaseModel):
    kind: Literal["human", "auto"]
    prompt: str | None = None
    check: str | None = None
    """For auto gates: script path relative to the workflow dir. Exit 0 = pass."""

    @model_validator(mode="after")
    def _check_fields(self) -> Gate:
        if self.kind == "auto" and not self.check:
            raise ValueError("auto gate requires `check`")
        return self


class Step(BaseModel):
    id: str
    persona: str
    instructions: str
    """Path to a markdown file, relative to the workflow dir."""
    env: str | None = None
    needs: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    gate: Gate | None = None
    timeout: str | None = None
    sees: list[str] | None = None
    """Workflow inputs this step is shown. `None` means the ones its own
    instruction text interpolates — a step that never writes `{question}` is
    never told the question. Set it explicitly to widen or narrow that."""


class WorkflowInput(BaseModel):
    type: str = "string"
    options: list[str] | None = None
    default: str | None = None
    required: bool = True


class Workflow(BaseModel):
    name: str
    description: str = ""
    inputs: dict[str, WorkflowInput] = Field(default_factory=dict)
    env: str = "default"
    steps: list[Step]
    path: Path
    """Workflow directory (contains workflow.yaml, steps/, templates/)."""

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def ordered_steps(self) -> list[Step]:
        """Topological order over `needs` (stable: keeps declaration order when free)."""
        done: list[str] = []
        pending = list(self.steps)
        while pending:
            progressed = False
            for s in list(pending):
                if all(n in done for n in s.needs):
                    done.append(s.id)
                    pending.remove(s)
                    progressed = True
            if not progressed:
                cycle = ", ".join(s.id for s in pending)
                raise ValueError(f"workflow {self.name}: cycle or unknown dependency among: {cycle}")
        return [self.step(i) for i in done]


class EnvSpec(BaseModel):
    name: str
    kind: Literal["kernel", "docker"] = "kernel"
    build: Path | None = None
    """Docker: directory with a Dockerfile, relative to cartridge root."""
    image: str | None = None
    """Docker: prebuilt image (alternative to `build`)."""
    gpu: Literal["required", "optional", "none"] = "none"
    requirements: list[str] = Field(default_factory=list)
    """Pip requirement strings this env must provide.

    Kernel envs verify them when provisioned; docker envs install them in their
    Dockerfile and the harness only validates the field. The harness never
    interprets the strings — see `dsagent.requirements`.
    """


class Cartridge(BaseModel):
    name: str
    version: str = "0.0.0"
    description: str = ""
    root: Path
    personas: dict[str, Persona]
    skills: dict[str, Skill]
    workflows: dict[str, Workflow]
    envs: dict[str, EnvSpec]

    def skills_for(self, persona: str) -> list[Skill]:
        return [self.skills[s] for s in self.personas[persona].skills]

    def traversal_skills(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.scope == "all"]
