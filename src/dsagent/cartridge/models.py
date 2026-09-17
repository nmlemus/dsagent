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
    section: str | None = None
    """Title of the report section this step writes, if the workflow says so.

    Optional, and inert when absent: a workflow that declares no sections still
    runs, and the harness never invents one. It is a *label*, carried on the
    step's events so the document can group what the step produced under a
    heading — the runner does not read it, and no behaviour depends on it.
    """


class WorkflowInput(BaseModel):
    type: str = "string"
    options: list[str] | None = None
    default: str | None = None
    required: bool = True
    guess: str | None = None
    """How a screen may offer a value for this input when nobody gave one.

    The only kind the harness knows is `unique_column`: "a column of the
    uploaded table that has no repeats and nothing missing". It knows nothing
    about *keys* — that this is what a key means is the cartridge's claim, made
    here, which is why the name of the input is never inspected.
    """


class Workflow(BaseModel):
    name: str
    description: str = ""
    inputs: dict[str, WorkflowInput] = Field(default_factory=dict)
    title_input: str | None = None
    """Which input, if any, a screen may use as the run's title.

    A workflow knows that its `question` is the thing a reader should see at the
    top of the report; the harness does not, and hard-coding the name of an input
    is how a domain leaks into it (invariant 1). Absent, screens fall back to the
    workflow's own name."""
    data_input: str | None = None
    """Which input names the table this run works on, for the screens that want
    to say so. Same reason: a harness that looks for `data_path` has learned
    something about a cartridge."""
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
    cartridge: str = ""
    """Which cartridge declared this env.

    Carried on the spec rather than threaded through `make_env`, because the
    only thing that needs it is a Docker image tag and the alternative is a
    third argument on a function every test stubs. A spec that knows its own
    provenance also makes `dsagent/ds-meridian` possible at all: without it two
    cartridges declaring `meridian` would build over each other's image.
    """
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
