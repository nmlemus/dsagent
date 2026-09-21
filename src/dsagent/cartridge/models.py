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
    title: str | None = None
    """What a screen calls this step. Presentation only: the runner keys on `id`.
    Absent, screens show the id."""
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
    label: str | None = None
    """What a form calls this input. Absent, the form shows the input's name."""
    placeholder: str | None = None
    help: str | None = None
    """One line under the field, if the cartridge has something to say."""


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
    title: str | None = None
    """What a screen calls this workflow. `name` stays the identifier the runner,
    the CLI and `run_workflow` use; the title is free to be in the cartridge's
    language and to change without renaming a directory."""
    summary: str | None = None
    """What a screen tells the *user* this workflow does.

    `description` is written for the orchestrator deciding whether to call it,
    and says harness things ("kernel env only — the smoke test"); this is the
    sentence a person reads before starting one. Absent, screens show
    `description`."""
    icon: str | None = None
    """A symbol name from the screen's own icon set. The harness passes it
    through; a screen falls back to a generic one it does not recognise."""
    badge: str | None = None
    """A short label beside the title, e.g. "Experimental"."""
    env: str = "default"
    steps: list[Step]
    path: Path
    """Workflow directory (contains workflow.yaml, steps/, templates/)."""

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Workflow:
        """A step id is the key everything else uses — `needs`, the event
        stream, `run.json`, a resumed run. `step()` returns the first match, so
        a duplicate ran the first step twice and never ran the second, silently.
        """
        seen: set[str] = set()
        for s in self.steps:
            if s.id in seen:
                raise ValueError(f"workflow {self.name}: duplicate step id '{s.id}'")
            seen.add(s.id)
        return self

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


class Noun(BaseModel, extra="forbid"):
    singular: str
    plural: str


class Nouns(BaseModel, extra="forbid"):
    """The three things every screen talks about, in the cartridge's words.

    A run is a "trabajo" to the harness; a cartridge may call it an análisis or
    a campaña. Screens build their sentences from these, so the harness never
    has to know which."""

    run: Noun = Field(default_factory=lambda: Noun(singular="trabajo", plural="trabajos"))
    resource: Noun = Field(default_factory=lambda: Noun(singular="recurso", plural="recursos"))
    deliverable: Noun = Field(
        default_factory=lambda: Noun(singular="entregable", plural="entregables")
    )


class Nav(BaseModel, extra="forbid"):
    """Labels for the five fixed sections. The sections themselves are the
    harness's structure; only their names belong to the cartridge."""

    overview: str = "Inicio"
    chat: str = "Asistente"
    runs: str = "Trabajos"
    resources: str = "Recursos"
    deliverables: str = "Entregables"


class HomeStep(BaseModel, extra="forbid"):
    title: str
    text: str = ""


class Home(BaseModel, extra="forbid"):
    eyebrow: str = "TU ESPACIO DE TRABAJO"
    headline: str = "Tu próximo resultado empieza aquí."
    intro: str = "Reúne recursos, ejecuta trabajos y revisa entregables en un solo lugar."
    stages: list[str] = Field(default_factory=lambda: ["Recursos", "Trabajo", "Entregables"])
    """Three words for the home illustration: what goes in, what happens, what
    comes out."""
    steps: list[HomeStep] = Field(
        default_factory=lambda: [
            HomeStep(title="Trae tus recursos", text="Los archivos con los que trabajará el equipo."),
            HomeStep(title="Define el objetivo", text="Elige un trabajo o conversa con el asistente."),
            HomeStep(title="Revisa los entregables", text="Aprueba cada revisión y explora el resultado."),
        ]
    )


class Assistant(BaseModel, extra="forbid"):
    headline: str = "¿En qué trabajamos hoy?"
    intro: str = "Un equipo de especialistas listo para ayudarte en este proyecto."
    placeholder: str = "Escribe tu mensaje…"
    suggestions: list[str] = Field(default_factory=list)
    """Opening prompts a screen offers; editable before sending."""
    instructions: str = ""
    """Extra instructions the chat client adds for the model — how to present
    results in this domain. Empty means the client adds nothing."""


class Resources(BaseModel, extra="forbid"):
    intro: str = "Sube una vez. Reutiliza en cada trabajo de este proyecto."
    formats: list[str] = Field(default_factory=list)
    """File extensions the screen suggests (and offers in its file picker).
    A hint, not a gate: the harness stores whatever is uploaded."""


class Experience(BaseModel, extra="forbid"):
    """How a cartridge presents itself in the workspace UI.

    The UI's *structure* is fixed by the harness — a home, an assistant, a list
    of runs, a library of resources, a shelf of deliverables. Everything a user
    reads inside that structure comes from here, so a cartridge for another
    domain changes the product's vocabulary without a line of frontend code.
    Unknown keys are rejected: a typo here would otherwise silently fall back
    to the generic copy and nobody would notice until a demo.
    """

    version: int = 1
    title: str = "DSAgent"
    tagline: str = "Tu equipo, en tu espacio de trabajo."
    nouns: Nouns = Field(default_factory=Nouns)
    nav: Nav = Field(default_factory=Nav)
    home: Home = Field(default_factory=Home)
    assistant: Assistant = Field(default_factory=Assistant)
    resources: Resources = Field(default_factory=Resources)


class Cartridge(BaseModel):
    name: str
    version: str = "0.0.0"
    description: str = ""
    root: Path
    personas: dict[str, Persona]
    skills: dict[str, Skill]
    workflows: dict[str, Workflow]
    envs: dict[str, EnvSpec]
    experience: Experience = Field(default_factory=Experience)

    def skills_for(self, persona: str) -> list[Skill]:
        return [self.skills[s] for s in self.personas[persona].skills]

    def traversal_skills(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.scope == "all"]
