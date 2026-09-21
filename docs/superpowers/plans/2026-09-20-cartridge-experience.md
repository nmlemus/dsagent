# Cartridge-driven UI experience — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The workspace UI keeps one stable structure (home, assistant, jobs, resources, deliverables) while every domain-specific word, suggestion, label and icon comes from the installed cartridge — proven by adding a small marketing cartridge without touching React.

**Architecture:** `cartridge.yaml` gains an `experience:` block (validated pydantic model with generic defaults) and `workflow.yaml` gains presentation fields on workflows, steps and inputs. `/cartridges` serves them; a project is bound to one cartridge, and the UI resolves the active experience from the active project through a React context. No new dependency on either side.

**Tech Stack:** Python 3.11+/pydantic 2/FastAPI (harness) · React 18/Vite/TypeScript (UI) · pytest with `TestClient` and the replay driver (no model).

**Spec:** the design agreed in chat on 2026-09-20 (structure stable, content per cartridge; contract proposal `experience:`; one cartridge per project; components by content type; DS migrated; marketing cartridge as acceptance test).

## Global Constraints

- Invariant 1: nothing under `src/dsagent/` names data science, a persona, a skill or a workflow. Harness defaults are generic ("trabajos", "recursos", "entregables").
- Invariant 2: no new field in `agents/*.md` frontmatter. Presentation lives in `cartridge.yaml` and `workflows/*/workflow.yaml`.
- Invariant 7: generated command skills are never hand-edited; the loader's prefix filter must use the cartridge name, not `ds-`.
- Tests in `tests/` need no model, kernel or Docker.
- UI chrome language stays Spanish (current product decision, SR-028); cartridge copy is the cartridge's choice.
- No commits in this session unless the user asks; leave a clean, verified working tree.

---

## File structure

| Responsibility | File |
|---|---|
| Experience contract + presentation fields | `src/dsagent/cartridge/models.py` |
| Load/validate `experience:`; generic generated-skill prefix | `src/dsagent/cartridge/loader.py` |
| `/cartridges` exposes experience, personas, presentation; run creation honours project cartridge | `src/dsagent/api.py` |
| Chat orchestrator scoped to the project's cartridge | `src/dsagent/serve.py` |
| DS cartridge migrated to the contract | `cartridges/ds/cartridge.yaml`, `cartridges/ds/workflows/*/workflow.yaml` |
| Marketing cartridge (acceptance test) | `cartridges/marketing/**` |
| Harness tests | `tests/test_experience.py`, `tests/test_loader.py`, `tests/test_runs_api.py` |
| UI types + experience context + hook | `ui/src/lib/types.ts`, `ui/src/lib/experience.ts`, `ui/src/lib/useCartridges.ts` |
| UI consumers | `App.tsx`, `Sidebar.tsx`, `WorkspaceViews.tsx`, `Launcher.tsx`, `LauncherModal.tsx`, `RunDock.tsx`, `ChatCenter.tsx`, `ProjectSwitcher.tsx`, `GateCard.tsx` |
| Docs | `docs/architecture.md` (§3.5), `docs/ROADMAP.md` (decisions log), `docs/ui-workspace-redesign-2026-09-19.md` (note) |

---

### Task 1: Experience contract in the harness

**Files:**
- Modify: `src/dsagent/cartridge/models.py`
- Modify: `src/dsagent/cartridge/loader.py`
- Create: `tests/test_experience.py`
- Modify: `tests/test_loader.py`

**Interfaces:**
- Produces `Experience` (pydantic) on `Cartridge.experience`, `Workflow.title/icon/badge`, `Step.title`, `WorkflowInput.label/placeholder/help`, and `Experience.model_dump()` as the wire shape.

- [ ] **Step 1: Failing tests**

```python
# tests/test_experience.py
from pathlib import Path
import pytest
from dsagent.cartridge import CartridgeError, load_cartridge
from dsagent.cartridge.models import Experience, Cartridge

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"

def _cartridge(tmp_path, experience_yaml: str) -> Path:
    root = tmp_path / "cart"
    (root / "agents").mkdir(parents=True)
    (root / "skills" / "notes").mkdir(parents=True)
    (root / "workflows" / "one" / "steps").mkdir(parents=True)
    (root / "agents" / "sam.md").write_text("---\nname: sam\nskills: [notes]\n---\nYou are Sam.\n")
    (root / "skills" / "notes" / "SKILL.md").write_text("---\nname: notes\ndescription: n\n---\nnotes\n")
    (root / "workflows" / "one" / "steps" / "a.md").write_text("Do {topic}.")
    (root / "workflows" / "one" / "workflow.yaml").write_text(
        "name: one\ntitle: Primer trabajo\nicon: file\nbadge: Beta\n"
        "inputs:\n  topic: {type: string, label: Tema, placeholder: 'Ej. lanzamiento', help: Una frase}\n"
        "title_input: topic\nsteps:\n  - {id: a, persona: sam, title: Preparar, instructions: steps/a.md, produces: [out.md]}\n"
    )
    (root / "cartridge.yaml").write_text(
        "name: cart\npersonas:\n  sam: {role: Writer, workflows: [one]}\nskills:\n  notes: {scope: all}\n"
        "workflows: [workflows/one]\n" + experience_yaml
    )
    return root

def test_experience_defaults_are_generic():
    e = Experience()
    assert e.nouns.run.plural == "trabajos"
    assert e.nav.resources == "Recursos"
    assert e.assistant.suggestions == []

def test_cartridge_without_experience_gets_defaults(tmp_path):
    c = load_cartridge(_cartridge(tmp_path, ""))
    assert c.experience == Experience()

def test_experience_and_presentation_fields_load(tmp_path):
    c = load_cartridge(_cartridge(tmp_path,
        "experience:\n  title: Estudio\n  nouns:\n    run: {singular: campaña, plural: campañas}\n"
        "  assistant:\n    suggestions: [Hola]\n"))
    assert c.experience.title == "Estudio"
    assert c.experience.nouns.run.plural == "campañas"
    assert c.experience.nouns.resource.plural == "recursos"   # untouched default
    wf = c.workflows["one"]
    assert (wf.title, wf.icon, wf.badge) == ("Primer trabajo", "file", "Beta")
    assert wf.steps[0].title == "Preparar"
    assert wf.inputs["topic"].label == "Tema"

def test_unknown_experience_key_is_rejected(tmp_path):
    with pytest.raises(CartridgeError, match="experience"):
        load_cartridge(_cartridge(tmp_path, "experience:\n  colour: red\n"))

def test_generated_skill_prefix_uses_cartridge_name(tmp_path):
    root = _cartridge(tmp_path, "")
    c = load_cartridge(root)
    assert (root / "skills" / "cart-one" / "SKILL.md").exists()
    assert "cart-one" not in c.skills
    load_cartridge(root)  # second load must not complain about the generated skill

def test_ds_experience_is_declared():
    c = load_cartridge(DS)
    assert c.experience.nouns.run.singular == "análisis"
    assert c.workflows["eda-to-report"].title == "Análisis exploratorio"
    assert c.workflows["eda-to-report"].step("profile").title == "Conocer los datos"
```

- [ ] **Step 2: Run** `pytest tests/test_experience.py -q` → fails (`Experience` import error).

- [ ] **Step 3: Models**

```python
# models.py additions
class Noun(BaseModel):
    singular: str
    plural: str

class Nouns(BaseModel):
    run: Noun = Noun(singular="trabajo", plural="trabajos")
    resource: Noun = Noun(singular="recurso", plural="recursos")
    deliverable: Noun = Noun(singular="entregable", plural="entregables")

class Nav(BaseModel):
    overview: str = "Inicio"; chat: str = "Asistente"; runs: str = "Trabajos"
    resources: str = "Recursos"; deliverables: str = "Entregables"

class HomeStep(BaseModel):
    title: str; text: str = ""

class Home(BaseModel):
    eyebrow: str = "TU ESPACIO DE TRABAJO"
    headline: str = "Tu próximo resultado empieza aquí."
    intro: str = "Reúne recursos, ejecuta trabajos y revisa entregables en un solo lugar."
    stages: list[str] = ["Recursos", "Trabajo", "Entregables"]
    steps: list[HomeStep] = [...3 generic...]

class Assistant(BaseModel):
    headline: str = "¿En qué trabajamos hoy?"
    intro: str = "Un equipo de especialistas listo para ayudarte."
    placeholder: str = "Escribe tu mensaje…"
    suggestions: list[str] = []
    instructions: str = ""      # extra system instructions for the chat model

class Resources(BaseModel):
    intro: str = "Sube una vez. Reutiliza en cada trabajo de este proyecto."
    formats: list[str] = []     # shown as hint; used for the file input `accept`

class Experience(BaseModel, extra="forbid"):
    version: int = 1
    title: str = "DSAgent"
    tagline: str = "Tu equipo, en tu espacio de trabajo."
    nouns: Nouns = Nouns(); nav: Nav = Nav(); home: Home = Home()
    assistant: Assistant = Assistant(); resources: Resources = Resources()
```

All nested models also `extra="forbid"`. Add `title: str | None`, `icon: str | None`, `badge: str | None` to `Workflow`; `title: str | None` to `Step`; `label/placeholder/help: str | None` to `WorkflowInput`; `experience: Experience = Field(default_factory=Experience)` to `Cartridge`.

- [ ] **Step 4: Loader** — parse `cy.get("experience")` via `Experience.model_validate`, wrap `ValidationError` in `CartridgeError(f"{root}: experience: {e}")`. Compute `name` before `_load_skills` and pass it so the generated-prefix filter is `f"{name}-"`.

- [ ] **Step 5: Run** `pytest tests/test_experience.py tests/test_loader.py -q` → the DS assertion fails until Task 2; everything else passes.

### Task 2: DS cartridge migrated to the contract

**Files:** `cartridges/ds/cartridge.yaml`, `cartridges/ds/workflows/eda-to-report/workflow.yaml`, `cartridges/ds/workflows/mmm-meridian/workflow.yaml`.

- [ ] Add `experience:` with the copy currently hard-coded in `WorkspaceViews.tsx`, `Sidebar.tsx`, `ChatCenter.tsx`, `Launcher.tsx` (Spanish). `assistant.instructions` = the `VIZ_INSTRUCTIONS` string. `resources.formats: [csv, parquet]`.
- [ ] Workflow titles/icons/badges and step titles/input labels from `workflowTitle`, `workflowDescription`, `stepTitle`, `fieldTitle`.
- [ ] `pytest tests/test_experience.py tests/test_loader.py tests/test_runs_api.py -q` → green.

### Task 3: API exposes the experience and binds runs to the project's cartridge

**Files:** `src/dsagent/api.py`, `src/dsagent/serve.py`, `tests/test_runs_api.py`.

**Interfaces:** `/cartridges` → `{cartridges: [{name, version, description, experience, personas:[{name, role, description}]}], workflows: [...+title, icon, badge; steps[].title; inputs[].label/placeholder/help]}`. `POST /runs` with `project` whose `cartridge` is set and differs from the workflow's cartridge → 400.

- [ ] Tests:

```python
def test_cartridges_carries_experience_and_personas(client):
    body = client.get("/cartridges").json()
    ds = body["cartridges"][0]
    assert ds["experience"]["nouns"]["run"]["singular"] == "análisis"
    assert [p["name"] for p in ds["personas"]] == ["ana", "marie", "noel", "pablo"]
    eda = next(w for w in body["workflows"] if w["name"] == "eda-to-report")
    assert eda["title"] == "Análisis exploratorio"
    assert eda["steps"][0]["title"] == "Conocer los datos"
    assert eda["inputs"]["question"]["label"]

def test_run_must_match_the_projects_cartridge(client, app):
    pid = client.post("/projects", json={"name": "Other", "cartridge": "marketing"}).json()["id"]
    r = client.post("/runs", json={"workflow": "eda-to-report", "inputs": {}, "project": pid})
    assert r.status_code == 400
    assert "cartridge" in r.text
```

- [ ] Implement in `workflow_shape` and `cartridges_route`; in the `POST /runs` handler resolve the project (when given) and compare. In `serve.py::chat_agent_for`, choose `[c for c in cartridges if c.name == project.cartridge] or cartridges` and build the orchestrator and its `workflow_tools` over that subset.
- [ ] `pytest -q` green.

### Task 4: Marketing cartridge

**Files (create):** `cartridges/marketing/.claude-plugin/plugin.json`, `cartridge.yaml`, `agents/lucia.md`, `agents/tomas.md`, `skills/briefs/SKILL.md`, `skills/copy/SKILL.md`, `workflows/campaign-brief/workflow.yaml`, `workflows/campaign-brief/steps/01-research.md`, `02-strategy.md`, `03-copy.md`. Test: `tests/test_marketing_cartridge.py`.

- [ ] Personas: `lucia` (Brand Strategist, skills `[briefs]`, workflows `[campaign-brief]`), `tomas` (Copywriter, skills `[briefs, copy]`). Skills: `briefs` scope all, `copy` scope `[tomas]`.
- [ ] Workflow `campaign-brief`: inputs `brief_path` (path, label "Brief de campaña"), `objective` (string, label "Objetivo de la campaña"), `audience` (string, optional). `title_input: objective`, `data_input: brief_path`. Steps: `research` (lucia → `artifacts/brief-analysis.md`), `strategy` (lucia, human gate → `artifacts/strategy.md`), `copy` (tomas → `deliverables/campaign-copy.md`). Env default kernel, no requirements.
- [ ] Experience in Spanish with marketing nouns (`campaña`, `material`, `pieza`), nav, home, assistant suggestions, resources formats `[md, pdf, txt]`.
- [ ] Test: loads, matrix consistent, experience nouns, `/cartridges` with both cartridges lists both experiences, `dsagent cartridge validate cartridges/marketing` passes.

### Task 5: UI — types, context, hook

**Files:** `ui/src/lib/types.ts`, `ui/src/lib/experience.ts` (new), `ui/src/lib/useCartridges.ts` (new).

**Interfaces:**
- `Experience` TS type mirrors the pydantic shape; `DEFAULT_EXPERIENCE` (generic, used before/without the API).
- `ExperienceContext` + `useExperience()`; `ExperienceProvider` value.
- `useCartridges(refresh)` → `{cartridges, workflows, loading, error}`.
- `experienceFor(cartridges, project?)` → the project's cartridge experience, else the first cartridge's, else default.
- `workflowsFor(workflows, project?)` → filtered by the project's cartridge when set.
- Helpers moved out of `WorkspaceViews`: `workflowTitle(w)`, `stepTitle(s)`, `runTitle(run, workflows)`, `workflowIcon(w): IconName`.

### Task 6: UI — consumers

- [ ] `App.tsx`: provide the context; nav labels from `experience.nav`; `document.title = experience.title`; launcher/search copy uses nouns.
- [ ] `Sidebar.tsx`: `NAV_IDS` fixed, labels from experience; brand `experience.title` and the note from `tagline`.
- [ ] `WorkspaceViews.tsx`: Overview/Analyses/DataLibrary/Deliverables copy from `experience.home/resources/nouns`; icons from workflow shape; accept formats.
- [ ] `Launcher.tsx`: labels/placeholders/help from input spec; step titles from shape; badge/icon from shape; drop the `eda-to-report` default-question hack; workflows filtered by project cartridge.
- [ ] `RunDock.tsx`, `GateCard.tsx`, `LauncherModal.tsx`: nouns.
- [ ] `ChatCenter.tsx`: headline/intro/placeholder/suggestions/instructions from experience; tool labels generic (`run_python: "ejecutando código"`, `run_workflow: "iniciando un trabajo"` → uses noun).
- [ ] `ProjectSwitcher.tsx`: cartridge select when more than one cartridge is loaded; sends `cartridge` in `createProject`.
- [ ] `grep -n "análisis\|datos\|eda-to-report\|mmm" ui/src` → only generic or experience-derived strings remain.
- [ ] `npm run typecheck --prefix ui && npm run build --prefix ui`.

### Task 7: Docs + decision log

- [ ] `docs/architecture.md`: new §3.5 "Experience (cartridge.yaml `experience:`) and presentation fields" with the YAML; note "one cartridge per project".
- [ ] `docs/ROADMAP.md` decisions log entry (2026-09-20).
- [ ] `docs/ui-workspace-redesign-2026-09-19.md`: replace the "generalización pendiente en SR-028/032" sentence with a pointer.
- [ ] `~/.devtrace/decisions/dsagent-v2.md` entry.

### Task 8: Verification

- [ ] `pytest -q` green; `ruff check src tests` no new findings.
- [ ] `dsagent cartridge validate cartridges/ds` and `cartridges/marketing`.
- [ ] Serve in replay with both cartridges: `dsagent serve --replay tests/fixtures/replay/run-eda-003 -c cartridges/ds -c cartridges/marketing`, build the UI, open in Chrome: DS project shows DS copy; create a marketing project → nav, home, launcher (campaign-brief with labelled inputs), assistant suggestions switch without a reload of code.

## Self-review

- Spec coverage: contract (T1), project↔cartridge (T3), nav/home/assistant/forms (T5–6), resources/viewers generalised (T6; viewers already type-driven by `FileViewer`), DS migrated (T2), marketing cartridge (T4), acceptance run (T8). Multimedia viewers (audio/video) are out of scope for this pass: `FileViewer` handles html/md/csv/images; a marketing deliverable is markdown.
- Names used consistently: `Experience`, `experienceFor`, `workflowsFor`, `workflowTitle`, `stepTitle`, `runTitle`, `useCartridges`, `useExperience`.
