# DSAgent v2 — Architecture

**Status:** Draft v0.1 · **Author:** Noel Moreno Lemus, PhD · **Date:** 2026-09-16

Decisions locked in this draft: own backend on Deep Agents (LangChain); new repo, porting selected pieces from `nmlemus/dsagent`; Docker-per-workflow execution; cartridge format is a superset of the Claude Code plugin layout.

---

## 1. Thesis

Coding harnesses (Claude Code, Copilot CLI, Codex, Deep Agents) have converged on the same primitives: a planning tool, a filesystem, subagents, progressive-disclosure skills (`SKILL.md`), and human-in-the-loop interrupts. What none of them ship is the *domain*: who the personas are, which skills each one is allowed to use, and which multi-step workflows exist. BMAD proved that the domain layer can be a drop-in package — and its latest releases now ship agents and workflows as plain Agent Skills, which confirms the packaging bet.

Data science is a genuine R&D process, not a coding task: hypotheses, data gates, model specs, long-running fits, human sign-off between stages, and a report at the end. It needs its own domain layer on top of any harness. DSAgent v2 is that layer, split into two things that must never be confused:

| Layer | What it is | Who changes it |
|---|---|---|
| **Harness** (`dsagent-core`) | A thin Deep Agents host: loads cartridges, wires personas as subagents, scopes skills, runs workflows, provisions execution environments, exposes CLI + HTTP. Domain-agnostic. | Rarely. Small enough to read in an afternoon. |
| **Cartridge** (`dsagent-cartridge-ds`, `…-bmad`, …) | Personas + skills + workflows + environment declarations. All the domain logic. | Constantly. This is the product. |

Swap the DS cartridge for a BMAD cartridge and the same binary becomes an SDLC team. Load both and you get a DS team that can also write software.

## 2. Non-goals (v2.0)

No UI beyond the CLI and an HTTP API. *(Revised 2026-09-16: the UI moved into M2 — see `docs/ui.md`. The API is AG-UI over SSE, not WebSocket.)* No multi-tenant auth. No hosted sandbox providers in the first cut (the sandbox protocol allows them later). No attempt to re-implement planning, context management or tool-calling loops — Deep Agents owns those.

## 3. Cartridge format

A cartridge is a directory that is **simultaneously a valid Claude Code plugin** and a DSAgent cartridge. Everything Claude Code understands lives in its standard places; everything DSAgent adds lives in one file, `cartridge.yaml`, plus a `workflows/` folder. Claude Code ignores the extras; DSAgent reads all of it.

```
dsagent-cartridge-ds/
├── .claude-plugin/
│   └── plugin.json            # standard Claude Code manifest (name, version, author)
├── cartridge.yaml             # DSAgent extension: persona↔skill matrix, workflows, envs
├── agents/                    # personas — standard Claude Code agent markdown
│   ├── marie.md
│   ├── noel.md
│   ├── ana.md
│   └── pablo.md
├── skills/                    # Agent Skills standard (SKILL.md + scripts/ references/ assets/)
│   ├── reports/
│   ├── eda/
│   ├── mmm/
│   └── ml/
├── workflows/                 # DSAgent extension: multi-step, gated, environment-aware runs
│   └── mmm-meridian/
│       ├── workflow.yaml
│       ├── steps/             # one markdown file per step (instructions for the persona)
│       └── templates/         # report/notebook templates
├── envs/                      # DSAgent extension: execution environments
│   └── meridian/
│       └── Dockerfile
├── hooks/hooks.json           # optional, Claude Code hooks
└── .mcp.json                  # optional, MCP servers the cartridge needs (BigQuery, etc.)
```

### 3.1 Personas (`agents/*.md`)

Plain Claude Code agent files. The frontmatter is the contract; the body is the persona's system prompt (identity, principles, communication style — BMAD-style, but without the menu-code ceremony).

```markdown
---
name: noel
description: Senior data scientist. Owns modeling, feature engineering, validation and ML deployment decisions. Invoke for anything predictive or causal beyond descriptive analysis.
model: sonnet
skills: [reports, eda, ml]
tools: [read_file, write_file, execute, task]
---
You are Noel, a senior data scientist with 20 years across CPG, energy and biotech.
You reason from the data-generating process first, models second. You never report a
metric without its uncertainty. You write all artifacts in English, and you push back
when a request skips the validation step.
```

In Claude Code this file registers `@dsagent-cartridge-ds:noel`. In DSAgent it becomes a Deep Agents subagent whose `skills` list is exactly the frontmatter list — nothing is inherited implicitly.

### 3.2 Skills (`skills/*/SKILL.md`)

Unchanged Agent Skills standard: `name` + `description` in frontmatter, progressive disclosure (metadata at startup, body on activation, `scripts/` and `references/` on demand). The existing v1 loader already parses this; it is ported as-is.

Scoping is declared once, in `cartridge.yaml`, and mirrored in each persona's frontmatter (the harness validates that both agree, so the cartridge stays honest when opened in Claude Code):

```yaml
# cartridge.yaml
name: ds
version: 0.1.0
description: Data science team — analysts, scientists and MMM specialists.
extends_plugin: .claude-plugin/plugin.json

personas:
  marie: { role: "Data Analyst",        skills: [reports, eda] }
  noel:  { role: "Data Scientist",      skills: [reports, eda, ml] }
  ana:   { role: "MMM Lead",            skills: [reports, mmm] }
  pablo: { role: "MMM Data Engineer",   skills: [reports, mmm, eda] }

skills:
  reports: { scope: all }              # traversal: every persona gets it
  eda:     { scope: [marie, noel, pablo] }
  mmm:     { scope: [ana, pablo] }
  ml:      { scope: [noel] }

workflows:
  - workflows/mmm-meridian
  - workflows/eda-to-report

envs:
  default:  { kind: kernel, requirements: [pandas>=2, matplotlib, markdown] }
  meridian: { kind: docker, build: envs/meridian, gpu: optional }
```

`scope: all` is what makes `reports` traversal. The harness resolves the matrix at load time and refuses to start if a persona claims a skill the matrix denies it.

### 3.3 Workflows (`workflows/*/workflow.yaml`)

Workflows are the part no harness gives you and the reason DSAgent exists. A workflow is a DAG of steps; each step names the persona that runs it, the environment it needs, the artifacts it must produce, and whether a human gate sits after it.

```yaml
name: mmm-meridian
description: End-to-end marketing mix model with Google Meridian, from raw media/sales data to a budget-optimization report.
inputs:
  data_source: { type: connector, options: [bigquery, csv, parquet] }
  kpi:         { type: string }
  date_range:  { type: daterange }
env: meridian                            # default env for every step unless overridden

steps:
  - id: ingest
    persona: pablo
    env: default                         # light work, kernel is enough
    instructions: steps/01-ingest.md
    produces: [data/raw.parquet, artifacts/data-profile.md]

  - id: data-gate
    persona: pablo
    env: default
    instructions: steps/02-data-gate.md
    produces: [artifacts/data-gate.md]
    gate:
      kind: human                        # HITL: stop and ask before spending GPU time
      prompt: "Data gate report ready. Approve to build the model spec?"

  - id: model-spec
    persona: ana
    instructions: steps/03-model-spec.md
    needs: [data-gate]
    produces: [artifacts/model-spec.md, model/spec.py]
    gate: { kind: human, prompt: "Review priors, adstock and saturation choices." }

  - id: fit
    persona: ana
    instructions: steps/04-fit.md
    needs: [model-spec]
    produces: [model/posterior.pkl, artifacts/diagnostics.md]
    timeout: 4h
    gate:
      kind: auto                         # machine gate: convergence check
      check: scripts/check_rhat.py       # exit 0 = pass

  - id: optimize
    persona: ana
    instructions: steps/05-optimize.md
    needs: [fit]
    produces: [artifacts/budget-optimization.md, artifacts/optimizer.html]

  - id: report
    persona: marie                       # a persona outside the MMM scope, using `reports`
    env: default
    instructions: steps/06-report.md
    needs: [optimize]
    produces: [report/mmm-report.html]
```

**A step only sees the inputs it uses.** By default the runner shows a step exactly the workflow inputs its own instruction text interpolates — a step that never writes `{question}` is never told the question, and it does not appear in the prompt's `## Inputs` block either. `sees: [name, ...]` on a step overrides that in either direction (`sees: []` shows nothing). Run 001 is why: with every input visible to every step, the profiling persona read the analysis question and answered it, then the gate step re-derived a verdict the profile had already written — about a third of the run spent on work that belonged to later steps. Naming an input in prose is not enough to see it; interpolate it or declare it. `dsagent cartridge validate` lists steps of an input-taking workflow that end up seeing nothing, which is usually that mistake.

**Invocation — same rule as BMAD.** BMAD exposes both agents and workflows as named skills (`bmad-agent-pm` loads a persona; `bmad-prd` runs a workflow) and a loaded persona can start any workflow from its menu. DSAgent does the same, so these three are equivalent and all end in the runner: the user types `/ds-mmm` (a command skill the harness auto-generates as `skills/ds-mmm/SKILL.md` from `workflow.yaml`, which is also what Claude Code sees); the user says "Ana, hazme un MMM" and Ana — whose `cartridge.yaml` entry lists `workflows: [mmm-meridian]` — calls `run_workflow`; or the orchestrator recognises the intent and routes it. A persona can only start workflows it is listed on, exactly like skill scope:

```yaml
personas:
  ana:   { role: "MMM Lead", skills: [reports, mmm], workflows: [mmm-meridian] }
  marie: { role: "Data Analyst", skills: [reports, eda], workflows: [eda-to-report] }
```

Step instructions are markdown, written for the persona, and are injected as the task message when the harness calls `task(persona, …)`. `produces` is verified on disk after every step; a missing artifact fails the step before the next one starts. An entry may be a glob (`artifacts/figures/*.png`) for a step that writes a variable number of files; a pattern is satisfied by at least one match and is otherwise treated exactly like a missing file, and everything it matches counts as a deliverable. Gates come in two kinds: `human` (a Deep Agents interrupt that surfaces over CLI/WebSocket) and `auto` (a script the harness runs in the step's env).

### 3.4 Environments (`envs/*`)

An environment is where a step's `execute` calls land. Two kinds in v2.0:

`kernel` is the persistent Jupyter kernel ported from v1 (`kernel/local.py`, `kernel/introspector.py`). Variables survive across steps and across chat turns; it is the right home for EDA and reporting.

`docker` builds (or pulls) an image and runs the step inside it, mounting the run's workspace at `/workspace`. A built image is tagged `dsagent/<cartridge>-<env>:latest`, which is why `EnvSpec` carries the cartridge that declared it: two cartridges may both declare an env called `meridian`, and without the namespace the second build silently replaces the first. Every `docker` invocation is a list of strings built by a function that runs nothing, and the thing that runs them is injected — so the mounts, the flags and the cleanup are ordinary unit tests, and only the test that genuinely needs a daemon needs one (`DSAGENT_DOCKER=1`). It implements Deep Agents' `SandboxBackendProtocol`, which only requires `execute()` (plus upload/download for file transfer); the base class derives read/write/ls/glob/grep from that. Meridian is the canonical case: `google-meridian` on Python 3.11/3.12 with the JAX backend, GPU strongly recommended for fits (`[and-cuda]` extra on Linux), CPU acceptable for small models and for the optimizer/report stages.

```dockerfile
# envs/meridian/Dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "google-meridian" pandas pyarrow plotly
# GPU variant: build-arg CUDA=1 → pip install "google-meridian[and-cuda]"
WORKDIR /workspace
```

An env declares the Python distributions it must provide:

```yaml
envs:
  default: { kind: kernel, requirements: [pandas>=2, matplotlib, markdown] }
```

`requirements` are opaque pip requirement strings. The harness never interprets them — which is what keeps invariant 1 intact: that `pandas` is needed is the `ds` cartridge's business, not the harness's. A kernel env verifies them when it is provisioned and refuses to start with the exact `pip install` command for whatever is missing, so a run fails before any persona is invoked rather than three tool calls into step 1. `dsagent cartridge install <path>` installs every kernel env's requirements into the interpreter running DSAgent. Docker envs declare theirs for documentation only: the image installs them and the harness only validates that the field is well formed.

The check is distribution metadata, not `import`: what a cartridge declares is what pip installs, and the import name is frequently not the distribution name (`scikit-learn` imports as `sklearn`). Resolving that would mean the harness carrying a table of package knowledge. Version specifiers are pip's business at install time.

Because the sandbox protocol is provider-agnostic, `kind: modal` or `kind: langsmith` can be added later without touching workflows.

#### What a Docker env has to honour (M2.3 design)

M2.5 and M2.6 added three things to the harness that were written against the kernel env, where "inside the env" and "on this machine" are the same place. In a container they are not, and each has to be settled before the backend is worth testing.

**The workspace is a bind-mount, never a copy.** Everything else here depends on it. The run directory is the single record a run keeps — `produces` is verified on the host after a step, the files endpoint serves from it, `show_chart` names a file in it, the zip is built from it — so a container that writes to its own copy produces a run whose evidence does not exist. `-v <run>/workspace:/workspace -w /workspace`, and the step's files are on disk the moment it writes them.

**Skills are already inside it.** `materialize_skills` copies each persona's granted skills to `<workspace>/.dsagent/skills/<persona>/`, which is *within* the bind-mount, and `run_skill_script` runs `<env.python> <path relative to the workspace root>` through `env.backend.execute()` with the container's working directory set to the mount. So the tool needs no change at all and no second mount to be reachable — a mount "at the same absolute path" would be solving a problem that does not exist. What is worth adding is the *read-only* guarantee: a second bind of the same directory with `:ro` over the first, so a persona cannot rewrite the skill it was granted. That is a containment property, not a plumbing one.

**Auto-gate scripts are not inside it.** `gate.check` resolves against the workflow directory (`wf.path / gate.check`) — in the cartridge, outside the workspace — and `_auto_gate` runs it with `subprocess.run(["python3", …])` on the host, ignoring the env entirely. In a Docker env that is the wrong machine, the wrong interpreter and the wrong dependencies: an R-hat check that needs the fitted posterior needs the image that produced it. The gate script is therefore **materialized like a skill** — copied to `<workspace>/.dsagent/gates/<workflow>/` when the run starts — and run through `env.backend.execute()` with `env.python`, in the step's own env. Materializing rather than mounting the cartridge keeps one rule for both env kinds: the kernel env runs the identical command and nothing about it changes.

**Chart validation stays on the host, and says so when it cannot happen.** `show_chart` writes nothing into the env: it reads a table the step already wrote and checks a Vega-Lite spec with `altair` and `vl-convert` in the harness's own process. That is correct — validation belongs where the answer is recorded, not where the data was computed — and it is another reason the workspace must be shared rather than copied. The consequence is that the *harness* needs the `[ui]` extra whenever a cartridge draws charts, even if every step runs in Docker. Today `validate_spec` returns "no problem" when either package is missing, which turns a missing install into silently unvalidated charts. It must be loud instead: a warning in `runner.log` the first time a chart is recorded unvalidated, and a note from `dsagent cartridge validate` when the harness cannot validate what this cartridge's personas are told to emit.

**One container per run, removed on the way out.** A container is provisioned on the first step that needs its env and reused by every later step in the same run — a fit and the optimizer that reads its posterior are the same machine — and removed in a `finally` on exit, failure or stop. Resuming after a server restart starts a fresh container: the workspace is on disk and the container held nothing that was not also there. The container id and image go into `runner.log` and onto a `dsagent.env` event, because "which image produced this" is part of what makes a run auditable.

## 4. Harness (`dsagent-core`)

The harness has five modules and deliberately nothing else.

**`cartridge/`** — loads one or more cartridge directories, parses `plugin.json` + `cartridge.yaml`, validates the persona↔skill matrix, discovers workflows and envs. Cartridges can be stacked; later ones can add personas and workflows but cannot silently override an earlier persona's skill scope (conflict = error).

**`host/`** — builds the Deep Agents graph. One `create_deep_agent` call: the orchestrator's `system_prompt` is generated from the cartridge (roster of personas, their roles, available workflows), `subagents=[…]` is one dict per persona (`name`, `description`, `system_prompt` from the agent body, `skills` from its frontmatter, `model` from frontmatter or default), `backend` is the env of the current step, `interrupt_on` carries the gates. Workflows are exposed as a `run_workflow(name, inputs)` tool (to the orchestrator and to every persona listed on that workflow) plus `list_workflows()`, and as auto-generated command skills (`/ds-<workflow>`) so the same entry point exists in Claude Code.

Skills are mounted for the file tools at a **virtual** path, `/skills/<persona>/`, backed by the copies materialized under `<workspace>/.dsagent/skills/<persona>/`. That path exists only inside the backend's file tools: `execute` and `run_python` run in the env, where it does not resolve. So a skill's *scripts* are run through **`run_skill_script(skill, script, argv)`**, a tool every persona agent is given. It resolves the script inside the persona's own materialized skill directory, runs it with the env's interpreter (`Env.python` — DSAgent's own interpreter for the kernel env, not whatever `python3` is on PATH, since that is where the cartridge's requirements are installed) from the workspace root, and returns the exit code and output. Resolving it there also enforces the matrix: naming a skill the persona was not granted is an error, not a path that happens not to exist. Step instructions must never spell out a skill-script path.

**`runner/`** — executes a workflow: topological order over `needs`, one `task()` per step aimed at the right persona, env switch per step, `produces` verification, gate handling, resumability (a run is a directory with `run.json` state; re-running continues from the last completed step). This is the successor of v1's `core/planner.py` + `core/executor.py`, but it never plans — the DAG is declared, not invented.

Every step also records what it actually did, into its `run.json` entry: tool calls counted by name, `input_tokens`/`output_tokens`, the `SKILL.md` files it read, and the workspace files it created or modified with their mtimes. It is read back from LangChain's standard message surface (`tool_calls`, `usage_metadata`) and from the workspace's own mtimes, so no provider-specific code enters the harness and a message carrying neither simply contributes nothing. This is what makes a run readable after the fact — which persona reached for which skill, what it cost, and in what order artifacts appeared (the last of which is also the input to the canvas design in `docs/ui.md`).

`run.json` is the record *after* the fact; while a step runs the runner also streams what it sees to an optional `on_event` callback, as `RunnerEvent`s named `dsagent.step`, `dsagent.tool` and `dsagent.file`. Each step event carries that step's `produces`, which is the declared contract the runner already verifies on disk, so a consumer can tell a deliverable from a working file without guessing from extensions. This is why a step's persona is executed with `.stream(stream_mode=["updates", "values"])` rather than `.invoke()`: steps run for minutes, and a consumer that only hears from one when it ends cannot show progress through the part worth watching. The last `values` chunk is the same final state `.invoke()` would have returned, so telemetry and `produces` verification are unchanged. Nothing in the runner knows about AG-UI or HTTP — `docs/ui-slice.md` describes the adapter that turns these into AG-UI `CUSTOM` events, and the payload schemas are pinned there.

**`envs/`** — `KernelBackend` (ported) and `DockerBackend` (new), both behind the sandbox protocol.

**`cli.py` / `serve.py`** — the two front ends. `cli.py` is `dsagent cartridge validate|list`, `run`, `chat` and `serve`. `serve.py` is the HTTP surface, and it replaces the v1 FastAPI + WebSocket server rather than porting it: the transport is AG-UI over SSE, via `ag_ui_langgraph.add_langgraph_fastapi_endpoint` and `copilotkit.LangGraphAGUIAgent`, so there is no hand-rolled protocol to maintain (`docs/ui.md`). It mounts the orchestrator at `POST /agent` and serves a run's workspace at `GET /runs/{run_id}/files/{path}`, resolved inside the workspace and refusing anything that escapes it or lives under `.dsagent/`.

The two front ends differ in exactly two ways, and share `runner/tools.py` for everything else: how a gate is asked, and where runner events go. In the terminal a gate is `typer.confirm` and events are printed; under `serve` a gate is a LangGraph `interrupt()` carrying the payload in `docs/ui-slice.md` §3, and events go out through `dispatch_custom_event`. Interrupts need a checkpointer, so the served graph is built with one (in-memory for the M2.2 slice); `run.json` remains what survives a process restart, and the checkpointer only has to outlive a gate answer. `build_orchestrator` takes `middleware` and `checkpointer` so that wiring lives in `serve.py` and the harness core never imports either package — the `ui` extra is optional.

Everything in v1's `core/engine.py` (924-line hand-rolled LLM loop), `core/planner.py`, `core/context.py`, `prompts/` and `memory/summarizer.py` is dropped: Deep Agents provides the loop, the todo/planning middleware, context management and summarization.

### 4.1 Mapping to Deep Agents primitives

| DSAgent concept | Deep Agents primitive |
|---|---|
| Persona | entry in `subagents=[…]` with `name`, `description`, `system_prompt`, `skills`, `model` |
| Skill scope | `skills=[…]` on the subagent; orchestrator gets only `scope: all` skills |
| Workflow step | `task(persona, instructions)` issued by the runner, not by the model |
| Human gate | `interrupt_on` / LangGraph interrupt, surfaced by the API layer |
| Environment | `backend=` (KernelBackend or DockerBackend via `SandboxBackendProtocol`) |
| Run workspace | backend filesystem rooted at the run directory |
| Session memory | Deep Agents `memory` (AGENTS.md-style) per run + v1 session store for transcripts |

### 4.2 Why Deep Agents and not the Claude Agent SDK

Both would work. Deep Agents wins for this project on three counts: it is model-agnostic (v1 users run OpenAI, Gemini and Ollama through LiteLLM and that portability should survive), its sandbox protocol is the cleanest fit for "Docker per workflow", and its skills loader already speaks the same `SKILL.md` standard as Claude Code, so a cartridge round-trips without translation. The cost is a LangGraph dependency and a younger API; the harness is small enough that a later swap is a rewrite of `host/`, not of the cartridges.

## 5. The DS cartridge (v0.1 scope)

Personas: **Marie** (Data Analyst: descriptive analysis, dashboards, stakeholder reports), **Noel** (Data Scientist: modeling, validation, ML), **Ana** (MMM Lead: model spec, priors, fit, optimization), **Pablo** (MMM Data Engineer: ingestion, data gates, connectors). Ana and Pablo are placeholders for the names in the original sketch; the roles are what matter.

Skills: `reports` (traversal — report structure, executive summary rules, chart conventions, templates), `eda` (profiling, null/outlier gates, the v1 EDA skill upgraded), `mmm` (Meridian usage, adstock/saturation guidance, prior elicitation checklist, optimizer interpretation), `ml` (feature engineering, CV strategy, leakage checks, model cards).

Workflows: `mmm-meridian` (above) as the flagship; `eda-to-report` (ingest → profile → gate → report) as the two-persona smoke test that runs entirely in the kernel env and exercises every harness feature except Docker.

## 6. Migration from `nmlemus/dsagent`

| v1 module | Fate |
|---|---|
| `kernel/*` | Port → `envs/kernel.py` behind the sandbox protocol |
| `skills/loader.py`, `registry.py`, `models.py` | Port → `cartridge/skills.py` (add scope validation) |
| `skills/installer.py` | Port → `dsagent cartridge add` (git/URL install) |
| `session/*`, `server/*` | Port with light changes (runs replace sessions as the unit) |
| `core/hitl.py` | Replace with Deep Agents interrupts; keep the WebSocket surface |
| `core/engine.py`, `planner.py`, `context.py`, `prompts/*`, `memory/*` | Drop |
| `tools/mcp_manager.py` | Replace with cartridge `.mcp.json` + LangChain MCP adapters |
| `utils/notebook.py` | Port → `reports` skill script (notebook export is a skill concern now) |

Package name stays `datascience-agent` on PyPI with a 2.0 major; the CLI keeps `dsagent`.

## 7. Milestones

**M1 — Harness skeleton (kernel env only).** Cartridge loader, persona→subagent wiring, skill scoping, `eda-to-report` workflow end-to-end with a human gate, CLI. Proves the cartridge model.

**M2 — Docker env + Meridian.** `DockerBackend`, `mmm-meridian` workflow running against the Meridian sample dataset, auto-gate on R-hat, HTML report.

**M3 — Portability proof.** Install the DS cartridge as a Claude Code plugin and run Marie/Noel there with no changes; ship a minimal BMAD-style cartridge (three SDLC personas, one workflow) to prove the harness is domain-agnostic.

**M4 — API + connectors.** FastAPI/WebSocket parity with v1, BigQuery connector through `.mcp.json`, run resumability across restarts.

## 8. Open questions

Persona names for the MMM pair (Ana/Pablo are placeholders). How to price/meter Docker GPU time per run for the Aiuda Labs delivery engine (relevant because the v1 dogfood tracked cost per engagement).

---

*Sources consulted for current facts: LangChain Deep Agents docs (overview, skills, subagents, sandboxes), Claude Code plugins reference, BMAD Method docs (skills-and-agents reference), Google Meridian install guide.*
