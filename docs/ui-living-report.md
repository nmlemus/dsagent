# M2.6 — The living report: DSAgent's product UI

**Date:** 2026-09-17 (builds on M2.5, PR #75: runs API, event log, cost, persistence) · **Status:** spec,
pending Noel's approval of the HTML mockup · **Base:** `v2` after PR #75 (M2.5) is merged ·
**Research:** `docs/research/ui-landscape-2026-09.md` — read it first; every choice below
cites it.

## 0. The idea

Every AI data product is chat + notebook or chat + dashboard, and PNG charts read as
dated. Nobody shows a *team*, nobody gates *before* execution, nobody makes a run
auditable, nobody shows what each step cost. Those four are exactly what the harness
already does. The UI's job is to make them visible and pleasant, and to fix the one thing
that is below table stakes: charts and tables must be interactive objects the agent
emits, not images.

**The run is a living report.** When an operator starts a run they are opening a
document that the team writes in front of them, section by section, while a rail on the
left shows who is working, on what, what it costs, and where it needs a decision. Charts
in that document are Vega-Lite objects: hover, zoom, filter, change type, "ask Ana to
change this". When the run ends, the document *is* the deliverable — interactive,
versioned, shareable, exportable — and the same run can be replayed step by step.

Three principles, from the research: separate the conversation from the activity (the
chat is never the only view of a run); progressive disclosure (summary → step → tool
calls → raw, never a wall of logs, never a blank screen); every approval says exactly
what is approved.

## 1. Screens

### 1.1 Home — runs
A list of runs (workflow, dataset, status badge `running / awaiting input / done /
failed`, started, duration, cost, personas involved as small avatars), newest first,
with a live progress chip on running runs. Primary action **New run**. Empty state: a
one-paragraph explanation and the New run button. Secondary: a Workflows tab that shows
each workflow of the loaded cartridge as a card (personas, steps, gates, what it
produces) — this is what a stakeholder reads to understand what the team can do.

### 1.2 New run — drop data, get a plan
No wizard. One surface: a drop zone for a file (CSV/parquet/xlsx) or a path, and a text
box "What do you want to know?". On drop, the app profiles the file locally (rows, columns,
dtypes, a 20-row preview) and the orchestrator proposes: the workflow, the inputs
(`key_column` guessed from unique columns, `question` from the text), the personas that
will work and the gates that will ask for a decision, and an estimated cost from the last
runs of that workflow. The operator edits inline and presses **Start**. The proposal is
the *plan artifact* (research C: plan-first gate); it is stored with the run.

### 1.3 Run — the living report
Two regions plus a drawer.

**Team rail (left, 320 px).** The run header: workflow, dataset, status badge, elapsed,
cost so far with cached share, tokens. Below, the step list as named phases (never a
percent bar): each step shows persona avatar and name, status, elapsed, and, expanded,
its tool-call count, the `produces` promises with ✓/○/✗, its narration log (persona
"thinking", collapsed by default), its cost. The step that is running pulses; a gate
renders **inline at its step** as a card (see 1.5). A hand-off marker between steps
shows what one persona passed to the next (the produced artifacts). At the bottom, the
conversation: a compact chat with the orchestrator, for questions about the run ("why did
the gate flag fog?") — it never shows persona narration and never starts a new run
without saying so.

**Document (center, fluid).** The report being written. It starts with the plan
(objective, dataset summary, the steps to come, greyed) and fills in as steps finish:
each step contributes a section (profile → data summary and quality; gate → the checks
table with the verdict; analyze → findings with charts; report → executive summary at
the top). Sections arrive with a subtle reveal, not a re-render. Every chart is a
`<ChartCard>` (1.4), every table a `<TableCard>`. Each section has "ask about this" and
a comment box that sends a message to the orchestrator anchored to that section. A
"Sources" strip under each finding lists the artifacts and code it came from (click →
drawer). While a step is running, its future section shows the persona at work: name,
current tool, live counter ("run_python × 4 · reading eda skill…").

**Drawer (right, collapsible, 420 px).** Progressive-disclosure detail for whatever was
clicked: a file (the M2.2 viewers), a step's tool calls with arguments and outputs, the
code a chart came from, the run log. This is where raw lives; it is never the default.

### 1.4 Charts and tables — the core of the milestone
Personas stop writing PNGs. They call two harness tools:

```python
show_chart(spec: dict, data_ref: str, title: str, chart_id: str | None = None,
           section: str | None = None) -> {"chart_id", "spec", "data_url", "rows"}
show_table(data_ref: str, title: str, columns: list[str] | None = None,
           table_id: str | None = None, section: str | None = None) -> {...}
```

`spec` is Vega-Lite v5 with `data: {"name": "table"}`; `data_ref` is a parquet or CSV
written by `run_python` into the workspace. The harness validates the spec against the
Vega-Lite schema (altair ships it), compiles it once with altair as a smoke test, and on
failure returns the validator message to the persona for repair (max 3 rounds — the
VegaChat loop, research B). Rows are never inlined by the model; the tool result carries
`data_url` (the files endpoint) and `rows`. Specs and their versions are kept in graph
state `charts: {chart_id: {spec, data_ref, version, section}}` so they survive resume and
reach the frontend via AG-UI state.

Frontend `<ChartCard>`: `react-vega` / `vega-embed` with tooltips on, `bind: "scales"`
pan-zoom for quantitative axes, brush selection where the spec allows, a toolbar (chart
type among the mark's sensible alternatives, aggregate, filter by a categorical field,
download PNG/SVG, view spec, view code) and **"Ask <persona> to change…"** which sends a
message with the `chart_id`. Local toolbar edits patch the spec on the client and post
a `STATE_DELTA` so the agent and the report see the user's version. Brush selections post
"user selected N rows where …" as context on the next message. `<TableCard>`: TanStack
Table + react-virtual, sort/filter/column chooser, loads parquet with hyparquet (first
50k rows, "load more"), a **Pivot** button that mounts Perspective on the same Arrow
buffer on demand. Both cards render a skeleton while the tool call streams.

Plotly is the escape hatch: `spec["$schema"] == "plotly"` renders with `plotly.js-
cartesian`, lazy-loaded, for 3D/geo. The `reports` skill's HTML renderer embeds
`vega-embed` plus the final specs and a data snapshot so the exported report stays
interactive; `vl-convert` produces PNG/SVG for PDF.

### 1.5 Gates that say what they approve
A gate card shows: the step and persona, the message, **what exactly is being approved**
(the artifacts of the step, rendered inline: the data-gate table, the model-spec
summary), the cost so far and the estimated cost of what follows, and, on re-entry after
a rejection, a diff between the previous artifact version and the new one. Approve /
Reject with a note. The header shows how long the gate waited once answered. The plan
artifact records every decision with who and when (the run's audit trail, 1.6).

### 1.6 Run finished — the deliverable
The document gets its executive summary at the top, a summary card (delivered
artifacts, duration, cost, gates and their wait), **Share** (copy link; viewers see the
same document), **Export** (interactive HTML — the report the `reports` skill already
renders, now with live charts — PDF via vl-convert, a zip of all artifacts, the run log
as JSON), and **Replay**: a scrubber that replays the run's event log at 10× on the same
screen — the auditable-run story, and the best demo tool we will have. Versions: every
run of the same workflow on the same dataset lists as a version; a diff view compares
findings and charts between two runs.

### 1.7 Failure, rejection, resume
A failed step shows the persona, what was promised and missing, and the error in plain
words; "Retry from this step" uses the runner's resume. A rejected gate shows the note
and "Resume", which reopens the gate. Stopping a run is a first-class button and never
costs more than what already ran.

## 2. Visual system

The Aiuda Labs system as the product skin: cream `#FAF8F4` background with the subtle
noise overlay; navy `#142850` for text and the rail; orange `#E8440A` only for the
primary action and the running step; emerald `#0A7B5A` for done/pass; a muted red for
fail. Satoshi 500/700/900 for UI, Instrument Serif for report and section titles,
JetBrains Mono for paths, ids, numbers and code. Fonts self-hosted in `ui/public/fonts`
(client data on screen; no third-party requests). One `tokens.css`; no Tailwind; restyle
CopilotKit through its CSS variables so nothing reads as default. 8-pt spacing, generous
whitespace, visible keyboard focus, readable at 1280 px and usable at 1024 px. Light
mode only. Motion: section reveal and chart mount use a 200 ms fade/slide; the running
step pulses; nothing else animates.

## 3. Backend the UI needs (harness, tested, domain-agnostic)

3.1 **Chart/table tools** (`envs` contribute them like `run_python`): `show_chart`,
`show_table` as in 1.4, with Vega-Lite validation (`altair` + `jsonschema` added to the
`[ui]` extra; the kernel env requires them via the cartridge's `requirements`), repair
loop, `charts` in graph state, and a `dsagent.chart` runner event so the CLI and the
event log carry them too. The harness knows nothing about what a chart *means*.

3.2 **Sections**: `dsagent.step` events gain `section` (from a new optional
`Step.section` in `workflow.yaml`: title + order); the document is reconstructed from
step + chart + file events. Cartridge change: `eda-to-report` declares sections and its
`analyze`/`report` steps say "use `show_chart` for every figure; PNGs only as a fallback
for PDF".

3.3 **Runs API**: `GET /runs`, `GET /runs/{id}` (run.json + gates + totals + plan),
`POST /runs` (multipart upload into `data/`, local profile), `POST /runs/{id}/plan`
(ask the orchestrator for the proposal, no execution), `POST /runs/{id}/start`,
`POST /runs/{id}/stop`, `GET /cartridges`.

3.4 **Event log per run**: runner appends every event to `<run_dir>/events.jsonl`;
`GET /runs/{id}/events` (full) and `?after=n` as SSE. The UI restores from the log on
load and follows live; replay reads the same log. Reload, second tab and CLI runs all
show the same thing.

3.5 **SQLite checkpointer** for the served graph so a gate can be answered after a
restart (`--memory` keeps the in-memory saver).

3.6 **Cost**: `prices.yaml` (model → input/cached/output rates) loaded by serve;
`cost_usd` per step and total in `run.json` and in the run header; an estimate on the
plan from the last three runs of the workflow.

3.7 **Exports**: `GET /runs/{id}/export.html` (the report with live charts),
`export.pdf` (vl-convert fallbacks), `export.zip`, `events.json`.

Each is a PR-sized commit with tests in the existing style (fake agent, stub env,
TestClient). No change to invariants, runner contract or cartridge format beyond the
optional `Step.section`.

## 4. Out of scope

A2UI / json-render panels (a follow-up once ChartCard/TableCard exist as catalog
components); MCP Apps; auth/multi-user; Docker env; mmm-meridian; dark mode; mobile;
i18n; editing artifacts in place; replacing CopilotKit; scheduling.

## 5. Working rules for the autonomous run

- Branch `m26-living-report` off `v2`. One commit per task from §7, messages
  `<area>: <what>`. Push after every task. One PR at the end; never merge to `v2`.
- Green before every commit: `pytest`, `ruff check src tests`, `dsagent cartridge
  validate`, `npm run build`, `npm run typecheck`, `npm run lint`.
- **Real-model budget: 10 runs** for the whole milestone (≈$0.5–1 each), only
  `eda-to-report` on `tests/data/seattle-weather.csv`. The first real run happens after
  task 3 (to record a fixture that contains chart events); the rest are for §6 evidence
  and for at most two prompt iterations on the `reports`/`eda` skills to make personas
  use `show_chart` well. Everything else develops against `serve --replay`.
- Browser verification against `next build && next start`; screenshots per task into
  `docs/runs/ui-product/`.
- Keep `docs/ui-product-log.md` (per task: what, verified how, decisions, deferred) and
  the ROADMAP decisions log.
- Write `BLOCKED.md` and stop if: a §3 change would need a runner-contract or invariant
  change; a required CopilotKit/AG-UI/Vega capability does not exist; the budget is
  exhausted before §6 passes; a task fails three attempts. Otherwise do not wait.

## 6. Definition of done — the demo

From a clean checkout with a real model; every line true, evidence (screenshot or log)
in `docs/runs/ui-product/DEMO.md`.

1. Home lists runs with status, duration, cost and personas; Workflows tab shows the
   team and the gates; empty state on a fresh install.
2. New run: drop `seattle-weather.csv`, type the question → the proposal appears with
   workflow, inputs, personas, gates and a cost estimate → Start. No chat typing.
3. Within 5 s the document shows the plan and Marie at work in the first section; the
   rail shows step 1 running with a live tool counter. Nothing is blank.
4. Sections fill in as steps finish, with a reveal, without losing scroll position.
5. `analyze` produces **at least three interactive Vega-Lite charts**: hover shows
   values, a quantitative chart pans/zooms, changing the mark type from the toolbar
   re-renders locally, a brush selection posts context to the chat, and "Ask Noel to
   change this" yields a modified chart within the run's conversation.
6. A table card shows the profile with sort/filter; Pivot opens Perspective on it.
7. The gate card appears inline at `data-gate` with the checks table rendered as what is
   being approved and the cost so far/estimate; Approve continues; the header later shows
   the wait.
8. Reload mid-run: the whole screen restores and keeps following.
9. The run ends **with no error**; the executive summary lands at the top; Export HTML
   opens a report whose charts are still interactive; Export zip contains every
   deliverable; Replay scrubs the run at 10×.
10. Ask in the chat "which finding should I be most careful with?" — answered from the
    artifacts without a new run.
11. Reject a gate with a note on a second run, Resume, Approve — done; the rejection is
    in the audit trail; the gate card showed a diff of `data-gate.md` on re-entry.
12. Kill `dsagent serve` at a gate, restart, answer — the run completes.
13. Every screen is in the Aiuda visual system; self-hosted fonts; no default CopilotKit
    or browser control visible.

## 7. Task order

Tasks 1 and 2 were delivered by PR #75 (M2.5): event log + replay fixture, runs API,
SQLite checkpointer, `prices.yaml` cost, launcher, home ledger, stepper, gate card,
Aiuda tokens and self-hosted fonts. Do not rebuild them — extend them. What M2.5 did
*not* deliver and this milestone owes: `show_chart`/`show_table` and every chart as
Vega-Lite (figures are still PNG `<img>`), the living document (canvas is still a file
list), `<ChartCard>`/`<TableCard>` with local edits and brush-to-context, plan proposal
before start, sections, exports with live charts, share, versions/diff. Missing from
§3.3 as delivered: `POST /runs/{id}/plan`, `POST /runs/{id}/stop`, `GET /cartridges`
(check `src/dsagent/api.py` first).

1. ~~Event log~~ — done in M2.5.
2. ~~Runs API, checkpointer, cost~~ — done in M2.5; add `/plan`, `/stop`, `/cartridges`.
3. `show_chart` / `show_table` tools with validation and `charts` state (§3.1); cartridge:
   sections and "use show_chart" in `eda-to-report` (§3.2). Then **real run 1** to
   record a fixture with charts; iterate the skills at most twice.
4. App shell: keep M2.5 tokens/fonts; re-lay the run screen as rail / document / drawer
   (see the HTML mockup `docs/mockups/living-report.html`); home, workflows tab.
5. Document: sections from events, reveal, persona-at-work placeholder, sources strip,
   ask-about-this.
6. `<ChartCard>` and `<TableCard>` with toolbar, local edits → state delta, ask-to-change,
   Perspective on demand.
7. Team rail: steps, promises, narration, cost, hand-offs, inline gate card with
   what-is-approved, diff on re-entry, wait time.
8. New run: drop zone, local profile, plan proposal, start; stop; failure/resume
   presentation.
9. Finished run: summary, share, exports (HTML with live charts, PDF, zip), replay,
   versions/diff.
10. Demo script with evidence, DEMO.md, log, PR.
