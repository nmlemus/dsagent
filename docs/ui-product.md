# M2.5 — UI product pass ("sellable")

**Date:** 2026-09-17 · **Status:** spec, approved for autonomous execution · **Base:** `v2` after M2.2.1 items 1–3.

## 0. Why

M2.2 proved the mechanism: events, gates, canvas, DAG. It looks like a prototype. M2.5
turns it into something a stakeholder can watch for ten minutes, understand without a
narrator, and want. "Sellable" is defined in §7 as a demo script with acceptance checks —
that is the exit condition, not a feeling.

Everything in `CLAUDE.md` still holds. The harness stays domain-agnostic; the cartridge
owns the domain; Deep Agents owns the loop. This milestone is UI plus the minimum backend
the UI needs (§4). No A2UI, no MCP Apps, no Docker — those stay where the ROADMAP has them.

## 1. Who is in front of the screen

**The operator** — an analyst at the client, or an Aiuda consultant on a call — who
starts runs, answers gates and reads results. Non-technical: never types `data_path=`.
**The stakeholder** — watches over the operator's shoulder or gets the link to a finished
run. Needs to see progress, who did what, what it cost, and the report.

## 2. Screens

### 2.1 Home — runs
A list of runs, newest first: workflow, dataset name, status (running / awaiting gate /
done / failed), started, duration, cost. A running run shows a live progress bar
(steps done / total). Click → run screen. Primary action: **New run**. Empty state
explains what a run is and points to New run.

### 2.2 New run — launcher
Step 1 pick a workflow (cards from the cartridge: name, description, personas involved,
steps and gates preview). Step 2 provide inputs: a form generated from the workflow's
declared `inputs` (`path` → file upload with drag-and-drop, `string` → text, `options` →
select, `daterange` → two dates, defaults prefilled, required marked). Step 3 confirm:
summary + "Start". The chat is not required to start a run. Uploads land in the run's
`data/` before the run starts.

### 2.3 Run — the main screen
Three regions, resizable:
- **Left: conversation.** The orchestrator's chat for this run's thread. Persona
  narration never appears here (already enforced). The chat can be used to ask about
  the run ("why did the gate flag fog?") — the orchestrator answers from the artifacts.
- **Top-right: progress.** The DAG as a horizontal stepper with the current step
  highlighted, per-step persona avatar/initial, elapsed, tool-call count, produces
  ticks, expandable narration log, and the gate card *inline at its step* when waiting.
  A run-level header: status, elapsed, tokens, cost so far, cached share.
- **Bottom-right: canvas.** Deliverables and working files; the viewers from M2.2 plus
  parquet (server-side preview endpoint, first 200 rows) and interactive HTML (Plotly)
  via `sandbox="allow-scripts"` on a separate origin path. Focus rules from M2.2 stay.
  "Download" on every file; "Download all" as zip on a done run.

Reloading the page restores the whole screen from the server (§4.2). A run started in
another tab or by the CLI is visible and attachable.

### 2.4 Run finished — report view
Same screen; the canvas opens the final report full-width with a "Report" toggle, and
the header gains a summary card: what was delivered, duration, cost, gates answered and
by how long they waited. A "Share" button copies a link to the run.

### 2.5 Failure and gates
A failed step shows *what* failed in plain words (the step's error, not a stack trace),
which persona, which artifact was missing, and a "Retry from this step" action (uses the
existing resume). A rejected gate shows the note and a "Resume" that reopens the gate.

## 3. Visual system

Apply the Aiuda Labs design system as the product skin (it is the company's canonical
system): background cream `#FAF8F4`, accent orange `#E8440A`, emerald `#0A7B5A` for
success/done, navy `#142850` for text and headers; Satoshi (weights 500/700/900) for UI,
Instrument Serif for the run/report titles, JetBrains Mono for paths, ids and numbers.
Subtle noise overlay on the app background; dark footer only on the home screen. Fonts
are self-hosted in `ui/public/fonts` (no third-party font requests — client data may be
on screen). One `tokens.css` with CSS variables; no Tailwind, no component library beyond
what CopilotKit already ships for the chat (restyle it through its CSS variables).

Rules: one accent colour used sparingly (primary actions, the running step); status
colours only for status; generous whitespace; 8-pt spacing scale; everything readable at
1280 px and usable at 1024 px; keyboard focus visible; light mode only in this milestone.
The empty canvas during the first 60–90 s of a run shows the persona at work (name,
step, live tool counter, "reading the eda skill…") instead of a blank pane.

## 4. Backend the UI needs (harness, small, tested)

4.1 **Runs API** on `dsagent serve`: `GET /runs` (list, from `run.json` files), `GET
/runs/{id}` (run.json + gate records + telemetry totals), `POST /runs` (create a run dir,
accept multipart uploads into `data/`, return `run_id`), `POST /runs/{id}/start`
(invokes the orchestrator with a run request on a new thread — the same path the chat
takes), `GET /cartridges` (workflows with their declared inputs, personas, steps, gates).

4.2 **Event log per run**: the runner appends every `RunnerEvent` to
`<run_dir>/events.jsonl`. `GET /runs/{id}/events` returns it; `GET
/runs/{id}/events?after=<n>` streams new ones (SSE). The UI reconstructs the screen from
the log on load and keeps following. This is what makes reload, second tab and CLI runs
all show the same thing.

4.3 **SQLite checkpointer** for the served graph (`.dsagent/checkpoints.sqlite`), so a
gate can be answered after a restart. Keep in-memory as a `--memory` flag.

4.4 **Parquet preview**: `GET /runs/{id}/preview/{path}?rows=200` → JSON rows/columns via
pandas (the kernel env already requires pandas through the cartridge; the endpoint
degrades to 415 when the interpreter lacks it).

4.5 **Interactive HTML**: `/runs/{id}/files/{path}` served with a `sandbox` header policy
that allows scripts only under a second route prefix `/runs-x/…` used by the iframe.

4.6 **Cost in telemetry**: `run.json` gains `cost_usd` per step and total, computed from a
`prices.yaml` the serve command loads (model → input/cached/output rates). The harness
does not hardcode prices; the file ships with the repo and is editable.

Each of these is a PR with unit tests in the existing style (fake agent, stub env,
TestClient). None changes the runner's contract, the invariants or the cartridge format.

## 5. Out of scope (do not drift)

A2UI / json-render panels; MCP Apps; auth or multi-user; Docker env; mmm-meridian;
dark mode; mobile; i18n (UI copy in English); editing artifacts in place; replacing
CopilotKit.

## 6. Working rules for the autonomous run

- Branch `m25-ui-product` off `v2`. One commit per task from §8, message `<area>: <what>`.
  Open **one PR** at the end; do not merge to `v2`. Push the branch after every task so
  progress is visible.
- `pytest`, `ruff check src tests`, `dsagent cartridge validate`, `npm run build`,
  `npm run typecheck` green before every commit. Add `npm run lint` (eslint via Next) and
  keep it green.
- Real-model runs: at most **6** in total for the whole milestone, each ≈ $0.5–1,
  always `eda-to-report` on `tests/data/seattle-weather.csv`. Use them for the
  verification runs in §7, not for development. Development uses recorded event logs:
  first task is to capture one real run into `ui/fixtures/run-eda-003.events.jsonl`
  (+ the run's workspace files) and build a `dsagent serve --replay <dir>` mode that
  serves it at 10× speed, so every screen can be developed and screenshot without a
  model.
- Browser verification against `next build && next start` (never `next dev`), with
  screenshots into `docs/runs/ui-product/` named by screen and task.
- Keep a running `docs/ui-product-log.md`: per task, what changed, what was verified,
  decisions taken and anything deferred. Add decisions to the ROADMAP decisions log.
- Stop and write a `BLOCKED.md` at the repo root if: a §4 change would need to alter the
  runner's contract or an invariant; a CopilotKit/AG-UI API required for a screen does
  not exist; the real-model budget is exhausted before §7 passes; or a task exceeds
  three attempts. Otherwise do not wait for approval between tasks.

## 7. Definition of done — the demo script

Run this script from a clean checkout (`pip install -e ".[ui,anthropic]"`, `npm ci`,
`dsagent serve`, `npm run build && npm run start`) with a real model. Every line must be
true; record the evidence (screenshot or log line) in `docs/runs/ui-product/DEMO.md`.

1. Open the app: the home screen lists previous runs with status, duration and cost; an
   empty state is shown on a fresh install.
2. New run → pick `eda-to-report` → upload `seattle-weather.csv` by drag-and-drop → the
   form shows `question` prefilled and `key_column` optional → Start. No chat typing.
3. Within 5 s the run screen shows step 1 running with Marie's activity; the canvas is
   not blank.
4. Files appear as they are written; figures append without stealing focus.
5. The gate card appears inline at `data-gate` with the gate report rendered below;
   Approve continues the run; the header later shows how long the gate waited.
6. Reload the page mid-run: the screen restores fully and keeps following.
7. The run ends **without any error**; the report opens full-width; "Download all"
   yields a zip with every deliverable.
8. The header shows tokens, cached share and cost for the run; the home screen shows
   the same cost in the list.
9. Ask in the chat "which finding should I be most careful with?" — the orchestrator
   answers from the artifacts without starting a new run.
10. Reject a gate on a second run with a note, then Resume and Approve — the run
    completes; the rejection and note are visible in the step's history.
11. Kill `dsagent serve` while a run waits at a gate; restart; the gate is still
    answerable and the run completes (SQLite checkpointer).
12. Every screen renders in the Aiuda visual system: self-hosted fonts, the four
    colours, no unstyled CopilotKit defaults, no browser default form controls visible.

## 8. Task order

1. Replay fixture + `serve --replay` (unblocks everything else).
2. Runs API + event log (§4.1, §4.2) with tests.
3. Home screen + run restore on load (from the event log).
4. Launcher (cartridges endpoint, upload, form from inputs, start).
5. Run screen layout + visual system (tokens, fonts, CopilotKit restyle).
6. Progress region: stepper, header metrics, inline gate card, narration.
7. Canvas: parquet preview, interactive HTML, downloads, zip.
8. Failure/reject/resume presentation.
9. SQLite checkpointer + cost in telemetry (§4.3, §4.6).
10. Demo script run, evidence, DEMO.md, log, PR.
