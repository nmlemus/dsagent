# Harness UI (v3) — design

**Date:** 2026-09-17 · **Author:** Noel Moreno Lemus (with Claude Code) · Design + build.
Supersedes the deleted Next.js + CopilotKit UI (M2.5/M2.6). Companion to `docs/ui-slice.md`
(event/interrupt contract) and `docs/ui-product.md` (run-belongs-to-server model).

## 1. Goal

A specialized *harness* UI on top of the existing `dsagent serve` API: simple to use, powerful
underneath. Run-centric — launch a workflow, watch it run, answer gates, read the artifacts —
with the agent able to render **dynamic** widgets (A2UI) where a fixed panel would be too rigid.

## 2. What already exists (no backend rework)

`dsagent serve` (one FastAPI app) exposes everything the UI needs:

- **Chat / AG-UI:** `POST /agent` (SSE), `GET /agent/health`.
- **Runs:** `GET /cartridges` (workflows + input schemas), `GET|POST /runs`, `GET /runs/{id}`,
  `POST /runs/{id}/start|stop|gate`, `POST /runs/{id}/plan`, `PUT /runs/{id}/data/{name}`,
  `DELETE /runs/{id}`.
- **Live:** `GET /runs/{id}/events` (SSE, rebuildable from `events.jsonl`), `/events.json?after=`.
- **Canvas:** `GET /runs/{id}/files/{path}`, `/preview/{path}`, `/log`, `/export.html`,
  `/download`, `/chart-theme`, `/gate-version/{step}/{version}/{path}`.

Events on the wire (`docs/ui-slice.md` §3): `dsagent.step`, `dsagent.tool`, `dsagent.file`,
`dsagent.note`; human gates arrive as AG-UI **interrupts** on `/agent` and as `awaiting_gate`
step status on the run event stream. Answered via `POST /runs/{id}/gate` or the interrupt.

## 3. Verified facts (grounding, not guesses)

- `copilotkit==0.1.96` (pinned in the `[ui]` extra) already ships `copilotkit.a2ui` with
  `create_surface`, `update_components`, `update_data_model`, `render`, `a2ui_prompt`,
  `BASIC_CATALOG_ID`, `DEFAULT_DESIGN_GUIDELINES`, `DEFAULT_GENERATION_GUIDELINES`, and
  `CopilotKitMiddleware(a2ui_params=…)`. A2UI **v0.9**. No backend version bump required.
- The middleware injects a `render_a2ui` tool: the model calls it with `surfaceId`,
  `components` (a tree; exactly one `id="root"`), and optional `data` (path-bound data model).
  Basic catalog components: Text, Image, Row, Column, List, Button, Card, Divider, plus
  chart/metric. Custom catalogs are host-registered via `a2ui_params`.
- Frontend transport is **open** (CopilotKit deepagents quickstart says a Node CopilotRuntime
  is required; LangChain's CopilotKit doc says the React app can point `runtimeUrl` straight at
  the FastAPI AG-UI route). **Resolved by the spike (§8, step 0)**, not by assumption.

## 4. Architecture

One **Vite + React + TypeScript SPA** in `ui/`, built to `ui/dist/`, **served by
`dsagent serve`** at `/` with SPA fallback. API paths keep their prefixes; the SPA fallback
must not shadow `/agent`, `/runs`, `/cartridges`, `/chart-theme`. One process, one URL, no CORS.

- **Dev:** `npm run dev` (Vite :5173) proxies `/agent`, `/runs`, `/cartridges`, `/chart-theme`
  to `:8000`. Production is the single server.
- **Client:** `@copilotkit/react-core` + `@copilotkit/react-ui` as the AG-UI client and A2UI
  renderer *only* — not as an app framework. `@ag-ui/client` is the fallback renderer.
- **Invariant 1 stays intact.** The static mount and the A2UI catalog are domain-agnostic
  (generic UI components; a launcher form generated from a workflow's declared inputs).
  Guidance telling a persona *when to emit which widget* lives in `cartridges/ds/`
  (agent frontmatter/body + `a2ui_params` guidelines), never in `src/dsagent/`.

## 5. Layout

Three columns, plus a chat pane.

- **Left rail** — run history (`GET /runs`), a **New Run** launcher (workflow picker from
  `GET /cartridges`; form auto-generated from declared inputs; data upload via
  `PUT …/data/{name}`; `POST /runs` → `…/start`; optional `…/plan` for a dry-run estimate).
- **Center — run view** — the workflow DAG (nodes = steps, keyed by step id; status/persona/
  env/cost from `dsagent.step`; ticks/links from `produces_matched`, not `produces`), persona
  **narration** (`dsagent.note`), and **gate cards** (approve/reject).
- **Right — canvas** — workspace files as they land (`dsagent.file`, append-only; focus
  changes only on step `done`), viewers (md, csv/parquet via `/preview`, png `<img>`, html in
  a sandboxed iframe with **empty** sandbox), the living report (`/export.html`), `/download`.
- **Chat pane** — `/agent`. Persona narration is withheld from the chat transcript (it shows
  in the run view instead); A2UI surfaces render inline here and in gate/step panels.

## 6. Data flow

- Control: REST (`/runs`, `/start`, `/stop`, `/gate`, `/data`, `/plan`).
- Live: subscribe to `GET /runs/{id}/events` (SSE). Rebuild from the log's replay, then follow —
  this is what makes reload / second tab / CLI-started runs converge (`docs/ui-product.md` §4.2).
  A reducer keyed by step id folds `dsagent.step`; `dsagent.file` appends; `dsagent.note`
  attributes narration to the step by id (the event carries it — no time-window guessing).
- Chat + A2UI: `/agent` via CopilotKit. Gates that arrive as interrupts resolve through the
  CopilotKit interrupt hook; gates on a CLI/launcher run resolve via `POST /runs/{id}/gate`.
- Canvas: `/files`, `/preview`, `/chart-theme`.

## 7. A2UI generative islands

The frontend registers an A2UI catalog (basic catalog + harness-neutral additions: a table, a
stat row, a form). Surfaces render **inside** gate cards, step-result panels, and chat. The
catalog id is shared between backend (`a2ui_params`) and frontend; a tiny `GET /a2ui/catalog`
may expose it so there is one source of truth. Charts prefer Vega-Lite **v5** to match the
backend's `altair 5.5` validation (`docs/ui-slice.md`).

## 8. Build order (spike-first)

0. **Spike** — Vite React + CopilotKit → `/agent`; render one A2UI surface end-to-end with no
   Node runtime. Decide runtime-vs-direct with evidence; if a shim is needed, prefer a minimal
   embedded runtime, else the `@ag-ui/client` renderer. Output is a decision + a working shell.
1. **Serve** — `serve.py` mounts `ui/dist` (StaticFiles) with SPA fallback that does not shadow
   API routes; `TestClient` test (`GET /` serves the app, unknown client path → index.html,
   `/agent/health` and a files-traversal 404 still behave). No model/kernel/Docker.
2. **Shell** — left rail + run launcher (REST, generated form).
3. **Run view** — SSE reducer, DAG, narration.
4. **Gate cards** — REST + interrupt.
5. **Canvas** — file list + viewers + preview + report + download.
6. **A2UI** — catalog + islands; cartridge-side persona guidance in `cartridges/ds/`.
7. **Chat pane** — `/agent`, narration withheld.

## 9. Testing

- Harness: static-serve + SPA-fallback test in `tests/` (no model/kernel/Docker; `FakeAgent`/
  stub-env pattern where a driver is needed).
- Frontend: thin; the spike is the chain-of-custody proof. A full run render is an
  integration check under `tests/integration/`, gated on `DSAGENT_INTEGRATION=1`.

## 10. Risks

- **Transport (highest):** Node-runtime-vs-direct — retired by the spike before any panel is built.
- **A2UI client maturity:** react-ui A2UI renderer version must match the v0.9 surfaces the
  Python side emits; pin deliberately, like the backend `[ui]` pins.
- **Frontend deps are new:** approved implicitly by the stack choice; kept minimal
  (react, react-dom, vite, typescript, @copilotkit/react-core, @copilotkit/react-ui,
  @ag-ui/client; md/chart libs chosen conservatively during the build).
