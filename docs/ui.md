# UI — research and decision (2026-09-16)

**Requirement (Noel):** a modern chat UI (Claude / ChatGPT class) with a canvas, where the
agent decides at runtime what gets rendered — the frontend cannot know in advance what
is coming.

## Landscape, as of September 2026

**Transport between agent and UI — AG-UI.** Event-based protocol (RUN_STARTED,
TEXT_MESSAGE_*, TOOL_CALL_*, STATE_SNAPSHOT/DELTA, CUSTOM) over HTTP/SSE or WebSocket.
First-party integrations for LangGraph, CrewAI, Google ADK, Microsoft Agent Framework,
Pydantic AI, Mastra, AWS Strands/AgentCore; community for Claude Agent SDK and OpenAI
Agents. For our stack the bridge is `ag-ui-langgraph` (`LangGraphAgent`,
`add_langgraph_fastapi_endpoint`) plus CopilotKit's `CopilotKitMiddleware()` and its
`LangGraphAGUIAgent` subclass, which streams Deep Agents' todos, files and subagent
activity to the frontend automatically. (This paragraph first placed `LangGraphAGUIAgent`
in `ag-ui-langgraph`; it is in `copilotkit`. Verified signatures in `docs/ui-slice.md`.)
This is the *de facto* standard; building our own WebSocket protocol (v1 style) would
be a mistake.

**Three levels of "the agent decides what to render", from safest to freest:**

1. *Typed tool renders.* The agent calls a known tool (`show_table`, `show_chart`,
   `show_report`, `request_approval`) and the frontend maps it to a React component
   (`useRenderToolCall` in CopilotKit, `makeAssistantToolUI` in assistant-ui).
   Deterministic, testable, covers 80 % of a DS product.
2. *Declarative UI from a catalog.* The agent emits JSON constrained to a catalog of
   components the host owns; the host renders it natively. Two contenders:
   **A2UI** (Google; v0.9.1 stable, v1.0 RC; React/Angular/Lit/Flutter renderers;
   CopilotKit renders it out of the box; MIME `application/a2ui+json` over MCP) and
   **json-render** (Vercel Labs; Apache-2.0; Zod-typed catalog, streamed spec compiler,
   actions/state binding; backend-agnostic, ~15k stars). Both solve "unknown layout,
   known vocabulary".
3. *Free HTML in a sandboxed iframe.* **MCP Apps** (hosted today by ChatGPT, Claude,
   VS Code, Goose, Postman) or a plain artifacts panel (assistant-ui ships a
   Claude-artifacts example: side panel + sandboxed iframe). Right for reports,
   Plotly dashboards, notebooks — exactly what the DS cartridge produces.

The June-2026 consensus for custom products: keep the outer shell (auth, navigation,
destructive actions, approvals) in developer-owned code; let the agent generate
read-only/low-risk panels declaratively; put rich documents in an iframe.

**Chat shells.** CopilotKit (~28k stars, MIT; React GA, Angular/Vue/React Native;
makers of AG-UI; renders A2UI and MCP Apps) is the heavy, batteries-included option.
assistant-ui (~8k stars, MIT; headless React primitives; has a LangGraph runtime and
the artifacts example) is the light, composable option. Vercel AI SDK + AI Elements is
excellent but pulls toward Vercel's runtime. TanStack AI is alpha. Chainlit's status is
uncertain after leadership changes.

**Do not fork:** `langchain-ai/open-canvas` (archived Feb 2026) and
`langchain-ai/deep-agents-ui` (archived Jun 2026).

## Decision

Backend: expose the orchestrator and every workflow run over **AG-UI** via
`ag-ui-langgraph` + `CopilotKitMiddleware`. The runner emits AG-UI events for step
start/end and gates (gates become HITL interrupts the UI answers with approve/reject).
This replaces the v1-style "FastAPI + WebSocket parity" item.

Frontend: **Next.js + CopilotKit**, two-pane layout — chat left, canvas right.
The canvas has three content sources, all driven by the agent:

- **Workspace artifacts** (files under the run workspace, streamed by the filesystem
  middleware): HTML reports and Plotly dashboards in a sandboxed iframe, markdown
  rendered, PNG figures, parquet/CSV as a virtual table, notebooks. This is the DS
  "canvas" most of the time.
- **Typed tool renders**: workflow progress (DAG with step status), gate cards,
  data-gate tables, metric tiles, model cards.
- **Declarative panels via A2UI** for anything the agent wants to compose that has no
  dedicated component. Start with A2UI (native in CopilotKit); keep json-render as the
  fallback if A2UI's catalog proves too rigid.

Why CopilotKit over assistant-ui: the Deep Agents middleware, A2UI rendering and HITL
hooks exist today and remove months of glue; the cost is a heavier dependency and
CopilotKit's opinionated runtime proxy. If that becomes a problem the AG-UI event
stream is the stable seam — assistant-ui can consume the same backend.

## Canvas observations from run 001

The first real `eda-to-report` run (`docs/runs/eda-to-report-001.md`, 8 m 53 s, 13 files)
is the only empirical evidence we have about what the canvas has to render. Four things
it settled:

1. **Nothing appears for the first 87 seconds** — 16 % of the run. A canvas fed only by
   workspace files shows an empty panel for a minute and a half while the first persona
   profiles the data. Step and tool events have to carry the UI until the first artifact
   lands; files alone are not a progress indicator.
2. **Figures arrive as a burst** — five PNGs in 41 seconds, one roughly every 10 s. This
   is the one moment in the run where the canvas visibly streams, so images should be
   appended as they appear rather than re-rendering a list on every event.
3. **The last two files are 8 seconds apart and one supersedes the other**
   (`report/findings.md`, then `report/findings.html`). A canvas that auto-focuses the
   newest file would flash the markdown and then replace it. Focus on step completion,
   not on every file event.
4. **Two of the thirteen files were working files** the persona never meant to deliver.
   The canvas needs a deliverable/working distinction or it will show a scratch CSV with
   the same weight as the final report.

For (4), **`produces` is the deliverable signal and the runner already has it** — it is
the declared contract the runner verifies on disk at the end of every step. The
`dsagent.step` `CUSTOM` event should carry the step's `produces` list, so the frontend can
mark those paths as deliverables without a heuristic on file extensions or directories.

## Slice for M2.2 (thin vertical slice, before Docker/Meridian)

1. `dsagent serve`: FastAPI app with `add_langgraph_fastapi_endpoint` for the
   orchestrator; runner emits `CUSTOM` events `dsagent.step`, `dsagent.tool` and
   `dsagent.file`. There is no `dsagent.gate` event — a gate is a LangGraph `interrupt()`,
   which AG-UI carries as an interrupt, not as `CUSTOM`. See `docs/ui-slice.md`.
2. `ui/` (Next.js + CopilotKit): chat + canvas; canvas shows workspace files as they
   appear (iframe for `.html`, markdown, images, table for `.csv/.parquet`).
3. Gate card: `request_approval` HITL → approve/reject → runner resumes.
4. Run `eda-to-report` end to end from the browser.

A2UI panels and MCP Apps come after the slice works.

## Sources

AG-UI docs (docs.ag-ui.com); CopilotKit repo and "Frontend for LangChain Deep Agents"
blog; Google Developers Blog "A2UI + MCP Apps"; sunpeak "MCP Apps vs A2UI (June 2026)";
vercel-labs/json-render README; assistant-ui artifacts example; dev.to "I evaluated
every AI chat UI library in 2026"; langchain-ai/open-canvas and deep-agents-ui repos.
