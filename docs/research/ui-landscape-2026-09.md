# UI landscape research — September 2026

Three parallel investigations (≈20 sources each) run on 2026-09-17 to reset the UI
direction after the M2.2 slice. Condensed; sources at the end of each part.

## Part A — AI data-analysis products

**Table stakes now** (present in nearly every product): a prompt bar as front door with
follow-up context; inspectable code/SQL per answer (hidden code is a complaint);
interactive charts (Plotly in Julius, Vega-Lite in Looker and Databricks agents, native
chart/Explore cells in Hex, Liveboards in ThoughtSpot); post-hoc pivot/filter/drill on
results; a semantic/context layer with verified queries; CSV/Excel + PDF/PPT export;
share-by-link with inherited permissions; scheduled runs and Slack/Teams delivery;
model/effort choice; a handoff from casual chat to rigorous notebook.

**What differentiates the leaders:** Hex's cell-by-cell pending-changes review with
confirm/undo and diff views (the only praised approval UX); Genie "Deep Analysis" and
Looker "Thinking" — a visible research plan that ends in a cited report; agents as
reusable, permissioned, schedulable assets (Deepnote, Looker Data Agents); human-readable
interpretation before execution (Spotter's search tokens, Claude-in-Excel's explanation
before apply); "data never leaves" as the enterprise pitch.

**Loudest complaints:** confident inaccuracy / shadow analytics (Julius, Copilot);
message and query meters (Julius $45→$375 cliff, Spotter 25 queries/user); unpredictable,
unattributable spend (Databricks "cheap until it isn't", Snowflake "no fixed ceiling",
Fabric capacity throttling); heavy semantic-model prerequisites; auto-apply without
review (Fabric "Fix with Copilot", Copilot Agent Mode).

**White space a new entrant can own:** (1) a visible *team* of personas with roles,
per-persona artifacts, handoffs and a critic that must sign off — nobody shows a team;
(2) approval gates as first-class workflow nodes *before* execution (plan, joins, compute
spend, external sharing) — Hex only reviews after the fact; (3) auditable, replayable
runs: plan → per-step code, inputs, row counts, model, who approved what, diff between
runs; (4) cost transparency per step, before and after, with a budget gate; (5)
self-hostable with BYO/local models for shops that are not on Snowflake/Databricks;
(6) the canvas as the unit of work, with one-click publish as report/dashboard/schedule.

Sources: upsolve Julius review; Coefficient Julius pricing; Hex docs and Fall-2025 launch;
Deepnote Agent Workspace + G2; Databricks Genie Agents blog, WisdomAI Genie review,
Databricks community cost thread; upsolve Spotter, thoughtspot.com, Querio comparison;
WisdomAI Cortex guide, Snowflake VQR docs; Toosio Akkio; AIChief Powerdrill and Rows;
Nexacu Copilot-Excel 2026, Microsoft Learn Fabric Copilot; Google Cloud Looker
Conversational Analytics; Research.com DataGPT; Fabi.ai, Querio, WisdomAI PR.

## Part B — Making the agent emit interactive charts and tables

**Chart grammars an LLM can emit.** Vega-Lite: 15–40-line specs, data by reference
(`data.name`/`url`), formal JSON Schema, tooltips/brush/pan-zoom via `params`, the only
grammar with published LLM benchmarks (VegaChat: 0.3 % empty / 0 % error after
schema-validate + ≤5 repair loops, arXiv 2601.15385); Databricks picked it for agents as
"compact, self-validating, secure by design vs generated code"; BSD-3, ≈250–300 kB gz.
Plotly JSON: zero-effort interactivity, very good LLM validity, but traces inline the
data (4–10× payload), no schema-validation culture, 365 kB–1.4 MB gz. ECharts: strong on
big data, loosely typed (hallucinated keys silently ignored). Observable Plot: a JS API,
not a grammar. Recharts-via-JSON: no official schema. Flint (Microsoft Research, Jul
2026): semantic-type spec compiling to VL/ECharts/Plotly — young (v0.4), worth watching.

**Tables.** TanStack Table (MIT, 45 kB, headless, virtualise with react-virtual, no
pivot); AG Grid Community (pivot is Enterprise-only); Glide Data Grid (canvas, millions
of rows, no built-in sort/filter); **Perspective** (FINOS, Apache-2.0, WASM engine with
pivot/group/filter/expressions, ingests Arrow/CSV/JSON, React bindings, Python server) —
the only OSS pivot table out of the box, at a multi-MB bundle. Browser data loading:
apache-arrow JS; **hyparquet** (10 kB, HTTP range requests, streams rows); duckdb-wasm
(full SQL over parquet, ≈6 MB, when the frontend must re-aggregate).

**Generative-UI frameworks.** Google A2UI v0.9.1 (v1.0 targeted Q4 2026, stable React
renderer since Q2 2026, streaming JSON healing, catalog model, client→server data sync);
Vercel json-render v0.19 (Zod catalog, streaming patch compiler, `$state`/`setState`,
TypeScript-only prompt generation); Thesys C1 → "OpenUI" (proprietary, hosted, skip);
CopilotKit's three tiers on AG-UI — controlled (`useRenderTool` on backend tool calls),
declarative (A2UI catalog), open-ended (MCP Apps iframes); assistant-ui tool UIs;
LangGraph native GenUI (tied to LangSmith deployment); MCP Apps (spec 2026-01-26, iframe,
right for third-party widgets). All can carry a Vega-Lite spec as one catalog component;
none renders charts natively beyond simple bars.

**How assistants do it.** ChatGPT: a fixed in-house renderer for bar/line/scatter/pie,
static fallback otherwise, tables expandable with cell references in follow-ups. Claude:
model-written React (Recharts/D3) in a sandboxed iframe, inline since March 2026.
Gemini: Canvas apps. Consumer assistants generate *code*; enterprise/agent platforms
(Databricks, Palantir, Elastic, Lightdash) standardise on *Vega-Lite specs*.

**Round-trip editing.** Keep `charts: {id: {spec, data_ref, version}}` in graph state;
AG-UI `STATE_SNAPSHOT/DELTA` push it down and echo UI edits back; local edits are
mechanical on VL (swap `mark`, add `transform.filter`, change `encoding.x.field`);
"ask to change" re-emits through the validate-then-repair loop; brush selections post
back as context ("user selected rows X").

**Recommended stack:** `show_chart(spec, data_ref)` / `show_table(data_ref, …)` persona
tools emitting Vega-Lite v5 with data by reference (parquet in the workspace), validated
server-side with altair's schema + repair loop, Plotly as escape hatch for 3D/geo; typed
tool render over AG-UI (skeleton while streaming, `vegaEmbed` on result); A2UI reserved
for long-tail panels; `ACTIVITY_*` events for progress; frontend `<ChartCard>` with
toolbar (type, aggregate, filter, edit JSON, ask agent) and selection listener; TanStack
+ react-virtual ≤100k rows, Perspective on "pivot"; the HTML report embeds vega-embed +
spec + data snapshot so it stays interactive; `vl-convert` renders PNG/SVG for PDF.

Sources: Google A2UI v0.9 blog and roadmap; A2UI React renderer issue #347; A2UI + MCP
Apps; vercel-labs/json-render; openui.com; CopilotKit generative-ui repo, spectrum, A2UI
and tool-rendering docs; AG-UI tool-call, state and activity specs; assistant-ui
generative UI; LangChain GenUI docs; MCP Apps spec; VegaChat (arXiv 2601.15385);
Databricks Vega-Lite blog; Vega-Lite streaming tutorial; Flint; plotly.js bundles; 2026
viz-library comparison; recharts #5982; TanStack vs AG Grid 2026; glide-data-grid;
finos/perspective; hyparquet; duckdb-wasm docs; TDS on ChatGPT charts; Claude inline
visuals; Gemini visualizations guide.

## Part C — Agentic-app UX patterns

**Layouts.** Chat + canvas/artifact (Claude, OpenAI Canvas; Anthropic merged chat,
Cowork and Artifacts into one window on 2026-09-16); chat + live preview (Lovable, v0,
Bolt — the preview *is* the progress); timeline + workspace (Devin's Progress tab with
Shell/IDE/Desktop and "Side Chats"; Manus); agent kanban/grid (Cursor 2.0/3, Linear
AgentSessions). Consensus: separate the conversation from the activity tracker; "chat
for long-horizon tasks" is the number-one anti-pattern.

**Progress.** Named phases and checklists, never a percent bar; three-level disclosure
(summary → step → tool calls → raw I/O; no mid-run visibility = 3× abandonment);
thinking streams and "sites browsed" (Gemini); elapsed-time chip and file-touch cards
(Manus); cost per checkpoint (Replit, after the "$70 in a night" backlash; Perplexity
Computer charged for cancelling).

**Approvals.** Plan-first gates with an editable plan (Gemini, ChatGPT deep research,
Claude Code plan mode); inline elicitation cards typed Thought / Action / Elicitation /
Response / Error with a session state `awaitingInput` (Linear — the cleanest schema
found); risk-tiered confirmations and "watch mode" (ChatGPT agent); approvals bound to an
exact action version, invalidated when the plan changes; pause-before-takeover (Devin);
progressive delegation (operator → supervisor).

**Multiple agents.** The least mature area: swim lanes per agent, Cursor's grid and
best-of-n, Linear's agents as assignable teammates with avatars, Notion's agents as
workspace members, Hex's "agents attached to a notebook with a human who reads SQL in the
loop". Handoffs as structured work items (goal, effects, pending approvals, one owner).

**Durable outputs.** Versioned artifacts with restore and share-with-viewer's-permissions
(Cowork); live artifacts as a file system; exports to Docs/PDF/PPTX/notebook; checkpoints
as a rollback timeline (Replit); intent-linked checkmarks (Manus).

**2026 delight.** Live-updating document the agent writes into, section by section,
with comments to the agent on completed parts (Claude Documents); clickable sources while
it runs; follow-up chips; "add X to my report" without re-running; every clickable
component bound to the same tool call the agent can execute (Builder.io); cross-device
progress and notifications.

**Anti-patterns.** Chat for everything; hiding reasoning entirely or dumping full traces;
approval on every operation or none on irreversible ones; state lost on reload;
unexplained agency; no undo; single-level control; runaway sub-agents burning credits;
watermarked outputs.

**Ten recommendations for a multi-persona run screen:** three-pane (timeline / live
workspace / persona + gate rail); plan-as-artifact with one Start gate; typed feed events
with a run-level state badge; named phases with elapsed and cumulative cost; gate cards
that state exactly what is approved with a diff; risk-tiered autonomy per persona; swim
lanes and handoff cards; pause / take over / resume, stopping never costs more; the report
as a live artifact with comment-to-agent, "ask about this chart", versions and share;
everything persisted server-side with notifications and exports.

Sources: Zylos agentic-UX patterns; Kagan catalog vol. 13; Devin session tools docs;
AI UX Playground Manus teardown; TechCrunch on Claude's merged interface; Linear agent
interaction; Cursor 2.0 changelog; AgentPatterns Cursor; Cybernews and Zieminski on
Perplexity Computer; OpenAI ChatGPT agent; MacRumors deep-research viewer; Google Deep
Research; Replit checkpoints; The Register on Replit Agent 3; Claude Help artifacts;
Eigent live artifacts; TechCrunch on Notion agents; Ondelva data-agent comparison;
makeyouragent.ai; Builder.io generative UI; Design Buddies; UX.raspberry; Claude Directory
plan mode; Better Stack Bolt/v0/Lovable.
