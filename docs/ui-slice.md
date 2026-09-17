# M2.2 — UI vertical slice, design

**Date:** 2026-09-16 · Design only, no code. Extends the decision in `docs/ui.md`;
corrects two things that document got wrong.

Everything in §1 was verified by installing the packages in a scratch venv and importing
them: `ag-ui-langgraph` **0.0.45**, `ag-ui-protocol` **0.1.22**, `copilotkit` **0.1.96**,
`fastapi` **0.141.1**, resolving `langgraph` **1.2.11** / `langchain-core` **1.6.3** —
the same versions the project venv already has, so the `[ui]` extra adds no core churn.
npm side: `@copilotkit/react-core|react-ui|runtime` **1.72.0**, `@ag-ui/client` **0.0.59**.

## 1. Verified APIs

**Correction 1 — `LangGraphAGUIAgent` is not in `ag-ui-langgraph`.** `docs/ui.md` and the
ROADMAP both name it there; importing it raises `ImportError: cannot import name
'LangGraphAGUIAgent' from 'ag_ui_langgraph'`. The real signatures:

```python
# ag_ui_langgraph
LangGraphAgent.__init__(self, *, name: str, graph: CompiledStateGraph,
    description: Optional[str] = None, config: RunnableConfig | None | dict = None,
    enable_legacy_on_interrupt_event: bool = True, emit_interrupt_outcome: bool = False,
    emit_raw_events: bool = True, emit_subagent_events: Optional[bool] = None,
    subagent_visibility: Optional[str] = None)

add_langgraph_fastapi_endpoint(app: FastAPI | APIRouter, agent: LangGraphAgent,
    path: str = "/", **kwargs)     # registers POST {path} (SSE) and GET {path}/health

# copilotkit  — LangGraphAGUIAgent lives here and subclasses the above
LangGraphAGUIAgent.__init__(self, *, name: str, graph: CompiledStateGraph,
    description: Optional[str] = None, config: RunnableConfig | None | dict = None)
CopilotKitMiddleware.__init__(self, *, expose_state: bool | Iterable[str] = False,
    a2ui_params: Optional[A2UIToolParams] = None)      # an AgentMiddleware

# deepagents 0.7.14 — both hooks we need are already there
create_deep_agent(..., middleware: Sequence[AgentMiddleware] = (),
    checkpointer: None | bool | BaseCheckpointSaver = None, ...) -> CompiledStateGraph
```

The CopilotKit subclass drops the extra flags; they are plain instance attributes, so
`agent.emit_interrupt_outcome = True` after construction works (verified).

**Correction 2 — `get_stream_writer()` does not reach the bridge.** The bridge consumes
`self.graph.astream_events(**kwargs)` (`agent.py:2389`) and `get_stream_kwargs` passes
only `input`, `subgraphs`, `version="v2"`, `config`, `context` — **never `stream_mode`**.
Probed on langgraph 1.2.11: a `get_stream_writer()` write shows up in
`.astream(stream_mode="custom")` and **not** as `on_custom_event` in `astream_events`;
only `dispatch_custom_event` / `adispatch_custom_event` produce `on_custom_event`.

So the runner emits with `langchain_core.callbacks.manager.dispatch_custom_event`. The
sync form is enough: verified that a sync `@tool` running in a `ToolNode` under an
**async** `astream_events` run reaches the stream (contextvars, Python ≥ 3.11 — we
require 3.11). From there `agent.py:3629` turns any `on_custom_event` into
`CustomEvent(type=EventType.CUSTOM, name=event["name"], value=event["data"])`, and
`copilotkit.LangGraphAGUIAgent._dispatch_event` passes unrecognised names straight to
`super()`. That is the whole path.

## 2. Run inside a tool

`run_workflow` stays the sync orchestrator tool it is today; `WorkflowRunner` gains an
`on_event` callback that the tool wires to `dispatch_custom_event`.

**Persona events already propagate — that is not optional.** Probed: a persona graph
invoked *inside* the tool has its own `on_tool_start` / `on_tool_end` surface in the
parent's `astream_events`, and `config={"callbacks": []}` does **not** detach it. So the
bridge will emit `TOOL_CALL_*` for every `read_file` and `run_python` a persona makes,
carrying no step or persona attribution. `dsagent.tool` exists to add that attribution
and to give the frontend a dedupe key, not to be the only source of tool events.
Personas run with `.stream(stream_mode="updates")` so the runner sees node boundaries:
it emits `dsagent.tool` when the model node yields tool calls, and re-snapshots the
workspace after each tool node for `dsagent.file`.

**Human gates: `interrupt()` inside the tool, with one correction to resume.** LangGraph
re-executes the whole task from the top on resume, and matches resume values
*positionally* — the `Interrupt.id` is derived from the call position, not the payload
(a two-gate probe gave both gates the same id `69b854ab…`). The runner today skips a
`done` step **and its gate**, which shifts that sequence. The probe result was
`gate1:skipped gate2:APPROVE-1`: gate 2 silently consumed gate 1's answer.

> **Rule.** The sequence of `interrupt()` calls must be identical on every re-entry. The
> runner calls the gate's `interrupt()` for every gate it reaches, decided or not, and
> discards the return value when `run.json` already records a decision. Step *work* stays
> skipped — that is the existing idempotency and it is what makes re-entry cheap.

This needs `StepRecord.gate` (decision + timestamp) split from `StepRecord.status`, which
today means both "work done" and "gate passed". Built in PR 2, with one refinement the
implementation forced: the rule applies to **human** gates only. An auto gate calls no
`interrupt()`, so re-running one buys no sequence stability while a convergence check
costs minutes — it is skipped once its record reads `approve`. Unit test (`FakeAgent`, no model): a
two-gate workflow stops at gate 1; re-entering with `approve` does **not** re-invoke
step 1's agent and stops at gate 2; re-entering with `reject` records `reject` on gate 2,
not `approve`.

**`run_id` must be deterministic per invocation.** The same re-execution that motivates
the gate rule also re-runs the top of `run_workflow`, where the tool mints
`RUNS_DIR / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"` (`cli.py:229`). On re-entry that
is a *new, empty* run directory, so `resume` finds no `run.json`, every step reads as
`pending`, and the run redoes — and re-pays for — work it already did. The gate rule alone
does not save it: it protects the interrupt sequence, not the state the sequence indexes into.

The id is derived from the tool call, not from the clock. `tool_call_id` is the right
source: verified that injecting it with
`Annotated[str, InjectedToolCallId]` yields the same value on the original call and on the
re-entry (`toolu_01ABC` both times), because the id belongs to the `AIMessage` that the
checkpoint replays. So `run_id = f"{workflow}-{tool_call_id}"`, and `resume=True` whenever
that directory already exists. `thread_id` plus a counter in graph state is the fallback if
a caller ever reaches `run_workflow` without a tool call; it is strictly more machinery for
the same guarantee, so it stays the fallback. Test in PR 2: two entries of the same tool
call open the same `run_dir` and the second does not re-invoke a completed step's agent.

**Checkpointer.** `InMemorySaver` for the slice — one `dsagent serve` process,
`thread_id` from AG-UI `RunAgentInput.thread_id`. `run.json` already survives restarts;
the checkpointer only has to outlive the gate answer. `SqliteSaver` is a follow-up, and
the moment we want a gate answered tomorrow rather than in five minutes.

## 3. Event payloads

Three `CUSTOM` events. `name` is the name below, `value` the object. Every payload carries
`run_id` (the run-directory name) so one stream can carry more than one run.

```jsonc
// dsagent.step
{"run_id": "eda-to-report-20260916-121016", "workflow": "eda-to-report",
 "step": "profile", "persona": "marie", "env": "default", "index": 0, "total": 4,
 "status": "started|done|failed|awaiting_gate", "needs": [],
 "produces": ["artifacts/data-profile.md", "artifacts/data-profile.json"],
 "produces_matched": {"artifacts/data-profile.md": ["artifacts/data-profile.md"],
                      "artifacts/data-profile.json": ["artifacts/data-profile.json"]},
 "ts": 1789564216.4, "error": null}

// dsagent.tool
{"run_id": "…", "step": "profile", "persona": "marie", "tool": "run_skill_script",
 "tool_call_id": "toolu_01…", "phase": "started|finished",
 "args_preview": {"skill": "eda", "script": "profile.py"}, "ts": …}

// dsagent.file
{"run_id": "…", "step": "analyze", "path": "artifacts/figures/01_temp_by_category.png",
 "kind": "deliverable|working", "change": "created|modified",
 "size": 48213, "mtime": …, "ts": …}
```

`produces` rides on **every** `dsagent.step` status, including `started`, so the canvas can
mark a path as a deliverable before the file exists — the run-001/002 requirement. It is the
promise, as declared, and since a `produces` entry may be a glob it is not necessarily a
path. `produces_matched` is what has actually landed: each entry mapped to the real files it
names at event time — empty lists on `started`, filled on every other status. Links and
done/missing ticks read `produces_matched`; `produces` is what shows a promise that has no
file behind it yet. Keyed by entry rather than flattened, so a tick is per promise: with two
patterns a flat list cannot say whether both were satisfied.
`kind` on a file event is `deliverable` iff the path is in that step's `produces`.
`tool_call_id` is the dedupe key against the raw `TOOL_CALL_*` the bridge emits for the
same call. `args_preview` is truncated to 500 chars and never carries file contents.

**The gate is not a CUSTOM event** — it is the interrupt, so `dsagent.gate` drops off the
ROADMAP. `ag_ui_langgraph.interrupts.lg_interrupt_to_agui` reads `reason`, `message`,
`tool_call_id`/`toolCallId`, `response_schema`/`responseSchema` and `expires_at` off the
interrupt value and keeps everything else under `metadata.langgraph.raw`:

```python
interrupt({"reason": "dsagent.gate",
           "message": "Data gate report is in artifacts/data-gate.md. Proceed to analysis?",
           "response_schema": {"type": "object", "required": ["decision"], "properties": {
               "decision": {"enum": ["approve", "reject"]}, "note": {"type": "string"}}},
           "run_id": "…", "workflow": "eda-to-report", "step": "data-gate",
           "persona": "marie", "produces": ["artifacts/data-gate.md"]})
```

A single resolved answer becomes `Command(resume=payload)`
(`_build_command_from_agui_resume`), so `interrupt()` returns the `{"decision": …}` object
the card sent, verbatim.

## 4. Frontend (`ui/`)

Next.js app router. `app/api/copilotkit/route.ts` uses
`copilotRuntimeNextJSAppRouterEndpoint` with
`new CopilotRuntime({ agents: { dsagent: new HttpAgent({ url: "http://localhost:8000/agent" }) } })`
(`agents?: Record<string, AbstractAgent>`). Layout: `<CopilotKit>`, chat left, canvas right.

Use the **v2** hooks. In 1.72.0 `useCoAgent`, `useRenderToolCall`, `useLangGraphInterrupt`,
`useFrontendTool` and `useHumanInTheLoop` all live under `src/v1-deprecated/`.

| Panel | Hook |
|---|---|
| Gate card | `useInterrupt({ enabled: e => e.value?.reason === "dsagent.gate", renderInChat: false, render: ({interrupt, resolve, cancel}) => <GateCard/> })`; approve calls `resolve({decision: "approve"})` |
| DAG progress | `const { agent } = useAgent(); agent.subscribe({ onCustomEvent: ({ event }) => … })`, filtered to `event.name === "dsagent.step"`, reduced into a map keyed by step id. Ticks and links read `produces_matched`, not `produces` — a glob is not a path |
| Persona narration | the same subscriber's `onTextMessageStartEvent` / `onTextMessageEndEvent`, attributed to whichever step is between its `started` and its end. The chat withholds those ids through the `messageView` slot |
| Canvas file list | the same subscriber on `dsagent.file` — append-only (run 002: four figures in 14 s), and focus changes only on `dsagent.step` `status=done`, never on a file event (run 001, obs. 3) |

File serving: `GET /runs/{run_id}/files/{path:path}` on the same FastAPI app, resolved
against `<run_dir>/workspace`, 404 unless `Path.resolve().is_relative_to(workspace)`, and
`.dsagent/` denied. `.md` / `.csv` / `.parquet` are fetched and rendered client-side, `.png`
is an `<img>`, `.html` goes in `<iframe sandbox src=…>` with an **empty** sandbox — run
002's report is self-contained and needs no scripts. Plotly dashboards will need
`allow-scripts` plus a separate origin; deferred, not designed here.

### Persona narration in the chat

A persona's own commentary reaches the browser as an ordinary assistant message: the
bridge attributes messages only inside a subagent window, which it opens solely for a tool
literally named `task` (`agent.py:426`), and the runner launches persona steps itself. So
`subagentRunId` is empty and nothing on the wire says marie wrote this and noel wrote that.
Left in the transcript it reads as one voice changing personality mid-conversation —
visible in `docs/runs/ui-canvas-002.png`.

Three ways to keep it out of the chat, least invasive first:

1. **Rendering override.** `<CopilotChat messageView={…}>` is a documented slot;
   `CopilotChatMessageView` takes the `messages` it should draw, so a wrapper filters and
   delegates. Nothing about the run, the thread or the backend changes — the withheld
   messages are still in the thread and still reach the model as context. **Chosen.**
2. **Owning the view.** `<CopilotChatView messages={…}>` accepts messages directly, but
   `CopilotChatProps` omits that prop, so this means driving input, suggestions,
   attachments and scroll ourselves.
3. **Tagging in `serve`.** Set something on the message the frontend could key off. A
   backend change for a presentation problem, and there is no field for it — `name` and
   `metadata` exist on the AG-UI message but the bridge fills neither from a nested graph.

Attribution is by time, because the protocol offers nothing better: a message that starts
between a step's `started` and its end belongs to that step's persona. The same subscriber
already tracks that, so it costs one `useRef`.

Before deciding what to do with the unattributed persona `TOOL_CALL_*`, PR 5 checks
`emit_subagent_events` / `subagent_visibility` (`"inline"` default, `"attributed"`,
`"hidden"`) on the agent. Expect them not to help: the bridge opens a subagent window only
for a tool literally named `task` (`agent.py:426`, `if name != "task"`), and the runner
launches persona steps itself from inside `run_workflow` rather than through Deep Agents'
`task`. If that holds, `"hidden"` cannot suppress them and `"attributed"` cannot label
them, and `dsagent.tool` is the only attribution mechanism we have. Confirm against a real
run before building on it.

## 5. PRs, smallest first (base `v2`)

1. **harness: runner emits step/tool/file events.** `RunnerEvent` + `on_event` on
   `WorkflowRunner`; the CLI prints them. No AG-UI dependency at all. Tests assert order
   and that `produces` rides on every step event.
2. **harness: stable gate-interrupt sequence on re-entry.** Split `StepRecord.gate` from
   `status`; unconditional gate call; the two-gate resume test from §2.
3. **harness: dispatch runner events as LangChain custom events.** Thin adapter on
   `on_event`; unit test captures the dispatcher and checks the three names and payload keys.
   Plus a **timing test**, which is the one that matters: a tool that emits, sleeps 2 s, then
   emits, consumed through `astream_events`, asserting the consumer sees the first event
   *before* the tool returns. Events that only flushed at tool end would make the whole
   design pointless for a six-minute tool. Probed green on langgraph 1.2.11 — first event at
   t=0.005 s, tool return at t=2.012 s — so the test pins behaviour we have, rather than
   hoping for it.
4. **harness: `dsagent serve`.** `[ui]` extra (`ag-ui-langgraph`, `copilotkit`, `uvicorn`);
   `build_orchestrator(..., middleware=[CopilotKitMiddleware()], checkpointer=InMemorySaver())`;
   `LangGraphAGUIAgent` + `add_langgraph_fastapi_endpoint`; the files endpoint. Tests use
   `TestClient` on `/agent/health` and on the files endpoint including a traversal 404.
5. **ui: Next.js shell.** CopilotKit route and chat pane only, no canvas.
6. **ui: canvas.** File list plus the four viewers.
7. **ui: gate card.**
8. **ui: DAG progress.**
9. **docs: run 003 from the browser,** with the screenshot the ROADMAP asks for.

Open, to settle with code in hand: whether `emit_interrupt_outcome = True` (default
`False`) is needed for `useInterrupt`'s standard path or the legacy `on_interrupt` flow is
enough — decide in PR 7; and whether the unattributed persona `TOOL_CALL_*` in the chat
pane read as useful detail or as noise — PR 5 will show.
