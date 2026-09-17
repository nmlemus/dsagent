# Roadmap

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done. One task per PR.

## M1 — Harness skeleton (kernel env only) — DONE 2026-09-16

- [x] Cartridge loader + matrix validation + generated command skills
- [x] Persona agents and orchestrator on Deep Agents
- [x] Runner: DAG, produces, human/auto gates, resume, dry-run, input templating
- [x] Kernel env (LocalShellBackend + persistent Jupyter `run_python`)
- [x] CLI: cartridge validate/list, run, chat
- [x] DS cartridge: 4 personas, 4 skills, 2 workflows, Meridian Dockerfile
- [x] 12 unit tests (no model needed)

## M2 — Real runs + Docker + Meridian

### M2.1 First real run of `eda-to-report` — DONE 2026-09-16
- [x] First commit and push as branch `v2` of `nmlemus/dsagent` (decided 2026-09-16; `main` stays v1 until 2.0 ships)
- [x] Add `tests/integration/test_eda_to_report.py` (skipped unless `DSAGENT_INTEGRATION=1`) using a small public CSV
- [x] Env requirements: `EnvSpec.requirements` declared in `cartridge.yaml`; kernel envs verify them when provisioned and fail fast with the exact `pip install`; `dsagent cartridge install <path>` installs them into the current interpreter; Docker envs keep theirs in the Dockerfile (harness validates the field only)
- [x] Per-step telemetry in `run.json`: tool calls by name, token usage, skills read, and workspace files touched — so the first real run can be read instead of guessed at
- [x] Run it with a real model; capture what the personas actually do in `docs/runs/eda-to-report-001.md` (prompt gaps, tool misuse, cost, wall time)
- [x] Harness fixes from run 001: `Step.sees` (a step is shown only the inputs it interpolates), cache-token detail in telemetry, canvas observations in `docs/ui.md`
- [x] Fix step instructions / skills based on that run — iteration 2 verified by run 002: D1–D8 all fixed, one new deviation (D9, `analyze` fitted trend lines against its own no-modeling rule) recorded in `docs/runs/eda-to-report-002.md`

### M2.2 UI vertical slice (see docs/ui.md) — DONE 2026-09-16
- [x] **Runner: stream step events instead of only `log()`** — `RunnerEvent` + `on_event` on `WorkflowRunner`, emitting `dsagent.step` / `dsagent.tool` / `dsagent.file`; personas run with `.stream()` so tool and file events arrive while the step is still working. Each step event carries `produces`, so a consumer can tell a deliverable from a working file (runs 001/002). No AG-UI dependency
- [x] Design PR: `docs/ui-slice.md` — verified package APIs, run-inside-a-tool design, event schemas, frontend plan, PR breakdown
- [x] Runner: stable gate-`interrupt()` sequence on re-entry (`StepRecord.gate` split from `status`) and a deterministic `run_id` derived from `tool_call_id`, with the two-gate resume test and a re-entry test asserting the same `run_dir` is reopened
- [x] `dsagent serve`: FastAPI + `ag-ui-langgraph` (`add_langgraph_fastapi_endpoint`) + `copilotkit` (`LangGraphAGUIAgent`, `CopilotKitMiddleware()`) exposing the orchestrator, plus `/runs/{id}/files/{path}`. Optional `[ui]` extra; base dependencies unchanged
- [x] Dispatch runner events as LangChain custom events (`dsagent/runner/dispatch.py`), which `ag-ui-langgraph` forwards as AG-UI `CUSTOM` with the same name and value. Pinned by a test that runs the runner inside a sync tool under an async `astream_events` consumer, and by a timing test: the first event arrives while the tool is still working
- [x] Human gates become LangGraph `interrupt()`s answered from the UI (`ask_human` → `interrupt`), on the resume rules already built. `GateRequest` gives the hook the step context a gate card needs
- [x] `ui/` Next.js + CopilotKit shell: chat left, empty canvas right, `/api/copilotkit` relaying to `dsagent serve` via `HttpAgent`. Verified from the browser against a real model (`docs/runs/ui-shell-001.png`)
- [x] Canvas contents: workspace files from `dsagent.file` — deliverables and working files grouped by `kind`, sandboxed iframe for `.html`, react-markdown for `.md`, `<img>` for figures, a table for `.csv/.tsv`, `<pre>` for json/text. Verified end to end (`docs/runs/ui-canvas-001.png`, `-002.png`). Parquet still has no browser reader; it links out
- [x] Gate card: `useInterrupt` in the right pane — step, persona, message, `produces` links to `/runs/{id}/files/{path}`, approve/reject with a note. Verified end to end from the browser (`docs/runs/ui-gate-001.png`)
- [x] Workflow progress render: DAG panel at the top of the canvas — per-step persona, status, live elapsed, `produces` ticks from `produces_matched`, tool counts. It also owns persona narration: a persona's assistant messages are withheld from the chat by a `messageView` rendering override and shown against their step (`docs/runs/ui-dag-001.png`)
- [x] Runner: `produces` glob support — a pattern is a contract, satisfied by at least one match; matched files are `kind: deliverable`. `analyze` declares `artifacts/figures/*.png`
- [x] Run `eda-to-report` end to end from the browser — `docs/runs/eda-to-report-003.md`, with the gate approved from the card and screenshots in `docs/runs/eda-to-report-003/`. $0.47 against run 002's $0.85; all four run-001 canvas observations answered
- [ ] Then: A2UI panels for agent-composed views; MCP Apps later if needed *(deferred past M2.2 — the slice does not need them)*

### M2.2.1 UX debt — from run 003, unstarted

The slice works end to end; these are what an operator hits while using it. Ordered as
`docs/runs/eda-to-report-003.md` orders them.

- [x] **A successful run ends in a red error.** Fixed: the served graph runs at `recursion_limit=150` (`dsagent serve --recursion-limit` to change it). The limit is per invocation and applies to the orchestrator alone — a persona is compiled with `checkpointer=False` and spends its own budget. Measured: ~3 super-steps fixed plus ~2 per model↔tool round, and `CopilotKitMiddleware` adds an `after_model` node to each round, so LangGraph's default 25 buys about ten rounds
- [ ] **A served run leaves no readable log.** `dsagent serve` passes `log=lambda m: None`, so a run directory has `run.json` and no `runner.log`; the CLI writes one. Reading a run after the fact is worse from the browser than from the terminal
- [ ] **`produces` ticks read ○ while the files are visibly landing.** `produces_matched` is empty on `started` by design, so mid-step the DAG row and the file list disagree in front of the operator
- [ ] **A reload loses the run.** Canvas state is reduced from the live event stream only; refreshing mid-run shows an empty canvas while the run continues server-side, with no way to re-attach. Related: nothing outside the gate card names the run (see M4's multi-run management)
- [ ] **The human wait is invisible.** Run 003 spent 37 s waiting for the gate decision and no surface shows it — exactly the number needed to judge whether a gate earns its cost
- [ ] **The error toast shows a stack trace** from a minified bundle. It should name the step and the reason
- [ ] **A failed step's presentation is unexercised.** The DAG row renders `error`, but no run has failed in the browser, so that path has never been looked at

### M2.5 UI product pass (see docs/ui-product.md) — in progress on `m25-ui-product`

Turning the M2.2 slice into something a stakeholder can watch for ten minutes and want.
§7 of the spec is the exit condition; `docs/ui-product-log.md` is the working log.

- [x] Replay fixture + replay engine: the runner writes `events.jsonl` and `runner.log`
      per run, a gate is announced before it is asked, persona narration becomes
      `dsagent.note`, and `ui/fixtures/run-eda-003` replays run 003 at 10× with no model
- [x] Runs API + event-log SSE (§4.1, §4.2) and `dsagent serve --replay`
- [ ] Home screen + run restore on load
- [ ] Launcher: cartridges endpoint, upload, form from declared inputs, start
- [ ] Run screen layout + the Aiuda visual system
- [ ] Progress region: stepper, header metrics, inline gate card, narration
- [ ] Canvas: parquet preview, interactive HTML, downloads, zip
- [ ] Failure / reject / resume presentation
- [ ] SQLite checkpointer + cost in telemetry (§4.3, §4.6)
- [ ] Demo script run, evidence, `docs/runs/ui-product/DEMO.md`, PR

### M2.3 Docker env — next
- [ ] `DockerBackend` integration test behind `DSAGENT_DOCKER=1` (build `envs/meridian`, `execute("python -c 'import meridian'")`)
- [ ] Workspace bind-mount + file ownership sanity (non-root user in image)
- [ ] Auto-gate scripts run *inside* the step env, not on the host (`_gate` currently shells out on the host)
- [ ] Env lifecycle: reuse container across steps of the same run; always `rm -f` on exit/failure

### M2.4 `mmm-meridian` on the Meridian sample dataset
- [ ] `cartridges/ds/skills/mmm/scripts/load_sample.py` — fetch Google's simulated dataset into `data/`
- [ ] Run steps ingest → data-gate → model-spec with a real model; review Ana's spec by hand
- [ ] Run fit (CPU, small draws) → check_rhat auto gate → optimize → report
- [ ] Document the run in `docs/runs/mmm-meridian-001.md` with cost and wall time

## M3 — Portability proof
- [ ] Install `cartridges/ds` as a Claude Code plugin; run Marie and Noel there with zero changes
- [ ] Minimal `cartridges/sdlc/` (3 BMAD-style personas, 1 workflow) proving the harness is domain-agnostic
- [ ] `dsagent cartridge add <git-url>` (port v1 installer)

## M4 — Connectors + hardening

- [ ] Runner: stream personas with `updates` only and reconstruct the final state, instead of `["updates", "values"]`. `values` mode copies the whole transcript on every node, which is wasted allocation on a long step; the runner only needs the last one
- [ ] Multi-run management in the UI (list runs, open past run, resume paused run)
- [ ] BigQuery connector via cartridge `.mcp.json` + LangChain MCP adapters
- [ ] Run resumability across process restarts (already in `run.json`; needs API surface)
- [ ] chore: `dsagent --version` prints "Missing command" (eager callback vs `no_args_is_help`)

## Decisions log

- 2026-09-17 — A run belongs to the server, not to the browser tab that started it. `POST /runs`
  creates the directory, `POST /runs/{id}/start` drives the orchestrator on a background thread,
  and the browser follows `events.jsonl` over SSE and answers gates over `POST /runs/{id}/gate`.
  Reload, a second tab and a CLI-started run cannot show the same screen if the run exists only
  inside one browser's event stream (`docs/ui-product.md` §4.2, §7.6, §7.11). The run still goes
  through the orchestrator — the path the chat takes — so it exists in a thread that can be asked
  about afterwards; the only thing the server decides for it is which directory it writes to.
- 2026-09-17 — The run id travels to `run_workflow` in the graph config (`dsagent_run_id`), read
  with `ensure_config()` and **not** with a `config: RunnableConfig` parameter. Under
  `from __future__ import annotations` that parameter's annotation is a string, LangChain does not
  recognise the injection, and the tool is silently handed `None` — the run then lands in a
  directory named after the tool call while the operator's uploaded dataset sits in another.
  Probed both ways on langchain-core 1.6.3.
- 2026-09-17 — A run keeps the inputs it started with: on resume they are merged from `run.json`
  *before* validation, not after. A re-entering caller is the accident-prone half — `--resume`
  without the original `-i`, or a model retyping `inputs={}` — and this is what makes the
  launcher's form, rather than the model, authoritative about what a run is running on.
- 2026-09-17 — File uploads are a raw-body `PUT /runs/{id}/data/{input}`, not multipart. Multipart
  means adding `python-multipart` for a form with one file in it, and a drag-and-drop already has
  the `File` in hand. The input name is in the path, so a workflow with two datasets needs no new
  convention. Revisit if a form ever needs several files in one request.
- 2026-09-17 — The runs API lives in `src/dsagent/api.py`, not in `serve.py`. FastAPI resolves route
  annotations against module globals, so importing `Request` inside the registering function made
  every route take `request` as a query parameter (422 "Field required"). `api.py` is imported only
  from `build_app`, which already requires the `[ui]` extra.

- 2026-09-17 — The runner writes `<run_dir>/events.jsonl` and `<run_dir>/runner.log` itself,
  rather than leaving both to whichever front end drove the run. `dsagent serve` passed
  `log=lambda m: None`, so a browser-driven run left nothing readable while a CLI run did
  (M2.2.1 item 2). The event log is what a late reader gets — a reload, a second tab, a
  stakeholder on a link, a run started by the CLI — and reconstructing the screen from it is
  what makes those the same screen. `RunState.save` became atomic in the same change: the runs
  API reads `run.json` while the run is rewriting it.
- 2026-09-17 — A human gate is announced on the event stream *before* anyone is asked
  (`dsagent.step` with `status: awaiting_gate` and a `gate` object whose `decision` is null),
  and again once answered. The interrupt is still how the browser holding it answers; it was
  also the only thing that knew the run had stopped, which left every other reader watching a
  run go silently quiet. The same pending question is written to `run.json` (`RunState.gate`),
  because a run waiting for a person is the one most likely to still be waiting when the
  process dies (§7.11), and `GateRecord.asked_at` makes the human wait a number rather than a
  thing nobody can see (M2.2.1 item 5).
- 2026-09-17 — `ask_human` may return `GateAnswer(decision, note)`. The gate card always
  collected a note on a rejection and the runner dropped it on the floor; §7.10 asks for that
  note to be visible in the step's history, which cannot be built from a decision alone.
- 2026-09-17 — Persona narration is a runner event (`dsagent.note`), not an inference from the
  message stream. M2.2 attributed it by time — a message starting inside a step's window was
  that step's — because nothing on the wire says who is talking. The runner opened the window,
  so it says so directly; and unlike a message stream a note survives a reload, a second tab
  and a run nobody was attached to. The `messageView` filter stays: it is what keeps a
  chat-started run's persona messages out of the transcript.
- 2026-09-17 — The replay fixture is run 003's own record, reconstructed by
  `tools/make_replay_fixture.py`, not a fresh capture. Real: step boundaries, the 37-second
  gate wait, file mtimes, per-step tool counts and token usage, each persona's closing summary.
  Reconstructed: the order and individual timestamps of tool calls inside a step, which
  `run.json` does not record. It buys the same fixture for none of the six-run budget, which §7
  needs more than a fifth recording of a run we already have.

- 2026-09-16 — v2 lives as branch `v2` in `nmlemus/dsagent` (not a new repo). `main` keeps v1 and the `datascience-agent` PyPI line until 2.0 ships, then `v2` merges to `main` as 2.0.0.
- 2026-09-16 — Backend on Deep Agents (not Claude Agent SDK): model-agnostic, sandbox protocol fits Docker-per-workflow, same SKILL.md standard as Claude Code.
- 2026-09-16 — Cartridge = Claude Code plugin + `cartridge.yaml`. Workflows exposed three ways (persona request, `/ds-<wf>` command skill, orchestrator intent), all ending in `run_workflow`; a persona can only start workflows listed on it.
- 2026-09-16 — Gates are runner-level (not LangGraph interrupts) so runs pause/resume from `run.json` without a checkpointer. Revisit when the API needs tool-level approvals.
- 2026-09-16 — UI is part of the product, not M4. Transport = AG-UI via `ag-ui-langgraph` + CopilotKit middleware (no custom WebSocket protocol). Frontend = Next.js + CopilotKit, chat + canvas; canvas = workspace artifacts (iframe/markdown/table) + typed tool renders + A2UI declarative panels. open-canvas and deep-agents-ui are archived — do not fork. Details in `docs/ui.md`.
- 2026-09-16 — `artifacts/data-profile.json` is owned by `eda/scripts/profile.py` and carries `schema_version`. The profiling persona must run the script and may only extend the markdown; the gate step evaluates thresholds from the JSON and never recomputes. Run 001 had the persona hand-roll the profiler and silently rename `top` to `top5`, which made the skill's script dead code with no contract. The integration test now compares the artifact's keys against the script's own output rather than against the keys both happen to share.
- 2026-09-16 — Step 03 of `eda-to-report` may fit a trend when a finding depends on one, and must report the slope with an interval. Run 002 showed the previous "no modeling" line contradicting the chart rule added in the same iteration — a title may not assert an untested trend, which cannot be satisfied without testing it. Resolved in favour of testing: a trend that cannot be tested is not a headline finding. Predictive and causal modelling stay out of the step.
- 2026-09-16 — A skill's scripts are run with the `run_skill_script(skill, script, argv)` tool, never by path. `/skills/<persona>/` is a virtual mount for the file tools only; `execute` runs in the env, where it does not resolve, so a hardcoded path fails. The tool resolves inside the persona's materialized skill directory (enforcing the matrix) and runs with `Env.python`, which for the kernel env is DSAgent's interpreter rather than PATH's `python3` — that is where the cartridge's requirements are installed.
- 2026-09-16 — A step is shown only the workflow inputs its own instructions interpolate (`Step.sees` overrides). Run 001 showed the opposite default teaches a persona the finish line: the profiling step read `question` and answered it, and the gate step then re-derived a verdict already on disk. Naming an input in prose does not make it visible — `dsagent cartridge validate` reports steps that end up seeing nothing.
- 2026-09-16 — Default model is `anthropic:claude-sonnet-5` (was `claude-sonnet-4-6`). Newer generation and cheaper ($2/$10 vs $3/$15 per MTok — run 001 would have cost $1.72 instead of $2.57). Opus 4.8 was rejected for the default: 2.5× the cost, and it runs *without* thinking unless the caller passes `thinking={"type": "adaptive"}`, which the harness deliberately does not do (invariant 6 — Deep Agents owns the loop). Sonnet 5 runs adaptive thinking when the parameter is omitted. **A stronger model is opted into per persona via frontmatter `model:` (or per session with `dsagent chat --model`), never made the harness default** — the default carries every step of every workflow, so it is chosen for cost and for behaving well unconfigured.
- 2026-09-16 — Steps record their own telemetry into `run.json` (tool calls by name, token usage, skills read, workspace files touched with mtimes). Sourced from LangChain's standard `tool_calls`/`usage_metadata` and from workspace mtimes, never from a provider SDK. Subagent activity that Deep Agents does not surface in the step result is not counted.
- 2026-09-16 — Env dependencies are declared per env as `EnvSpec.requirements` (opaque pip strings the harness never interprets, so invariant 1 holds). Kernel envs verify at provisioning and fail fast with the exact `pip install`; `dsagent cartridge install <path>` installs them into the current interpreter; Docker envs keep theirs in the Dockerfile and the harness only validates the field. Verification is distribution metadata rather than importability, because import name ≠ distribution name and a mapping would put package knowledge in the harness.
- 2026-09-16 — Skill scoping materialised on disk per persona (copy, not symlink) and mounted via `CompositeBackend` at `/skills/<persona>/`.
- 2026-09-16 — M2.2 transport facts, each verified by importing the package (`docs/ui-slice.md`).
  `LangGraphAGUIAgent` is in `copilotkit`, not `ag-ui-langgraph` (which exports `LangGraphAgent`);
  `docs/ui.md` had it wrong. The bridge reads `astream_events` and passes no `stream_mode`, so
  `get_stream_writer()` writes never reach it — the runner emits with `dispatch_custom_event`.
  A gate is a LangGraph `interrupt()` carried as an AG-UI interrupt, so the planned `dsagent.gate`
  CUSTOM event is dropped; the three CUSTOM events are `dsagent.step` / `dsagent.tool` / `dsagent.file`.
- 2026-09-16 — A gate's `interrupt()` is called on every re-entry, decided or not, and its return
  value discarded when `run.json` already records a decision. LangGraph matches resume values
  positionally and derives `Interrupt.id` from the call position, so skipping a decided gate — which
  the runner does today, since `status == "done"` covers both work and gate — shifts the sequence and
  feeds gate *n*'s answer to gate *n+1* (reproduced with two gates). Step *work* stays skipped.
- 2026-09-16 — `run_workflow` derives its `run_id` from the tool call (`run_id = f"{workflow}-{tool_call_id}"`,
  injected with `InjectedToolCallId`), never from the clock. LangGraph re-executes the tool from the top on
  resume, and the current timestamped `run_dir` would mint a fresh empty run on every re-entry — losing
  `run.json` and re-paying for every completed step. The tool call id is stable across the original call and
  the re-entry (verified). `thread_id` + a state counter stays the fallback for a caller without a tool call.
- 2026-09-16 — `StepRecord.gate` (decision, note, ts) is split from `StepRecord.status`. `status` is the step's
  work (pending/running/done/failed); `gate` is whether anyone agreed to go on; the run is what reads
  `awaiting_gate`. A **human** gate is asked on every entry, decided or not, and its answer discarded when the
  record already reads `approve` — that is what keeps the `interrupt()` sequence stable under `dsagent serve`.
  An **auto** gate is skipped once approved instead: it calls no `interrupt()`, so re-running it buys no
  stability and a convergence check costs minutes.
- 2026-09-16 — `_fill` substitutes with the `_PLACEHOLDER` regex, not `str.format_map`, so it and
  `visible_input_names` share one definition of what a placeholder is. Step instructions are prose written for a
  persona and prose contains braces — JSON, dict literals, CSS, f-string examples — all of which format syntax
  either mangles or raises on. `mmm-meridian`'s `fit` step documents a JSON artifact and could not run at all.
- 2026-09-16 — `dsagent serve` is an optional `[ui]` extra, pinned exactly for `ag-ui-langgraph` and
  `copilotkit` (pre-1.0, and their APIs already moved once under this design) and floored for `fastapi`
  and `uvicorn`. Installing it does not move `deepagents`, `langgraph` or `langchain-core`. The harness
  core imports neither package: `build_orchestrator` takes `middleware` and `checkpointer`, and `serve.py`
  supplies `CopilotKitMiddleware()` and `InMemorySaver()`.
- 2026-09-16 — `ask_human` takes a `GateRequest` (run_id, workflow, step, persona, produces, prompt)
  rather than a bare prompt. A terminal needs only the prompt; a gate card in a browser has to say which
  step of which run is waiting and what it produced, and under `serve` the request becomes the
  `interrupt()` payload. `dsagent chat` and `dsagent serve` now share one `run_workflow`
  (`runner/tools.py`) and differ only in that hook and in where events go.
- 2026-09-16 — CopilotChat renders **nothing** for a tool call with no registered renderer. Observed in the
  shell: the orchestrator's `list_workflows` call is in the AG-UI stream as `TOOL_CALL_START/ARGS/END/RESULT`
  and appears nowhere in the chat, not even collapsed. So the unattributed persona `TOOL_CALL_*` are invisible
  by default rather than noisy, and PR 6/8 must *opt in* to rendering them (`useRenderTool` /
  `useDefaultRenderTool`) rather than suppress them. `dsagent.tool` stays the attributed source for the DAG panel.
- 2026-09-16 — `emit_interrupt_outcome=True` is **not** needed: on the defaults the bridge already emits the
  legacy `CUSTOM name=on_interrupt`, and CopilotKit's `useInterrupt` handles it. Setting it True adds the
  standard `RUN_FINISHED` outcome *alongside* the legacy event rather than replacing it, so it is available
  if the frontend later wants the typed AG-UI `Interrupt`. The catch that actually matters: **the legacy
  event's `value` is a JSON string, not an object**, so the obvious predicate
  `event.value?.reason === "dsagent.gate"` never matches and the card silently never renders while the run
  waits at the gate forever. `ui/app/gate-card.tsx` parses it.
- 2026-09-16 — Persona agents are compiled with `checkpointer=False`. A graph without its own checkpointer
  *inherits the caller's* when it runs inside one, under a namespace derived from the call's position in the
  task. `dsagent serve` re-executes `run_workflow` on resume and the runner skips finished steps, so step N+1's
  persona lands in step N's slot and replays its finished conversation — returning in milliseconds with the
  wrong persona's messages. Caught in the first real browser run: `analyze` failed its `produces` carrying
  `profile`'s telemetry and marie's `skills_read`. Same positional-identity trap as the gate-sequence bug, one
  level down; only reachable under `serve`, because the CLI has no checkpointer.
- 2026-09-16 — `/runs/*` is rewritten through Next onto `dsagent serve` rather than fetched cross-origin. The
  canvas reads `.md` and `.csv` with `fetch`, which CORS governs; `<iframe>` and `<img>` do not, which is why the
  first canvas showed figures and failed on text with "Failed to fetch". A rewrite makes every artifact
  same-origin, so nothing is relaxed on the Python side and there is no CORS policy to get wrong later. The
  report iframe keeps `sandbox=""`, which denies same-origin access anyway. `fileUrl` is now relative and
  `NEXT_PUBLIC_DSAGENT_ORIGIN` is gone.
- 2026-09-16 — Canvas focus follows step completion, never a file event, and lands on the step's *last*
  `produces`. Confirmed live: `report` wrote `findings.md` then `findings.html` three seconds apart and the pane
  went straight to the HTML, and the four figures of the analyze burst appended without stealing focus — run
  001's observations 2 and 3, now enforced rather than noted. A user click pins the pane until the next step
  finishes.
- 2026-09-16 — `produces` entries may be globs. A step that writes a variable number of files cannot name them
  — `analyze` produced five figures in run 001 and four in run 002, so they went undeclared and the canvas
  listed them as working files. A pattern is still a contract: at least one match or the step fails, exactly as
  a missing literal does. Matching is `Path.glob`, not `fnmatch` (ignores the separator) or `PurePath.match`
  (matches from the right) — either would let `figures/*.png` claim `artifacts/figures/x.png`. The step event's
  `produces` stays unexpanded, because it is the declared contract; the `kind` on a file event is what resolves
  it. `03-analyze.md` now asks for *between one and five* figures rather than "up to five", so the prose
  requires what the contract enforces.
- 2026-09-16 — `dsagent.step` gains `produces_matched`: each declared entry mapped to the real files it names at
  event time, empty on `started` and filled on every other status. `produces` stays the promise, patterns and all.
  Keyed by entry rather than flattened, because with two patterns a flat list cannot say whether both were met.
- 2026-09-16 — Persona narration is withheld from the chat by a `messageView` rendering override, and shown in the
  DAG row for its step. The bridge attributes messages only inside a subagent window, opened solely for a tool named
  `task` (`agent.py:426`), which the runner does not use — so nothing on the wire says which persona is talking and
  the transcript reads as one voice changing personality mid-conversation. Attribution is by time: a message that
  starts between a step's `started` and its end is that step's. The slot was chosen over owning `CopilotChatView`
  (would mean driving input, suggestions, attachments and scroll) and over tagging in `serve` (a backend change for
  a presentation problem). Nothing about the run or the thread changes — the withheld messages still reach the model.
- 2026-09-16 — M2.2 is done: `eda-to-report` runs end to end from the browser, gate approved from the card
  (`docs/runs/eda-to-report-003.md`). All four canvas observations from run 001 are answered, and the run cost
  $0.47 against 002's $0.85 — model variance in `analyze`'s turn count, not anything the milestone did. The
  slice left seven operator-facing rough edges, tracked as M2.2.1; the first is that a successful run still
  ends in a red `terminated` error, because the orchestrator graph exhausts LangGraph's default recursion limit
  after the work is finished. A2UI panels are deferred past the slice. M2.3 (Docker env) is next.
