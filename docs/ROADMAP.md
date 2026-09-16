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

### M2.2 UI vertical slice (see docs/ui.md) — moved up: the product is the UI
- [x] **Runner: stream step events instead of only `log()`** — `RunnerEvent` + `on_event` on `WorkflowRunner`, emitting `dsagent.step` / `dsagent.tool` / `dsagent.file`; personas run with `.stream()` so tool and file events arrive while the step is still working. Each step event carries `produces`, so a consumer can tell a deliverable from a working file (runs 001/002). No AG-UI dependency
- [x] Design PR: `docs/ui-slice.md` — verified package APIs, run-inside-a-tool design, event schemas, frontend plan, PR breakdown
- [x] Runner: stable gate-`interrupt()` sequence on re-entry (`StepRecord.gate` split from `status`) and a deterministic `run_id` derived from `tool_call_id`, with the two-gate resume test and a re-entry test asserting the same `run_dir` is reopened
- [x] `dsagent serve`: FastAPI + `ag-ui-langgraph` (`add_langgraph_fastapi_endpoint`) + `copilotkit` (`LangGraphAGUIAgent`, `CopilotKitMiddleware()`) exposing the orchestrator, plus `/runs/{id}/files/{path}`. Optional `[ui]` extra; base dependencies unchanged
- [x] Dispatch runner events as LangChain custom events (`dsagent/runner/dispatch.py`), which `ag-ui-langgraph` forwards as AG-UI `CUSTOM` with the same name and value. Pinned by a test that runs the runner inside a sync tool under an async `astream_events` consumer, and by a timing test: the first event arrives while the tool is still working
- [x] Human gates become LangGraph `interrupt()`s answered from the UI (`ask_human` → `interrupt`), on the resume rules already built. `GateRequest` gives the hook the step context a gate card needs
- [x] `ui/` Next.js + CopilotKit shell: chat left, empty canvas right, `/api/copilotkit` relaying to `dsagent serve` via `HttpAgent`. Verified from the browser against a real model (`docs/runs/ui-shell-001.png`)
- [ ] Canvas contents: workspace files from `dsagent.file` (iframe for `.html`, markdown, PNG, table for `.csv/.parquet`)
- [ ] Gate card (`request_approval` render) → approve/reject → runner resumes
- [ ] Workflow progress render: DAG with step status from `dsagent.step` events
- [ ] Run `eda-to-report` end to end from the browser; screenshot in `docs/runs/`
- [ ] Then: A2UI panels for agent-composed views; MCP Apps later if needed

### M2.3 Docker env
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
- [ ] Runner: `produces` glob support (`artifacts/figures/*.png`)
- [ ] chore: `dsagent --version` prints "Missing command" (eager callback vs `no_args_is_help`)

## Decisions log

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
