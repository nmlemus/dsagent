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
- [ ] **Runner: stream step events (start/tool/end) instead of only `log()`** — the foundation for both `chat` progress and the AG-UI bridge. Each event carries the step's `produces`, so a consumer can tell a deliverable from a working file (runs 001/002)
- [ ] `dsagent serve`: FastAPI + `ag-ui-langgraph` (`LangGraphAGUIAgent`, `add_langgraph_fastapi_endpoint`) exposing the orchestrator; `CopilotKitMiddleware()` in the graph
- [ ] Map that event stream onto AG-UI `CUSTOM` events `dsagent.step` and `dsagent.gate`; human gates become HITL interrupts answered from the UI
- [ ] `ui/` Next.js + CopilotKit: chat left, canvas right; canvas lists workspace files as the filesystem middleware streams them (iframe for `.html`, markdown, PNG, table for `.csv/.parquet`)
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
