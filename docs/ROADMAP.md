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

### M2.1 First real run of `eda-to-report`
- [ ] First commit and push as branch `v2` of `nmlemus/dsagent` (decided 2026-09-16; `main` stays v1 until 2.0 ships)
- [ ] Add `tests/integration/test_eda_to_report.py` (skipped unless `DSAGENT_INTEGRATION=1`) using a small public CSV
- [ ] Run it with a real model; capture what the personas actually do in `docs/runs/eda-to-report-001.md` (prompt gaps, tool misuse, cost, wall time)
- [ ] Fix step instructions / skills based on that run (expect 2–3 iterations)
- [ ] Runner: stream step events (start/tool/end) instead of only `log()`, so `chat` and a future API can show progress
- [ ] Runner: `produces` glob support (`artifacts/figures/*.png`)

### M2.2 Docker env
- [ ] `DockerBackend` integration test behind `DSAGENT_DOCKER=1` (build `envs/meridian`, `execute("python -c 'import meridian'")`)
- [ ] Workspace bind-mount + file ownership sanity (non-root user in image)
- [ ] Auto-gate scripts run *inside* the step env, not on the host (`_gate` currently shells out on the host)
- [ ] Env lifecycle: reuse container across steps of the same run; always `rm -f` on exit/failure

### M2.3 `mmm-meridian` on the Meridian sample dataset
- [ ] `cartridges/ds/skills/mmm/scripts/load_sample.py` — fetch Google's simulated dataset into `data/`
- [ ] Run steps ingest → data-gate → model-spec with a real model; review Ana's spec by hand
- [ ] Run fit (CPU, small draws) → check_rhat auto gate → optimize → report
- [ ] Document the run in `docs/runs/mmm-meridian-001.md` with cost and wall time

## M3 — Portability proof
- [ ] Install `cartridges/ds` as a Claude Code plugin; run Marie and Noel there with zero changes
- [ ] Minimal `cartridges/sdlc/` (3 BMAD-style personas, 1 workflow) proving the harness is domain-agnostic
- [ ] `dsagent cartridge add <git-url>` (port v1 installer)

## M4 — API + connectors
- [ ] FastAPI + WebSocket parity with v1 (`server/` port): runs, gates over WS, artifacts
- [ ] BigQuery connector via cartridge `.mcp.json` + LangChain MCP adapters
- [ ] Run resumability across process restarts (already in `run.json`; needs API surface)

## Decisions log

- 2026-09-16 — v2 lives as branch `v2` in `nmlemus/dsagent` (not a new repo). `main` keeps v1 and the `datascience-agent` PyPI line until 2.0 ships, then `v2` merges to `main` as 2.0.0.
- 2026-09-16 — Backend on Deep Agents (not Claude Agent SDK): model-agnostic, sandbox protocol fits Docker-per-workflow, same SKILL.md standard as Claude Code.
- 2026-09-16 — Cartridge = Claude Code plugin + `cartridge.yaml`. Workflows exposed three ways (persona request, `/ds-<wf>` command skill, orchestrator intent), all ending in `run_workflow`; a persona can only start workflows listed on it.
- 2026-09-16 — Gates are runner-level (not LangGraph interrupts) so runs pause/resume from `run.json` without a checkpointer. Revisit when the API needs tool-level approvals.
- 2026-09-16 — Skill scoping materialised on disk per persona (copy, not symlink) and mounted via `CompositeBackend` at `/skills/<persona>/`.
