# DSAgent v2

A thin harness on top of [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
that loads **cartridges**: swappable bundles of personas, skills, workflows and
execution environments. The harness is domain-agnostic; the DS cartridge in
`cartridges/ds` turns it into a data science team. Swap in a BMAD-style cartridge
and it becomes an SDLC team.

See `docs/architecture.md` for the design. This repo is milestone **M1**: harness
skeleton, kernel env, the `eda-to-report` workflow end to end, CLI. The Docker env
and the `mmm-meridian` workflow are wired but need a Docker daemon (M2).

## Layout

```
src/dsagent/
  capabilities/ versioned host providers + immutable per-run composition
  cartridge/   models + loader (+ matrix validation, /ds-<workflow> command generation)
  host/        cartridge → Deep Agents (persona agents, orchestrator with subagents)
  runner/      declared-DAG workflow runner: envs per step, produces checks, gates, resume
  envs/        kernel (local shell + persistent Jupyter) and docker (BaseSandbox) backends
  cli.py       dsagent cartridge validate|list · run · chat · serve
  serve.py     FastAPI: the orchestrator over AG-UI, plus a run's files over HTTP
cartridges/ds/ the data science cartridge — also a valid Claude Code plugin
```

## Quick start

```bash
pip install -e ".[dev,anthropic]"
export ANTHROPIC_API_KEY=...            # or DSAGENT_MODEL=openai:gpt-... + OPENAI_API_KEY

dsagent cartridge validate cartridges/ds
dsagent cartridge list cartridges/ds

# walk the DAG without calling a model
dsagent run eda-to-report -i data_path=data/sales.csv --dry-run

# real run (pauses at the human gate; -y auto-approves)
mkdir -p .dsagent/runs/demo/workspace/data && cp your.csv .dsagent/runs/demo/workspace/data/sales.csv
dsagent run eda-to-report --run-id demo -i data_path=data/sales.csv

# resume after approving a gate
dsagent run eda-to-report --run-id demo -i data_path=data/sales.csv --resume

# chat with the orchestrator ("Ana, hazme un MMM" → run_workflow; "/ds-mmm-meridian" works too)
dsagent chat -c cartridges/ds
```

## Serving the UI

`dsagent serve` puts the same orchestrator behind [AG-UI](https://docs.ag-ui.com),
which is what the web UI in `ui/` will talk to. It needs the `ui` extra:

```bash
pip install -e ".[ui,anthropic]"
dsagent serve                            # 127.0.0.1:8000, --host/--port to change
```

Principal routes:

| Route | What it is |
|---|---|
| `POST /agent` | The conversational orchestrator over AG-UI. Workflow tools submit jobs to the execution service and return a run id. `GET /agent/health` reports liveness. |
| `POST /runs`, `POST /runs/{id}/start` | Create and start a workflow directly, without asking an orchestrator model to launch it. |
| `GET /runs/{id}`, `GET /runs/{id}/events` | Durable state and live progress over SSE. |
| `POST /runs/{id}/gate` | Answer with `{"gate_id": "<current gate UUID>", "decision": "approve", "note": ""}`. The decision survives server restarts. |
| `GET /runs/{run_id}/files/{path}` | One file from a run's workspace, so the canvas can render the artifacts those events announce. Resolved inside `<run_dir>/workspace` — anything escaping it, and anything under `.dsagent/`, is a 404. |

The UI lives in [`ui/`](ui/) — Next.js + CopilotKit, chat left and canvas right:

```bash
# terminal 1
dsagent serve                          # http://127.0.0.1:8000/agent

# terminal 2
cd ui && npm install && npm run dev     # http://localhost:3000
```

`npm run build` is the UI's check for now; there are no frontend tests yet.
`DSAGENT_URL` points the UI at a different agent, and `NEXT_PUBLIC_DSAGENT_ORIGIN`
is where the *browser* reaches it for artifact links.

A run's workspace starts empty, and its directory is named after the tool call, so
nothing can put a dataset in it beforehand. `--seed DIR` copies a directory into
each new run's workspace:

```bash
mkdir -p /tmp/seed/data && cp tests/data/seattle-weather.csv /tmp/seed/data/
dsagent serve --seed /tmp/seed     # data_path is then `data/seattle-weather.csv`
```

When a workflow reaches a human gate, the run pauses and a gate card appears in the
right pane with the step, its persona, and links to what it produced. Approving
resumes the run.

Launcher and chat use the same job owner. Closing a tab does not stop a job;
restarting the server preserves pending gates and accepted decisions. A job whose
process died during a step becomes `interrupted` and requires explicit retry after
reviewing possible external effects. This implementation coordinates processes
on one POSIX host with local storage; it is not a distributed queue. See the
[execution service ADR](docs/execution-service.md).

Workflow jobs now run in supervised local subprocesses. **Cancel execution** stops
the active worker and cleans up its registered resources. `Step.timeout` accepts
durations such as `30s`, `5m` or `1h` (default: one hour). Unverified cleanup blocks
retries and is exposed in the UI. See [process control](docs/process-control.md)
for recovery behavior and the single-host boundary.

## Cartridge contract

A cartridge is a Claude Code plugin (`.claude-plugin/plugin.json`, `agents/`,
`skills/`) plus `cartridge.yaml` (persona ↔ skill matrix, workflows, envs and
required executable capabilities),
`workflows/<name>/workflow.yaml` and `envs/<name>/Dockerfile`. The loader refuses to
start when an agent's frontmatter `skills:` disagrees with the matrix, so the
cartridge behaves the same inside Claude Code and inside DSAgent. Command skills
`skills/<cartridge>-<workflow>/SKILL.md` are regenerated on every load.
Capabilities are resolved and version-checked before execution; every run pins
the resolved manifest in `run.json`. See
[`docs/capability-composition.md`](docs/capability-composition.md).

## Tests

```bash
pytest
```

The runner tests use a fake agent and a stub env: no model, kernel or Docker needed.
