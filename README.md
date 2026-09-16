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

Two routes:

| Route | What it is |
|---|---|
| `POST /agent` | The orchestrator as an AG-UI SSE stream. A workflow's progress arrives as `CUSTOM` events named `dsagent.step`, `dsagent.tool` and `dsagent.file`; a human gate arrives as an interrupt the client answers with `{"decision": "approve"}`. `GET /agent/health` reports liveness. |
| `GET /runs/{run_id}/files/{path}` | One file from a run's workspace, so the canvas can render the artifacts those events announce. Resolved inside `<run_dir>/workspace` — anything escaping it, and anything under `.dsagent/`, is a 404. |

The UI lives in [`ui/`](ui/) — Next.js + CopilotKit, chat left and canvas right:

```bash
# terminal 1
dsagent serve                          # http://127.0.0.1:8000/agent

# terminal 2
cd ui && npm install && npm run dev     # http://localhost:3000
```

`npm run build` is the UI's check for now; there are no frontend tests yet.
`DSAGENT_URL` points the UI at a different agent.

Each browser tab is one `thread_id`, which is what a gate resumes into. The slice
checkpoints in memory, so a `serve` restart loses in-flight gates; `run.json` still
has every finished step, and the run resumes from there. Event schemas and the
frontend plan are in [`docs/ui-slice.md`](docs/ui-slice.md).

## Cartridge contract

A cartridge is a Claude Code plugin (`.claude-plugin/plugin.json`, `agents/`,
`skills/`) plus `cartridge.yaml` (persona ↔ skill matrix, workflows, envs),
`workflows/<name>/workflow.yaml` and `envs/<name>/Dockerfile`. The loader refuses to
start when an agent's frontmatter `skills:` disagrees with the matrix, so the
cartridge behaves the same inside Claude Code and inside DSAgent. Command skills
`skills/<cartridge>-<workflow>/SKILL.md` are regenerated on every load.

## Tests

```bash
pytest
```

The runner tests use a fake agent and a stub env: no model, kernel or Docker needed.
