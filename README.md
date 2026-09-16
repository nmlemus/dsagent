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
  cli.py       dsagent cartridge validate|list · run · chat
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
