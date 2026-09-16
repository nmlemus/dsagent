# CLAUDE.md — dsagent-v2

Read `docs/architecture.md` once before touching anything. `docs/ROADMAP.md` is the
task list; work top-down, one task per branch/PR, and tick it when done.

## Invariants (do not break these)

1. **The harness is domain-agnostic.** Nothing under `src/dsagent/` may mention data
   science, MMM, Meridian, personas by name, or any skill by name. If you need that,
   it belongs in `cartridges/ds/`.
2. **A cartridge is a valid Claude Code plugin plus `cartridge.yaml`.** Never add a
   field to `agents/*.md` frontmatter that Claude Code would reject; DSAgent-only
   config goes in `cartridge.yaml`, `workflows/` or `envs/`.
3. **Skill scope lives in `cartridge.yaml` and is mirrored in agent frontmatter.**
   The loader refuses to start when they disagree — keep it that way. Do not add
   implicit inheritance of skills.
4. **The runner never plans.** Workflows are declared DAGs; the model runs *inside*
   a step, never chooses the next step. Gates are runner-level.
5. **Every step's `produces` is verified on disk.** Do not relax this to "warn".
6. **Deep Agents owns the agent loop.** Do not reimplement planning, context
   management, tool loops or summarization. If Deep Agents lacks something, wrap it
   in middleware; don't fork the loop.
7. **Command skills `skills/<cartridge>-<workflow>/` are generated.** Never hand-edit
   them; fix `workflow.yaml` or `generate_command_skills()`.

## Stack

Python ≥ 3.11 · `deepagents` 0.7.x · pydantic 2 · typer · jupyter-client. Install with
`pip install -e ".[dev,anthropic]"`. Model via `DSAGENT_MODEL` (default
`anthropic:claude-sonnet-4-6`).

## Commands

```
pytest                                  # must stay green; runner tests use a fake agent, no model
dsagent cartridge validate cartridges/ds
dsagent cartridge list cartridges/ds
dsagent run eda-to-report -i data_path=data/sales.csv --dry-run
```

## Conventions

- Tests for harness behaviour go in `tests/` and must not need a model, a kernel or
  Docker (use the `FakeAgent` / stub env pattern in `tests/test_runner.py`).
  Integration runs that need a model go under `tests/integration/` and are skipped
  unless `DSAGENT_INTEGRATION=1`.
- Step instructions (`workflows/*/steps/*.md`) are written *to the persona*, in
  English, and may use `{input_name}` placeholders.
- Keep `docs/architecture.md` in sync when a decision changes; note the change in
  `docs/ROADMAP.md` under "Decisions log".
- Commit messages: `<area>: <what>` (areas: harness, cartridge-ds, docs, tests).
