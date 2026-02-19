# Repository Guidelines

## Project Structure & Module Organization
- `src/dsagent/` holds the core library code (agents, core runtime, schema, tools, utils).
- `tests/` contains pytest suites (e.g., `test_context.py`, `test_mcp.py`).
- `docs/` and `examples/` provide reference material and usage samples.
- `demo/` contains demo assets or runnable examples.
- `pyproject.toml` is the single source for packaging, tooling, and configuration.

## Build, Test, and Development Commands
- `uv venv --python 3.11` creates a local virtual environment.
- `uv sync --all-extras` installs dev + optional extras for local development.
- `uv run pytest` runs the test suite under `tests/`.
- `uv run ruff check .` runs lint checks; add `--fix` to auto-apply fixes.
- `uv run mypy src/` performs static type checking on the library.

## Coding Style & Naming Conventions
- Python 3.10+ codebase; follow PEP 8 with `ruff` rules (line length 100).
- Type hints are required for functions; add docstrings for public APIs.
- Use standard module names and keep package code under `src/dsagent/`.
- Prefer descriptive, action-oriented names (e.g., `run_logger.py`, `test_notebook.py`).

## Testing Guidelines
- Tests are written with `pytest` and `pytest-asyncio`.
- Place tests in `tests/` and name files/functions `test_*.py` / `test_*`.
- New features and fixes should include or update tests; no explicit coverage target is defined.

## Commit & Pull Request Guidelines
- **Do not commit directly to `main`.** For every change: document in a GitHub issue → create a branch → implement → test → push → open a PR. See `.cursor/rules/workflow-branch-pr.mdc`.
- Follow Git Flow: branch from `develop` (or `main`) and PR back to `develop`/`main`.
- Branch names: `feature/*`, `bugfix/*`, `fix/*`, `release/vX.Y.Z`, `hotfix/*`.
- Commit messages use `type: short description` (e.g., `feat: add streaming API endpoint`).
- PRs should include a clear description, pass tests, and update docs when needed.

## Configuration Tips
- Optional extras exist for API and MCP integrations; use `uv sync --all-extras` during dev.
- CLI entrypoint is `dsagent` (see `src/dsagent/cli.py`).
