# DSAgent

[![Upload Python Package](https://github.com/nmlemus/dsagent/actions/workflows/python-publish.yml/badge.svg)](https://github.com/nmlemus/dsagent/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/datascience-agent)](https://pypi.org/project/datascience-agent/)
[![Python](https://img.shields.io/pypi/pyversions/datascience-agent)](https://pypi.org/project/datascience-agent/)
[![CodeQL Advanced](https://github.com/nmlemus/dsagent/actions/workflows/codeql.yml/badge.svg)](https://github.com/nmlemus/dsagent/actions/workflows/codeql.yml)
[![License](https://img.shields.io/github/license/nmlemus/dsagent)](https://github.com/nmlemus/dsagent/blob/main/LICENSE)

An AI-powered autonomous agent for data science with persistent Jupyter kernel execution, session management, and conversational interface.

```
    ____  _____  ___                    __
   / __ \/ ___/ /   | ____ ____  ____  / /_
  / / / /\__ \ / /| |/ __ `/ _ \/ __ \/ __/
 / /_/ /___/ // ___ / /_/ /  __/ / / / /_
/_____//____//_/  |_\__, /\___/_/ /_/\__/
                   /____/
```

## Features

- **Conversational Interface**: Interactive chat with persistent context and sessions
- **Dynamic Planning**: Agent creates and follows plans with step tracking
- **Persistent Execution**: Code runs in a Jupyter kernel with variable persistence across messages
- **Session Management**: Save and resume conversations with full kernel state
- **Multi-Provider LLM**: Supports OpenAI, Anthropic, Google, Ollama via LiteLLM
- **MCP Tools**: Connect to external tools (web search, databases, etc.) via Model Context Protocol
- **Human-in-the-Loop**: Configurable checkpoints for plan and code approval
- **Notebook Generation**: Automatically generates clean, runnable Jupyter notebooks
- **Agent Skills**: Extensible skill system for specialized tasks (EDA, ML, etc.)

## Installation

```bash
pip install datascience-agent
```

With optional features:
```bash
pip install "datascience-agent[api]"   # FastAPI server support
pip install "datascience-agent[mcp]"   # MCP tools support
```

For development:
```bash
git clone https://github.com/nmlemus/dsagent
cd dsagent
uv sync --all-extras
```

### Docker

Configuration uses the same environment variables as the CLI and server (see [Configuration](docs/getting-started/configuration.md)). The container listens on `PORT` (default 8000).

```bash
# Run API server (default: port 8000)
docker run -d -p 8080:8080 \
  -e PORT=8080 \
  -e DSAGENT_DEFAULT_MODEL=gpt-4o \
  -e OPENAI_API_KEY=sk-your-key \
  nmlemus/dsagent:latest

# Run interactive CLI
docker run -it \
  -e OPENAI_API_KEY=sk-your-key \
  -v "$(pwd)/workspace:/workspace" \
  nmlemus/dsagent:latest \
  dsagent chat

# One-shot task
docker run --rm \
  -e OPENAI_API_KEY=sk-your-key \
  -v "$(pwd)/workspace:/workspace" \
  nmlemus/dsagent:latest \
  dsagent run "Analyze data/sales.csv" --data ./data/sales.csv
```

For Docker deployment details, see [docs/DOCKER.md](docs/DOCKER.md) and [docs/guide/docker.md](docs/guide/docker.md).

## Quick Start

### 1. Setup (First Time)

Run the setup wizard to configure your LLM provider:

```bash
dsagent init
```

This will:
- Ask for your LLM provider (OpenAI, Anthropic, Google, local, etc.)
- Store your API key securely in `~/.dsagent/.env`
- Automatically select a default model based on provider:
  - OpenAI → `gpt-4o`
  - Anthropic → `claude-sonnet-4-5`
  - Google → `gemini/gemini-2.5-flash`
  - Local → `ollama/llama3`
- Optionally configure MCP tools (web search, etc.)

To use a different model, set `DSAGENT_DEFAULT_MODEL` or `LLM_MODEL` in `~/.dsagent/.env`, or use the `--model` flag:
```bash
dsagent --model gpt-4o-mini
```

### 2. Start Chatting

```bash
dsagent
```

This starts an interactive session where you can:
- Chat naturally with the agent
- Execute Python code with persistent variables
- Analyze data files
- Generate visualizations
- Resume previous sessions

### 3. One-Shot Tasks

For batch processing or scripts:

```bash
dsagent run "Analyze sales trends" --data ./sales.csv
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `dsagent` | Start interactive chat (default) |
| `dsagent chat` | Same as above, with explicit options |
| `dsagent run "task"` | Execute a one-shot task |
| `dsagent serve` | Run REST + WebSocket API server |
| `dsagent init` | Setup wizard for configuration |
| `dsagent skills list` | List installed skills |
| `dsagent skills install <source>` | Install a skill from GitHub or path |
| `dsagent skills remove <name>` | Remove a skill |
| `dsagent skills info <name>` | Show skill details |
| `dsagent mcp list` | List configured MCP servers |
| `dsagent mcp add <template>` | Add an MCP server from template |
| `dsagent mcp remove <name>` | Remove an MCP server |

### Examples

```bash
# Interactive chat with specific model
dsagent --model claude-sonnet-4-5

# One-shot analysis
dsagent run "Find patterns in this data" --data ./dataset.csv

# Resume a previous session
dsagent --session abc123

# With MCP tools (web search)
dsagent --mcp-config ~/.dsagent/mcp.yaml

# Human-in-the-loop mode
dsagent --hitl plan
```

For complete CLI documentation, see [docs/CLI.md](docs/CLI.md).

## Python API

DSAgent provides two agents for different use cases:

### ConversationalAgent (Interactive)

For building chat interfaces and interactive applications:

```python
from dsagent import ConversationalAgent, ConversationalAgentConfig

config = ConversationalAgentConfig(model="gpt-4o")
agent = ConversationalAgent(config)
agent.start()

# Chat with persistent context
response = agent.chat("Load the iris dataset")
print(response.content)

response = agent.chat("Train a classifier on it")
print(response.content)  # Has access to previous variables

agent.shutdown()
```

### PlannerAgent (Batch)

For one-shot tasks and automated pipelines:

```python
from dsagent import PlannerAgent

with PlannerAgent(model="gpt-4o", data="./data.csv") as agent:
    result = agent.run("Analyze this dataset and create visualizations")
    print(result.answer)
    print(f"Notebook: {result.notebook_path}")
```

For complete API documentation, see [docs/PYTHON_API.md](docs/PYTHON_API.md).

## Supported Models

DSAgent uses [LiteLLM](https://docs.litellm.ai/) to support 100+ LLM providers:

| Provider | Models | API Key |
|----------|--------|---------|
| OpenAI | `gpt-4o`, `o1`, `o3-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5`, `claude-opus-4` | `ANTHROPIC_API_KEY` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| DeepSeek | `deepseek/deepseek-r1` | `DEEPSEEK_API_KEY` |
| Ollama | `ollama/llama3.2` | None (local) |

For detailed model setup, see [docs/MODELS.md](docs/MODELS.md).

## MCP Tools

Connect to external tools via the Model Context Protocol:

```bash
# Add web search capability
dsagent mcp add brave-search

# Use it in chat
dsagent --mcp-config ~/.dsagent/mcp.yaml
```

Available templates: `brave-search`, `filesystem`, `github`, `memory`, `fetch`, `bigquery`

For MCP configuration details, see [docs/MCP.md](docs/MCP.md).

## Session Management

Sessions persist your conversation history and kernel state:

```bash
# List sessions
dsagent chat
> /sessions

# Resume a session
dsagent --session <session-id>

# Export session to notebook
> /export myanalysis.ipynb
```

## Output Structure

Each run creates organized output:

```
workspace/
└── runs/{run_id}/
    ├── data/           # Input data (copied)
    ├── notebooks/      # Generated Jupyter notebooks
    ├── artifacts/      # Charts, models, exports
    └── logs/           # Execution logs
```

## Included Libraries

DSAgent comes with essential data science libraries pre-installed:

| Category | Libraries |
|----------|-----------|
| **Core** | numpy, pandas, scipy |
| **DataFrames** | polars, pyarrow |
| **Visualization** | matplotlib, seaborn, plotly |
| **Machine Learning** | scikit-learn, xgboost, lightgbm, pycaret |
| **Feature Selection** | boruta |
| **Statistics** | statsmodels |

## Documentation

- [CLI Reference](docs/CLI.md) - All commands: chat, run, serve, init, mcp, skills
- [Configuration](docs/getting-started/configuration.md) - Environment variables and `.env`
- [HTTP API](docs/api/http-api.md) - REST and WebSocket API reference
- [Python API](docs/PYTHON_API.md) - ConversationalAgent and PlannerAgent
- [Model Configuration](docs/MODELS.md) - LLM provider setup
- [MCP Tools](docs/MCP.md) - External tools integration
- [Agent Skills](docs/guide/skills.md) - Extensible skill system
- [Docker Guide](docs/DOCKER.md) - Container deployment

## License

MIT
