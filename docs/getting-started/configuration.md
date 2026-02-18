# Configuration

DSAgent uses a single configuration source: `dsagent.config` (see `src/dsagent/config.py`). All settings support environment variables with the `DSAGENT_` prefix and optional `.env` files.

## Configuration Methods

Configuration is loaded in this order (later sources override earlier):

1. Default values in code
2. `~/.dsagent/.env` — global (CLI loads this first)
3. `./.env` — local project (CLI overrides global with this)
4. Environment variables already set in the shell
5. Command-line arguments (highest priority for CLI)
6. API request fields (e.g. `model` in create-session) for the server

This allows API keys in `~/.dsagent/.env` and per-project overrides in `./.env`.

## Environment Variables

All `DSAGENT_*` variables are read by both CLI and API server (where applicable).

### LLM and model

| Variable | Description | Default |
|----------|-------------|---------|
| `DSAGENT_DEFAULT_MODEL` | Default LLM model (recommended) | `gpt-4o` |
| `LLM_MODEL` | Legacy default model (CLI compatible) | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `LLM_API_BASE` | Custom API endpoint (e.g. LiteLLM proxy) | - |
| `OLLAMA_API_BASE` | Ollama server URL | `http://localhost:11434` |

### Agent settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DSAGENT_MAX_ROUNDS` | Maximum agent iterations | `30` |
| `DSAGENT_TEMPERATURE` | LLM temperature (0.0–2.0) | `0.3` |
| `DSAGENT_MAX_TOKENS` | Max tokens per response | `4096` |
| `DSAGENT_CODE_TIMEOUT` | Code execution timeout (seconds) | `300` |
| `DSAGENT_WORKSPACE` | Workspace directory | `./workspace` |
| `DSAGENT_SESSIONS_DIR` | Sessions directory name under workspace | `workspace` |
| `DSAGENT_SESSION_BACKEND` | Session store: `sqlite` or `json` | `sqlite` |

### API server

| Variable | Description | Default |
|----------|-------------|---------|
| `DSAGENT_API_KEY` | If set, API and WebSocket require `X-API-Key` header | - (disabled) |
| `DSAGENT_CORS_ORIGINS` | Comma-separated origins or `*` | `*` |
| `DSAGENT_REQUIRE_API_KEY` | If `true`, server refuses to start without `DSAGENT_API_KEY` | `false` |
| `DSAGENT_MAX_UPLOAD_MB` | Max upload size per file in MB; `0` = no limit | `50` |

## Configuration File

Create a `.env` file in `~/.dsagent/` for persistent **global** configuration:

```bash
# ~/.dsagent/.env - Global config (API keys, default model)

# Default model (DSAGENT_* preferred; LLM_MODEL still supported)
DSAGENT_DEFAULT_MODEL=gpt-4o

# API keys — add the providers you use
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GROQ_API_KEY=gsk_your-key-here
GOOGLE_API_KEY=your-key-here

# Optional agent settings
DSAGENT_MAX_ROUNDS=30
DSAGENT_TEMPERATURE=0.3
```

Create a `.env` file in your **project directory** to override settings for that project:

```bash
# ~/my-project/.env - Project-specific overrides

DSAGENT_DEFAULT_MODEL=groq/llama-3.3-70b-versatile
# or LLM_MODEL=claude-sonnet-4-20250514
```

The setup wizard (`dsagent init`) creates the global file automatically.

## Command-Line Options

Override settings per-session:

```bash
# Use a different model
dsagent --model claude-sonnet-4-5

# Custom workspace
dsagent --workspace /path/to/workspace

# Resume a session
dsagent --session abc123

# Enable human-in-the-loop
dsagent --hitl plan
```

## Human-in-the-Loop Modes

Control when the agent asks for approval:

| Mode | Description |
|------|-------------|
| `none` | No approval required (default) |
| `plan` | Approve plan before execution |
| `full` | Approve plan and each code block |
| `plan_answer` | Approve plan and final answer |
| `on_error` | Ask for guidance on errors |

```bash
dsagent --hitl plan
```

## MCP Tools Configuration

MCP tools are configured in `~/.dsagent/mcp.yaml`:

```yaml
mcpServers:
  brave-search:
    command: npx
    args: ["-y", "@anthropic/mcp-brave-search"]
    env:
      BRAVE_API_KEY: "your-brave-api-key"
```

Add tools using the CLI:

```bash
dsagent mcp add brave-search
dsagent mcp add filesystem
```

See [MCP Tools](../guide/mcp.md) for available tools and configuration.

## Workspace Structure

DSAgent organizes output under the workspace directory (`DSAGENT_WORKSPACE`, default `./workspace`). Session storage uses `DSAGENT_SESSIONS_DIR` (default `workspace`) and the backend (`DSAGENT_SESSION_BACKEND`: `sqlite` or `json`).

Typical layout:

```
workspace/                    # or DSAGENT_WORKSPACE
├── .dsagent/
│   └── sessions.db          # SQLite sessions (when session_backend=sqlite)
└── runs/
    └── {session_id}/        # Per-session workspace
        ├── data/            # Input/uploaded data
        ├── artifacts/       # Charts, models, exports
        ├── notebooks/       # Generated Jupyter notebooks
        └── logs/            # Execution logs, events
```

## Proxy Configuration

For corporate environments or custom endpoints:

```bash
# Use a proxy server
export LLM_API_BASE="https://your-proxy.com/v1"
export OPENAI_API_KEY="your-proxy-key"

# Route through LiteLLM proxy
dsagent --model openai/gpt-4o
```

## Docker Configuration

See [Docker Guide](../guide/docker.md) for container-specific configuration options.
