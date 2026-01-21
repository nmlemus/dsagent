# CLI Reference

DSAgent provides a unified command-line interface with subcommands for different use cases.

## Quick Reference

```bash
dsagent                          # Start interactive chat (default)
dsagent chat                     # Same as above
dsagent run "task"               # Execute one-shot task
dsagent init                     # Setup wizard
dsagent skills list              # List installed skills
dsagent skills install <source>  # Install a skill
dsagent skills remove <name>     # Remove a skill
dsagent mcp list                 # List MCP servers
dsagent mcp add <template>       # Add MCP server
dsagent mcp remove <name>        # Remove MCP server
dsagent --version                # Show version
dsagent --help                   # Show help
```

---

## `dsagent` / `dsagent chat`

Start an interactive conversational session with the agent.

```bash
dsagent [options]
dsagent chat [options]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | LLM model to use | `gpt-4o` or `$LLM_MODEL` |
| `--workspace` | `-w` | Workspace directory | `./workspace` |
| `--session` | `-s` | Session ID to resume | New session |
| `--hitl` | | Human-in-the-loop mode | `none` |
| `--live-notebook` | | Save notebook after each execution | Off |
| `--notebook-sync` | | Bidirectional sync with Jupyter | Off |
| `--mcp-config` | | Path to MCP servers YAML config | None |

### HITL Modes

| Mode | Description |
|------|-------------|
| `none` | Fully autonomous (default) |
| `plan` | Pause for plan approval before execution |
| `full` | Pause for both plan and code approval |
| `plan_answer` | Pause for plan and final answer approval |
| `on_error` | Pause only when errors occur |

### Examples

```bash
# Start with default settings
dsagent

# Use Claude model
dsagent --model claude-sonnet-4-5

# Resume a previous session
dsagent --session 20260108_143022_abc123

# Require plan approval
dsagent --hitl plan

# Enable live notebook sync
dsagent --live-notebook

# With MCP tools
dsagent --mcp-config ~/.dsagent/mcp.yaml

# Combine options
dsagent -m gpt-4o --hitl plan --live-notebook
```

### Interactive Commands

Once in a chat session, you can use slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/new` | Start a new session |
| `/sessions` | List all sessions |
| `/session <id>` | Switch to a session |
| `/export [file]` | Export session to notebook |
| `/clear` | Clear the screen |
| `/model <name>` | Change the model |
| `/skills` | List installed skills |
| `/skill <name>` | Show skill details |
| `/status` | Show current status |
| `/quit` | Exit the session |

---

## `dsagent run`

Execute a one-shot task without interactive mode.

```bash
dsagent run "task description" [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `task` | The task to execute (required) |

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--data` | `-d` | Path to data file or directory | None |
| `--model` | `-m` | LLM model to use | `gpt-4o` |
| `--workspace` | `-w` | Workspace directory | `./workspace` |
| `--max-rounds` | `-r` | Maximum agent iterations | `30` |
| `--quiet` | `-q` | Suppress verbose output | Off |
| `--hitl` | | Human-in-the-loop mode | `none` |
| `--mcp-config` | | Path to MCP config | None |

### Examples

```bash
# Analyze a CSV file
dsagent run "Analyze this dataset and find trends" --data ./sales.csv

# Code generation (no data)
dsagent run "Write a REST API client for GitHub"

# With specific model
dsagent run "Build ML model" -d ./dataset -m claude-sonnet-4-5

# Custom output directory
dsagent run "Create visualizations" -d ./data -w ./output

# Quiet mode with max rounds
dsagent run "Complex analysis" -d ./data -q -r 50

# With MCP tools
dsagent run "Search for Python best practices" --mcp-config ~/.dsagent/mcp.yaml
```

### Output

Each run creates an isolated workspace:

```
workspace/
└── runs/{run_id}/
    ├── data/           # Input data (copied)
    ├── notebooks/      # Generated notebooks
    ├── artifacts/      # Images, charts, models
    └── logs/
        ├── run.log     # Human-readable log
        └── events.jsonl # Structured events
```

---

## `dsagent init`

Interactive setup wizard for first-time configuration.

```bash
dsagent init [--force]
```

### Options

| Option | Description |
|--------|-------------|
| `--force` | Overwrite existing configuration |

### What it configures

1. **LLM Provider**: Choose from OpenAI, Anthropic, Google, local (Ollama), or LiteLLM proxy
2. **API Keys**: Securely store in `~/.dsagent/.env`
3. **Default Model**: Automatically selected based on provider:
   - OpenAI → `gpt-4o`
   - Anthropic → `claude-sonnet-4-5`
   - Google → `gemini/gemini-2.5-flash`
   - Local → `ollama/llama3`
4. **MCP Tools**: Optionally configure web search, filesystem access, etc.

To use a different model after setup, either:
- Edit `LLM_MODEL` in `~/.dsagent/.env`
- Use the `--model` flag: `dsagent --model gpt-4o-mini`

### Example

```bash
$ dsagent init

DSAgent Setup Wizard

Step 1: LLM Provider
Select your LLM provider [openai/anthropic/local/litellm]: openai
Enter your OpenAI API key: ********

Step 2: MCP Tools (optional)
Would you like to configure MCP tools? [y/N]: y
Add Brave Search? [Y/n]: y
Enter your Brave Search API key: ********

Writing configuration...
  Created /Users/you/.dsagent/.env
  Created /Users/you/.dsagent/mcp.yaml

Setup complete!
```

---

## `dsagent mcp`

Manage MCP (Model Context Protocol) server configurations.

### `dsagent mcp list`

List all configured MCP servers.

```bash
$ dsagent mcp list

Configured MCP Servers
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name         ┃ Transport ┃ Command                                    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ brave_search │ stdio     │ npx -y @modelcontextprotocol/server-brave… │
│ filesystem   │ stdio     │ npx -y @modelcontextprotocol/server-files… │
└──────────────┴───────────┴────────────────────────────────────────────┘

Config file: /Users/you/.dsagent/mcp.yaml
```

### `dsagent mcp add <template>`

Add an MCP server from a template.

```bash
dsagent mcp add brave-search
dsagent mcp add filesystem
dsagent mcp add github
dsagent mcp add memory
dsagent mcp add fetch
dsagent mcp add bigquery
```

#### Available Templates

| Template | Description | Required |
|----------|-------------|----------|
| `brave-search` | Web search via Brave API | `BRAVE_API_KEY` |
| `filesystem` | Local file system access | Paths to allow |
| `github` | GitHub repository access | `GITHUB_TOKEN` |
| `memory` | Persistent memory/knowledge base | None |
| `fetch` | Fetch and parse web content | None |
| `bigquery` | Google BigQuery access | Toolbox path, `BIGQUERY_PROJECT` |

### `dsagent mcp remove <name>`

Remove an MCP server by name.

```bash
dsagent mcp remove brave_search
```

---

## `dsagent skills`

Manage agent skills for extended capabilities. Skills extend DSAgent with reusable knowledge packages. See [Skills](skills.md) for complete documentation.

### `dsagent skills list`

List all installed skills.

```bash
$ dsagent skills list

Installed Skills
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Name          ┃ Version ┃ Description                              ┃ Scripts ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ eda-analysis  │ 1.0.0   │ Comprehensive exploratory data analysis  │ 3       │
└───────────────┴─────────┴──────────────────────────────────────────┴─────────┘

Skills directory: /Users/you/.dsagent/skills
```

### `dsagent skills install <source>`

Install a skill from GitHub or a local directory.

```bash
# From GitHub
dsagent skills install github:dsagent-skills/eda-analysis
dsagent skills install github:user/repo/path/to/skill

# From local directory
dsagent skills install ./my-local-skill

# Force reinstall
dsagent skills install --force github:dsagent-skills/eda-analysis
```

### `dsagent skills remove <name>`

Remove an installed skill.

```bash
dsagent skills remove eda-analysis
```

### `dsagent skills info <name>`

Show detailed information about a skill.

```bash
dsagent skills info eda-analysis
```

---

## Environment Variables

DSAgent reads configuration from environment variables and `.env` files.

### Search Order for `.env`

1. Current working directory: `./.env`
2. Project root: `{project}/.env`
3. User config: `~/.dsagent/.env`

### Configuration Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_MODEL` | Default model | `gpt-4o` |
| `LLM_API_BASE` | Custom API endpoint | `http://localhost:4000/v1` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google API key | `...` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | `...` |
| `BRAVE_API_KEY` | Brave Search API key | `...` |

### Using LiteLLM Proxy

For routing through a LiteLLM proxy server:

```bash
export LLM_API_BASE="http://localhost:4000/v1"
export LLM_MODEL="openai/claude-sonnet-4-5"
export OPENAI_API_KEY="your-proxy-key"

dsagent
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (configuration, execution, etc.) |

---

## Tips

### Model Selection

```bash
# Quick model switching
dsagent -m gpt-4o           # OpenAI GPT-4o
dsagent -m claude-sonnet-4-5 # Anthropic Claude
dsagent -m ollama/llama3.2  # Local Ollama
```

### Session Workflow

```bash
# Start a session, note the session ID
dsagent

# Later, resume it
dsagent -s 20260108_143022_abc123

# Or use interactive commands
dsagent
> /sessions
> /session 20260108_143022_abc123
```

### Data Analysis Workflow

```bash
# Analyze data with output in specific directory
dsagent run "EDA and visualization" -d ./data.csv -w ./analysis

# Check outputs
ls ./analysis/runs/*/artifacts/
ls ./analysis/runs/*/notebooks/
```
