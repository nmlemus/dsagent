# Docker Guide

DSAgent provides Docker images for easy deployment of both the CLI and API server.

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nmlemus/dsagent
   cd dsagent
   ```

2. **Set your API key:**
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   # Or for other providers:
   # export ANTHROPIC_API_KEY=sk-ant-...
   # export GOOGLE_API_KEY=...
   ```

3. **Start the API server:**
   ```bash
   docker-compose up -d
   ```

4. **Access the API:**
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Using Docker Hub

```bash
# Pull the image
docker pull nmlemus/dsagent:latest

# Run API server
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  nmlemus/dsagent:latest

# Run interactive CLI
docker run -it \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/workspace:/workspace \
  nmlemus/dsagent:latest \
  dsagent chat
```

## Configuration

### Environment Variables

#### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | LLM model to use | `gpt-4o` |
| `OPENAI_API_KEY` | OpenAI API key (for gpt-* models) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for claude-* models) | - |
| `GOOGLE_API_KEY` | Google API key (for gemini/* models) | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `LLM_API_BASE` | Custom API endpoint (for proxies) | - |
| `OLLAMA_API_BASE` | Ollama API endpoint | `http://host.docker.internal:11434` |

#### Agent Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DSAGENT_MAX_ROUNDS` | Maximum agent iterations | `30` |
| `DSAGENT_TEMPERATURE` | LLM temperature | `0.3` |
| `DSAGENT_MAX_TOKENS` | Max tokens per response | `4096` |
| `DSAGENT_CODE_TIMEOUT` | Code execution timeout (seconds) | `300` |

#### API Server

| Variable | Description | Default |
|----------|-------------|---------|
| `DSAGENT_API_KEY` | API authentication key | - (disabled) |
| `DSAGENT_CORS_ORIGINS` | CORS allowed origins | `*` |

#### MCP Tools (Optional)

| Variable | Description |
|----------|-------------|
| `BRAVE_API_KEY` | Brave Search API key |
| `FINANCIAL_DATASETS_API_KEY` | Financial Datasets API key |

### Using a .env File (Recommended)

Create a `.env` file in the same directory as `docker-compose.yml`:

```bash
# .env
# LLM Configuration
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-openai-key

# Or for other providers:
# LLM_MODEL=claude-sonnet-4-5
# ANTHROPIC_API_KEY=sk-ant-your-key

# Agent Settings (optional)
DSAGENT_MAX_ROUNDS=30
DSAGENT_TEMPERATURE=0.3

# API Server (optional)
DSAGENT_API_KEY=my-secret-api-key

# MCP Tools (optional)
BRAVE_API_KEY=your-brave-key
```

Then start with docker-compose:
```bash
docker-compose up -d
```

Or with docker run:
```bash
docker run -d -p 8000:8000 --env-file .env nmlemus/dsagent:latest
```

**Note:** Only set the API key for your chosen provider.

### Volumes

| Path | Description |
|------|-------------|
| `/workspace` | Session data, notebooks, artifacts |
| `/home/dsagent/.dsagent` | User configuration |

## Usage Examples

### API Server Mode

```bash
# Start server with default settings
docker-compose up -d

# Start with specific model
LLM_MODEL=claude-sonnet-4-5 ANTHROPIC_API_KEY=sk-ant-... docker-compose up -d

# View logs
docker-compose logs -f dsagent

# Stop server
docker-compose down
```

### Interactive CLI Mode

```bash
# Using docker-compose
docker-compose run --rm dsagent-cli

# Or directly with docker
docker run -it \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/workspace:/workspace \
  nmlemus/dsagent:latest \
  dsagent chat
```

### One-Shot Task

```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data.csv:/workspace/data.csv \
  nmlemus/dsagent:latest \
  dsagent run "Analyze this dataset" --data /workspace/data.csv
```

## LaTeX Support (Docker Only)

LaTeX is available in the `:full` variant of the Docker image for generating PDF reports and presentations.

### Image Variants

| Tag | Size | LaTeX | Use Case |
|-----|------|-------|----------|
| `dsagent:latest` | ~1GB | No | Standard data science tasks |
| `dsagent:full` | ~1.5GB | Yes | When you need PDF report generation |

```bash
# Build without LaTeX (default)
docker build -t dsagent:latest .

# Build with LaTeX
docker build -t dsagent:full --build-arg INSTALL_LATEX=true .
```

**Available LaTeX tools (`:full` only):** `pdflatex`, `xelatex`, `latexmk`

### Example: Generate PDF Report

Ask the agent to create a report:
```
Create a PDF report summarizing the analysis with charts
```

The agent can:
1. Create a `.tex` file with your analysis results
2. Include generated charts from `artifacts/`
3. Compile to PDF using `!pdflatex report.tex`

### Manual LaTeX Compilation

```bash
# Inside the container
docker exec -it dsagent /bin/bash

# Compile LaTeX
cd /workspace/artifacts
pdflatex report.tex
```

**Note:** LaTeX adds ~500MB to the image size. It's only available in the Docker image, not in pip installations.

## Building Locally

```bash
# Build the image
docker build -t dsagent:local .

# Run with local build
docker run -it \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  dsagent:local \
  dsagent chat
```

## Docker Compose Profiles

The `docker-compose.yml` includes two services:

1. **dsagent** (default): API server on port 8000
2. **dsagent-cli** (profile: cli): Interactive CLI

```bash
# Start API server only
docker-compose up -d

# Start CLI session
docker-compose --profile cli run --rm dsagent-cli
```

## Security Notes

- The container runs as a non-root user (`dsagent`)
- API key authentication is optional but recommended for production
- Use `DSAGENT_API_KEY` to enable API authentication
- Restrict `DSAGENT_CORS_ORIGINS` in production

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs dsagent
```

### API key not working

Ensure the environment variable is passed correctly:
```bash
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" ...
```

### Permission denied on workspace

The container user needs write access:
```bash
chmod 777 ./workspace
# Or run as root (not recommended):
docker run --user root ...
```
