---
hide:
  - navigation
  - toc
---

# DSAgent

<p align="center" style="font-size: 1.4em; color: #666;">
  AI-powered autonomous agent for data science tasks
</p>

<p align="center">
  <a href="https://pypi.org/project/datascience-agent/"><img src="https://img.shields.io/pypi/v/datascience-agent" alt="PyPI"></a>
  <a href="https://pypi.org/project/datascience-agent/"><img src="https://img.shields.io/pypi/pyversions/datascience-agent" alt="Python"></a>
  <a href="https://github.com/nmlemus/dsagent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/nmlemus/dsagent" alt="License"></a>
  <a href="https://hub.docker.com/r/nmlemus/dsagent"><img src="https://img.shields.io/docker/pulls/nmlemus/dsagent" alt="Docker"></a>
</p>

---

<div class="grid cards" markdown>

-   :material-chat-processing:{ .lg .middle } __Conversational Interface__

    ---

    Interactive chat with persistent context and session management. Just describe what you want to analyze.

-   :material-file-tree:{ .lg .middle } __Dynamic Planning__

    ---

    Agent creates and follows structured plans with step-by-step execution and progress tracking.

-   :material-language-python:{ .lg .middle } __Persistent Execution__

    ---

    Code runs in a Jupyter kernel with variable persistence across messages. No context lost.

-   :material-cloud-sync:{ .lg .middle } __Multi-Provider LLM__

    ---

    Supports OpenAI, Anthropic, Google, DeepSeek, Ollama via LiteLLM. Use any model you prefer.

</div>

## Quick Start

=== "pip"

    ```bash
    pip install datascience-agent
    dsagent init  # Configure your LLM
    dsagent       # Start chatting
    ```

=== "Docker"

    ```bash
    docker run -it \
      -e OPENAI_API_KEY=$OPENAI_API_KEY \
      -v $(pwd)/workspace:/workspace \
      nmlemus/dsagent:latest \
      dsagent chat
    ```

=== "From Source"

    ```bash
    git clone https://github.com/nmlemus/dsagent
    cd dsagent
    uv sync --all-extras
    dsagent init
    ```

## Example Session

```
$ dsagent

DSAgent | Model: gpt-4o | Session: abc123

You: Load the iris dataset and train a classifier

Agent: I'll load the iris dataset and train a Random Forest classifier.

[Executing code...]

✓ Loaded iris dataset (150 samples, 4 features)
✓ Split into train/test (80/20)
✓ Trained RandomForestClassifier
✓ Accuracy: 96.7%

The model achieved 96.7% accuracy on the test set. Would you like me to:
1. Show the confusion matrix?
2. Analyze feature importance?
3. Try a different algorithm?

You: Show feature importance

Agent: [Executing code...]

📊 Saved: artifacts/feature_importance.png

The most important features are:
1. petal length (44.2%)
2. petal width (42.1%)
3. sepal length (9.8%)
4. sepal width (3.9%)
```

## Features

| Feature | Description |
|---------|-------------|
| **Session Management** | Save and resume conversations with full kernel state |
| **Notebook Generation** | Automatically generates clean, runnable Jupyter notebooks |
| **MCP Tools** | Connect to external tools (web search, databases) via Model Context Protocol |
| **Human-in-the-Loop** | Configurable checkpoints for plan and code approval |
| **Docker Ready** | Official images with LaTeX support for PDF reports |

## Supported Models

| Provider | Models | Setup |
|----------|--------|-------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5`, `claude-opus-4` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` | `GOOGLE_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat`, `deepseek/deepseek-r1` | `DEEPSEEK_API_KEY` |
| Ollama | `ollama/llama3.2`, `ollama/codestral` | Local (no key) |

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Installation__

    ---

    Get DSAgent installed and configured in minutes.

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Learn the basics with a hands-on tutorial.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-api:{ .lg .middle } __Python API__

    ---

    Build applications with the DSAgent SDK.

    [:octicons-arrow-right-24: API Reference](api/overview.md)

-   :material-docker:{ .lg .middle } __Docker Deployment__

    ---

    Deploy DSAgent in containers.

    [:octicons-arrow-right-24: Docker Guide](guide/docker.md)

</div>
