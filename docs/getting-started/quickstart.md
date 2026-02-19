# Quick Start

This guide will get you analyzing data with DSAgent in under 5 minutes.

## Prerequisites

- DSAgent installed ([Installation Guide](installation.md))
- An API key from your LLM provider

## 1. Configure Your LLM

If you haven't run the setup wizard yet:

```bash
dsagent init
```

Or set your API key directly:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== "Google"

    ```bash
    export GOOGLE_API_KEY="..."
    ```

## 2. Start a Chat Session

Launch the interactive chat:

```bash
dsagent
```

You'll see the DSAgent prompt:

```
DSAgent | Model: gpt-4o | Session: a1b2c3

Type your message or /help for commands.

You:
```

## 3. Analyze Some Data

Let's try a simple analysis. Type:

```
You: Create a sample dataset with sales data and analyze it
```

The agent will:

1. Generate sample data
2. Perform exploratory analysis
3. Create visualizations
4. Provide insights

All code executes in a persistent Jupyter kernel, so variables are available for follow-up questions.

## 4. Follow Up

Continue the conversation naturally:

```
You: Show me the monthly trends

You: Which product category has the highest growth?

You: Create a forecast for next quarter
```

## 5. Work with Your Own Data

Analyze your own files:

```
You: Load and analyze /path/to/my/data.csv
```

Or start with data from the command line:

```bash
dsagent run "Analyze this sales data" --data ./sales.csv
```

## 6. Export Your Work

Export the session as a Jupyter notebook:

```
You: /export my_analysis.ipynb
```

The notebook contains all code executed during the session, ready to share or modify.

## Common Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/sessions` | List previous sessions |
| `/export <file>` | Export to notebook |
| `/clear` | Clear conversation |
| `/quit` | Exit DSAgent |

## Example: Complete Analysis

Here's a complete example session:

```
You: Load the iris dataset from sklearn and do a complete analysis

Agent: I'll load the iris dataset and perform a comprehensive analysis.

[Executing code...]

✓ Loaded iris dataset
  - 150 samples, 4 features
  - 3 classes: setosa, versicolor, virginica

📊 Summary Statistics:
  - Sepal Length: 4.3 - 7.9 (mean: 5.84)
  - Sepal Width: 2.0 - 4.4 (mean: 3.06)
  - Petal Length: 1.0 - 6.9 (mean: 3.76)
  - Petal Width: 0.1 - 2.5 (mean: 1.20)

📈 Saved: artifacts/iris_pairplot.png

Key findings:
1. Setosa is clearly separable from other species
2. Petal measurements are more discriminative than sepal
3. Versicolor and virginica overlap slightly

Would you like me to train a classifier?

You: Yes, try multiple algorithms and compare them

Agent: [Executing code...]

📊 Model Comparison:

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Random Forest      | 96.7%    | 0.967    |
| SVM (RBF)          | 96.7%    | 0.966    |
| Logistic Regression| 100.0%   | 1.000    |
| KNN (k=5)          | 96.7%    | 0.967    |

Logistic Regression achieved perfect accuracy on this test split!

📈 Saved: artifacts/model_comparison.png
📈 Saved: artifacts/confusion_matrices.png

You: /export iris_analysis.ipynb

✓ Exported to: workspace/notebooks/iris_analysis.ipynb
```

## Next Steps

- [CLI Reference](../guide/cli.md) - All command-line options
- [Python API](../api/overview.md) - Build applications with DSAgent
- [MCP Tools](../guide/mcp.md) - Add web search and other tools
- [Docker Deployment](../guide/docker.md) - Run DSAgent in containers
