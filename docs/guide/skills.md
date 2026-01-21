# Skills

Skills extend DSAgent with reusable knowledge packages that the agent can use to perform specialized tasks. Each skill contains instructions and optional scripts that help the agent solve specific types of problems.

## Overview

```
~/.dsagent/
├── skills/                 # Installed skills directory
│   ├── eda-analysis/
│   │   ├── SKILL.md        # Instructions and metadata
│   │   └── scripts/        # Reusable Python scripts
│   ├── ml-training/
│   └── my-custom-skill/
└── skills.yaml             # Registry of installed skills
```

Skills are **not** bundled with DSAgent. You install only what you need, similar to MCP servers.

## Quick Reference

```bash
dsagent skills list              # List installed skills
dsagent skills install <source>  # Install a skill
dsagent skills remove <name>     # Remove a skill
dsagent skills info <name>       # Show skill details
```

---

## Installing Skills

### From GitHub

```bash
# Full path to skill in a repository
dsagent skills install github:anthropics/claude-cookbooks/skills/custom_skills/creating-financial-models

# Short form for repos with skill at root
dsagent skills install github:dsagent-skills/eda-analysis
```

### From Local Directory

```bash
dsagent skills install ./my-local-skill
```

### Example Installation

```bash
$ dsagent skills install github:dsagent-skills/eda-analysis

Installing skill from: github:dsagent-skills/eda-analysis

Successfully installed: eda-analysis

  Description: Comprehensive exploratory data analysis for datasets
  Version: 1.0.0
  Dependencies: pandas, matplotlib, seaborn
  Scripts: 3

The agent will automatically use this skill when relevant.
```

---

## Using Skills

Once installed, skills are **automatically available** to the agent. The agent:

1. Sees all installed skills in its context
2. Recognizes when a skill is relevant to your request
3. Uses the skill's instructions and scripts to help you

### Example Conversation

```
You: "Do a quick EDA on my sales data"

Agent: I'll use the eda-analysis skill to analyze your data.

[Agent executes skill scripts, shows statistics and visualizations]

The dataset has 50,000 rows and 15 columns. Here are the key findings...
```

### Interactive Skill Commands

In chat sessions, you can also use slash commands:

| Command | Description |
|---------|-------------|
| `/skills` | List available skills |
| `/skill <name>` | Show skill details |

---

## Managing Skills

### List Installed Skills

```bash
$ dsagent skills list

Installed Skills
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Name          ┃ Version ┃ Description                              ┃ Scripts ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ eda-analysis  │ 1.0.0   │ Comprehensive exploratory data analysis  │ 3       │
│ ml-training   │ 1.2.0   │ Machine learning model training          │ 5       │
└───────────────┴─────────┴──────────────────────────────────────────┴─────────┘

Skills directory: /Users/you/.dsagent/skills
```

### View Skill Details

```bash
$ dsagent skills info eda-analysis

eda-analysis
Version: 1.0.0

Description:
  Comprehensive exploratory data analysis for datasets

Author: DSAgent Team

Tags: eda, analysis, pandas

Python Dependencies:
  - pandas>=2.0
  - matplotlib>=3.7
  - seaborn>=0.12

Scripts:
  - basic_stats.py - Generate basic statistics
  - correlation.py - Correlation analysis and heatmaps
  - missing_analysis.py - Missing value analysis

Location:
  /Users/you/.dsagent/skills/eda-analysis

Instructions:
  When the user asks for basic data analysis or EDA, use this skill's scripts...
```

### Remove a Skill

```bash
$ dsagent skills remove eda-analysis

Removing skill: eda-analysis
Successfully removed: eda-analysis
```

---

## Creating Custom Skills

Skills follow the [Agent Skills](https://github.com/agentskills/agentskills) standard.

### Skill Structure

```
my-skill/
├── SKILL.md              # Required: metadata + instructions
├── scripts/              # Optional: reusable Python scripts
│   ├── script1.py
│   └── script2.py
└── examples/             # Optional: usage examples
```

### SKILL.md Format

```markdown
---
name: my-skill
description: Brief description of what the skill does
version: "1.0.0"
author: Your Name
tags:
  - category1
  - category2
compatibility:
  python:
    - pandas>=2.0
    - numpy>=1.20
---

# My Skill

Detailed instructions for the agent on how to use this skill.

## When to Use

Describe scenarios when this skill should be activated.

## Available Scripts

### script1.py

Explain what this script does and what variables it expects.

**Required variables:**
- `df`: The DataFrame to process

**Example usage:**
\`\`\`python
exec(open('~/.dsagent/skills/my-skill/scripts/script1.py').read())
\`\`\`
```

### Key Elements

| Element | Required | Description |
|---------|----------|-------------|
| `name` | Yes | Unique identifier for the skill |
| `description` | Yes | Brief description (shown in listings) |
| `version` | No | Semantic version (default: "1.0.0") |
| `author` | No | Skill author/maintainer |
| `tags` | No | Categories for organization |
| `compatibility.python` | No | Required Python packages |

### Best Practices

1. **Clear instructions**: Write detailed instructions for the agent
2. **Parameterized scripts**: Use expected variables rather than hardcoded values
3. **Error handling**: Include try/except in scripts
4. **Documentation**: Explain all scripts and their requirements
5. **Examples**: Include usage examples in the SKILL.md

---

## Available Skills

Community-maintained skills are available at:

- **[dsagent-skills](https://github.com/dsagent-skills)** - Official DSAgent skills
- Skills compatible with [Agent Skills](https://github.com/agentskills/agentskills) standard

### Recommended Skills

```bash
# Exploratory Data Analysis
dsagent skills install github:dsagent-skills/eda-analysis

# Machine Learning Training
dsagent skills install github:dsagent-skills/ml-training

# Data Loading (multiple sources)
dsagent skills install github:dsagent-skills/data-loading

# Visualization
dsagent skills install github:dsagent-skills/visualization
```

---

## Troubleshooting

### Skill Not Found

```bash
$ dsagent skills install github:user/nonexistent

Installation error: Could not find skill at github:user/nonexistent
```

Verify the repository URL and path are correct.

### Already Installed

```bash
$ dsagent skills install github:dsagent-skills/eda-analysis

Skill 'eda-analysis' is already installed. Use --force to overwrite.
```

Use `--force` to reinstall:

```bash
dsagent skills install --force github:dsagent-skills/eda-analysis
```

### Validation Error

Skills must have a valid `SKILL.md` with required frontmatter. Check that:
- The `name` field is present
- The `description` field is present
- YAML frontmatter is correctly formatted
