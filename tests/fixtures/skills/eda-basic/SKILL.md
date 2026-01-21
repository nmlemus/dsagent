---
name: eda-basic
description: Basic exploratory data analysis for pandas DataFrames
version: "1.0.0"
author: DSAgent Team
tags:
  - eda
  - analysis
  - pandas
compatibility:
  python:
    - pandas>=2.0
    - matplotlib>=3.7
---

# EDA Basic Skill

This skill provides basic exploratory data analysis capabilities for pandas DataFrames.

## Usage Instructions

When the user asks for basic data analysis or EDA, use this skill's scripts.

### Available Scripts

#### 1. `basic_stats.py` - Generate basic statistics

**Required variables:**
- `df`: The pandas DataFrame to analyze

**Example usage:**
```python
# Make sure 'df' is defined with your DataFrame
exec(open('~/.dsagent/skills/eda-basic/scripts/basic_stats.py').read())
```

This script will output:
- Shape of the DataFrame
- Data types
- Missing value counts
- Basic statistics (describe)

#### 2. `plot_distributions.py` - Plot column distributions

**Required variables:**
- `df`: The pandas DataFrame to analyze
- `columns` (optional): List of columns to plot. If not set, plots all numeric columns.

**Example usage:**
```python
# Define df and optionally columns
df = your_dataframe
columns = ['price', 'quantity']  # Optional
exec(open('~/.dsagent/skills/eda-basic/scripts/plot_distributions.py').read())
```

## When to Use This Skill

Use this skill when:
- User asks for "basic analysis" or "EDA"
- User wants to understand the structure of their data
- User asks for statistics or distributions
