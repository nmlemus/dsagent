"""Plot distributions for numeric columns.

Required variables:
- df: pandas DataFrame to analyze

Optional variables:
- columns: List of columns to plot (defaults to all numeric columns)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Verify df exists
if 'df' not in dir():
    raise NameError("Variable 'df' must be defined before running this script")

# Get columns to plot
if 'columns' not in dir() or columns is None:
    columns = df.select_dtypes(include=['number']).columns.tolist()

if len(columns) == 0:
    print("No numeric columns to plot!")
else:
    # Calculate grid size
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Handle single plot case
    if len(columns) == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        df[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

        # Add statistics
        mean = df[col].mean()
        median = df[col].median()
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
        ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"Plotted distributions for {len(columns)} columns")
