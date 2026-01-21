"""Basic statistics for a DataFrame.

Required variables:
- df: pandas DataFrame to analyze
"""

import pandas as pd

# Verify df exists
if 'df' not in dir():
    raise NameError("Variable 'df' must be defined before running this script")

print("=" * 60)
print("BASIC DATA ANALYSIS")
print("=" * 60)

# Shape
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

# Data types
print("\nData Types:")
print("-" * 40)
for col, dtype in df.dtypes.items():
    print(f"  {col}: {dtype}")

# Missing values
print("\nMissing Values:")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
if missing.sum() == 0:
    print("  No missing values!")

# Basic statistics
print("\nNumeric Statistics:")
print("-" * 40)
print(df.describe().round(2).to_string())

# Categorical columns summary
cat_cols = df.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    print("\nCategorical Columns:")
    print("-" * 40)
    for col in cat_cols:
        unique = df[col].nunique()
        print(f"  {col}: {unique} unique values")
        if unique <= 10:
            top = df[col].value_counts().head(5)
            for val, count in top.items():
                print(f"    - {val}: {count}")

print("\n" + "=" * 60)
