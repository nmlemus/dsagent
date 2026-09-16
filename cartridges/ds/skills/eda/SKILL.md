---
name: eda
description: Exploratory data analysis and data quality gates — profiling checklist, null/outlier/duplicate thresholds, and the gate report format. Use before any modeling or reporting on a new dataset.
---

# EDA and data gate

## Profiling checklist (run in this order)

1. Load with explicit dtypes; record shape, memory, source path and load time.
2. Per column: dtype, null %, distinct count, min/max (numeric), top-5 values
   (categorical), min/max date and gaps (datetime).
3. Duplicates: exact rows, and duplicates on the business key if one is known.
4. Outliers: for numeric columns report values beyond 1.5×IQR and beyond 3σ; do
   not drop them here, flag them.
5. Time coverage: expected vs. observed periods; list missing periods.
6. Save the profile as `artifacts/data-profile.md` (tables) and the raw stats as
   `artifacts/data-profile.json`.

`scripts/profile.py <file> <out_dir>` does 1–6 for CSV/Parquet.

## Gate thresholds (defaults; a workflow may override in its step instructions)

| Check | Fail when |
|---|---|
| Null share on a required column | > 20 % |
| Exact duplicate rows | > 1 % |
| Missing time periods | any, inside the requested range |
| Key column not unique at declared grain | any violation |

## Gate report

`artifacts/data-gate.md`: one table with check / value / threshold / result, then
a single final line — `GATE: PASS` or `GATE: FAIL — <reason>`. Workflow gates read
that line.
