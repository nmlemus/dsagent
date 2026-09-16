"""Profile a CSV/Parquet file: writes data-profile.json and data-profile.md.

**The JSON is this script's output and its schema is the contract.** Later steps
read it and evaluate gate thresholds from it, so it must be produced by running
this script — not hand-written, not "extended" with renamed keys. Add prose and
extra tables to the markdown instead.

Usage:
    python profile.py data/raw.csv artifacts/
    python profile.py data/raw.csv artifacts/ --key-column date
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1


def profile(path: str, out_dir: str, key_column: str | None = None) -> dict:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "column": c,
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 2),
            "distinct": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            info.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
                        outliers_iqr=int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()))
        else:
            info["top"] = {str(k): int(v) for k, v in s.value_counts(dropna=True).head(5).items()}
        cols.append(info)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": str(p), "rows": int(len(df)), "columns": int(df.shape[1]),
        "exact_duplicates_pct": round(float(df.duplicated().mean() * 100), 2),
        "key_uniqueness": _key_uniqueness(df, key_column),
        "columns_detail": cols,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data-profile.json").write_text(json.dumps(summary, indent=2))
    (out / "data-profile.md").write_text(_markdown(p, summary, cols))
    return summary


def _key_uniqueness(df: pd.DataFrame, key_column: str | None) -> dict:
    """Always present so the schema is stable; nulls when no key was declared."""
    if not key_column or key_column == "None":
        return {"column": None, "unique": None, "duplicates": None}
    if key_column not in df.columns:
        sys.exit(f"--key-column {key_column!r} is not a column: {', '.join(map(str, df.columns))}")
    duplicates = int(df[key_column].duplicated().sum())
    return {"column": key_column, "unique": duplicates == 0, "duplicates": duplicates}


def _markdown(p: Path, summary: dict, cols: list[dict]) -> str:
    key = summary["key_uniqueness"]
    key_line = (
        f" · Key `{key['column']}`: {key['duplicates']} duplicate(s)"
        if key["column"] else " · Key column: not declared"
    )
    md = [f"# Data profile — `{p.name}`", "",
          f"Rows: {summary['rows']:,} · Columns: {summary['columns']} · "
          f"Exact duplicates: {summary['exact_duplicates_pct']} %{key_line}", "",
          "| column | dtype | null % | distinct | notes |", "|---|---|---|---|---|"]
    for c in cols:
        note = (f"min {c['min']:.4g}, max {c['max']:.4g}, IQR outliers {c['outliers_iqr']}"
                if "min" in c else "top: " + ", ".join(f"{k} ({v})" for k, v in c["top"].items()))
        md.append(f"| {c['column']} | {c['dtype']} | {c['null_pct']} | {c['distinct']} | {note} |")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("out_dir", nargs="?", default="artifacts")
    ap.add_argument("--key-column", default=None,
                    help="column that must be unique at the declared grain")
    a = ap.parse_args()
    s = profile(a.path, a.out_dir, a.key_column)
    print(f"profiled {s['rows']} rows × {s['columns']} cols → {a.out_dir}/data-profile.*")
