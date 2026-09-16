"""Profile a CSV/Parquet file: writes data-profile.md and data-profile.json.

Usage: python profile.py data/raw.csv artifacts/
"""

import json
import sys
from pathlib import Path

import pandas as pd


def profile(path: str, out_dir: str) -> dict:
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
        "source": str(p), "rows": int(len(df)), "columns": int(df.shape[1]),
        "exact_duplicates_pct": round(float(df.duplicated().mean() * 100), 2),
        "columns_detail": cols,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data-profile.json").write_text(json.dumps(summary, indent=2))
    md = [f"# Data profile — `{p.name}`", "",
          f"Rows: {summary['rows']:,} · Columns: {summary['columns']} · Exact duplicates: {summary['exact_duplicates_pct']} %", "",
          "| column | dtype | null % | distinct | notes |", "|---|---|---|---|---|"]
    for c in cols:
        note = (f"min {c['min']:.4g}, max {c['max']:.4g}, IQR outliers {c['outliers_iqr']}"
                if "min" in c else "top: " + ", ".join(f"{k} ({v})" for k, v in c["top"].items()))
        md.append(f"| {c['column']} | {c['dtype']} | {c['null_pct']} | {c['distinct']} | {note} |")
    (out / "data-profile.md").write_text("\n".join(md) + "\n")
    return summary


if __name__ == "__main__":
    s = profile(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "artifacts")
    print(f"profiled {s['rows']} rows × {s['columns']} cols → {sys.argv[2] if len(sys.argv) > 2 else 'artifacts'}/data-profile.*")
