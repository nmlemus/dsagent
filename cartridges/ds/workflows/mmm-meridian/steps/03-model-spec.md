Write the model specification for a Meridian MMM on `data/mmm_input.parquet` with
KPI `{kpi}`.

Follow the model-spec checklist in the `mmm` skill item by item. Every prior must
cite where it comes from (past test, benchmark, or "uninformative — flag for
review"). Produce:

- `artifacts/model-spec.md` — the checklist answered in prose, with a table of
  channels × (max lag, ROI prior mean, ROI prior sd, source).
- `model/spec.py` — a runnable module that builds `InputData` and `ModelSpec`
  from the table and exposes `build() -> (input_data, model_spec)`.

Do not fit anything in this step; a human reviews the spec first.
