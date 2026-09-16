Ingest the MMM inputs from `{data_source}` at `{data_path}` for KPI `{kpi}`
(date range: `{date_range}`; "None" means use everything available).

Read the `mmm` skill (modeling-table schema and `references/input_schema.md`) and
the `eda` skill. Build `data/mmm_input.parquet` in long `geo × time` format with the
column groups the skill defines, and keep raw extracts under `data/raw/`.

Then profile the modeling table (run the `eda` skill's `scripts/profile.py`) and
extend the profile with: calendar completeness per geo, channels with zero spend for
more than 25 % of periods, and any currency/unit inconsistencies you detect.
Deliver `artifacts/data-profile.md`.
