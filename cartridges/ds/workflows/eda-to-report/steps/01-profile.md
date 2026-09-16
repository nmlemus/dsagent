Profile the dataset at `{data_path}` (path is workspace-relative; if it is absolute, copy it to `data/raw.<ext>` first).

Read the `eda` skill and follow its checklist. Prefer running its `scripts/profile.py`
via `run_python` or `execute`, then extend the profile with anything the script does
not cover (time coverage and gaps if there is a date column; key uniqueness if a
`key_column` input was given).

Deliver `artifacts/data-profile.md` and `artifacts/data-profile.json`.
