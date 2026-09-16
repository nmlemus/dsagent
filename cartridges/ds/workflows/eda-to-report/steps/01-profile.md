Profile the dataset at `{data_path}` (path is workspace-relative; if it is absolute, copy it
to `data/raw.<ext>` first). The declared key column is `{key_column}`.

**Run the `eda` skill's `scripts/profile.py` — do not write your own profiler.** Use the
`run_skill_script` tool, never a path: skill files are not reachable from the shell.

```
run_skill_script(skill="eda", script="profile.py", argv=["{data_path}", "artifacts/"])
```

When the key column named above is not `None`, append `"--key-column", "{key_column}"` to
`argv` so the script records key uniqueness. When it is `None`, leave them out.

`artifacts/data-profile.json` belongs to the script. Its schema is the contract the next
step reads and evaluates thresholds from, so do not hand-write it, do not add keys to it,
and do not rename any of its keys. If the script fails, fix the call and run it again;
report the failure rather than working around it.

You may extend `artifacts/data-profile.md` — the script writes a baseline table, and
anything the script does not cover belongs there in prose: time coverage and gaps if there
is a date column, distribution notes, anything a reader needs to interpret the numbers.

**Do not evaluate gate thresholds and do not write a `GATE:` line.** Step `data-gate` owns
the verdict; if you pre-empt it, it does the work twice. Flagging something as worth a
closer look is fine; deciding pass or fail is not yours.

`data/` holds the run's inputs — do not write there unless a `produces` path says so.
Working files go under `artifacts/scratch/`.

Deliver `artifacts/data-profile.md` and `artifacts/data-profile.json`.
