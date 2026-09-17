Evaluate every gate threshold in the `eda` skill against `artifacts/data-profile.json`
and write `artifacts/data-gate.md`.

**Read the numbers from the profile JSON. Do not reload the dataset and do not recompute
anything the profile already contains** — null shares, duplicate share, distinct counts,
outlier counts and `key_uniqueness` are all in there. The profile is the evidence; your
job is the verdict on it. Recomputing costs a second pass over the data and invites the
two of you to disagree.

Use `key_uniqueness` for the key check: `unique: false` (or a non-zero `duplicates`) fails
the grain. When `column` is `null` no key was declared — record the check as not
applicable rather than inventing one.

Cover, one row per check: null share per column, exact duplicate rows, key uniqueness,
and time coverage / missing periods where a date column exists. Give each row its value,
its threshold and its denominator.

The file must end with exactly one of:

- `GATE: PASS`
- `GATE: FAIL — <one-line reason>`

If it fails, still write the report; the human will decide whether to continue. State
clearly which columns or periods are affected and what a fix would look like. Anything
that does not breach a threshold but changes how the data can be read belongs in a
"Flags" section above the verdict — that is what the analysis step will carry as caveats.

**Show the checks as a table.** The person deciding this gate reads your four rows
before answering it, so write them to `artifacts/scratch/gate-checks.csv` (check, value,
threshold, result) and emit them with `show_table`:

```
show_table(data_ref="artifacts/scratch/gate-checks.csv",
           title="Data gate — <n> of <n> checks")
```

That table is what the gate card shows as *what is being approved*. Write it before you
finish, or the decision is made on the prompt alone.

`data/` is read-only here. Working files go under `artifacts/scratch/`.
