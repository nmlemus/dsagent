Using `artifacts/data-profile.json` and the gate thresholds in the `eda` skill,
evaluate every check and write `artifacts/data-gate.md`.

The file must end with exactly one of:

- `GATE: PASS`
- `GATE: FAIL — <one-line reason>`

If it fails, still write the report; the human will decide whether to continue.
State clearly which columns or periods are affected and what a fix would look like.
