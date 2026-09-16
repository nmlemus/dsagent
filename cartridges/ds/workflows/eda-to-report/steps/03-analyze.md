Answer this question with the data: **{question}**

Read `artifacts/data-profile.md` and `artifacts/data-gate.md` first so you know what to
trust. Use `run_python` (state persists) for the analysis. Produce up to five figures
under `artifacts/figures/` following the chart rules in the `reports` skill.

Write `artifacts/findings.md` with exactly three sections, in this order:

1. **Headline findings** — exactly three, one sentence each, action-oriented.
2. **Evidence** — one section per finding: the chart it rests on, then numbers with their
   denominators.
3. **Caveats** — the ones that come from the gate report, and only those that change how a
   finding should be read.

Nothing else. **"What the data cannot answer" and "Next steps" belong to the report
step** — if you write them here, the report step has to merge two versions of them.

No modeling in this step: descriptive and comparative analysis only.

`data/` is read-only. Working files go under `artifacts/scratch/`.
