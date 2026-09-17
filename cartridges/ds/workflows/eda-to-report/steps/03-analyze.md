Answer this question with the data: **{question}**

Read `artifacts/data-profile.md` and `artifacts/data-gate.md` first so you know what to
trust. Use `run_python` (state persists) for the analysis. Produce between one and five figures
under `artifacts/figures/` following the chart rules in the `reports` skill — every
finding in *Evidence* rests on one, so a run with no figure has nothing to show.

Write `artifacts/findings.md` with exactly three sections, in this order:

1. **Headline findings** — exactly three, one sentence each, action-oriented.
2. **Evidence** — one section per finding: the chart it rests on, then numbers with their
   denominators.
3. **Caveats** — the ones that come from the gate report, and only those that change how a
   finding should be read.

Nothing else. **"What the data cannot answer" and "Next steps" belong to the report
step** — if you write them here, the report step has to merge two versions of them.

Descriptive and comparative analysis, with one exception: **if a finding claims a
direction over time, fit the trend and report its slope with an interval.** The chart
rules in the `reports` skill do not let you assert a trend you have not tested, and a
trend you cannot test is not a headline finding — say "no clear trend" instead. Nothing
beyond that: no predictive models, no causal claims, no train/test work.

`data/` is read-only. Working files go under `artifacts/scratch/`.
