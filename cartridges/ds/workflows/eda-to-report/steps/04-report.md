Turn `artifacts/findings.md`, `artifacts/data-profile.md` and `artifacts/data-gate.md`
into the stakeholder report.

**Read the `reports` skill and only that skill.** Everything you need about the data is
already written down in the three artifacts above; the gate thresholds do not need
re-reading, because the gate report states its own verdict and flags.

Follow the `reports` structure. Two sections are yours to write rather than copy: "What
this data cannot answer" (from the gate's flags and the profile's coverage) and "Next
steps". The three headline findings come from `findings.md` unchanged unless they
contradict the gate.

Write `report/findings.md`, then render it to `report/findings.html` with the `reports`
skill script:

```
python /skills/marie/reports/scripts/render_html.py report/findings.md report/findings.html
```

Check the HTML opens (file exists, non-empty, contains the three headline findings and the
figures as embedded images).

`data/` is read-only. Working files go under `artifacts/scratch/`.
