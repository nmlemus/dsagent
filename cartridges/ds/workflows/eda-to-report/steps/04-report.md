Turn `artifacts/findings.md`, `artifacts/data-profile.md` and
`artifacts/data-gate.md` into the stakeholder report using the `reports` skill
structure.

Write `report/findings.md`, then render it to `report/findings.html` with the
`reports` skill script (`scripts/render_html.py`), which embeds the figures.
Check the HTML opens (file exists, non-empty, contains the three headline findings).
