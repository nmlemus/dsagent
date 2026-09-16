---
name: reports
description: How this team writes reports — structure, executive summary rules, chart conventions and the HTML/Markdown templates. Use whenever producing a report, memo or findings document for stakeholders.
---

# Reports

Every report in this team follows the same shape so stakeholders can read any of
them in two minutes.

## Structure

1. **Headline findings** — exactly three, one sentence each, action-oriented
   ("Retail media is saturated above $120k/week; shift 15% to CTV").
2. **Context** — what question was asked, what data was used (source, period, grain,
   row count), and what the data cannot answer.
3. **Evidence** — one section per finding: chart + two or three sentences.
4. **Data quality notes** — nulls, outliers, gaps, anything the gate flagged.
5. **Next steps** — what you would do with one more week.
6. **Appendix** — method details, parameter tables, full metric tables.

## Rules

- Numbers carry their denominator and their uncertainty when one exists.
- Charts: one message per chart, title states the message, axes labelled with units,
  no dual axes, no 3D, colorblind-safe palette. Save as PNG under
  `artifacts/figures/` and reference them from the report.
- Prefer tables for fewer than six numbers; prefer charts for trends and comparisons.
- Write in English, plain and direct. No hedging phrases that add nothing.

## Output

Write `report/<name>.md` and, when asked for HTML, render it with the snippet in
`scripts/render_html.py` (markdown → self-contained HTML with embedded images).
