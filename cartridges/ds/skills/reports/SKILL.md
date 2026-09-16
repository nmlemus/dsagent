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
- **A title never claims a trend the intervals do not support.** If the confidence
  intervals overlap across the periods, or the series does not move in one direction,
  the title says what is actually there — "no clear trend", "flat within noise",
  "higher in every year" — not "rising". The hedge in your prose does not travel with
  the image; the title is what a reader quotes.
- **A series with fewer than 30 observations in a period carries its `n` in the label,
  or is left out.** A four-point line built on five days per point is a shape, not a
  finding. Prefer omitting it to drawing it with a caveat nobody reads.
- Prefer tables for fewer than six numbers; prefer charts for trends and comparisons.
- Write in English, plain and direct. No hedging phrases that add nothing.

## Output

Write `report/<name>.md` and, when asked for HTML, render it with the snippet in
`scripts/render_html.py` (markdown → self-contained HTML with embedded images).
