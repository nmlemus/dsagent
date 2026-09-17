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
  no dual axes, no 3D, colorblind-safe palette.
- **Emit every figure with `show_chart`, never as an image.** A PNG is a decision
  nobody downstream can revisit: the reader cannot hover a point, zoom a range or
  ask what a category holds, and the numbers are gone. `show_chart` takes a
  Vega-Lite spec plus the path of the table it draws — see "Emitting a chart"
  below. Write a PNG only when something outside the report needs one (a PDF, a
  slide), and then as well as the chart, not instead of it.
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

## Emitting a chart

Aggregate with `run_python`, write the aggregate to a file, then reference it:

```python
by_cat = (df.groupby("weather")
            .agg(n=("date", "size"), precip_mean=("precipitation", "mean"))
            .reset_index())
by_cat.to_parquet("artifacts/scratch/by_category.parquet")
```

```python
show_chart(
    spec={
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container", "height": 240,
        "data": {"name": "table"},
        "params": [{"name": "hover", "select": {"type": "point", "on": "pointerover"}}],
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {"field": "weather", "type": "nominal", "sort": "-y", "axis": {"labelAngle": 0}},
            "y": {"field": "precip_mean", "type": "quantitative",
                  "title": "mean precipitation (mm/day)"},
            "tooltip": [{"field": "weather"}, {"field": "n", "title": "days"},
                        {"field": "precip_mean", "title": "mm/day"}],
        },
    },
    data_ref="artifacts/scratch/by_category.parquet",
    title="Only rain and snow record precipitation",
)
```

Rules that are specific to this tool:

- **Never inline rows in the spec.** `"data": {"name": "table"}`, always; the rows
  come from `data_ref`. Aggregate first and reference the aggregate — a chart that
  references the raw dataset makes the reader's browser do the `groupby`.
- **Make it interactive where it earns its keep**: a `point` select on
  `pointerover` for hover; `{"select": "interval", "bind": "scales"}` to pan and
  zoom a quantitative axis; an interval selection on `x` when a reader would want
  to pick a range. Give every encoded field a `tooltip`.
- The `title` is the sentence above the chart, and it follows the trend rule above:
  it states the message, and never a trend the intervals do not support.
- **On a layered chart, a selection `param` goes inside one layer, never at the
  top level.** Vega-Lite copies a top-level param into *every* layer, and Vega
  then refuses the result with `Duplicate signal name: "<param>_tuple"`. The spec
  satisfies the schema and draws nothing. Put `params` on the layer the reader
  interacts with — usually the one with the points.
- A spec that fails validation comes back with the validator's message — the
  schema's, or the compiler's, since every spec is drawn once headless before it
  is recorded. Fix it and call again **with the same `chart_id`**: that is what
  makes the second call a correction rather than a second chart.
- A table a reader will look *through* rather than *at* — a column profile, a check
  table, a ranked list — goes through `show_table` on the same file.

## Output

Write `report/<name>.md` and, when asked for HTML, render it with the snippet in
`scripts/render_html.py` (markdown → self-contained HTML with embedded images).
The charts are not in that file: they belong to the run, and the report screen
renders them from the run's own record.
