# Run 001 — `eda-to-report`, first real model run

**Date:** 2026-09-16 02:30:00 → 02:38:53 (local) · **Wall time:** 8 m 53 s (533 s)
**Model:** `anthropic:claude-sonnet-4-6` (the `DSAGENT_MODEL` default; no per-persona override)
**Env:** `default` (kernel) · **Cartridge:** `ds` v0.1.0 · **Harness:** `v2` @ `3241ae5`
**Command:** `DSAGENT_INTEGRATION=1 pytest tests/integration -s` — one pass, no retry
**Result:** `done`, all four steps `done`, all six integration assertions passed
**Dataset:** `tests/data/seattle-weather.csv` (1 461 rows × 6 cols, 2012-01-01 → 2015-12-31)
**Question:** *How do precipitation and temperature differ across the weather categories, and how did they change between 2012 and 2015?*
**Run directory:** `.dsagent/runs/eda-to-report-integration-20260916-023000/`

## Cost

| Step | Persona | Wall | Input tok | Output tok | Cost |
|---|---|---:|---:|---:|---:|
| profile | marie | 138.3 s | 238 573 | 6 661 | $0.816 |
| data-gate | marie | 101.7 s | 99 294 | 5 689 | $0.383 |
| analyze | noel | 153.6 s | 161 468 | 8 826 | $0.617 |
| report | marie | 139.7 s | 218 322 | 6 936 | $0.759 |
| **Total** | | **533.2 s** | **717 657** | **28 112** | **$2.57** |

At Sonnet 4.6 list rates ($3.00 / $15.00 per MTok). **This is an upper bound**: our telemetry
sums `usage_metadata.input_tokens`, which includes cache reads, and cache reads bill at a
reduced rate. We do not currently capture `input_token_details`, so the cached fraction is
unknown — see change #6.

Input tokens are 25× output. That ratio is inherent to an agent loop (every turn resends the
transcript), which makes prompt-cache behaviour, not output length, the cost lever here.

## What each step did

| Step | Tool calls | Skills read | Files written |
|---|---|---|---|
| profile | `run_python` 11, `read_file` 6, `ls` 4 | `eda` | `data/weather_category_stats.csv`, `data/weather_year_category_stats.csv`, `artifacts/data-profile.json`, `artifacts/data-profile.md` |
| data-gate | `read_file` 4, `run_python` 2, `glob` 1, `execute` 1, `write_file` 1 | `eda` | `artifacts/data-gate.md` |
| analyze | `run_python` 8, `read_file` 4, `ls` 1, `write_file` 1 | `reports` | `figures/fig1…fig5.png`, `artifacts/findings.md` |
| report | `read_file` 7, `ls` 6, `execute` 6, `write_file` 1 | `reports`, `eda` | `report/findings.md`, `report/findings.html` |

No persona read a skill outside its matrix scope. Marie used `eda` and `reports`; Noel used
`reports`. `ml` and `mmm` were never touched.

## File-appearance timeline

Offsets from run start. This is the sequence a canvas would have to render live.

```
+  0s  ▶ profile (marie)
+ 87s  ● data/weather_category_stats.csv          ← not requested
+ 87s  ● data/weather_year_category_stats.csv     ← not requested
+ 87s  ● artifacts/data-profile.json
+116s  ● artifacts/data-profile.md
+138s  ◀ profile   ▶ data-gate (marie)
+225s  ● artifacts/data-gate.md
+240s  ◀ data-gate  ▶ analyze (noel)
+265s  ● artifacts/figures/fig1_precip_by_category.png
+275s  ● artifacts/figures/fig2_temperature_by_category.png
+284s  ● artifacts/figures/fig3_day_counts_by_year.png
+295s  ● artifacts/figures/fig4_rain_intensity_by_year.png
+306s  ● artifacts/figures/fig5_tempmax_trend_by_category.png
+373s  ● artifacts/findings.md
+394s  ◀ analyze   ▶ report (marie)
+500s  ● report/findings.md
+508s  ● report/findings.html
+533s  ◀ report — run done
```

Four observations for the canvas design in `docs/ui.md`:

1. **The first 87 s produce nothing.** That is 16 % of the run with an empty canvas. A canvas
   fed only by workspace files shows a blank panel for a minute and a half. Step and tool
   events (`dsagent.step`) have to carry the UI until the first artifact lands.
2. **Figures arrive as a 41-second burst**, one every ~10 s. The canvas should append images
   as they appear rather than re-render a list; this is the one moment in the run where it
   visibly *streams*.
3. **The last two files are 8 s apart** and one supersedes the other. A canvas that auto-focuses
   the newest file would flash `findings.md` and then replace it with `findings.html`. Prefer
   focusing on step completion, not on every file event.
4. **13 files, 5 of them images, 1 HTML, 2 of them junk.** The canvas needs a notion of
   *deliverable* vs *working file*, or it will show `weather_category_stats.csv` with the same
   weight as the final report. `produces` is the natural signal and the runner already knows it.

## Deviations from step instructions and skills

### D1 — `profile` ran the data gate, which is `data-gate`'s job (cost: ~$0.38)

`steps/01-profile.md` never mentions gates. The produced `artifacts/data-profile.md` ends with:

> ## 4. Data Quality Gate
>
> | Check | Value | Threshold | Result |
> |---|---|---|---|
> | null_pct:date | 0.0 | 20.0 | **PASS** |
> … | … | … | … |
>
> `GATE: PASS`

Marie read the `eda` skill, which does describe the gate thresholds and the report format, and
applied all of it immediately. Step 02 then re-derived the same nine checks from scratch —
99 294 input tokens to restate a verdict already sitting on disk.

### D2 — `profile` also answered the analysis question, which is `analyze`'s job

`artifacts/data-profile.md` §3 is headed:

> ## 3. Weather Category Statistics
>
> *(Relevant to the analysis question: precipitation & temperature by category and year)*

and §5:

> ## 5. What This Data Cannot Answer

Step 01 asks for a profile. §3 is the comparative analysis step 03 asks for, and §5 is the
`reports` skill's Context section. **Root cause is harness-level, not prompt-level:**
`WorkflowRunner._task_message` renders *every* workflow input into every step's prompt, so
`question` was in front of Marie during profiling. Personas run ahead when you show them the
finish line.

### D3 — `profile` wrote outside `produces`, into the input directory

`data/weather_category_stats.csv` and `data/weather_year_category_stats.csv`. Step 01 says:

> Deliver `artifacts/data-profile.md` and `artifacts/data-profile.json`.

Two undeclared files, placed next to the raw input rather than under `artifacts/`. Nothing
downstream reads them. Harmless here, but `data/` is the one directory that should be treated
as read-only.

### D4 — the `eda` skill's `scripts/profile.py` was never used, and the JSON schema silently diverged

Step 01 says:

> Prefer running its `scripts/profile.py` via `run_python` or `execute`, then extend the profile
> with anything the script does not cover

Evidence it did not run: the script writes `data-profile.json` and `data-profile.md` in the same
call, but telemetry has them **30 s apart** (+87 s and +116 s); and the step made zero `execute`
calls. Marie wrote her own profiler instead.

The result is a superset that renames one key. The script emits `top` for categorical columns;
the artifact contains:

```json
{"column": "weather", "dtype": "str", "null_count": 0, "null_pct": 0.0,
 "distinct": 5, "top5": {"rain": 641, "sun": 640, "fog": 101, "drizzle": 53, "snow": 26}}
```

`top` → `top5`, plus added `null_count`, `median`, `std`, `q1`, `q3`, `outliers_3sigma`. **The
script is dead code and there is no contract on the artifact.** Anything written against
`profile.py`'s schema breaks on this run's output, silently. Our own integration test did not
catch it because it asserts only the keys both shapes happen to share.

### D5 — `analyze` added report-shaped sections

Step 03 asks for:

> three headline findings, then evidence per finding (chart reference + numbers with
> denominators), then caveats that come from the gate

`artifacts/findings.md` has those three, plus `## What the Data Cannot Answer` and
`## Next Steps` — sections 2 and 5 of the `reports` skill. Mild, and the report step then had to
merge two versions of them.

### D6 — `report` re-read the `eda` skill it was not pointed at

Step 04 names only the `reports` skill. `skills_read` for the step is `["…/reports/SKILL.md",
"…/eda/SKILL.md"]`. It plausibly wanted the gate thresholds for the appendix; it cost tokens in
the most expensive-per-token step.

### D7 — a chart title claims more than the data supports

`fig5_tempmax_trend_by_category.png` is titled:

> Sun-day highs consistently exceed rain and fog;
> **both sun and rain show slight warming 2012→2015**

Rain's mean `temp_max` runs 12.81 → 13.63 → 14.21 → 13.35 °C: it peaks in 2014 and falls back,
and the plotted 95 % CIs overlap in all four years. The report *text* hedges correctly
("four years of data are insufficient to confirm a statistically significant trend"), but the
`reports` skill says a chart title *states the message*, and this one states a stronger message
than the figure shows. The hedge did not propagate into the chart.

### D8 — a thin stratum was plotted without its caveat

The same figure plots `fog` for 2012, where n = 5 days; its CI spans roughly 14.5–27.7 °C. The
gate flagged thin strata explicitly ("`snow` is a thin stratum in 2013–2015 … Do not draw trend
conclusions") and step 03 says caveats come from the gate. The rule was applied to snow and not
generalised to fog.

## What went right

Worth recording, because it is most of the run:

- **The gate earned its place.** It caught a real artifact of the dataset — precipitation is
  0.0 mm on *every* `drizzle`, `fog` and `sun` day, because the label is derived from the
  precipitation column — and the caveat propagated all the way into headline finding #1. That is
  the pipeline working exactly as designed.
- **Verdict line format was exact**: `GATE: PASS` as the final line, parseable.
- **Exactly three headline findings**, each with denominators ("191 → 144 days/year", "+1.2 °C").
- **Five figures, within the "up to five" budget**, each following the chart rules: title states
  the message, axes carry units, `n=` on category labels, no dual axes, colourblind-safe palette.
- **Report structure matches the `reports` skill section for section** (headline / context /
  evidence / data quality / next steps / appendix).
- **No modelling in `analyze`**, as instructed — time-series modelling appears only as a next step.
- **Skill scoping held**: no persona read a skill outside its matrix.
- **The HTML is genuinely self-contained** — 466 KB with all five figures inlined as data URIs.

## Judgement: is the report any good?

**Yes — I would send this to a stakeholder with one edit.** It is better than the median
human-written EDA report at the same effort. Specifically:

- The headline findings are claims, not descriptions, and each carries its arithmetic.
- Finding 1 is the *right* finding: it identifies a labelling artifact that would have produced a
  nonsense "drizzle days have no rain" conclusion in a naive analysis, and it says which
  comparisons remain valid.
- The numbers I spot-checked against the profile are correct (rain days 191/158/148/144; sun
  `temp_max` 20.23 → 21.40 = +1.17 ≈ "+1.2 °C").
- Uncertainty is handled honestly in the prose.

The one edit: **D7's chart title**. Everything else is presentation polish. The weakest part of
the deliverable chain is not the report — it is that two of four steps did work that belonged to
another step, and that the profile artifact has no schema contract.

## Proposed changes, in priority order

Not applied here — this PR only records the run.

1. **Stop leaking future inputs into early steps** *(harness)*. `_task_message` renders all
   workflow inputs into every step. Give `Step` a `sees: [input_names]` (default: the ones
   its instruction text actually interpolates) so `profile` never sees `question`. This is the
   root cause of D2 and, indirectly, of the duplicated work in D1.
2. **Make `steps/01-profile.md` explicitly not the gate** *(cartridge)*. One line: "Do not
   evaluate gate thresholds or emit a `GATE:` line — step `data-gate` owns that." Then make
   step 02 read the profile rather than recompute it. Expected saving ≈ $0.38/run, 100 s.
3. **Give `profile.py` a schema contract, or delete it** *(cartridge)*. Either the step *must*
   run the script (and extend the JSON under a separate key), or the skill stops shipping a
   script nobody runs. Today it is dead code whose output shape silently disagrees with the
   artifact. Add a test asserting the artifact matches the script's schema.
4. **Declare `data/` read-only in the step instructions** *(cartridge)*, and have working files
   go to `artifacts/scratch/`. Fixes D3 and gives the canvas a clean deliverable/working split.
5. **Propagate the hedge into chart titles** *(cartridge, `reports` skill)*. Add to the chart
   rules: "if the finding is not statistically supported, the title says so or states no trend";
   and "any series with n < 30 in a period carries its n in the label or is omitted". Fixes D7
   and D8.
6. **Capture `input_token_details` in step telemetry** *(harness)*. Without cache-read and
   cache-creation counts, every cost figure here is an upper bound and we cannot tell whether
   caching is working at all. Cheapest possible change with the highest reporting value.
7. **Trim step 03's output contract** *(cartridge)*. Say explicitly that "What the data cannot
   answer" and "Next steps" belong to the report, not to `findings.md`. Fixes D5.
8. **Consider `produces` as the canvas's deliverable signal** *(harness/UI)*. The runner already
   verifies it; exposing it in the `dsagent.step` event gives the UI a free way to distinguish
   the report from a scratch CSV.

## Artifacts committed

- [`eda-to-report-001/findings.html`](eda-to-report-001/findings.html) — the delivered report, self-contained
- [`eda-to-report-001/findings.md`](eda-to-report-001/findings.md) — its markdown source
- [`eda-to-report-001/figures/`](eda-to-report-001/figures/) — the five figures as produced
