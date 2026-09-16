# Run 002 — `eda-to-report`, after iteration 2

**Date:** 2026-09-16 12:10:16 → 12:16:17 (local) · **Wall time:** 6 m 02 s (361 s)
**Model:** `anthropic:claude-sonnet-5` (the new `DSAGENT_MODEL` default)
**Env:** `default` (kernel) · **Cartridge:** `ds` v0.1.0 · **Harness:** `v2` @ `344af31`
**Command:** `DSAGENT_INTEGRATION=1 pytest tests/integration -s` — one pass, no retry
**Result:** `done`, all four steps `done`, all seven integration assertions passed
**Dataset / question:** unchanged from run 001
**Run directory:** `.dsagent/runs/eda-to-report-integration-20260916-121016/`

## Cost

Sonnet 5 rates, verified against the pricing docs: base input **$2**/MTok, cache read
**$0.20**/MTok (0.1×), 5-minute cache write **$2.50**/MTok, output **$10**/MTok.

| Step | Persona | Wall | Input | of which cache read | cache write (5m) | uncached | Output | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| profile | marie | 55.1 s | 102 976 | 88 925 (86.4 %) | 14 031 | 20 | 3 742 | $0.090 |
| data-gate | marie | 27.6 s | 54 502 | 49 073 (90.0 %) | 5 417 | 12 | 2 328 | $0.047 |
| analyze | noel | 186.0 s | 462 277 | 432 379 (93.5 %) | 29 856 | 42 | 16 611 | $0.327 |
| report | marie | 91.8 s | 506 061 | 415 464 (82.1 %) | 90 565 | 32 | 7 457 | $0.384 |
| **Total** | | **360.5 s** | **1 125 816** | **985 841 (87.6 %)** | **139 869** | **106** | **30 138** | **$0.85** |

**87.6 % of input is cache reads**, which is the number run 001 could not see. Charging
everything at base rate would give $2.55; the cache is worth **$1.70, or 67 % of the bill**.
Only 106 tokens across the whole run were genuinely uncached — every other input token was
either written to cache once or read from it.

## vs run 001

| | Run 001 | Run 002 | Δ |
|---|---:|---:|---:|
| Model | `claude-sonnet-4-6` | `claude-sonnet-5` | |
| Wall time | 533 s | **361 s** | **−32 %** |
| Input tokens | 717 657 | 1 125 816 | +57 % |
| Output tokens | 28 112 | 30 138 | +7 % |
| Cached share of input | *not measured* | **87.6 %** | — |
| Cost, everything at base rate | $2.57 | $2.55 | −1 % |
| Cost, cache accounted | *unmeasurable* | **$0.85** | — |
| Figures | 5 | 4 | |
| Files written | 13 | 11 | −2 (both were strays) |

Read the two cost rows carefully. Run 001's $2.57 was always a ceiling — it had the same
kind of caching, we simply could not measure it. The honest comparison is ceiling to
ceiling: **$2.57 → $2.55, essentially flat**, because Sonnet 5's lower per-token price
absorbed a 57 % increase in input tokens. The **$0.85** is what run 002 actually cost.

### Where the extra input tokens went

| Step | 001 input | 002 input | Δ |
|---|---:|---:|---:|
| profile | 238 573 | **102 976** | **−57 %** |
| data-gate | 99 294 | **54 502** | **−45 %** |
| analyze | 161 468 | 462 277 | **+186 %** |
| report | 218 322 | 506 061 | **+132 %** |

The two steps iteration 2 targeted both dropped by about half, and they dropped for the
reasons intended: `profile` ran the script instead of writing a profiler and never saw the
question, `data-gate` read the JSON instead of reloading the dataset. Wall time follows —
55 s and 28 s, against 138 s and 102 s.

The growth is in `analyze` and `report`, neither of which iteration 2 touched, and it is
almost entirely cache reads (93.5 % and 82.1 %). `analyze` made 16 `run_python` calls
against run 001's 8 — it did more analysis, including the trend fits discussed under D9
below — and every one of those turns resends a growing transcript. This is the agent-loop
cost shape, not a regression: those two steps cost $0.33 and $0.38.

## What each step did

| Step | Tool calls | Skills read | Files written |
|---|---|---|---|
| profile | `read_file` 4, `ls` 3, **`run_skill_script` 1**, `run_python` 2, `edit_file` 2 | `eda` | `artifacts/data-profile.json`, `artifacts/data-profile.md` |
| data-gate | `read_file` 3, `ls` 3, `write_file` 1 | `eda` | `artifacts/data-gate.md` |
| analyze | `run_python` 16, `read_file` 4, `ls` 3, `write_file` 1 | `eda`, `reports` | `figures/01…04.png`, `artifacts/findings.md` |
| report | `read_file` 10, `ls` 6, **`run_skill_script` 1**, `grep` 3, `run_python` 2, `write_file` 1 | `reports` | `report/findings.md`, `report/findings.html` |

**Confirmed from telemetry, as asked:**

1. **`run_skill_script` was called in `profile` and in `report`** — once each, for
   `profile.py` and `render_html.py`. No `/skills/...` path appears anywhere in the run.
2. **`data-gate` did not re-read the dataset.** Its tool calls are three `read_file`,
   three `ls` and one `write_file` — **no `run_python`, no `execute`**. It never loaded
   pandas. In run 001 the same step made two `run_python` calls and one `execute`.
3. The JSON is the script's: keys are exactly `schema_version, source, rows, columns,
   exact_duplicates_pct, key_uniqueness, columns_detail`, the categorical detail key is
   `top` (not run 001's invented `top5`), and `key_uniqueness` reads
   `{"column": "date", "unique": true, "duplicates": 0}` — so `--key-column` was passed.

`profile`'s two `run_python` calls and two `edit_file` calls are the persona extending
`data-profile.md` with what the script does not cover (time coverage, a `temp_max >=
temp_min` check). That is what the step now asks for, and it left the JSON alone.

## File-appearance timeline

```
+  0s  ▶ profile (marie)
+  9s  ● artifacts/data-profile.json      ← the script, 9s in
+ 43s  ● artifacts/data-profile.md
+ 55s  ◀ profile   ▶ data-gate (marie)
+ 76s  ● artifacts/data-gate.md
+ 83s  ◀ data-gate  ▶ analyze (noel)
+196s  ● artifacts/figures/01_temp_by_category.png
+200s  ● artifacts/figures/02_precip_by_category.png
+206s  ● artifacts/figures/03_temp_trend_2012_2015.png
+210s  ● artifacts/figures/04_precip_trend_2012_2015.png
+258s  ● artifacts/findings.md
+269s  ◀ analyze   ▶ report (marie)
+324s  ● report/findings.md
+327s  ● report/findings.html
+360s  ◀ report — run done
```

Against run 001's canvas observations:

- **The empty opening shrank from 87 s to 9 s.** Running the script produces the first
  artifact almost immediately. The observation still holds — a canvas needs step events
  before the first file — but the blank window is now 2.5 % of the run instead of 16 %.
- **The figure burst got tighter**: four figures in 14 s (001: five in 41 s). Whatever the
  canvas does with images, it does it in one breath.
- **The 8-second supersede at the end is now 3 seconds** (`findings.md` → `findings.html`).
  Focusing on step completion rather than file events matters more, not less.
- **Every file written is a deliverable.** 11 files, zero strays. The deliverable/working
  split still needs `produces` in the event stream, but this run would not have exercised it.

## Deviations D1–D8 from run 001

| | Deviation | Status |
|---|---|---|
| D1 | `profile` ran the data gate | **fixed** |
| D2 | `profile` answered the analysis question | **fixed** |
| D3 | `profile` wrote undeclared files into `data/` | **fixed** |
| D4 | `profile.py` unused, JSON schema diverged | **fixed** |
| D5 | `analyze` added report-shaped sections | **fixed** |
| D6 | `report` re-read the `eda` skill | **fixed** |
| D7 | a chart title claimed an unsupported trend | **fixed** |
| D8 | a thin stratum plotted without its caveat | **fixed** |

All eight. The evidence, briefly:

- **D1** — `data-profile.md` contains no gate table and no `GATE:` line; its only headings
  are `# Data profile` and `## Notes for readers`. The verdict appears once in the run,
  in `data-gate.md`.
- **D2** — the notes are profiling: coverage, nulls, duplicates, outliers, class
  proportions. No by-category or by-year comparison answering the question.
- **D3** — `data/` contains exactly one file, the input CSV. No `artifacts/scratch/` was
  needed, so none was created.
- **D4** — schema matches the script exactly (see above). The new integration test
  `test_profile_json_is_exactly_the_script_schema` compares against the script's own
  output, top level and per column, and passed.
- **D5** — `findings.md` has exactly three sections: *Headline findings*, *Evidence*,
  *Caveats*. "What this data cannot answer" now appears only in the report, in §2 Context,
  which is where step 04 puts it.
- **D6** — `report`'s `skills_read` is `['/skills/marie/reports/SKILL.md']` and nothing else.
- **D7** — the trend chart is titled *"Seattle warmed every year, 2012-2015 (95% CI,
  n≈365/yr)"*, and it earns it: the series is monotone (15.3 → 16.1 → 17.0 → 17.4 °C) with
  non-overlapping endpoint intervals. The precipitation chart states no trend, matching
  finding 3's *"slope 0.04 mm/year, 95% CI crosses zero"*.
- **D8** — every category on the by-category charts carries its `n` in the axis label,
  including the case that prompted the rule: `snow (n=26)`.

## New deviation

### D9 — `analyze` fitted trend lines, and the step says not to model

`steps/03-analyze.md` says:

> No modeling in this step: descriptive and comparative analysis only.

Headline finding 3 reads:

> daily max and min temperature rose about 0.7°C/year and 0.5°C/year respectively from
> 2012 to 2015, **with 95% confidence intervals that exclude zero**, while daily
> precipitation showed no such trend (**slope 0.04 mm/year, 95% CI crosses zero**)

Fitted slopes with confidence intervals are inferential statistics, not description. The
numbers are correct — I recomputed them: +0.74, +0.51 and +0.04 per year — and the
conclusion is sound. But the step drew a line it was told not to cross.

**It is worth noticing why.** Iteration 2 added the rule *"a title never claims a trend the
intervals do not support"*. The honest way to satisfy that is to test the trend, so a rule
written to stop overclaiming pushed the persona into modeling. The two instructions
conflict, and the persona resolved the conflict in favour of being right. That is the
better failure, but it is still a contradiction in the cartridge, and it is mine.

Not a deviation, but worth recording: `analyze` read the `eda` skill as well as `reports`,
which step 03 does not ask for. Noel's own working rules say to read `eda` before touching
new data, so the persona was following its standing orders. Leave it.

## Judgement: is the report any good?

**Better than run 001, and run 001 was already good.** Two things changed in kind:

**Every numeric claim I checked is exact.** I recomputed the load-bearing ones against the
CSV: 44 of 641 `rain` days at 0.00 mm (6.9 %), all 26 `snow` days above zero, 794
sun/fog/drizzle days at exactly zero, slopes +0.74 / +0.51 / +0.04 per year. All correct as
written, denominators included.

**The analysis found something run 001 missed.** Run 001 concluded that drizzle, fog and
sun are dry by construction. Run 002 found the sharper version: 44 `rain` days also record
0.00 mm, so the label is not a clean function of the precipitation column in either
direction — and it says so in the caveats, where it belongs, rather than in a headline.

The headline findings are decisions, not observations ("`weather` can substitute for
temperature in quick screens", "treat the other three labels as dry by default", "budget
for continued warming, not more rain"). The caveats name the thin strata with their `n`,
distinguish mean from median on skewed precipitation, and state the single-station limit.

Weaknesses: the by-category bar charts use a single colour where the category *is* the
comparison, so they lean on the axis labels; one data label collides with an error bar in
figure 01. Both cosmetic. The substantive flaw is D9 — not because the statistics are
wrong, but because the report presents inferential results in a workflow whose analysis
step is declared descriptive.

## Proposed changes, in priority order

Not applied here.

1. **Resolve the D9 contradiction** *(cartridge)*. Either step 03 may fit a trend line when
   a finding depends on one — and says so, with a requirement to report the interval — or
   the chart rule stops implying a test and asks for the claim to be dropped instead. The
   first is better: a trend that cannot be tested should not be a headline finding.
2. **Look at `report`'s 90 565 cache-write tokens** *(cartridge)*. It is the largest write
   in the run, 6× `profile`'s, and the step re-reads three artifacts plus its skill. Worth
   checking whether it is re-reading files it already has in context.
3. **Give the by-category charts a colour per category** *(cartridge, `reports` skill)*,
   since the category is the comparison. Cosmetic but cheap.
4. **Expose `produces` in the step event** *(harness)*. Unchanged from run 001 — this run
   had no strays, so nothing forced the issue, but the canvas still needs the signal.
5. **Record the model in `run.json`** *(harness)*. Comparing two runs required reading the
   default out of the source at the commit each ran on. The step record should say which
   model produced it.

## Artifacts committed

- [`eda-to-report-002/findings.html`](eda-to-report-002/findings.html) — the delivered report, self-contained
- [`eda-to-report-002/findings.md`](eda-to-report-002/findings.md) — its markdown source
- [`eda-to-report-002/figures/`](eda-to-report-002/figures/) — the four figures as produced
- [`eda-to-report-002/runner.log`](eda-to-report-002/runner.log) — the runner's step log
