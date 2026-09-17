# Run 003 — `eda-to-report`, from the browser

**Date:** 2026-09-16 21:03:03 → 21:11:14 (local) · **Wall time:** 6 m 45 s of run, 8 m 11 s door to door
**Model:** `anthropic:claude-sonnet-5` · **Env:** `default` (kernel) · **Cartridge:** `ds` v0.1.0
**Harness:** `v2` @ `855f392` · **Started from:** the chat pane at `localhost:3000`, `dsagent serve` on `:8000`
**Gate:** approved from the gate card · **Result:** `done`, all four steps `done`
**Dataset / question:** unchanged from runs 001 and 002
**Run directory:** `.dsagent/runs/eda-to-report-toolu_01Y3oqHiu97DSpDnBExLcCdi/`

The first run driven entirely from the UI. Nothing was typed into a terminal after
`dsagent serve`: the workflow was started by asking for it in the chat, the data gate was
approved by clicking Approve on the card, and the report was read in the canvas.

## Cost

Sonnet 5 rates: base input **$2**/MTok, cache read **$0.20**/MTok, 5-minute cache write
**$2.50**/MTok, output **$10**/MTok.

| Step | Persona | Wall | Input | of which cache read | cache write | uncached | Output | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| profile | marie | 32.1 s | 55,523 | 43,898 (79.1 %) | 11,613 | 12 | 2,743 | $0.065 |
| data-gate | marie | 32.6 s | 63,248 | 57,476 (90.9 %) | 5,758 | 14 | 2,780 | $0.054 |
| analyze | noel | 246.1 s | 334,141 | 309,591 (92.7 %) | 24,516 | 34 | 11,761 | $0.241 |
| report | marie | 56.5 s | 107,509 | 95,239 (88.6 %) | 12,256 | 14 | 5,741 | $0.107 |
| **Total** | | **367.3 s** | **560,421** | **506,204 (90.3 %)** | **54,143** | **74** | **23,025** | **$0.47** |

**$0.47**, against run 002's $0.85 — and the base-rate ceiling is $1.35 against 002's $2.55.
The run got cheaper because it did less input: 560 k tokens against 1.13 M. See below.

## vs run 002

| | Run 002 (CLI) | Run 003 (browser) | Δ |
|---|---:|---:|---:|
| Wall time | 361 s | **367 s** | +2 % |
| Input tokens | 1,125,816 | **560,421** | **−50 %** |
| Output tokens | 30,138 | 23,025 | −24 % |
| Cached share of input | 87.6 % | **90.3 %** | +2.7 pt |
| Cost, cache accounted | $0.85 | **$0.47** | **−45 %** |
| Figures | 4 | 4 | — |
| Files written | 11 | 10 | −1 |
| `run_python` in `analyze` | 16 | 11 | −5 |

Wall time is flat and cost halved. The halving is almost entirely `analyze`: 462 k input
tokens in 002 against 334 k here, off five fewer `run_python` turns. Every `run_python`
call re-sends the step's growing transcript, so turn count is the cost driver in that step
— the same shape 002 identified, moving in the other direction. This is model variance
between two runs of an unchanged step, not an improvement anything in this milestone made;
two samples cannot tell those apart.

`profile` also dropped (103 k → 56 k). It made one `run_python` call here against two, and
two `ls` against three.

## What each step did

| Step | Tool calls | Skills read | Files written |
|---|---|---|---|
| profile | `read_file` 3, `ls` 2, **`run_skill_script` 1**, `run_python` 1, `edit_file` 1 | `eda` | `artifacts/data-profile.json`, `artifacts/data-profile.md` |
| data-gate | `read_file` 3, `ls` 4, `write_file` 1 | `eda` | `artifacts/data-gate.md` |
| analyze | `run_python` 11, `read_file` 4, `ls` 2, `execute` 2, `write_file` 1 | `eda`, `reports` | `figures/fig1…fig4.png`, `artifacts/findings.md` |
| report | `read_file` 5, `ls` 1, **`run_skill_script` 1**, `write_file` 1, `execute` 1, `run_python` 1 | `reports` | `report/findings.md`, `report/findings.html` |

Unchanged from 002 where it matters: `run_skill_script` in `profile` and `report`, no
`/skills/...` path anywhere, and `data-gate` reading the JSON rather than reloading the
dataset (no `run_python`, no `execute`).

`analyze` read `eda` as well as `reports` again. Step 03 does not ask for `eda`; Noel's
standing orders do. Left alone, as in 002.

## File-appearance timeline

```
+  0s  ▶ profile (marie)
+  4s  ● artifacts/data-profile.json      ← the script
+ 25s  ● artifacts/data-profile.md
+ 32s  ◀ profile   ▶ data-gate (marie)
+ 58s  ● artifacts/data-gate.md
+ 65s  ◀ data-gate — gate card appears
+102s  ‖ gate approved (37 s of human reading time)
+102s  ▶ analyze (noel)
+289s  ● artifacts/figures/fig1_precip_by_category.png
+297s  ● artifacts/figures/fig2_temp_by_category.png
+305s  ● artifacts/figures/fig3_temp_trend.png
+310s  ● artifacts/figures/fig4_precip_trend.png
+337s  ● artifacts/findings.md
+349s  ◀ analyze   ▶ report (marie)
+392s  ● report/findings.md
+394s  ● report/findings.html
+405s  ◀ report — run done
```

Four figures in 21 s; `findings.md` superseded by `findings.html` 2 s later. Both are the
shapes runs 001 and 002 flagged for the canvas, and both are now handled — see below.

## Deviations vs the step instructions

**D9 is fixed.** Step 03 now permits a trend when a finding depends on one and requires the
interval to be reported. Headline finding 3 reads *"+0.73 °C/year for `temp_max`, +0.51
°C/year for `temp_min`, both 95% CIs exclude zero … precipitation +0.04 mm/year, 95% CI
includes zero"* — a tested trend, stated with its interval. No contradiction left.

**D10 (new, minor) — "season-adjusted" names a method the numbers do not come from.** The
report calls the slopes *season-adjusted*. I recomputed: regressing the four annual means
on year gives **+0.74, +0.51, +0.04**, which is what the report shows to within rounding. A
genuine seasonal adjustment (harmonic term on daily data) gives **+0.62, +0.43, +0.10** —
different numbers. Averaging whole calendar years does neutralise the seasonal cycle, so
the label is defensible rather than wrong; it is just imprecise about a method a reader
might try to reproduce. Worth one sentence in the `reports` skill about naming the method
used.

**Not a deviation, worth noting:** `temp_min`'s interval is [+0.04, +0.98], p = 0.042 — it
clears zero by a hair on four annual points. The report presents it beside `temp_max`
(p = 0.009) without distinguishing their strength. Both statements are true; one is much
more fragile than the other.

Everything D1–D8 stayed fixed: `data-profile.md` carries no gate verdict and no `GATE:`
line, `data/` holds only the input CSV, the JSON matches the script's schema, `findings.md`
has exactly three sections (*Headline findings*, *Evidence*, *Caveats*), `report`'s
`skills_read` is `reports` alone, and every by-category chart carries its `n`.

## Judgement: is the report any good?

**Yes, and every load-bearing number is exact.** Recomputed against the CSV:

| Claim | Report | Recomputed |
|---|---|---|
| `snow` mean max / min | 5.6 / 0.1 °C | 5.57 / 0.15 |
| `sun` mean max / min | 19.9 / 9.3 °C | 19.86 / 9.34 |
| `sun`/`fog`/`drizzle` median precipitation | 0 mm | 0.00 each |
| share of precipitation on `rain` + `snow` days | "almost all" | 95.0 % + 5.0 % = **100 %** |
| `temp_max` trend, 95 % CI | +0.73, excludes 0 | +0.74, [+0.43, +1.05] |
| `temp_min` trend, 95 % CI | +0.51, excludes 0 | +0.51, [+0.04, +0.98] |
| precipitation trend | +0.04, CI includes 0 | +0.04, [−1.18, +1.26] |

The one place the report is *weaker* than the data: "almost all recorded precipitation"
falls on `rain` and `snow` days, when to one decimal place it is all of it. Understating is
the right direction to err.

The headline findings are decisions rather than observations — treat `weather` as a wet/dry
proxy, not a graded scale; use `sun` vs `snow` as the temperature bookends; budget for
continued warming but not more rain. The caveats name the single-station limit and the thin
strata with their `n`.

## What the UI showed

The four canvas observations from run 001, and whether the M2.2 slice answers them:

| # | Run 001 observation | Addressed? |
|---|---|---|
| 1 | Nothing appears for the first 87 s — files alone are not a progress indicator | **Yes.** The DAG panel draws a row the moment a step starts, with persona, live elapsed and tool counts. The first *file* landed at +4 s here, but the panel was populated from +0 s and stayed useful through `analyze`'s 187-second silence between start and first figure. |
| 2 | Figures arrive as a burst — append, don't re-render | **Yes.** The file list is append-only; a rewritten file updates its row. Four figures in 21 s appended without disturbing the pane. |
| 3 | The last two files supersede each other — focus on step completion, not file events | **Yes.** Focus moves only on `status: done`, to the step's last matched `produces`. `report` wrote `findings.md` then `findings.html` 2 s apart and the canvas went straight to the HTML. |
| 4 | Working files need distinguishing from deliverables | **Yes, but untested here.** `dsagent.file` carries `kind`, and the list groups by it. This run wrote 10 files and all 10 were declared — the figures are now covered by `artifacts/figures/*.png`, so the *Working files* group stayed empty. The mechanism is exercised in `tests/test_produces_globs.py`, not by this run. |

Two things the slice added that the run-001 list did not anticipate:

**The gate card lands on the right artifact by itself.** Focus moves to the gated step's
last deliverable, which is the file the gate is asking about — so `artifacts/data-gate.md`
was already rendered below the card. The 37 seconds between the card appearing and Approve
were spent reading the check table, not hunting for it.

**The personas are out of the chat.** Their commentary now sits in the step row it belongs
to (`marie worked — 2 notes`, `noel worked — 4 notes`). Compare `ui-canvas-002.png`, where
the same text filled the transcript with no indication that the voice had changed.

### UX debt, prioritised → M2.2.1

From the operator's seat, in the order I would fix them:

1. **A successful run ends in a red error.** The moment the orchestrator finished
   summarising, the chat showed `terminated` and the console logged
   `agent_run_error_event`. Server-side:
   `langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without hitting a
   stop condition`. The *workflow* was already `done` and every artifact written — this is
   the orchestrator graph exhausting LangGraph's default 25 super-steps, so the SSE stream
   ends in `RUN_ERROR` after the work succeeded. Every run will do this. Fix is a
   `recursion_limit` on the served graph's config; the number needs choosing, not guessing.
2. **A served run leaves no readable log.** `dsagent serve` passes `log=lambda m: None`, so
   `.dsagent/runs/<id>/` has `run.json` and no `runner.log` — unlike a CLI run, which
   writes one. Reading a run after the fact is *worse* from the browser than from the
   terminal, which is backwards.
3. **`produces` ticks read ○ while the files are visibly landing.** `produces_matched` is
   empty on `started` by design, so mid-step the DAG row says nothing has been produced
   while the file list beside it lists the files. Self-consistent, but the two panes
   disagree in front of the operator.
4. **A reload loses the run.** All canvas state is reduced from the live event stream, so
   refreshing mid-run shows an empty canvas while the run continues server-side, and there
   is no way to re-attach. Related: nothing outside the gate card names the run, so after
   it finishes you cannot tell which run you are looking at.
5. **The human wait is invisible.** `data-gate` shows 33 s; the gate was approved at +102 s.
   The 37 seconds the run spent waiting for a person appear nowhere, which is exactly the
   number you want when deciding whether gates are worth their cost.
6. **The error toast shows a stack trace.** `Show Details` on that `terminated` banner opens
   frames from a minified bundle. Whatever replaces it should say which step failed and why.
7. **A failed step's presentation is unexercised.** The DAG row renders `error`, but no run
   has failed in the browser yet, so the failure path has never been looked at with eyes.

## Artifacts committed

- [`eda-to-report-003/findings.html`](eda-to-report-003/findings.html) — the delivered report, self-contained
- [`eda-to-report-003/findings.md`](eda-to-report-003/findings.md) — its markdown source
- [`eda-to-report-003/figures/`](eda-to-report-003/figures/) — the four figures as produced
- [`eda-to-report-003/run.json`](eda-to-report-003/run.json) — the run state and per-step telemetry this document is built from
- [`eda-to-report-003/01-gate-card.png`](eda-to-report-003/01-gate-card.png) — the gate, with `data-gate.md` already rendered below it
- [`eda-to-report-003/02-dag-analyze.png`](eda-to-report-003/02-dag-analyze.png) — mid-`analyze`, noel's narration open
- [`eda-to-report-003/03-final-report.png`](eda-to-report-003/03-final-report.png) — the finished run, report in the canvas (and the `terminated` toast, item 1)

No `runner.log`: see UX debt item 2.
