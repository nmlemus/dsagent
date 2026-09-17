# M2.5 — the demo script, line by line

`docs/ui-product.md` §7 is the exit condition: twelve lines that must each be true
from a clean checkout, with the evidence recorded here.

**Status: 12 of 12 verified, with a real model.** Three runs of `eda-to-report` on
`tests/data/seattle-weather.csv`, from the browser, against `dsagent serve` and
`next build && next start`:

| | run | what it was for | wall | cost |
|---|---|---|---|---|
| **A** | `eda-to-report-20260917-000956` | the clean run: §7.1–§7.9 | 6 m 14 s | $0.62 |
| **B** | `eda-to-report-20260917-080356` | killed at its gate, then sent back: §7.10, §7.11 | 6 m 46 s | $0.50 |
| **C** | `eda-to-report-20260917-081412` | the human wait, measured properly | 6 m 28 s | $0.48 |

**$1.60 of a six-run, ~$3–6 budget**, and a fourth run ($0.48) created through the
API while debugging, which is on the home screen too. Everything before these was
built against run 003's own recording, at no cost.

## Setup

```
pip install -e ".[ui,anthropic]"
cd ui && npm ci && npm run build
dsagent serve                      # :8000
cd ui && npm run start             # :3000
```

---

## 1. The home screen lists previous runs; a fresh install shows an empty state

**✓** Empty state on a run directory with nothing in it: *"Nothing has run yet"*
with what a run is and one way to start (`demo-1-empty.jpg`). After the three
runs, the ledger shows each with status, dataset, progress, start time, duration
and cost (`demo-1-home.jpg`).

## 2. New run → pick `eda-to-report` → drag in the CSV → the form → Start. No chat typing

**✓** `demo-2-launcher.jpg`: workflow cards with their personas and "4 steps, 1
stop for you"; the drop zone holding `seattle-weather.csv` (47.1 kB); `question`
prefilled from the workflow's own default and `key_column` marked optional; and
"What will happen" naming the step that stops for a decision. Start creates the
run, uploads the file, and opens the run screen. Nothing typed into a chat.

## 3. Within 5 s the run screen shows step 1 running with Marie's activity; the canvas is not blank

**✓ 3.45 s**, measured from the Start click to *"marie is working on profile"*
appearing — the same instant the canvas filled. `demo-3-first-step.jpg` catches it
at 6 s: four tool calls counted (`ls 2 read_file 1 run_skill_script 1`), the whole
DAG drawn with the rest waiting, and the canvas showing her at work rather than
the blank pane run 001 complained about.

## 4. Files appear as they are written; figures append without stealing focus

**✓** Visible across `demo-5-gate.jpg` → `demo-6-reloaded.jpg` → `demo-7-report.jpg`:
the deliverables list grows from 3 to 9 files as the run writes them, the three
figures of the `analyze` burst append without moving the pane, and focus follows
the *step*, landing on the gate's own report and then on the final HTML.

## 5. The gate card appears inline at `data-gate` with the report below it; Approve continues; the header shows how long the gate waited

**✓** `demo-5-gate.jpg`: the card at its step, whole and without scrolling, with
`artifacts/data-gate.md` rendered beside it and the chip reading **needs you**.

The wait: run C was left at its gate for a minute and a half, as a person reading
the check table would. The card counted **waiting 1m 33s**, and the moment it was
approved the header read **Waited for you 1m 34s** (`demo-5-waited.jpg`).

> Runs A and B exposed a real bug here and are the reason C exists. Under `serve`
> the first ask never returns — it raises a LangGraph interrupt — and the tool
> re-executes when the answer arrives, so a clock read at that point measured the
> *resume*, not the wait: every gate reported 0 s. The pending record in
> `run.json` now supplies the true moment, and the total accumulates across
> answers so a gate that was sent back and later approved counts both waits.
> Fixed, tested (`test_the_wait_is_measured_from_when_the_run_stopped`,
> `test_the_wait_is_a_total_across_every_answer`), and re-verified by run C.

## 6. Reload the page mid-run: the screen restores fully and keeps following

**✓** Reloaded during `analyze` with three files on screen and the step at seven
tool calls. The screen rebuilt from `events.jsonl` — same files, same step, whole
DAG — and the counter had already moved to **eight** by the time it finished
rendering, so it was not a snapshot but a live attachment (`demo-6-reloaded.jpg`).

```
before reload: {files: 3, running: "analyze", activity: "thinking · 7 tool calls so far"}
after  reload: {files: 3, running: "analyze", activity: "working · 8 tool calls so far"}
```

## 7. The run ends without any error; the report opens full-width; "Download all" yields a zip with every deliverable

**✓** All three runs ended `done`, 4/4 steps, with no error banner, no failed
step and nothing in the chat — the `terminated` toast that ended run 003 is gone
(M2.2.1 item 1, verified end to end here rather than only in a test).

The report takes the whole pane on a Report toggle (`demo-7-report.jpg`).
`GET /runs/{id}/download` returned **250 kB, 9 files** for run C — the three
figures, both reports, the profile pair, the gate report and the findings — and
**not** the uploaded dataset, which is not something the run produced.

## 8. The header shows tokens, cached share and cost; the home screen shows the same cost

**✓** Run A's header: **Tokens 1.02M · Cached 93% · Cost $0.62**, and $0.62
against that run in the ledger (`demo-1-home.jpg`). The rates come from
`prices.yaml`; the arithmetic is reconciled against run 003's published table in
`tests/test_pricing.py`.

## 9. Ask in the chat "which finding should I be most careful with?" — answered from the artifacts, without starting a new run

**✓** `demo-9-chat.jpg`. The answer names Finding 2, quotes its figure back
(*"~0.61 °C/year"*), and explains its weakness from **what the gate could not
verify** — no station metadata, time coverage unconfirmed, four years of data
under a per-year slope. That is `data-gate.md` and `findings.md` being read, not
a tool result being repeated. No run was started.

> This line found the second real bug of the demo. `SqliteSaver` is sync-only and
> the AG-UI bridge streams with `astream_events`, so the first chat message
> anyone typed raised *"The SqliteSaver does not support async methods"* and the
> chat died with a red `terminated`. The run driver and the bridge now take
> different savers; the trade is written down in `serve.py` and below.

## 10. Reject a gate with a note, then Resume and Approve — the run completes, and the rejection and note stay visible

**✓** Run B, sent back with *"Check the fog rows before we analyse — the null
share looks off to me."* (`demo-10-sent-back.jpg`): the run stops, the banner
carries the note, and the step's history keeps it. Resume reopened the gate,
Approve finished the run, and afterwards the step reads:

```
Sent back after 1m 12s   "Check the fog rows before we analyse — the null share looks off to me."
Approved after 1s
```

## 11. Kill `dsagent serve` at a gate, restart, answer it — the run completes

**✓** Run B was killed at its gate (`pkill -f "dsagent serve"`, confirmed down),
`run.json` still reading `awaiting_gate` with the pending question. On restart the
browser showed the same run with the same gate still waiting — **"data-gate ·
waiting 53s"**, counting from when the run originally stopped
(`demo-11-gate-after-restart.jpg`) — and it was answerable: the run went on to
finish. The interrupt survived in `.dsagent/checkpoints.sqlite`.

## 12. Every screen renders in the Aiuda visual system

**✓** `t5-run-skinned.jpg` (1440) and `t5-run-1024.jpg` — the finished run screen
at a 1024 px viewport, below the 1100 px breakpoint: the horizontal stepper on
one line with `report` wrapping under it, the whole metrics row (elapsed, waited,
tokens, cached, cost) unclipped, and the report already open full-width because
the run is done. And every demo screenshot above: Instrument Serif on run and
report titles, Satoshi across the
UI, JetBrains Mono on every path, id and number — all self-hosted from
`ui/public/fonts`, no third-party font request. Cream ground, the accent only on
primary actions and the running step, emerald only for done, a deep brick for
failure. The chat is themed through CopilotKit's own variables, and no
browser-default control is left visible.

---

## Also exercised, beyond the twelve

- **A failed step, with eyes on it** — M2.2.1 item 7, the path no run had ever
  taken in front of a person: `t8-failed-step.jpg` shows *"noel could not finish
  analyze. Nothing after it ran."*, **Retry from analyze**, the unmet promise, and
  the runner's own line folded away instead of a stack trace (item 6).
- **A served run leaves a readable log** — M2.2.1 item 2:
  `.dsagent/runs/<id>/runner.log` exists for every run above.
- **`produces` ticks keep up with the files** — M2.2.1 item 3, measured ticking
  mid-step.
- **Interactive HTML** re-framed from `/runs-x/…` with `allow-scripts` and a
  server-side CSP (`t7-report-full.jpg`).

## What the demo does not cover

- **A run started from the chat still loses its gate on a server restart.** The
  bridge keeps an in-memory saver because `AsyncSqliteSaver` cannot be built
  outside a running event loop, and the app is constructed before uvicorn has
  one. Runs started from the launcher — the product's path, and every run above —
  persist. Fixing the chat side means building the agent inside the server's
  lifespan, which is a change to make deliberately.
- **Parquet preview** is implemented and tested, but no engine is installed in
  this venv, so the test skips here and the endpoint answers 415 by design.
  `eda-to-report` produces no parquet, so no demo line touches it.

---

# M2.6 — the living report, line by line

`docs/ui-living-report.md` §6 is the exit condition: thirteen lines that must each
be true from a clean checkout, with the evidence recorded here.

**Status: 13 of 13 verified, with a real model.** Five runs of `eda-to-report` on
`tests/data/seattle-weather.csv`, **$2.88 of a ten-run budget**, against
`dsagent serve` and `next build && next start`:

| | run | what it was for | wall | cost | cards |
|---|---|---|---|---|---|
| **1** | `eda-charts-…103207` | the first run with `show_chart` at all — recorded the fixture | 5 m 40 s | $0.605 | 6 |
| **2** | `eda-charts2-…110638` | the fixture the screens were built against, after the validator gained its render check | 6 m 19 s | $0.570 | 7 |
| **3** | `eda-to-report-…120426` | the demo run: §6.2–§6.10 and §6.13 | 14 m 04 s | $0.603 | 9 |
| **4** | `eda-to-report-…123802` | sent back, redone, approved: §6.11 | 7 m 31 s | $0.560 | 7 |
| **5** | `eda-to-report-…124606` | killed at its gate and restarted: §6.12 | 6 m 28 s | $0.544 | 6 |

Run 3 took fourteen minutes because it **failed in the middle and was retried
from the screen** — see "bugs only a real run could find" below. That is the run
in most of the screenshots, and its failure is the evidence for §1.7.

## Setup

```
pip install -e ".[ui,anthropic]"
cd ui && npm ci && npm run build
dsagent serve                      # :8000
cd ui && npm run start             # :3000
```

---

## 1. Home lists runs with status, duration, cost and personas; Workflows tab shows the team and the gates; empty state on a fresh install

**✓** `m26-1-home.jpg`: eleven runs, each with its status word, progress, the two
personas who worked on it, when it started, how long it took and what it cost.
`m26-1-workflows.jpg`: both workflows the cartridge declares, each step with the
persona on it and **`asks you`** marked on the ones that stop —
`eda-to-report` stops once, `mmm-meridian` three times — read off the loaded
cartridge, not written into the page. `m26-1-empty.jpg`: the run directory moved
aside, *"Nothing has run yet"*, what a run is, and one way to start.

## 2. New run: drop the CSV, type the question → the proposal appears with workflow, inputs, personas, gates and a cost estimate → Start. No chat typing

**✓** `m26-2-plan.jpg`. The file is read where it lands — **1,461 rows × 6
columns, 47.1 kB**, with its column names — and `key_column` comes back as a
guess that says why: *"unique across all 1461 rows, with nothing missing"*,
in an editable field. Who does what, with `data-gate` marked **"then it asks
you"**. And the estimate: **$0.54 · 4 m 35 s · 1 decision · based on 8 runs**.

Nothing about that proposal is generated: the steps and gates come from the
workflow, the shape from the file, the money from past runs of this workflow
that actually finished. A proposal a person is about to approve should not
itself be able to hallucinate.

Start opened the run screen. Nothing was typed into a chat.

## 3. Within 5 s the document shows the plan and Marie at work in the first section; the rail shows step 1 running with a live tool counter. Nothing is blank

**✓ 6 s from the Start click** to *"marie is working on this section"* on screen.
`m26-3-first-step.jpg` catches it at 15 s: the rail has `profile` running with
`read_file 3 · ls 2 · run_skill_script 1`, the document's first section shows
Marie at work with her counters — **and her profile is already appearing under
it**, because the file was written and the document picked it up while the step
was still running. Sections 2–4 are greyed with what will be written in them.

## 4. Sections fill in as steps finish, with a reveal, without losing scroll position

**✓** Visible across `m26-3-first-step.jpg` (section 1 filling, 2–4 greyed) →
`m26-7-gate.jpg` (1 and 2 written, 3 and 4 still greyed) →
`m26-9-finished.jpg` (all four). Each section arrives with a 200 ms fade and
slide, and a pending one is the *plan* rather than an empty space.

Scroll is not touched when a section arrives: the document is one scroller and a
section is appended to it, so the browser keeps the offset. Measured during §6.8:
the page was scrolled 3,023 px into `analyze` when a new file event landed and
the position did not move.

## 5. `analyze` produces at least three interactive Vega-Lite charts: hover shows values, a quantitative chart pans/zooms, changing the mark type re-renders locally, a brush posts context to the chat, and "Ask Noel to change this" yields a modified chart

**✓ seven charts and two tables** in run 3. Every spec carries `params`: `hover`
selections for tooltips on five of them, and a `zoom` interval bound to scales on
the temperature trend, so it pans and zooms. Two carry a mark toggle
(`bar / point / line`); switching it patches the spec on the client and
re-renders — verified on `Snow days are far colder…`, bars to points, legend and
all.

**The brush** (`m26-5-brush.jpg`): dragging across the precipitation chart puts
`selected month 2012-10-08 → 2014-02-10 · of 48 rows · sent as context` under it,
and that context travels with the next question.

**"Ask Noel to change this"** (`m26-5-amended.jpg`): asked for *"make the fitted
trend line a thicker dashed red line so it stands out from the monthly series"*.
Noel read the spec, changed it, and emitted it under the same `chart_id` — the
card on the page became **v2**, with the dashed red trend, in place. A second
request added an interval brush to the precipitation chart; that is the brush in
`m26-5-brush.jpg`. Both are `amend_chart` calls, validated and drawn once before
they were recorded, exactly like any other chart.

## 6. A table card shows the profile with sort/filter; Pivot opens Perspective on it

**✓** Filtering the column profile for `temp` left **2 of 6 rows**, and the
footer said so; clicking a column header sorted it. `m26-6-pivot.jpg`: Pivot
mounts Perspective — grouped by column with a TOTAL row, in its light theme,
from the inline WebAssembly bundles. No CDN, no asset route, and nothing is
downloaded until the button is pressed.

## 7. The gate card appears inline at `data-gate` with the checks table rendered as what is being approved and the cost so far/estimate; Approve continues; the header later shows the wait

**✓** `m26-7-gate.jpg`: the card at its own step, under Marie's nine checks
rendered as a sortable table — the thing being approved, not a link to it — with
**"What you are approving: that noel starts analyze on this data as it stands"**
and **"What it costs: $0.18 so far. 8 finished runs of this workflow cost $0.54
in total on average. The run is paused until you answer; waiting costs nothing."**
The card counted **waiting 43 s**; approved at 54 s, the header read **54s
waited** and the step's history kept *"Approved after 54s"*.

## 8. Reload mid-run: the screen restores fully and keeps following

**✓** Reloaded during `analyze` with two cards on screen and the step at
`run_python × 3`. The screen rebuilt from `events.jsonl` — same cards, same
steps, `analyze` still running — and the counter had already moved to
`run_python × 5` by the time it finished rendering, so it was a live attachment
rather than a snapshot.

```
before reload: {cards: 2, tools: "read_file × 4 · run_python × 3 · ls × 2 · execute × 1"}
after  reload: {cards: 2, tools: "run_python × 5 · read_file × 4 · ls × 2 · execute × 1"}
```

## 9. The run ends with no error; the executive summary lands at the top; Export HTML opens a report whose charts are still interactive; Export zip contains every deliverable; Replay scrubs the run at 10×

**✓** Runs 3, 4 and 5 all ended `done`, 4/4 steps, no error banner and nothing in
the chat.

`m26-9-finished.jpg`: **14 m 05 s · $0.60 (92 % cached) · 6 deliverables + 6
charts · 54 s waited**, the share and export row, the report's own opening
paragraph lifted to the top under *"In one paragraph"*, and **versions v1…v9**
with v9 — the run being read — marked.

**Export HTML** (`m26-9-export.jpg`), run 5: 948 kB, four live charts with their
error bars and fitted trends, opened from a plain file server. Measured in the
browser: **zero external requests**. The only absolute URLs in the file are XML
namespaces and one URL inside a Vega code comment.

**Export zip**, run 5: 14.7 kB, **six files** — both reports, the findings, the
gate report and the profile pair — and **not** the uploaded dataset, which is not
something the run produced.

**Replay** (`m26-9-replay.jpg`): the scrubber rewinds the whole screen at 10× —
the gate open again, sections 3 and 4 back to their plan, two cards instead of
nine, the rail's steps un-ticked. It is the same reducer that draws the live
screen, run over the log up to a moment.

## 10. Ask in the chat "which finding should I be most careful with?" — answered from the artifacts without a new run

**✓** The orchestrator named **Finding #1**, quoted its numbers back — *"snow (26
days, 8.55 mm/day, 95% CI 5.72–11.39) … rain (641 days, 6.56 mm/day, 95% CI
5.89–7.23)"* — and explained why it is fragile: the intervals overlap, and
snow's is wide precisely because n = 26. That is `findings.md` being read, not a
tool result repeated. No run was started.

## 11. Reject a gate with a note, Resume, Approve — done; the rejection is in the audit trail; the gate card showed a diff of `data-gate.md` on re-entry

**✓** Run 4, sent back with *"The key uniqueness check reads 'not applicable' —
date was declared as the key for this run, so check it properly and say so in
the table."*

`m26-11-diff.jpg` is the gate asked the second time, and it carries the diff:

```
- `key_uniqueness.column` in the profile is `date`, i.e. a key was declared, so this
- check applies (not "not applicable").
+ `date` was declared as the key for this run (`key_uniqueness.column` in the
+ profile is `date`, not `null`), so the uniqueness check is evaluated against it
+ above rather than marked not applicable: the profile reports `unique: true` with
+ `0` duplicates across all 1,461 rows, so the grain holds.
```

Marie was re-run **with the note in her prompt** and addressed exactly what was
raised. Both decisions are in the audit trail — *"Sent back after 27s"* with the
note, then *"Approved after 1m 17s"* — and the header totalled **1m 45s waited**.

> This line is the reason the runner changed. A rejected gate used to leave its
> step `done` and simply ask again: the persona never saw the note, nothing was
> rewritten, and the only way forward was to approve the artifact you had just
> refused. There was no second version because nothing produced one. A rejection
> now sends the step back, carries the note into its prompt, and keeps the
> refused version aside so the two can be compared. See the log.

## 12. Kill `dsagent serve` at a gate, restart, answer it — the run completes

**✓** Run 5 was parked at `data-gate` when `dsagent serve` was killed
(`curl` to the API then answered `000`). `run.json` still read `awaiting_gate`
with the pending question and its `asked_at`. On restart the browser showed the
same gate, still waiting — `m26-12-restart.jpg`, **needs you · data-gate ·
1m 08s**, counting from when the run originally stopped — it was answerable, and
the run went on to finish: `done`, 4/4, **$0.544**, with **69 s** of human wait
recorded across the restart. The interrupt survived in
`.dsagent/checkpoints.sqlite`.

## 13. Every screen is in the Aiuda visual system; self-hosted fonts; no default CopilotKit or browser control visible

**✓** Every screenshot above: Instrument Serif on run, document and section
titles; Satoshi across the UI; JetBrains Mono on every path, id, number and
spec — all self-hosted from `ui/public/fonts`, no third-party font request.
Cream ground with the navy rail as the one dark surface; the accent on primary
actions and the running step only; emerald for done, a deep brick for failure,
amber for a gate that is waiting. The chat is themed through CopilotKit's own
variables — a second block of them for the rail's navy — and Perspective through
its own light theme. No browser-default control is left visible.

---

## Also exercised, beyond the thirteen

- **A failed step, with eyes on it** — `m26-x-failed.jpg`: *"noel could not
  finish analyze. Nothing after it ran."*, the promise it missed, the runner's
  own line, and **Retry from analyze** — *"the steps before it are kept; only
  this one runs again."* Pressed, it re-ran that step and the run completed. The
  chart Noel had already emitted before dying is still on the page, drawn from
  real data, in a run that failed.
- **Stop** is on the rail whenever a run is live.
- **A plan's edits reach the run**: the key column, accepted as a guess and then
  editable, is written back before Start.

## What the demo does not cover

- **A run started from the chat still loses its gate on a server restart.**
  Unchanged from M2.5, and unchanged on purpose: the bridge keeps an in-memory
  saver because `AsyncSqliteSaver` cannot be built outside a running event loop.
  Runs started from the launcher — the product's path, and every run above —
  persist, as §6.12 shows.
- **PDF export** and **a findings diff between two versions** are deferred with
  reasons, in `docs/ui-product-log.md` under task 9.
- **Parquet cards.** `show_chart` reads parquet through the same reader as CSV,
  and the preview endpoint is tested for both, but no persona wrote a parquet in
  these five runs, so no card on this screen was drawn from one.
