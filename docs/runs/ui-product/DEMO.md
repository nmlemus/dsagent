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
