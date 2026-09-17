# M2.5 — the demo script, line by line

`docs/ui-product.md` §7 is the exit condition: twelve lines that must each be true
from a clean checkout, with the evidence recorded here.

**Status: 8 of 12 verified, 4 waiting on a real-model run.** Everything that can
be established without a model has been, against `dsagent serve --replay` — which
serves the same API, the same event stream and the same screens, differing only
in what drives the run. The four lines below marked ⏳ are the ones a recording
cannot honestly stand in for: they are about the live agent path (the run ending
clean, the chat answering from artifacts) or about killing a real server.

> **What is still needed:** `ANTHROPIC_API_KEY` in the environment that runs
> `dsagent serve`. Three runs of `eda-to-report` on `tests/data/seattle-weather.csv`
> finish this document — one clean run, one rejected-then-resumed, one killed at
> its gate — about $1.50 of the six-run budget, none of which has been spent.

## Setup

```
pip install -e ".[ui,anthropic]"
cd ui && npm ci && npm run build
dsagent serve                      # :8000
cd ui && npm run start             # :3000
```

Replay mode, for everything below that did not need a model:

```
dsagent serve --replay ui/fixtures/run-eda-003 --replay-speed 12
```

---

## 1. The home screen lists previous runs; a fresh install shows an empty state

**✓ verified.** `docs/runs/ui-product/t9-home-costed.jpg` — workflow, dataset,
status, progress, started, duration and cost, newest first, including runs the
CLI started and one that failed. Statuses seen on that screen: `done`,
`failed`, `waiting for you`.

The empty state is the `EmptyState` branch of `ui/app/page.tsx`, shown when the
run directory is empty. ⏳ *screenshot pending on the demo run, where the runs
directory is moved aside first.*

## 2. New run → pick `eda-to-report` → drag in the CSV → form → Start, no chat typing

**✓ verified.** `t4-launcher.jpg`: the workflow cards (personas, "4 steps, 1 stop
for you"), the drop zone holding `seattle-weather.csv` (47.1 kB), `question`
prefilled with the workflow's own default, `key_column` marked optional, and
"What will happen" naming the step that stops for a decision. Start creates the
run, uploads the file and goes to the run screen. Nothing is typed into a chat.

## 3. Within 5 s the run screen shows step 1 running with Marie's activity; the canvas is not blank

**⏳ needs a real run.** The mechanism is in place and visible in replay: the run
screen draws the declared DAG before any event arrives, and the canvas shows the
persona at work with a live tool counter rather than a blank pane
(`ui/app/runs/[runId]/page.tsx`, `Waiting`). Click-to-first-step on a client-side
navigation measures **1.8 s**. What a recording cannot prove is that a real
`profile` step reports itself inside five seconds.

## 4. Files appear as they are written; figures append without stealing focus

**✓ verified.** `t4-run-live.jpg` and `t6-run-done.jpg`. The replay plays run
003's real file mtimes, so the four-figure burst arrives over the same 21 seconds
it originally did; the list is append-only and focus moves only when a step ends
(run 001's observations 2 and 3, now enforced in `reduce`). Measured live, the
promise ticks fill *while* `analyze` is still running:

```
analyze met=0 unmet=2 files=3
analyze met=1 unmet=1 files=4      ← the figures glob ticks on the fourth file
analyze met=2 unmet=0 files=8
```

## 5. The gate card appears inline at `data-gate` with the gate report below it; Approve continues; the header shows how long the gate waited

**✓ verified.** `t6-gate-inline.jpg`: the card sits at its step, fully visible
without scrolling, with `artifacts/data-gate.md` rendered beside it and a live
"waiting 2m 08s". After approving, the step keeps **"Approved after 16s"**
(`t4-run-live.jpg`) and the run header carries **Waited for you 2m 15s**.

## 6. Reload the page mid-run: the screen restores fully and keeps following

**✓ verified.** Reloaded mid-`analyze` with four files on screen; after the
reload the screen rebuilt from `events.jsonl` and the run carried on to
completion — 10 files, four steps done, without a second start.

```
before reload: {files: 4, chips: 4, running: "analyze"}
after  reload: {files: 10, chips: 4, done: 4}
```

## 7. The run ends **without any error**

**⏳ needs a real run.** This is the M2.2.1 item-1 fix (`recursion_limit=150` on
the served graph, PR #74) end to end. `tests/test_serve_recursion.py` asserts a
gated workflow completes through the served graph without `RUN_ERROR`, but the
line as written is about a real run in a browser.

## 7b. "Download all" yields a zip with every deliverable

**✓ verified.** `GET /runs/{id}/download` returned a 378 kB zip of ten
deliverables through the proxy — the four figures, both reports, the profile and
the gate — and pointedly *not* the uploaded dataset. Asserted file-by-file in
`tests/test_runs_api.py`.

## 8. The header shows tokens, cached share and cost; the home screen shows the same cost

**✓ verified.** `t9-home-costed.jpg` (home, **$0.47**) and `t6-run-done.jpg`
(header: Tokens 560k, Cached 90%). The per-step costs match run 003's published
table — profile $0.065, data-gate $0.054, analyze $0.241 — and the arithmetic is
reconciled against that document in `tests/test_pricing.py`.

## 9. Ask in the chat "which finding should I be most careful with?" — answered from the artifacts, without starting a new run

**⏳ needs a real run.** The plumbing is built: the run screen publishes the open
run through `useAgentContext`, and the orchestrator has `list_run_files` and
`read_run_file` so it can read that run's own workspace rather than only the
sentence its tool returned.

## 10. Reject a gate with a note, then Resume and Approve — the run completes, and the rejection and note stay visible

**✓ verified.** `t8-sent-back.jpg` and `t8-gate-history.jpg`. Sending
`data-gate` back with *"The fog rows look wrong — check the null share before we
analyse."* stopped the run and showed the note twice; Resume reopened the gate;
Approve finished the run; and the step's history then reads **"Sent back after
4s"** with the note, followed by **"Approved after 0s"**.

## 11. Kill `dsagent serve` at a gate, restart, answer it — the run completes

**⏳ needs a real run.** The mechanism is `.dsagent/checkpoints.sqlite` and is
unit-tested at exactly the point that matters: `tests/test_checkpointer.py` stops
a graph at an `interrupt()`, throws the saver *and the graph* away, rebuilds both
over the same file and answers. The same test shows the in-memory saver losing
the question, which is the behaviour being replaced.

## 12. Every screen renders in the Aiuda visual system

**✓ verified.** `t5-run-skinned.jpg` (1440) and `t5-run-1024.jpg` (1024):
Instrument Serif on run and report titles, Satoshi across the UI, JetBrains Mono
on every path, id and number, all self-hosted from `ui/public/fonts` with no
third-party request. Cream ground, one accent used only for primary actions and
the running step, emerald only for done, and a deep brick — not the accent — for
failure. No CopilotKit default is left on screen: the chat is themed through its
own variables (`t5-run-skinned.jpg`, left pane). No browser-default form control
appears; the launcher's fields, the gate's note and the file controls are all
styled.

---

## Also exercised, beyond the twelve

- **A failed step, with eyes on it** (M2.2.1 item 7, never previously looked at):
  `t8-failed-step.jpg` — a real run, failed by a persona that did not write its
  declared figures, showing *"noel could not finish analyze. Nothing after it
  ran."*, a **Retry from analyze** button, the unmet promise, and the runner's raw
  line folded away rather than a stack trace (item 6).
- **Interactive HTML**: the report re-framed from `/runs-x/…` with
  `allow-scripts` and a server-side CSP (`t7-report-full.jpg` shows the full-width
  report; the footer toggles scripts).
- **The whole DAG at all times**, including steps that have not run yet, so a
  stopped run cannot look complete.
