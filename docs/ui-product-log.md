# M2.5 — UI product pass, working log

One entry per task of `docs/ui-product.md` §8: what changed, what was verified,
what was decided, what was deferred. Written as the work happens, on branch
`m25-ui-product`.

**Budget.** §6 allows six real-model runs for the whole milestone. Spent so far: **0**.
Development is against the replay fixture.

---

## Task 1 — replay fixture and the event log

**Branch base.** `v2` @ `82e12f4`, which already carries M2.2.1 item 1 (the
recursion limit, PR #74, merged). Items 2 and 3 of M2.2.1 were *not* on `v2` when
this branch opened, though §0 of the spec assumes them; item 2 is closed by this
task (below) and item 3 by task 6, where the ticks live.

### What changed

**The runner writes its own record.** `<run_dir>/events.jsonl` gets every
`RunnerEvent`, one JSON object per line, and `<run_dir>/runner.log` gets every
`log()` line. Both are written by the runner rather than by a front end, which is
M2.2.1 item 2: `dsagent serve` passes `log=lambda m: None`, so a browser-driven
run used to leave nothing readable behind while a CLI run did. The file is written
whether or not anyone is listening, because the run that nobody watched is exactly
the one somebody reads afterwards.

**A gate is announced before it is asked.** `dsagent.step` gains a `gate` object:
`{kind, prompt, asked_at, decision, note, decided_at}`. The runner emits
`status=awaiting_gate` with `decision: null` *before* calling `ask_human`, and
emits the step again with the answer filled in once one arrives. Until now the
only surface that knew a run had stopped was whoever was holding the interrupt, so
a second tab, a reloaded page and a stakeholder on a link all saw a run that had
silently gone quiet.

**`RunState.gate`** records the same pending question in `run.json`, and
`GateRecord` gains `asked_at`. `run.json` is the only thing that outlives the
process, and a run waiting for a person is the one most likely to still be waiting
when the process dies (§7.11). `ts - asked_at` is also the human wait — M2.2.1
item 5, the 37 seconds run 003 spent at its gate with nothing to show for it.

**A gate's note survives.** `ask_human` may now return `GateAnswer(decision, note)`
as well as a bare `GateDecision`; the note lands in `run.json` and on the step
event. The card already collected a note and the runner used to drop it, which
made §7.10 ("the rejection and note are visible in the step's history")
unbuildable.

**`dsagent.note` — persona narration as a runner event.** The runner emits what a
persona says as it says it, attributed to the step it is in. M2.2 read the same
text off the AG-UI message stream and attributed it *by time*, because nothing on
the wire says who is talking. Attribution by construction is strictly better, and
unlike a message stream it survives a reload, a second tab, and a run nobody was
attached to.

**`RunState.save` is atomic.** Same-directory temp plus `os.replace`. The runs API
reads `run.json` on every poll while the run is rewriting it; the replay tests hit
the truncated-file window within a minute of existing.

**`src/dsagent/runs.py`** — reading a run directory back: `list_runs`,
`summarize`, `read_events(after=…)` with a line-number cursor, `read_log`,
`deliverables`, `is_live`. No FastAPI, no `[ui]` extra; HTTP goes in front of it in
task 2.

**`src/dsagent/replay.py` + `ui/fixtures/run-eda-003/`** — a recorded run played
back into a fresh run directory at 10×, writing the same `events.jsonl` every
other reader consumes, so a replayed run and a real one are the same run to
everything downstream. The gate is the one thing not replayed: the recorded wait
is discarded and the run stands there until a person answers.

### The fixture is run 003, not a new run

`tools/make_replay_fixture.py` builds it from
`.dsagent/runs/eda-to-report-toolu_01Y3oq…/` — the actual run 003 directory, still
on this machine. Real: step boundaries, the 37-second gate wait, every file with
the mtime it got, per-step tool counts, token usage, and each persona's closing
summary. Reconstructed: the *order and individual timestamps* of tool calls within
a step, which `run.json` does not record, spread evenly across their step. Nothing
else is invented — no narration a persona did not write, no file the run did not
produce.

This spends **zero** of the six real-model runs. §6 says "capture one real run";
capturing run 003's record is the same artifact for no money, and the four runs
§7 needs are worth more than a fifth recording of a run we already have.

### Verified

`pytest` 192 passed / 7 skipped, `ruff check src tests` clean. New: 12 tests in
`tests/test_run_store.py` (event log, cursor, half-written tail, summaries,
pending gate surviving its asker, listing), 8 in `tests/test_replay.py` (whole
workflow, files landing, the gate holding, rejection, resume-reopens-the-gate),
3 new gate-payload tests and 2 narration tests in `tests/test_runner_events.py`.
The replay tests run against the committed fixture, so a fixture that stops
matching the schema fails the suite.

### Deferred

The `--replay` CLI flag lands with task 2: the flag is meaningless until the run
endpoints exist, and dead code in between would be worse than one commit's wait.

---

## Task 2 — the runs API, and who owns a run

### The decision this task turned on

**A run belongs to the server, not to the tab that started it.** `POST /runs`
creates the directory, `POST /runs/{id}/start` sets the orchestrator going on a
background thread, and the browser follows `events.jsonl` over SSE and answers
gates over HTTP. It is a viewer.

§4.1 asks for exactly this and §4.2 explains why: reload, a second tab, and a
CLI-started run all have to show the same screen, and none of them can if the run
only exists inside one browser's event stream. §7.6 and §7.11 are the same
requirement with the page and the server killed respectively.

The run still goes *through* the orchestrator — the path the chat takes, as §4.1
says — so the run exists in a thread that can be asked about afterwards. The only
thing the server decides for it is which directory to write to.

### What changed

- **`src/dsagent/api.py`** — `GET /cartridges`, `GET|POST /runs`,
  `GET /runs/{id}`, `PUT /runs/{id}/data/{input}`, `POST /runs/{id}/start`,
  `POST /runs/{id}/gate`, `GET /runs/{id}/events` (SSE) + `/events.json`,
  `GET /runs/{id}/log`. Its own module rather than more of `serve.py`, and for a
  reason worth writing down: FastAPI resolves route annotations against the
  *module* globals, so importing `Request` inside the function that registers the
  routes made every route take `request` as a query parameter (422, "Field
  required"). `api.py` is only imported from `build_app`, which already requires
  the `[ui]` extra, so FastAPI is a plain module-level import here.
- **`src/dsagent/driver.py`** — `GraphDriver`: one background thread per run,
  `answer_gate` resuming the graph with `Command(resume=…)`, and `_mark_failed`
  so a failure *outside* the runner (no credentials, a graph that gave up) still
  lands in `run.json` instead of leaving the screen spinning. `Replay` implements
  the same three methods, which is what `--replay` swaps in.
- **`dsagent serve --replay <fixture> [--replay-speed]`** — every route identical,
  no model loaded, no agent endpoint mounted.
- **Run-reading tools for the orchestrator** — `list_run_files(run_id)` and
  `read_run_file(run_id, path)`. Its own file tools are rooted in the chat
  workspace; a run lives in its own directory. Without these, §7.9 ("which finding
  should I be most careful with?") could only be answered from the sentence the
  tool result returned.

### Three traps, each found by a test

**`config: RunnableConfig` does not work under `from __future__ import annotations`.**
The run id travels to `run_workflow` in the graph config. Declared as a parameter,
the annotation is the *string* `"RunnableConfig | None"`, LangChain does not
recognise the injection, and the tool is handed `None` — silently. The run then
lands in a directory named after the tool call, with the operator's uploaded
dataset sitting in a different one. `ensure_config()` reads it correctly; probed
both ways on langchain-core 1.6.3.

**Inputs were resolved before the recorded ones were merged in.** A re-entering
caller — `--resume` without the original `-i`, or a model retyping `inputs={}` —
failed validation for an input the run directory had recorded all along. A run now
keeps the inputs it started with, and they are resolved after the merge. This is
also what makes the launcher's form, rather than the model, authoritative.

**The gate note was dropped in serve mode.** `interrupt_gate` returned a bare
decision, so "reject, because the fog rows look wrong" reached `run.json` as
"reject". §7.10 asks for that note in the step's history.

### Uploads are a raw-body `PUT`, not multipart

`PUT /runs/{id}/data/{input}` with the file as the body. Multipart would mean
adding `python-multipart`, and `CLAUDE.md` says not to add a dependency without
asking — for a form with exactly one file in it, when a drag-and-drop already has
the `File` in hand (`fetch(url, {method: "PUT", body: file})`). The input name is
in the path, so a workflow declaring two datasets needs no new convention, and the
filename is sanitised to its last segment and `[A-Za-z0-9-_.]`.

If multipart is wanted later it is a one-line dependency and a small route change.

### Verified

`pytest` 219 passed / 7 skipped, `ruff` clean, `dsagent cartridge validate` green.
New: 18 tests in `tests/test_runs_api.py` (driven against a replayed run through
`TestClient`) and 9 in `tests/test_driver.py` (the real `create_deep_agent` graph
with a stub model and fake personas — the run lands in the launcher's directory,
the form beats the model, the gate is answered from outside the asking thread, a
rejection resumes, a failure outside the runner reaches `run.json`).

End to end by hand against `dsagent serve --replay … --replay-speed 20`: create →
upload 48 kB CSV → start → the run parks at `data-gate` after two steps → approve
over HTTP → finishes in 47 s with 116 events, 560 k tokens carried over from the
recording, and `report/findings.html` served at 272 kB.

**A note on `TestClient` and SSE.** A `client.post(...)` issued while that same
client holds an open stream deadlocks — one portal, one request at a time — and so
does breaking out of a stream early. Both are test-harness artifacts, not server
behaviour, but they cost an hour: the stream test now answers its gate through the
driver and consumes to the end frame. The endpoint also checks
`request.is_disconnected()`, which is what a real client hanging up looks like.

### Deferred

The replay writes no `runner.log` (it is not the runner); `GET /runs/{id}/log`
says so rather than 404ing. Cost per run is still `null` everywhere — §4.6, task 9.

---

## Task 3 — home screen, and a run restored from its log

### What changed

**`/` is the run list**, `/runs/<id>` the run screen, `/new` the launcher (task 4).
The M2.2 shell — one page, chat plus canvas, state reduced from the live AG-UI
stream — is gone. `ui/app/lib/` now holds the typed API client, the reducer, and
the hooks; `ui/app/components/` the panels.

**Everything on the run screen is reduced from `GET /runs/{id}/events`.** One
endpoint hands over the whole backlog and then keeps streaming, so opening a
finished run, reloading mid-run, opening a second tab and attaching to a run the
CLI started are the same code path — §4.2's whole point, and M2.2.1 item 4.
`EventSource` reconnects on its own and resumes from `Last-Event-ID`, which the
server now honours.

**The visual system landed here rather than in task 5**, because a home screen
has to look like something. `tokens.css` carries §3's palette and the three
typefaces; the run list is a ledger (mono numbers, right-aligned, hairline rules)
rather than a stack of cards, so two runs can be compared down a column. Two
colours the spec's four do not cover: `--fail`, a deep brick that is *not* the
accent — orange is the primary action, and a failed step drawn in it reads as
something to click — and navy at low alpha for secondary text and rules, so
nothing introduces grey. Fonts are declared but not yet installed; the stacks
fall back to system faces until task 5.

**The proxy prefix moved from `/runs/*` to `/dsa/*`.** `/runs/<id>` is now a page
in this app, and a Next rewrite on that path would proxy the run screen away to
the API.

**`npm run lint`** added (ESLint flat config, `eslint-config-next`; `next lint`
is gone in Next 16). It found four real React-hygiene bugs, all fixed rather than
silenced: `setState` in three effect bodies, and `Date.now()` read during render.
The fixes are better code — the event stream accumulates into a local that the
connection's own `onopen` publishes, the file viewer keys its fetch result by URL
so "loading" is derived rather than assigned, and anything that counts up takes
its time from one `useClock` hook.

### The two hours that went into an iframe

The report rendered as a blank white pane. The `sandbox=""` attribute looked
guilty and was not: a hand-made iframe with the same attribute and the same URL
painted fine, and so did the real one the moment it was moved out of its
container. The cause was **`backdrop-filter: blur(6px)` on the sticky topbar** —
it puts the page on a compositing path where a sandboxed iframe *elsewhere on the
screen* loads its document and paints nothing. Removed; the comment in
`globals.css` says why, because the next person will want a blurred header too.

Two real fixes came out of the hunt anyway: the iframe now has a definite height
(`min-height: 0` up the flex chain), and the file list shows the whole
workspace-relative path, because `artifacts/findings.md` and `report/findings.md`
are two different files with one basename in every run so far.

### Verified

`npm run build`, `npm run typecheck`, `npm run lint` clean; `pytest` 219,
`ruff` clean. In the browser against `next build && next start` with
`dsagent serve --replay`: the home screen lists every run this machine has,
including the ones the CLI made (`docs/runs/ui-product/t3-home.jpg`), and a run
opened cold rebuilds its four steps, its promises, its ten files and its report
from `events.jsonl` alone (`t3-run-restored.jpg`).

### Deferred

Cost is `—` everywhere until task 9. The progress region shows two steps before
scrolling on a 900 px window; task 6 owns that. The chat pane is a placeholder in
replay mode, which is honest — there is no model behind a recording.
