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
