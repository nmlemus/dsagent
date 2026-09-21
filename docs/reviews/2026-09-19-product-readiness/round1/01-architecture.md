# Round 1 — arch-review (Opus 4.8) — verbatim

Lens: architecture & correctness, the 7 CLAUDE.md invariants, the `globals()` hack, the
per-project router, the single-source-of-truth claim. Reviewed 2026-09-18 at commit
`8ac2758`. No files modified.

---

Adversarial architecture review of the chat-persistence / per-project serve work. Findings ranked by severity, each with file:line, failure scenario, and fix. Invariants #4 (runner never plans) and #5 (produces verified, not warned) still hold — verified in runner.py:680-682 and driver/tools. #6 mostly holds (the hand-written /agent router only wraps LangGraphAGUIAgent.run, it does not reimplement the loop). The damage is concentrated in the per-project run-location seam and the SessionStore durability claim.

## CRITICAL

**C1 — Run-scoped orchestrator tools ignore the store, so they break for every non-default project.**
`src/dsagent/runner/tools.py:180,183,198,219-249` (`list_run_files`, `read_run_file`, `_run_workspace`, `_resolve_in_workspace`) and `src/dsagent/runner/charts.py:436,454,465,487` (the chart amend tools) all resolve a run as `runs_dir / run_id`. But `resolve_run_dir` (tools.py:89-91) puts a chat-started run under `store.runs_dir(project)` = `<projects_root>/<pid>/runs/<id>` for any non-default project. Scenario: user is in project "acme", asks the orchestrator to run `eda-to-report`; run lands at `projects/acme/runs/eda-to-report-<id>`. The orchestrator then calls `list_run_files(run_id)` to summarize → `runs_dir/<id>` does not exist → returns "unknown run <id>". Every follow-up ("which finding matters most?") fails, and the chart amend tools can't locate the run to amend. The tools' entire stated purpose (docstring lines 108-114) is dead outside the default project. Fix: thread `store` into `workflow_tools`'s inner tools and `amend_tools`, and resolve via `store.locate_run(run_id)` with the `runs_dir/<id>` fallback — the same pattern `driver._run_dir` and `run_dir_of` already use.

## HIGH

**H1 — SessionStore is not the "single durable source of truth" it claims; it loses updates and is written non-atomically.**
`src/dsagent/sessions.py:68-83`. `save()` is full-overwrite, last-writer-wins, no version/merge, and writes with a bare `write_text` (not temp+os.replace — contrast api.py:774-787 `_rewrite_state`, which IS atomic). Two concrete failures: (a) Concurrent tabs on the same chat — a feature the docstring at sessions.py:6 and serve.py:184 explicitly advertises. Tab A adds an exchange and saves; Tab B (stale transcript in memory) sends a message and saves its view → A's exchange is silently erased from the durable store. (b) A crash or a racing pagehide-flush-vs-debounced-save (ChatCenter.tsx:299-339 can fire both) mid-write leaves truncated JSON; `get()` catches ValueError → returns None → `project_of` falls back to "default", so a run started in that chat lands in the wrong project and the transcript is gone. This directly contradicts serve.py:231-241 ("single source of truth… can never durably diverge"). Fix: atomic temp+rename write; add an `updated_at`/version guard and reject stale saves (409), or merge by message id server-side.

**H2 — The serve.py "checkpoint always converges to the stored transcript" claim is false under concurrent tabs.**
`src/dsagent/serve.py:231-251`. The in-memory checkpoint is per-project, process-lifetime (created once in `chat_agent_for`, serve.py:265), and `add_messages` only ever *adds* by id — it never removes. Scenario following H1(a): the in-memory checkpoint retains Tab A's exchange (added under the thread_id) even after the SessionStore has dropped it. Now the checkpoint and the durable store diverge for the rest of the process lifetime, and the model is fed context the user's transcript no longer shows. "It can never durably diverge" is only true across a restart (which rebuilds from the store); within a process it diverges the moment two writers touch one thread. Fix: don't claim convergence; either key the chat graph with `checkpointer=False` (rebuild context purely from the client's messages every turn) or make the SessionStore authoritative on read by passing the full stored transcript and discarding checkpoint history.

**H3 — First turn of a new chat can run in the wrong project (race).**
`ui/src/components/ChatCenter.tsx:236-240` registers the session (`saveSession({project, title:"", messages:[]})`) fire-and-forget in the getSession-404 catch, not awaited, while the composer is immediately usable. `serve.py:288` resolves `project_of(thread_id)` per `POST /agent`, and `sessions.project_of` returns "default" when no record exists yet (sessions.py:64-66). If the user sends the first message before that registration round-trips (or if it failed — `.catch(()=>{})` swallows it), the first turn's orchestrator runs against the *default* workspace/env: it lists the default project's datasets, writes the persona's report into the default workspace, and any run it starts (`resolve_run_dir`, tools.py:89) is created under the default project. Fix: await the registration before mounting the thread/enabling send, or have `POST /agent` create-on-first-touch the session with the project the client declares (send project in the RunAgentInput/forwarded metadata) rather than defaulting silently.

## MEDIUM

**M1 — `preview` can't preview files of non-default-project runs.**
`src/dsagent/api.py:486` calls `resolve_run_file(runs_dir, run_id, path)` with no `store`, while `run_dir_of` (api.py:168-172) and `_profile_inputs` (api.py:355) correctly pass it. So the plan/profile works but the canvas preview 404s for the same run in a non-default project. Fix: pass `store` (same one-line fix as C1).

**M2 — `globals()["RunAgentInput"]=…` / `globals()["Request"]=…` is a fragile idiom.**
`src/dsagent/serve.py:278-279`. It mutates module globals at first `build_app` call to make FastAPI's `get_type_hints` resolve the annotations of a function-local endpoint (needed because `from __future__ import annotations` stringifies them and the imports are function-local to keep serve.py importable without the `[ui]` extra). It works for the single-startup case but: it leaks past `build_app`, a second `build_app` in-process (tests) re-sets them, and it couples the endpoint's correctness to a side effect 10 lines away. Cleaner: drop annotation-based body resolution entirely — `async def agent_endpoint(request: Request)` and parse `RunAgentInput.model_validate(await request.json())` inside. No globals, no reliance on hint resolution, and `Request` is the only annotation left (already resolvable via a local `Request = Request` or a top-level typing-only import).

**M3 — Default project's chat env is the same kernel object the driver's orchestrator uses; two graphs, one stateful kernel, no locking.**
`src/dsagent/serve.py:261` (`chat_env = env if project_id==DEFAULT_ID`) and serve.py:306 (driver built with the same `env`, same `workspace`). For the default project the chat orchestrator and the launcher-run driver share one KernelBackend whose variables persist across calls (architecture.md:189). A chat turn and a launcher run executing in the default env concurrently interleave kernel state. Non-default projects get their own env (make_env), so only `default` has this. Fix: give the default project its own chat env too, or document that the default env is single-writer and serialize.

**M4 — Resumed-gate coordination is timing/heuristic across two threads via a JSON file.**
`src/dsagent/driver.py:239-270` (`_walk_past_decided_gates`) polls `RunState.load` every `PARK_SECONDS/5` for ≤12 rounds and answers re-raised interrupts with `_standing_decision` (driver.py:58-68), which returns APPROVE if *any* step is done+approved. Two fragilities: (a) it races the runner rewriting run.json — it reads `state.gate` to decide "a person is being asked" vs "re-raising a decided gate", a field the runner is concurrently rewriting per step; (b) with several gates and a real kernel whose close takes seconds, 12 rounds × ~1s may expire before the graph finishes walking, leaving the run parked on an already-approved gate with no thread behind it. It's bounded and best-effort, but it's coordination-by-polling-a-file. Not easily fixed cheaply; at minimum make the bound proportional to the gate count and log when the walk gives up.

## LOW

**L1 — Invariant #1 leaks in comments (and, more substantively, in the harness UI).**
`src/dsagent/cartridge/models.py:141-142` and `src/dsagent/envs/docker.py:140` name "meridian"; `src/dsagent/host/build.py:171` and `src/dsagent/serve.py:57` name "marie"/"analyze". These are illustrative comments, not logic, so low — but invariant #1 says "may mention… Meridian" flatly. More real: the harness-shipped UI hardcodes the DS domain — `ChatCenter.tsx:363-365` ("delegate to Marie / Noel / Ana / Pablo"), `SUGGESTIONS` (ChatCenter.tsx:126-130, "Inventa ventas mensuales"), and "Message the DS team". `_mount_ui` (serve.py:347-361) ships this as part of the harness, so swapping the cartridge leaves the UI naming DS personas and suggesting sales data. Fix: source persona names and example prompts from the loaded cartridge(s), not string literals.

**L2 — `/runs` lists only the default (legacy) dir.**
`src/dsagent/api.py:192` `list_runs(runs_dir, …)` never sees non-default project runs (they're under `projects/<id>/runs`). If any screen uses `/runs` as a global list, non-default runs are invisible there. Likely intentional (project-scoped `/projects/{id}/runs` exists) but the asymmetry with the project-aware `run_dir_of` is a trap. Confirm intent or aggregate across projects.

**L3 — `_safe(session_id)[:128]` (sessions.py:21) can collide two thread ids to one file.** Client-generated thread ids; unlikely but silent. Low.

## Top 3 to fix first
1. **C1** — thread `store` into the run-reading and chart-amend tools (and M1's `preview`). One seam, breaks all non-default-project chat runs today.
2. **H1** — make SessionStore writes atomic and add a stale-write guard; it is billed as durable/source-of-truth but loses updates and can corrupt.
3. **H3** — close the new-chat project-registration race (await it, or resolve project from the client's declared project on `POST /agent`) so first turns don't silently execute in the default project.
