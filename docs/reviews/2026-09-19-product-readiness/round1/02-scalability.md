# Round 1 — scale-review (Opus 4.8) — verbatim

Lens: scalability, resources, concurrency (local single-user bar). Reviewed 2026-09-18 at
commit `8ac2758`. No files modified.

---

Adversarial scalability / resources / concurrency review of dsagent-v2 serve stack. Ten findings, ranked. Read the actual code; no files modified.

=== CRITICAL ===

1. Per-project Env (kernel/container) is never evicted — unbounded resident memory.
   src/dsagent/serve.py:254-272 (`chat_agents`/`chat_envs` dicts, `chat_agent_for`), released only at graceful shutdown in lifespan serve.py:158-169.
   Every distinct project you ever send a chat message to builds an Env on first use and caches it FOREVER (serve.py:257-268). A kernel Env is a live Jupyter/IPython process (kernel.py:26-34) — ~80-200MB RSS baseline, far more once pandas/numpy and DataFrames are loaded (an EDA on the 8.6MB Olist file easily pushes it past 500MB-1GB). A docker Env is a resident `sleep infinity` container (docker.py:99-120). There is no LRU, no idle reaper, no cap.
   Scenario: an analyst touches 40 projects over a week on a long-lived server → 40 resident kernels. At a conservative ~250MB each that is 10GB held with zero active work; with loaded frames, OOM. The default env is intentionally excluded from close (serve.py:163-164), correct, but every other project accumulates.
   Fix: LRU cache with a hard cap (e.g. keep the N most-recently-used chat envs, `close()` the evicted one and drop it from both dicts), and/or an idle-timeout reaper thread. Closing must remove from `chat_agents` too, not just `chat_envs`.

=== HIGH ===

2. Kernels/containers leak on any non-graceful shutdown, and can't be reaped by project.
   lifespan serve.py:159-169 is the ONLY path that closes chat envs; it runs only on a clean ASGI shutdown. On SIGKILL / OOM-kill / crash it never runs → docker containers survive as orphaned `sleep infinity` processes and kernel child processes are orphaned. Worse, container names are random (docker.py:169 `dsagent-{uuid4}`) and kernels have no on-disk handle, so a restart CANNOT find and reap the orphans it left behind. Repeated crash/restart cycles accumulate orphans indefinitely.
   Fix: register kernel shutdown via atexit; name run/chat containers deterministically (project-derived) and sweep stale `dsagent-*` containers on startup; or a `dsagent gc`.

3. `POST /agent` blocks the single event loop on first-use env build and per-turn seeding.
   The handler is `async def` (serve.py:287-296) but calls `chat_agent_for(project_id)` synchronously (serve.py:289) with no `await`/threadpool. Inside, first use runs `make_env` → for a kernel, `wait_for_ready(timeout=60)` (kernel.py:33) blocks the loop up to 60s; for docker, `ensure_image` + `docker run` (docker.py:250,177) blocks for the length of a build (minutes). And EVERY turn runs `seed_run` (serve.py:271, see #6) synchronously on the loop. While the loop is blocked, every other client freezes: all SSE run streams, every 4s/5s poll, `/agent/health`.
   Note on the "no lock" concern: because `chat_agent_for` has no `await`, under a single uvicorn worker it is effectively atomic w.r.t. the event loop, so the plain-dict check-then-set does NOT currently double-build. That safety is exactly what causes the blocking above. If anyone later makes env construction async (the natural fix), the check-then-set becomes a genuine double-build/leak race and needs an `asyncio.Lock` per project id.
   Fix: `await run_in_threadpool(chat_agent_for, project_id)` plus a per-project async lock guarding construction.

4. One kernel per project + `.clone()` shares it → concurrent same-project turns serialize and corrupt each other's output.
   serve.py:289 `chat_agent_for(project_id).clone()` clones the AG-UI wrapper but the underlying compiled graph — and the `run_python` kernel closure (kernel.py:90-101, single `kernel` nonlocal) — is shared. Two concurrent turns in one project (two tabs, or a launcher run + a chat turn) hit one kernel. `JupyterKernel.run` filters iopub by `msg_id` and `continue`s past others (kernel.py:44-45), which DISCARDS the other call's messages from the queue. Result: the second concurrent `run` loses its output and blocks to full `timeout` (600s, kernel.py:26) returning "[timeout]", while turns also serialize on the one kernel.
   Fix: single-flight lock per project around the turn, or a per-turn kernel, or at minimum a lock around `kernel.run`.

5. `list_project_artifacts` walks the ENTIRE workspace tree on every 5s poll and every turn.
   serve.py:514-533 uses `ws.rglob("*")` then skips top-level dirs AFTER descending into them — the `rel.parts[0] in ARTIFACT_SKIP_DIRS` check (serve.py:525) filters output only; rglob has already recursed through `.dsagent/skills/` (many files), `data/` (seeded datasets), `work/`, `model/`, `large_tool_results/`. Every file gets a `stat()` then the list is sorted. Route `/projects/{id}/artifacts` (serve.py:543-545) is polled every 5s per open chat (ChatCenter.tsx:177-189). The default project's workspace is the shared legacy `.dsagent/serve` dir which accumulates artifacts across ALL default chats forever, so this tree only grows.
   Scenario: 5,000 files (skills + datasets + accumulated EDA/run outputs) → 5,000 stats + a sort every 5s per open tab. Sync route runs in the anyio threadpool (default 40 workers), so it doesn't block the loop but it saturates the pool and thrashes disk.
   Fix: `os.walk` with in-place pruning of skip-dirs (never descend into `.dsagent`/`data`/etc.), or scan only known output dirs, plus an mtime-gated cache.

6. `seed_run` re-copies all datasets into the workspace on EVERY chat turn.
   serve.py:271 → datasets.py:53-66 unconditionally `shutil.copy2` every file in `data/` on every call to `chat_agent_for`. With the 8.6MB Olist file plus others, a 40-turn chat re-copies tens of MB per turn = GBs of redundant IO over the conversation, on the event loop (see #3). Also races: copy2 truncates/rewrites `data/x.csv` while a running kernel reads it → torn read.
   Fix: copy only when source size/mtime differs from dest; or seed once at env creation and on explicit dataset upload, not per turn.

=== MEDIUM ===

7. `SessionStore.list_for` parses every session's full transcript just to list titles.
   sessions.py:85-104 globs `*.json` and `json.loads` each file — but each file holds the FULL messages array (sessions.py:68-83), so listing headers parses every message of every chat, then discards all but title/updated_at. It also globs the whole sessions dir (all projects) each call. Route `/projects/{id}/sessions` (api_projects.py:80-84) is hit on project switch and after title changes.
   Scenario: 500 chats × ~200KB → ~100MB of JSON parsed per sidebar refresh.
   Fix: a lightweight header index/sidecar separate from messages.

8. Client PUTs the full transcript every turn; keepalive path silently drops chats over 64KB.
   ChatCenter.tsx:299-317 serializes ALL messages and PUTs the whole transcript each turn (debounced 600ms); server rewrites the entire JSON each time (sessions.py:81) → O(n) write per turn, O(n²) over a chat's life. On unmount/pagehide it saves with `keepalive: true` (ChatCenter.tsx:311,342-348). The Fetch keepalive spec caps total body at 64KB — inline vega-lite specs with embedded data blow past that in a few turns — so the pagehide/reload flush silently fails and the tail of a large chat is lost exactly when the debounced save hadn't fired.
   Fix: append-delta protocol or server-side merge; detect >64KB and fall back on the unmount path.

9. Polling: fixed intervals, no visibility gating, per-tab fan-out (but intervals DO clear on unmount).
   ProjectArtifacts 5s (ChatCenter.tsx:179-189) and SessionRuns 4s (ChatCenter.tsx:145-155) both clear their interval on unmount — no leak there. But there's no backoff and no `document.hidden` gating, so a hidden tab keeps firing #5 (rglob) and `list_runs` (runs.py:81-95 summarizes every run dir) forever. Multiple tabs multiply it.
   Fix: pause on `document.hidden`, back off when unchanged, or one shared poller.

10. No retention/GC anywhere — unbounded disk growth.
   Sessions dir (only user-deleted), the shared default workspace (#5), run dirs (each also holds a seeded-dataset copy + events.jsonl + logs), and built docker images all grow without bound. Amplifiers: `store.locate_run` scans all project dirs per lookup (projects.py:77-93) and `/projects` calls `run_dirs` per project (api_projects.py:19,23), so list cost grows with total runs.
   Fix: retention policy / TTL / size caps / a `dsagent gc`.

=== TOP 3 TO FIX FIRST ===
1. Finding 1 — bound and evict per-project chat envs (LRU + idle reaper). This is the one that OOMs a long-lived server.
2. Findings 3 + 6 — get env construction and per-turn `seed_run` off the event loop (threadpool + per-project lock) and make seeding incremental. One blocked build currently freezes every client.
3. Finding 5 — stop rglob'ing the whole (ever-growing, shared) workspace every 5s; prune during the walk and cache. Compounded by the default workspace never being cleaned (#10).
