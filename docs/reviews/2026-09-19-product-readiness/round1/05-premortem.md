# Round 1 — premortem (Opus 4.8) — verbatim

Lens: 3-months-later failure narratives (local bar). Reviewed 2026-09-18 at commit
`8ac2758`. No files modified. Superseded in scope by `../round2/06-premortem-product.md`.

---

PREMORTEM — dsagent-v2 (read-only; no files changed). 9 narratives, ranked by likelihood × blast radius. All evidence is in today's code.

════════════════════════════════════════
#1 — The session store shreds its own chats (non-atomic writes on the "single source of truth")
Likelihood: HIGH · Blast: DATA LOSS / SILENT CORRUPTION
Scenario: A user has a long, valuable chat. They reload the page — or a second poll fires — during a save. The page comes back with the chat empty, or truncated to half a message, or gone from the sidebar. Over weeks this is "the app randomly eats conversations."
Root cause: The whole architecture (serve.py:231-251) declares the SessionStore the ONE durable source of truth. But SessionStore.save writes it NON-ATOMICALLY: `self._path(session_id).write_text(json.dumps(rec), ...)` — sessions.py:81. There is no temp-file+os.replace, unlike RunState.save which does exactly that and documents *why* at runner.py:273-292 ("a plain overwrite has a window where the reader gets a truncated — or empty — file"). Sessions carry full message arrays including inline Vega specs with embedded data, so the write window is large; a concurrent `getSession`/`list_for` read (sessions.py:56-62, 85-104) during that window hits the half-written file, `json.loads` raises, and `get()` returns None → the client renders an empty/lost chat. A crash mid-write leaves the file permanently truncated. Additionally the OSError is SWALLOWED (sessions.py:82-83 `except OSError: pass`) and the route still returns 200 (api_projects.py:108-116), so a disk-full/permission failure means the chat was never saved and the client is told it succeeded.
Smallest fix: give SessionStore.save the same temp-file + os.replace it already wrote for run.json (runner.py:290-292), and let write failures surface (non-200) so the client's retry path — which exists, ChatCenter.tsx:313-315 — actually triggers.

════════════════════════════════════════
#2 — "It always converges via stable IDs" — except it doesn't, with two tabs (or two devices)
Likelihood: MED-HIGH · Blast: DATA LOSS
Scenario: User opens the same chat in two tabs (or phone + laptop). Tab A sends a turn; tab B (loaded earlier) sends a different turn. One turn silently disappears forever.
Root cause: The convergence claim (serve.py:231-251) assumes one writer. Persistence is last-write-wins with NO version/merge/etag: the client PUTs its entire message array (ChatCenter.tsx:299-317), and SessionStore.save unconditionally overwrites (sessions.py:68-83). Tab B loaded `[msgs]` at mount (ChatCenter.tsx:224-244), so after its turn it writes `[msgs, turnB]`, clobbering the `[msgs, turnA]` tab A wrote. The in-memory server checkpoint may briefly hold both (both tabs share one thread_id → one InMemorySaver, add_messages dedups), but it's ephemeral and the store — the thing reloads read from — is now missing turnA. The convergence direction is backwards: the checkpoint converges to the client, the client overwrites the store, and the store is the lossy one. Same failure with "client sends stale history after an out-of-band change" (e.g. a run card appended server-side).
Smallest fix: optimistic concurrency — store an `updated_at`/version, reject a PUT whose base version is stale (409), have the client reload+merge. At minimum, detect divergence instead of silently overwriting.

════════════════════════════════════════
#3 — Per-project kernels and in-memory checkpoints are never evicted (weeks-of-use leak)
Likelihood: HIGH (given the stated "weeks" horizon) · Blast: RESOURCE EXHAUSTION → CRASH
Scenario: A team uses this for a month, creating projects freely (the UI encourages it — "new project = one field", projects-sessions.md:113). RAM and process/FD count climb until the box OOMs or can't spawn kernels, and the server dies taking every in-flight run with it.
Root cause: `chat_agents` and `chat_envs` are dicts that only ever grow — serve.py:254-272. A project's kernel/container is built on first chat use (serve.py:261) and cached forever; there is NO idle eviction and NO LRU. They're closed ONLY at process shutdown (serve.py:158-169). Each cached agent also holds an `InMemorySaver` (serve.py:265, `checkpointer_for(ws, memory=True)`) that accumulates every thread's full transcript in RAM for the life of the process and is never pruned. So both live kernels and chat history in memory grow without bound in the number of projects-ever-touched × turns.
Smallest fix: an LRU/idle-timeout on `chat_envs`/`chat_agents` (close+evict the env, drop the agent) — a cap of N live projects with close() on eviction.

════════════════════════════════════════
#4 — Disk fills quietly; every chat turn re-copies the datasets
Likelihood: MED-HIGH · Blast: RESOURCE EXHAUSTION + LATENCY
Scenario: Months in, the disk is full and nothing runs; or every chat message in a project with a 200MB dataset takes seconds and thrashes the disk.
Root cause: (a) Nothing is ever pruned: run workspaces, `events.jsonl` (grows unbounded per run, runner.py:100-107), `gate-versions` copies (runner.py:905-933), and session JSONs all accumulate; runs are only deletable while `pending` (api.py:246-268), so every completed run is permanent by design with no retention policy anywhere. (b) `seed_run` deep-copies EVERY project dataset into the chat workspace on EVERY chat turn — serve.py:271 calls it inside `chat_agent_for`, which runs per request; seed_run shutil.copy2's all files (datasets.py:53-66). Same full copy happens into every new run workspace (tools.py:160-163, api.py:243). Large datasets → per-message copy cost and disk churn.
Smallest fix: skip the re-seed when the dataset mtime/size already matches the workspace copy (or seed once per project-session, not per turn); add a retention/size budget for runs+events.

════════════════════════════════════════
#5 — The next dependency/format bump bricks existing runs and chats
Likelihood: MED · Blast: DATA LOSS (dead runs) / OUTAGE
Scenario: A routine `pip install -U` or `npm update` lands, and every run on disk fails to load, or every saved chat renders blank, or `POST /agent` 500s on boot.
Root cause: Several hard couplings to internal shapes with no version field or migration: (a) `RunState.load` does `cls(**d)` with no unknown-key tolerance — runner.py:267-271 — so any renamed/removed dataclass field makes every existing run.json raise TypeError and become unresumable/unreadable. (b) Chat messages are persisted VERBATIM as the third-party AG-UI message shape (sessions.py:68-83, api_projects.py:93-116) and replayed through `fromAgUiMessages`/`toAgUiMessages` (ChatCenter.tsx:273, 300-305); a shape change across an assistant-ui/ag_ui upgrade means old sessions fail to parse, and the save path silently falls back to a *different* shape (`agent.messages`, ChatCenter.tsx:302-304). (c) The FastAPI body wiring depends on `globals()["RunAgentInput"]=...` (serve.py:277-280) and on copying the internals of `add_langgraph_fastapi_endpoint` (serve.py:281-296) — an upstream refactor breaks the agent route outright. (d) Gate resume relies on LangGraph matching interrupts POSITIONALLY (runner.py:807-822, documented "Reproduced on langgraph 1.2.11") — a change to interrupt identity silently misroutes a gate answer to the wrong gate.
Smallest fix: add a `version` field to run.json and session JSON and filter unknown keys on load (`cls(**{k:v for k,v in d.items() if k in fields})`); pin the three fragile deps tightly and add one load-old-fixture test per format.

════════════════════════════════════════
#6 — run.json lost-update race between the HTTP handlers and the runner
Likelihood: LOW-MED · Blast: SILENT CORRUPTION (run metadata)
Scenario: An operator edits inputs / requests a plan while the run is starting, or a driver failure lands as the runner writes a step — and a field (plan, inputs, or a step status) silently reverts.
Root cause: run.json has multiple independent load-modify-save writers with no locking: the runner (runner.py:273-292, called throughout), `_rewrite_state` in the HTTP handlers (api.py:774-787, used by plan/upload/stop), and `GraphDriver._mark_failed` (driver.py:340-364). os.replace makes each write atomic (no torn file), but two read-modify-write cycles interleave → last writer wins and the other's change is lost. The STOP flag was already moved OUT of run.json into a file for exactly this reason (runner.py:73-81), which shows the race is known — but plan/inputs edits still go through the racy path.
Smallest fix: funnel all run.json mutations through one lock/owner, or move operator-editable fields (plan, edited inputs) to their own file like STOP_FILE.

════════════════════════════════════════
#7 — No observability: every failure is a swallowed exception
Likelihood: MED · Blast: PROLONGS/HIDES ALL OF THE ABOVE
Scenario: Something above happens in production and there is no log, no metric, no error surfaced — the first signal is a user saying "my chat is gone."
Root cause: Silent-catch is pervasive and intentional. Server side: sessions.py:33-34, 61-62, 82-83, 102-103 all swallow; the driver records errors only onto the run and into an optional log that `serve` sets to a no-op (driver.py:133 default `lambda m: None`, and serve.py builds GraphDriver without a log at 305-308). Client side: essentially every call is `.catch(()=>{})` / `.catch(() => {})` — App.tsx:45-47,54-56,69-74,106; ChatCenter.tsx:147-148,181,232-240,314-315. There is no error boundary, no toast, no server error log for the chat path.
Smallest fix: wire GraphDriver's `log` to a real logger in serve.py; log (don't `pass`) in SessionStore; add one client-visible error surface for save failures.

════════════════════════════════════════
#8 — A chat-started run's gate is a dead end (and re-does paid work)
Likelihood: MED · Blast: ANNOYING + WASTED $ (GPU)
Scenario: A user asks the orchestrator in chat to "build the MMM," it hits a human gate, the user reloads or the server restarts — and the gate can never be answered; the run stalls forever, or a retry re-pays for completed steps.
Root cause: This is documented as a known hole (api.py:14-21): a chat-started run's interrupt lives in the chat graph's IN-MEMORY saver (serve.py:236-251), so `POST /runs/{id}/gate` — which resumes the *driver's* graph — returns 409, and the run "has to be approved from the chat that started it, and a server restart loses the question entirely." It is real, not hypothetical, and combined with #3's unbounded in-memory checkpoints it's the same store that also leaks. The MMM fit step is exactly the expensive one (docs/architecture.md:149-158, `timeout: 4h`, GPU) whose gate this affects.
Smallest fix: the fix is already named in the code (api.py:19-21) — build the AG-UI agent inside the server lifespan so it can hold an async SQLite saver, so chat gates survive restart and can be answered via the runs API.

════════════════════════════════════════
#9 — Open surface if ever bound beyond localhost
Likelihood: LOW (design is single-user/localhost, architecture.md:24) · Blast: SECURITY (HIGH if exposed)
Scenario: Someone runs `dsagent serve` on a shared host / port-forwards it "just to demo," and now anyone on the network can list/read/delete every project, session, run and file, and serves model-generated HTML from the origin.
Root cause: There is no auth on any route (serve.py/api.py/api_projects.py — none present), and destructive routes are open: DELETE /sessions/{id} (api_projects.py:118-122), DELETE /runs/{id} (api.py:246-268), DELETE project data (api_projects.py:72-78). Uploads read the full body into memory with no size cap (api.py:294, api_projects.py:66) — trivial memory-exhaustion. Path traversal is genuinely well-defended (resolve_run_file serve.py:408-437, resolve_project_file serve.py:496-511, tools.py:219-249) and the scripted-artifact CSP is thoughtful (serve.py:440-460), so the gap is purely the missing authn/z + no upload limit, which only bites once the "no multi-tenant auth" non-goal meets a real network.
Smallest fix: bind to 127.0.0.1 by default and refuse a non-loopback bind without an explicit token/flag; cap upload size.

════════════════════════════════════════
THE ONE THING TO FIX FIRST
#1 — make SessionStore.save atomic (temp-file + os.replace, exactly as RunState.save already does at runner.py:290-292) and stop swallowing its write errors. It is the single component the whole design designates as the durable source of truth for users' actual conversations, and today it is written with a plain, interruptible, silently-failing `write_text`. Everything else degrades a run or a resource; this one destroys the user's primary artifact — the chat — and does it invisibly. It's also the smallest change of the nine.

(Cross-cutting note the ranking reflects: the "single source of truth / converges via stable IDs" claim rests on one atomic, single-writer store. Today the store is neither atomic (#1) nor single-writer (#2), so the invariant it's built on doesn't hold.)
