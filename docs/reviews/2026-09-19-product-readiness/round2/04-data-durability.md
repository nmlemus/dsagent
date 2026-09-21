# Round 2 — data-durability (Fable 5.1) — verbatim (head + parts 1–3)

Lens: durability inventory of every persisted artifact, silent-corruption / lost-update
sequences, 12-month growth model, retention/backup, erasure map (GDPR), PII, target storage
model, versioning/migration, RPO/RTO. Reviewed 2026-09-19 at commit `8ac2758` against the live
`.dsagent/` on the dev machine (77 MB, 18 default runs, 3 projects, 7 sessions, one 1.5 MB
checkpoint DB + 4.1 MB WAL). No files modified.

---

Scope: every persisted artifact under `.dsagent/`, judged against "hosted multi-tenant SaaS and/or self-hosted product, real customer data".

## 1. Durability inventory

| Artifact | Writers | Readers | Schema ver. | Atomic write | Concurrent writers | Crash behaviour | Deleted by |
|---|---|---|---|---|---|---|---|
| `projects/sessions/<id>.json` (full transcript) | `SessionStore.save` `sessions.py:68-83` via `PUT /sessions/{id}` `api_projects.py:93-116` | `get` `:56-62`, `list_for` `:85-104` (reads **every** file per list), `project_of` (every `/agent` turn, `serve.py:288`) | none | **No** (`write_text`), OSError swallowed, route returns 200 | Last-write-wins full snapshot, no version | Torn JSON → `get` returns None → 404 → client PUTs `messages: []` (`ChatCenter.tsx:227-240`) → **loss made permanent** | `DELETE /sessions` unlinks JSON only |
| `<run>/run.json` (`RunState`) | runner main thread `runner.py:551,566,577,655,697,869,881,898`; chart-tool thread `:504-507`; HTTP `_rewrite_state` `api.py:299,347,368,394`; driver `_mark_failed` `driver.py:340-364`; amend `charts.py:438-441` | `RunState.load` `runner.py:267-271`, `read_state` `runs.py:98-100` (polled every 4 s per tab) | none | Yes (per-thread temp + `os.replace`) | 3 independent load-modify-save writers + 2 threads serializing one mutable object | Good on disk; in-memory races (see §2) | `DELETE /runs/{id}` only while `pending` (`api.py:246-268`) |
| `<run>/events.jsonl` | `_append` `runner.py:375-386` (open/append/close per line, OSError swallowed, no fsync); amend `charts.py:446-447` | `read_events` `runs.py:162-184` re-scans from line 1 every 250 ms per SSE stream (`api.py:441`) | none (event names only) | Line-append; partial last line + next append → both events lost silently | Two appenders (runner + amend) — OK per POSIX `O_APPEND` for <4 KB lines, not guaranteed for larger | Partial line dropped; cursor is a line number so an unparsable line silently shifts nothing but is lost | With run dir only |
| `<run>/runner.log`, `<run>/session` marker, `<run>/stop` | runner; `mark_session` `sessions.py:25-35` (swallows); `api.py:390` | `runs.py:195-197,149` | none | No | — | — | With run dir; marker dangles after session delete |
| `<run>/gate-versions/<step>/v<n>/` | `_keep_version` `runner.py:905-933` (`copy2`, failures logged) | `api.py:510-532` | none | No | — | — | With run dir |
| `<run>/workspace/` (incl. `data/` seeded copy of every project dataset) | `seed_run` `datasets.py:53-66`, `_store_upload` `api.py:750-761`, model code, Docker bind-mount | files route, zip, export, `produces` | — | `copy2`/`write_bytes` = truncate-then-write | Model code vs seed copy | Torn dataset possible | With run dir (never, post-start) |
| `projects/<id>/project.json` | `_write` `projects.py:174-177` | `_read` | none | **No** | — | Torn → project silently disappears from `list()` (`:153-155`) with all its runs unreachable by project routes | **No `DELETE /projects` exists** |
| `projects/<id>/data/<name>` (datasets) | `store_dataset` `datasets.py:28-32` | `seed_run` (every `/agent` turn `serve.py:271` and every run create) | none; no size/sha256 manifest | **No** (`write_bytes` in place, overwrites same name) | Upload vs. concurrent `seed_run` read → torn copy in a run | Disk-full/crash → truncated CSV with the right name, seeded into every future run, analysed without error | `DELETE …/data/{name}` removes this copy only |

### Inventory remainder (part 1/3)

**`projects/<id>/workspace/` (chat workspace: `report/`, `artifacts/`, `data/`, `large_tool_results/`)**
- Writers: model code from ALL chats of a project sharing one env + one Jupyter kernel (`serve.py:257-272` — one `chat_agent_for(project_id)`, `.clone()` per request, same `chat_env`). deepagents' overflow clip writes raw tool output to `large_tool_results/<tool_call_id>` at the env backend root (`.venv/.../deepagents/middleware/_overflow_clip.py:11`, `_message_eviction.py:123-153`). `seed_run` re-copies every project dataset into `workspace/data/` on every `POST /agent` (`serve.py:269-271`).
- Readers: `GET /projects/{id}/artifacts` (skips `large_tool_results`, `serve.py:489`) and `GET /projects/{id}/files/{path}` (`resolve_project_file` `serve.py:496-511` blocks only `.dsagent` — so `large_tool_results/<id>` IS served).
- Schema: none. Atomic: no. Concurrency: chat A and chat B overwrite each other by path (no versioning, no artifact→session attribution). Deleted by: nothing.
- Live evidence: `.dsagent/serve/large_tool_results/toolu_01A81y…` = 95 KB of raw customer rows (`customer_id, age, tenure_months, …`), written 2026-09-18, never cleaned.

**`serve/.dsagent/checkpoints.sqlite` (+ `-wal`, `-shm`) — launcher runs**
- Writer/reader: `SqliteSaver` on one shared connection with `check_same_thread=False` (`serve.py:393-403`), driver `get_state` (`driver.py:299-308`). Path = `<serve workspace>/.dsagent/checkpoints.sqlite` — INSIDE the default project's chat workspace, which is the kernel's cwd and `LocalShellBackend` root (`kernel.py:87`, `serve.py:261,191`).
- Schema: LangGraph's. Atomic: SQLite. Never pruned; no delete touches it. Live: 266 checkpoints / 450 writes across 12 `run:*` threads, 25–167 KB per thread; DB 1.5 MB, WAL 4.1 MB. Blobs contain dataset column names (2 checkpoints, 4 writes match Titanic columns) — tool outputs with data are persisted here.

**In-memory chat checkpoints** (`InMemorySaver`, `serve.py:265,233-251`): per-process, every super-step of every turn snapshots the full state; unbounded growth for process lifetime; retains deleted transcripts until restart; an in-flight chat gate is lost on restart (documented at `api.py:14-21`).

**Legacy `.dsagent/serve` workspace**: adopted in place as the `default` project's chat workspace (`serve.py:189-192`) and as `default`'s runs dir parent (`projects.py:67-72`). Contains `artifacts/ model/ report/ work/ data/ large_tool_results/` from every pre-projects chat plus the checkpoint DB. `.dsagent/legacy-serve-data-backup-20260918/` and the orphan `.dsagent/projects/sessions.json` (old session→project map, zero readers in `src/` or `ui/`) show migrations were done by hand; there is no migration tool.

**Exports / logs**: `GET /runs/{id}/export.html` and `/download` are generated per request, not stored (good), but not audited and out of erasure reach. `runner.log` (`runner.py:349-356`) is appended per line, OSError swallowed, never rotated, not covered by any delete. `run_dir/session` marker (`sessions.py:25-35`, swallows OSError) dangles after a session delete.

**Data root**: everything is CWD-relative (`cli.py:43 RUNS_DIR = Path(".dsagent/runs")`, `:286 --workspace .dsagent/serve`, `serve.py:178`). Starting the server from another directory silently creates an empty second store. `.dsagent` is gitignored; no backup/restore anywhere.

## 2. Silent-corruption and lost-update paths (concrete sequences)

1. **Chat wiped by the client after a torn write (BLOCKER).** `PUT /sessions/x` → `write_text` (`sessions.py:81`) interrupted by crash / disk full / SIGKILL on deploy → truncated JSON. User reopens → `get` hits `ValueError` → None → 404 (`api_projects.py:87-91`) → `ChatCenter.tsx:232-239` treats ANY rejection as "new chat" and PUTs `{title:"", messages:[]}` → the recoverable truncated file is replaced by a valid empty one. Same path on a 500 or proxy timeout. And `save` swallows OSError and returns 200 (`sessions.py:82-83`, `api_projects.py:116`), so the client clears `dirty` and never retries a failed save.

2. **Two tabs lose turns (BLOCKER).** Tabs A and B load chat X. A sends 3 turns; each debounce writes A's full snapshot (`ChatCenter.tsx:299-317`). B sends one turn; B's snapshot (built from its pre-A load) overwrites the file. A's 3 turns are gone; A still displays them until reload. No version, no `If-Match`; `updated_at` is client-supplied (`api_projects.py:114`).

3. **Truncated dataset propagates silently (BLOCKER).** `store_dataset` (`datasets.py:28-32`) does `write_bytes` in place: truncate-then-write of the same name. A `seed_run` running concurrently (every `/agent` POST in that project, `serve.py:271`; every `POST /runs`, `api.py:243`) `copy2`s a partial file into a run's `workspace/data/`; or the process dies mid-write and a partial CSV with the correct name is seeded into every later run and chat. No size/sha256 is recorded anywhere, so nothing can notice; reports are computed on wrong data with no error.

4. **Wrong run acted on across projects (IMPORTANT).** `new_run_id` is unique per directory at one-second granularity (`api.py:143-157`). Two projects create `eda-to-report-20260918-195400` in the same second. `locate_run` (`projects.py:77-93`) returns the first match (default first, then `iterdir` order). `POST /runs/{id}/gate|stop`, `/files`, `DELETE` act on the other project's run.

5. **`RunState.load` brittle both ways (REQUIRED).** `cls(**d)` `runner.py:271`, `StepRecord(**…)` `:298`, `GateRecord(**gate)`, `ChartRecord(**stored)` `charts.py:457,493`. Add a field in v+1 then roll back → every run.json written by v+1 raises `TypeError`; `list_runs` swallows (`runs.py:92-93`) so the run vanishes from the list and `_walk_past_decided_gates` returns silently (`driver.py:256-258`). Remove/rename a field → every existing run.json fails to load. No `schema_version` in run.json, session JSON, project.json, events.jsonl or chart records.

6. **One `RunState` object, two threads (IMPORTANT).** `_on_chart` on the tool thread mutates `self._state.charts` and calls `save()` (`runner.py:504-507`) while the runner thread may be in `asdict(self)` inside its own `save()` (`:291`, called at `:655,697,869,881`) → `RuntimeError: dictionary changed size during iteration`, or a snapshot missing the chart the tool just confirmed to the persona. The earlier per-thread temp-file fix (`:283-288`) addressed the file collision, not the shared object.

7. **HTTP load-modify-save races (IMPORTANT).** `upload_input` (`api.py:285-301`) and `plan_run` with edited inputs (`:338-347`) both `read_state` → mutate → `_rewrite_state` (`:774-787`); concurrent calls drop one input. `stop_run` (`:392-394`) and `driver._mark_failed` (`driver.py:348-364`) rewrite status from a stale read. Same "three writers, one document" shape the code already diagnosed for the stop flag (`runner.py:73-81`).

8. **Chat artifacts overwritten across chats (IMPORTANT).** Shared project workspace (`serve.py:189-192,257-272`): chat A writes `report/customers-by-state.md`; chat B (or the same persona a week later) writes the same path; the "Archivos del proyecto" link in A now opens B's content. No artifact versioning or attribution.

9. **Orphan sessions (NICE).** Every abandoned "New chat" leaves a ~130-byte untitled file (`ChatCenter.tsx:236-239`); `list_for` hides untitled (`sessions.py:94`). On disk: 5 of 7 session files are these.

10. **Event-line concatenation (NICE).** Crash after a partial `events.jsonl` line; the next `_append` (`runner.py:375-386`) continues the same line; `read_events` drops the unparsable line (`runs.py:178-180`): two events lost, possibly the `awaiting_gate` step event a reconnecting browser rebuilds from.

## 3. Twelve-month growth (one active customer) (part 2/3)
Assumptions: 10 runs/day, 30 chat turns/day, 250 working days → 2,500 runs, 7,500 turns in ~500 chats; a 50 MB dataset (the live demo one is 8.6 MB; real customer files are larger).

- **Dataset copies**: 2 project-level (`data/` + chat `workspace/data/`) + 1 per run (`seed_run`, `api.py:243`, `tools.py:160-163`) = ~2,502 byte-identical copies → **125 GB/yr** at 50 MB, 1.25 TB at 500 MB. None content-addressed. Plus every `/agent` POST re-copies all datasets (`serve.py:271`): 30 × 50 MB = 1.5 GB/day of pointless I/O per project. Live evidence: the same 48,219-byte `seattle-weather.csv` exists 20 times under `.dsagent/`; the 8.6 MB olist file 3 times in one project with one run.
- **run.json 12–28 KB + events.jsonl ~60 KB (~160 lines, 120 tool events) + runner.log** → ~225 MB/yr. Small, but `list_runs` reads every run.json + `session` marker on every call (`runs.py:81-95`) and it is polled every 4 s per tab (`Sidebar.tsx:50-60`, `ChatCenter.tsx:145-155`); each SSE stream rescans `events.jsonl` from byte 0 every 250 ms (`api.py:441`, `runs.py:174-184`).
- **checkpoints.sqlite**: 25–170 KB per run thread, never pruned → ~250 MB/yr in one file on one shared connection, WAL already 3× the DB.
- **Session JSON**: 1.2 KB/message without charts (live: 26 KB / 22 messages). `VIZ_INSTRUCTIONS` tells the model to inline `data.values` (`ChatCenter.tsx:114-119`), so a 200-turn chat with 40 charts is 1–2 MB; 500 chats ≈ up to 1 GB, each rewritten whole on every 600 ms debounce, and the whole transcript is re-sent to the model every turn (storage and token bill scale together).
- **`large_tool_results`**: ~100 KB per spill, several per turn on big files → GBs/yr in a directory nothing lists, prunes or protects.
- **`InMemorySaver`**: every super-step of every turn snapshots full state; by year end the server process holds GBs of transcript. This is an OOM, not a storage line.
- gate-versions: copies of `produces` per rejection, MBs. `run.json.plan.profile.preview`: 20 raw rows per run (`api.py:636-643`; verified in `eda-to-report-20260917-120426/run.json`).

## 4. Retention, backup, restore — today
- Retention: **none** for any class (runs, checkpoints, `large_tool_results`, orphan sessions, gate-versions, logs, chat workspace). Started runs are undeletable by design (`api.py:252-257`, 409).
- Backup: none. Store is CWD-relative, gitignored, single disk. No `dsagent export/import`, no documented dump. Restore: undefined.
- Whole-project export: none. Per-run HTML/zip only (`api.py:543-610`). No session export, no dataset bundle, no machine-readable transcript in a schema you own (stored verbatim in `@ag-ui/client` message shape, `sessions.py:45-48`, `ChatCenter.tsx:302`).

## 5. Erasure map — where one dataset and one chat end up, and whether deletion removes it

| Copy | `DELETE /projects/{p}/data/{name}` (`api_projects.py:72-78`) | `DELETE /sessions/{id}` (`:118-122`) | Removed by anything |
|---|---|---|---|
| `projects/<p>/data/<name>` | yes | — | yes |
| `projects/<p>/workspace/data/<name>` (chat seed; `seed_run` only adds, never removes, `datasets.py:53-66`) | **no** | no | no |
| every `runs/<r>/workspace/data/<name>` | no | no | only while `pending` |
| `runs/<r>/run.json` → `plan.profile.preview` (20 raw rows) | no | no | no |
| `runs/<r>/gate-versions/**` | no | no | no |
| `runs/<r>/events.jsonl` (`args_preview` = first 120 chars of EVERY tool arg incl. `run_python` literals, `runner.py:1066-1081`; `dsagent.note` persona narration about the data; chart `columns`) | no | no | no |
| `runs/<r>/runner.log` | no | no | no |
| `checkpoints.sqlite` run threads (verified: tool outputs with dataset columns) | no | no | no |
| `InMemorySaver` chat threads | no | no (still in RAM) | restart |
| `projects/<p>/workspace/large_tool_results/<id>` — raw rows, HTTP-served | no | no | no |
| session JSON: tool messages up to ~11 KB each (`read_file` of reports, `ls` output) + model-inlined `data.values` | — | yes | yes |
| runs launched by that chat (`session` marker) | — | no (marker dangles) | no |
| chat workspace `report/`, `artifacts/` written by that chat | — | no | no |
| downloaded `export.html` / zip | out of reach; and not audited | | |
| **Project as a whole** | — | — | **no `DELETE /projects` route exists** (`api_projects.py`) |

Conclusion: "delete my dataset" cannot be honoured; "delete this chat" leaves most derived content; "delete the project" is impossible via the API; started runs are undeletable — incompatible with GDPR Art. 17. Portability (Art. 20): partial for runs, absent for chats/projects/datasets.

## 6. PII handling
None. No classification, no encryption at rest, no per-tenant key, no owner/tenant column on any record (sessions/projects/runs carry no principal at all — `sessions.py:72-78`, `projects.py:33-40`, `runner.py:226-265`), no audit log of reads/exports/downloads, no redaction in events/logs/checkpoints. Model-executed code has host-level filesystem access under the kernel env (`kernel.py:87-101`: unrestricted Python, cwd = workspace), so any file the process can read is reachable from a prompt. Data sent to the model provider is out of scope here but must appear in the DPA.

## 7. Target storage model (part 3/3)

**Relational (Postgres; SQLite through the same SQLAlchemy models for single-node self-host):**
```
tenants(id, name, kms_key_id, created_at)
users(id, tenant_id, email, created_at)
projects(id ULID, tenant_id, name, goal, cartridge, cartridge_version, created_by, created_at, updated_at, deleted_at, schema_version)
datasets(id ULID, project_id, name, bytes, sha256, content_type, object_key, created_by, created_at, deleted_at)
sessions(id ULID, project_id, tenant_id, owner_id, title, created_at, updated_at, version, deleted_at)
session_events(session_id, seq bigint, kind, payload jsonb, created_at, PK(session_id, seq))   -- append-only; transcript = fold(events)
runs(id ULID, project_id, session_id NULL, workflow, cartridge, cartridge_version, status, inputs jsonb, plan jsonb, gate_wait, cost_usd, created_at, started_at, finished_at, version, schema_version, deleted_at)
run_steps(run_id, step_id, status, gate jsonb, usage jsonb, model, cost_usd, error, started_at, finished_at, PK(run_id, step_id))
run_events(run_id, seq bigint, name, payload jsonb, ts, PK(run_id, seq))   -- replaces events.jsonl as truth
charts(run_id, chart_id, version, kind, title, spec jsonb, data_ref, step, persona, ts, PK(run_id, chart_id, version))
artifacts(id, tenant_id, run_id NULL, session_id NULL, path, kind, object_key, bytes, sha256, version, created_at, deleted_at)  -- gate-versions = artifacts with version>1
langgraph checkpoint tables (PostgresSaver) + thread_owners(thread_id, tenant_id, run_id|session_id)
audit_log(tenant_id, actor, action, subject_type, subject_id, ts, detail jsonb)   -- includes export/download
deletion_jobs(id, tenant_id, subject_type, subject_id, requested_by, requested_at, completed_at, manifest jsonb)
```

**Object storage (S3-compatible; directory tree for self-host), per-tenant prefix and per-tenant KMS key, datasets content-addressed:**
```
tenants/{t}/projects/{p}/datasets/{dataset_id}/{sha256}                 # immutable; runs reference by id, never copy
tenants/{t}/projects/{p}/runs/{run}/workspace/{path}                    # synced at each `produces` verification + terminal state
tenants/{t}/projects/{p}/runs/{run}/gate-versions/{step}/v{n}/{path}
tenants/{t}/projects/{p}/runs/{run}/logs/runner.log
tenants/{t}/projects/{p}/sessions/{session}/artifacts/{path}            # per session, not per project
```
**Stays on files (ephemeral, reproducible):** the live run workspace on the worker (bind-mounted into the container), materialized skills, kernel scratch, `large_tool_results` as a per-turn temp dir deleted at turn end. Rule: nothing that is the only copy of customer data or run truth lives on a worker disk or inside a model-executable root.

**Schema versioning + migration:** `schema_version` int on every JSON document and Alembic for tables; pydantic models with `extra="ignore"` plus one `upgrade_vN_to_vN+1` per bump in a single module; writers always emit current version; server refuses to start on a NEWER version than it knows (no silent downgrade); CI fixture dir holding one document per historical version, all must load. Ship `dsagent migrate` (idempotent, `--dry-run` report) whose migration 0001 imports today's `.dsagent/` layout — including the adopted `default`/`serve` workspace — so current data is the first migration, not a hand copy.

**Write/concurrency semantics:**
- Sessions: append-only. Client `POST /sessions/{id}/events {after_seq, events[]}`; server assigns `seq`, bumps `version`; `GET ?after=seq`. Stale tab → 409 with the missing events, client merges. Phase 0 fallback: `If-Match: <version>` on the snapshot PUT, 412 → refetch.
- Runs: single owner (the runner worker) writes `runs`/`run_steps`/`run_events` in one transaction per step boundary; pre-start HTTP mutations use `UPDATE … WHERE version=?`; post-start mutations (stop, gate answer, amend) go through a `run_commands` mailbox the runner consumes (the `stop` file generalized).
- Datasets: immutable by content hash; re-upload = new dataset id, runs pin the id; delete = tombstone + purge job.

**Retention defaults (configurable per tenant):** datasets until deleted, 30-day soft-delete; run metadata/events indefinite, workspace + gate-versions 180 days then cold tier, purge on request; run checkpoint threads deleted at terminal + 7 days; chat checkpoints at idle 24 h; `large_tool_results` end of turn; orphan untitled sessions 7 days; audit log ≥ 1 year; erasure jobs complete ≤ 24 h with a receipt (promise 30 days, deliver 1).

**Guarantees you could honestly promise after Phase 1:** metadata + transcripts RPO ≤ 5 min (managed Postgres PITR), RTO ≤ 4 h; datasets + artifacts RPO ≤ 1 h (versioned bucket, cross-region replication), RTO ≤ 4 h; a run interrupted mid-step resumes from the last verified step boundary — the in-flight step's work is lost and re-run, say so; a gate waiting at crash survives (Postgres checkpointer); deletion ≤ 24 h with receipt. **Today:** RPO unbounded (no backup, CWD-relative store), RTO undefined, a chat can be lost by a second tab or a network blip, no deletion request can be fully honoured.

## 8. Top 5 blockers
1. Chat loss: client wipes on any non-404 + non-atomic save returning 200 (`ChatCenter.tsx:227-240`, `sessions.py:79-83`, `api_projects.py:116`).
2. Last-write-wins transcripts, no version (`ChatCenter.tsx:297-350`, `sessions.py:68`).
3. Erasure impossible: no project delete, undeletable runs (`api.py:246-268`), dataset/transcript content in ≥ 8 places nothing cleans incl. HTTP-served `large_tool_results` and never-pruned checkpoints.
4. Non-atomic, unverified dataset storage/copying → silent wrong analysis (`datasets.py:28-32,53-66`, `serve.py:271`).
5. Store location/isolation: CWD-relative root, checkpoint DB inside the model-executable default workspace, no backup/restore (`cli.py:43,286`, `serve.py:178,393`, `kernel.py:87`).

## 9. Phased path
**Phase 0 — demo-safe (1–2 weeks, files stay):** atomic writes (temp + fsync + rename) for sessions/project/datasets; client registers only on a real 404 and never PUTs `[]` over an error; `save` raises → 500; server `version` + `If-Match`/412 on session PUT; `schema_version` + tolerant loaders on run.json/session/project; `DSAGENT_HOME` absolute root; checkpoints outside any workspace and the default chat workspace ≠ serve workspace; deepagents offload dir per turn + deny the prefix in file routes; prune checkpoint thread at terminal; ULID run ids; `DELETE /projects/{id}` cascade and deletable finished runs (rename to trash + purge); dataset delete purges seeded copies; dataset manifest with sha256; seed only on manifest change; drop `plan.profile.preview` rows; lock around `RunState.save`; orphan-session sweep; migration step removing `projects/sessions.json` and importing the legacy dirs.

**Phase 1 — first paying customer (6–10 weeks):** Postgres metadata per §7 (SQLite behind same models for self-host); object storage with content-addressed datasets and per-session artifact prefixes; `PostgresSaver`/`AsyncPostgresSaver` for driver and chat bridge, agents built in lifespan; append-only `session_events` in your own transcript schema (strip inlined `data.values` to artifact refs); `run_events` in DB with offset-based SSE; single-owner run writes + command mailbox; erasure job with receipt + audit log; `dsagent export/import project`; backups with a rehearsed restore; retention sweeps; encryption at rest with per-tenant keys; the RPO/RTO statement in the contract.

**Phase 2 — scale:** workspace materialization by reflink/CSI volume instead of copies; event fan-out via DB notify or a stream instead of file polling; checkpoint compaction; cold-tier archiving; per-tenant storage quotas and cost attribution; data-residency regions; legal hold; automated periodic restore drill that fails CI when RTO is missed.
