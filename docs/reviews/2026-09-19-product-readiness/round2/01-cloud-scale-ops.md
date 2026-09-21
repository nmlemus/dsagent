# Round 2 — cloud-scale-ops (Fable 5.1) — verbatim

Lens: 12-factor / cloud-native readiness, statefulness inventory, horizontal-scaling test,
target topology, day-2 ops, per-tenant cost model. Reviewed 2026-09-19 at commit `8ac2758`.
No files modified.

---

Cloud readiness verdict: dsagent-v2 cannot run as a multi-replica hosted service today; it is a well-structured single-node appliance. Self-host as a "one VM per customer" appliance is reachable in weeks; multi-tenant SaaS needs a control-plane/worker/sandbox split (months). Details, with evidence, below. I modified no files.

## 0. What is already right (keep it)

- The run directory as record of truth (`run.json` + `events.jsonl` + `workspace/`) is a durable, restart-safe job log. The SSE endpoint *tails a file*, not memory (`api.py:419-465`), so it is replica-agnostic as long as the file is reachable. This is the seed of a worker/storage split.
- Chat transcript is owned by the client+SessionStore, and the AG-UI request carries the full history; the in-process checkpoint is explicitly disposable (`serve.py:231-251`). That is *almost* a stateless chat tier.
- Per-step telemetry from `usage_metadata` and `prices.yaml` (`runner.py:1034-1055`, `pricing.py`) is provider-agnostic and attributable to run → project. Right primitive for a billing meter.
- `Driver` is a protocol (`driver.py:46-54`); `GraphDriver` and `Replay` are the two implementations. A queue-backed driver can slot in without touching the API.
- Docker env builds argv as values with an injected runner (`envs/docker.py`); swapping "docker on this host" for a remote sandbox API is a contained change.

## 1. 12-factor audit

| Factor | State | Evidence |
|---|---|---|
| Config | Relative paths baked in: `RUNS_DIR = Path(".dsagent/runs")` `cli.py:43`; `--workspace` default `.dsagent/serve` `cli.py:286`; projects root derived as `runs_dir.parent / "projects"` `serve.py:178`; `prices.yaml` searched upward from cwd `pricing.py:123`; UI dist derived from `__file__` `serve.py:338-343` (breaks in a wheel install). Only env vars: `DSAGENT_MODEL`, `DSAGENT_UI_DIST`. No `DSAGENT_DATA_DIR`, no `PORT`, no settings object. | REQUIRED, both, S |
| Secrets | Provider API key via LangChain env var; then leaked into the sandbox: `LocalShellBackend(inherit_env=True)` `envs/kernel.py:87` gives every model-run shell the server's environment. | BLOCKER for SaaS (security agent's lane; noted for the sandbox-tier design) |
| Stateless processes | No. See §2. Live kernels, run threads, InMemorySaver graphs, driver thread table all in-process. | BLOCKER |
| Backing services | None: filesystem for everything, SQLite for checkpoints. No DB, no queue, no object store, no cache. | BLOCKER |
| Build/release/run | No Dockerfile for the harness, no compose, no CI dir, no Helm. Cartridge images built at run time on the serving host (`ensure_image` `envs/docker.py:243-256`, invoked from `make_docker_env` inside a run thread). | REQUIRED, both, M |
| Port binding | Fine (`--host/--port`). | — |
| Concurrency | Threads inside the API process (`driver.py:310-316`, daemon threads) plus sync tools in LangChain's executor. No worker model, no backpressure, no limit on concurrent runs. | BLOCKER |
| Disposability | Broken. Lifespan closes only chat envs (`serve.py:158-169`); default env never closed (`cli.py:339` returns from `uvicorn.run` with no cleanup); per-run envs live in `WorkflowRunner._envs` on daemon threads and are never closed on SIGTERM → orphan ipykernels and `sleep infinity` containers holding GPUs (`envs/docker.py:120`). Uvicorn is started with no `timeout_graceful_shutdown`, and SSE loops never end on their own → a k8s rollout ends in SIGKILL. No auto-resume after restart: a run reads `running` forever with `live:false` until a human clicks Start (`api.py:738-747`). | BLOCKER, both, M |
| Dev/prod parity | Good in one sense (Replay driver, FakeAgent) but the kernel env means "prod" = the API server's own Python. | — |
| Logs | No `logging` module anywhere in `src/dsagent` (grep is empty). Runner narration goes to `runner.log` per run; the driver's `log` is a no-op lambda under serve (`driver.py:133`), and `emit()` swallows consumer exceptions into that no-op (`runner.py:372-373`). No request ids, no run/tenant correlation on stdout. | REQUIRED, both, S |
| Admin/migrations | `run.json` has no schema version; compat is ad-hoc reads (`runs.py:121-127`). The "default project adopts the legacy runs dir" is a live migration hack (`projects.py:67-72`, `locate_run` scans every project dir `projects.py:77-93`). | REQUIRED, both, M |
| Health | `/agent/health` returns a constant (`serve.py:298-300`). No readiness (model creds, disk, docker daemon, kernel start). | REQUIRED, both, S |

## 2. Statefulness inventory

| State | Where | Class | Target |
|---|---|---|---|
| Per-project chat graph + `InMemorySaver` (holds the live turn's interrupt) | `serve.py:254-267` `chat_agents` dict | must move | Postgres checkpointer (`langgraph-checkpoint-postgres`) keyed by thread; or keep ephemeral *and* make chat gates go through the run gate path |
| Per-project live kernel (variables, DataFrames) | `serve.py:255,261`, `envs/kernel.py:98-101` | must become a managed sandbox | sandbox tier with lease + idle TTL; state is per-project-session, not per-replica |
| Default serve env (kernel) + serve workspace | `cli.py:322`, `serve.py:306` | same | same |
| Driver thread table, error table, start lock | `driver.py:145-147` | must become a job | queue + worker; `is_running` becomes a heartbeat/lease in the DB |
| Run checkpoints (interrupt state for launcher runs) | `checkpoints.sqlite` `serve.py:364-405` (`check_same_thread=False`) | must move | Postgres checkpointer; one SQLite file cannot be shared by two replicas (NFS SQLite corrupts) |
| `run.json`, `events.jsonl`, `runner.log`, `gate-versions/`, `stop` file | run dir | split | metadata+events → Postgres (events table with monotonic index = today's line number); artifacts → object storage; stop flag → DB field/queue message |
| Run workspace (bind-mounted into containers) | `<run>/workspace` | split | worker-local scratch (block volume) during the step; synced to object storage at step end (that is when `produces` is verified anyway) |
| Project datasets, copied into every run and every chat turn | `datasets.py:53-66`, `api.py:243`, `tools.py:161`, `serve.py:271` | must move | object storage; mount/stream into sandbox; stop copying per run (storage is O(runs × datasets)) |
| Sessions (full transcripts, one JSON each; `list_for` parses all, `project_of` parses full transcript per `/agent` call) | `sessions.py:56-104`, `serve.py:288` | must move | Postgres `sessions` + `messages`; last-writer-wins across tabs today (`sessions.py:68-83`) |
| Projects metadata | `projects.py` | must move | Postgres, with `tenant_id` |
| Materialized skills per workspace | `host/build.py:41-73` (`copytree` on every `build_orchestrator`) | fine ephemeral | bake into the sandbox image or a read-only volume |
| `prices.yaml` | repo file | must move | DB table per tenant/plan; today one global rate card |
| Cartridge images | docker daemon on the API host | must move | registry; build in CI or a build job, never in the request/run path |
| UI static dist | `serve.py:347-361` | fine | CDN/ingress |

## 3. Horizontal scaling test (2 replicas behind an LB, today)

1. **Chat turn on B while kernel/checkpoint live on A.** `agent_endpoint` builds a fresh graph on B from the client's full message list — the transcript survives (by design). The *kernel* does not: B starts a fresh kernel, "the DataFrame from last turn" is gone, and files the persona wrote on A are invisible on B unless the disk is shared. A chat gate (`ask_human` interrupt) mid-turn: the resume must land on A; on B `InMemorySaver` has no thread → the turn fails. Result: works only with sticky sessions + shared disk, and even then per-project state is per-replica.
2. **SSE run stream.** File tail; with shared storage it works cross-replica. Without it, B streams nothing; `is_live` (run.json says `running`) keeps the stream open forever and `driver.is_running` on B is always False (`api.py:447`).
3. **Gate answer POSTed to B while the graph thread ran on A.** `_awaiting` reads run.json (OK with shared disk) → `_interrupted` calls `graph.get_state` against B's SQLite (`driver.py:299-308`) → no thread → 409. With the SQLite file on shared NFS it "works" by accident: B finds the interrupt and spawns the resume thread on B — the run migrates replicas silently, on a database that must not live on NFS. Chat-started runs: always 409 regardless (`api.py:14-21`).
4. **Run started on A, A redeployed.** Daemon thread dies mid-step; per-run kernel/container orphaned (never `close()`d); run.json stays `running`; `live:false` after restart; nobody resumes it. A human clicks Start → `resume=True` → the step restarts from scratch, because a persona graph is `checkpointer=False` (`host/build.py:172`): a 4h fit that is 3h in is re-paid in full. Also: `Step.timeout` (`cartridge/models.py:60`, `workflow.yaml:44` `timeout: 4h`) is declared and never enforced anywhere in `runner.py`.
5. **Bonus: chat-started runs are bound to an HTTP connection.** `run_workflow` is a sync tool executed inside the AG-UI `astream_events` request (`dispatch.py:28-35`). Browser closes the tab → Starlette cancels the generator → the async graph task is cancelled; the executor thread keeps running the workflow (cannot be cancelled) but its result and any interrupt land in a graph nobody is streaming. This is the "must approve from the chat that started it" caveat in `api.py:14-21`, and in the cloud it means a whole workflow's fate depends on a browser tab and an LB idle timeout.

## 4. Findings, ranked

BLOCKERS
- B1 `driver.py:123-316` — runs execute on daemon threads inside the API process; no queue, no lease, no retry, no reaper, no cancellation, no concurrency cap. Path: both. Change: a `runs` table (status, lease_owner, lease_expiry, attempt) + queue (Postgres `SKIP LOCKED` is enough for Phase 1; a real broker later) + a `dsagent worker` process that claims a run, heartbeats, drives the graph, and releases. `GraphDriver` becomes `QueueDriver` behind the same `Driver` protocol. Effort: L.
- B2 `serve.py:254-272`, `envs/kernel.py` — execution sandbox is the API process (kernel env: `LocalShellBackend` in the server's own Python, inheriting its env). Path: both (SaaS: security + isolation; self-host: still the reason the API can't scale). Change: an execution tier: one sandbox per (tenant, project session) for chat and one per run for workflows, with a lease, an idle TTL, a warm pool, and a network policy. Implement `SandboxBackendProtocol` against it (`kind: k8s-job`/`kind: modal`), which `architecture.md §3.4` already anticipates. Effort: XL.
- B3 `serve.py:364-405`, `serve.py:265` — checkpoint state is SQLite-on-local-disk (runs) and process memory (chat); the gate cannot be answered on another replica and chat gates die with the process. Change: Postgres checkpointer for both graphs (async saver inside lifespan, as `api.py:21` itself suggests); chat gates should additionally be recorded like run gates so answering is an API action, not a stream action. Effort: M.
- B4 disposability — no graceful drain, envs leaked on shutdown, no auto-resume (`serve.py:158-169`, `cli.py:339`, `runner.py:516-519` only reached via `finally` on the thread). Change: SIGTERM handler that (a) stops claiming new runs, (b) sets in-flight runs to `interrupted` with the step to redo, (c) closes envs and removes containers, (d) exits before the grace period; plus a reaper that re-queues runs whose lease expired and marks their sandbox for deletion. Effort: M.
- B5 no tenant anywhere — no `tenant_id` on project/session/run/cost; `locate_run` scans all projects by run id (`projects.py:77-93`); run ids are global and guessable (`api.py:143-157`, `workflow-YYYYMMDD-HHMMSS`); every route resolves a run by bare id (`api.py:165-176`). Path: SaaS. Change: tenant in the auth principal, `tenant_id` column on every table, every query scoped, ids opaque. Effort: L (touches every route).

REQUIRED
- R1 `datasets.py:53-66` copy-per-run and per-turn (`serve.py:271`, on the event loop; `api.py:243`; `tools.py:161`). Dataset storage is O(runs × datasets) and a large upload blocks the loop each turn. Change: object storage + per-sandbox mount/stream, seeded once per sandbox. Both. M.
- R2 step-level resumability only (`host/build.py:172`, `runner.py:557-567`) and `Step.timeout` unenforced. A 4h GPU step that dies is re-paid in full and can run forever. Change: enforce `timeout` in the worker (kill sandbox, mark step failed, re-queue by policy); for the fit step, make the *cartridge* checkpoint its own progress into the workspace (Meridian trace files) and instruct the persona to resume from them — the harness cannot checkpoint a model fit, but it must guarantee the workspace survives (object storage sync at step end + periodic sync during long steps). Both. M.
- R3 GPU scheduling: `--gpus all` (`envs/docker.py:119`) on whatever host runs the API; no queue, no per-run GPU allocation, image built in the run path (`envs/docker.py:243-256`). Change: GPU worker pool (k8s node pool with `nvidia.com/gpu: 1` requests, or a managed GPU-job service); images built in CI and pulled from a registry; the worker for a docker step *is* a k8s Job with the workspace volume. Both. L.
- R4 observability: no structured logs, no metrics, no traces, no correlation ids (`driver.py:133`, `runner.py:372`). Change: `logging`/structlog JSON to stdout with `tenant_id, project_id, run_id, step, thread_id` bound; OpenTelemetry (LangChain has callbacks; LangSmith/Langfuse can be enabled today with env vars for the model side); Prometheus counters for runs by status, gate wait, step duration, tokens by model, sandbox count. Both. M.
- R5 billing meter incomplete: only workflow steps are priced (`runner.py:725-735`); chat turns and `task` subagent calls are unmetered (no `usage_metadata` read in `serve.py`/`host/build.py`); compute-hours and GPU-hours are not metered at all; `prices.yaml` is global; `_estimate` reads the default runs dir only (`api.py:703-720`). Change: a `usage_events` table written from a LangChain callback on *every* model call (chat and runs) and from the sandbox lease (cpu-seconds, gpu-seconds, storage-bytes), keyed by tenant/project/run/step. SaaS blocker, self-host required. M.
- R6 API/ingress hygiene: SSE loops poll disk every 250 ms per client (`api.py:64`), UI polls `/projects/{p}/runs` every 4 s from two components, artifacts `rglob` every 5 s, run detail every 3 s plus on every event (`Sidebar.tsx:50-56`, `ChatCenter.tsx:145-190`, `RunDock.tsx:25-33`); `/projects` counts run dirs of every project per call (`api_projects.py:23`); `list_runs` opens every run.json and marker file per call. No ETag, no caching. Change: events from DB with `LISTEN/NOTIFY` or Redis pub/sub; a single `/projects/{p}/stream` for list-level changes; artifact index maintained by the worker at step end, not recomputed by rglob. Both. M.
- R7 config/paths (§1). Both. S.
- R8 Session store: whole-transcript PUT per save, last-writer-wins, `list_for` parses every file. Both. S–M (falls out of Postgres move).
- R9 `tools.py:180-207` and `charts.py:411-440` resolve `runs_dir / run_id` without the store — chat cannot read or amend a run in a non-default project. Correctness today, tenancy bug tomorrow. S.

IMPORTANT
- I1 No retention/GC: runs immutable after start (`api.py:246-268` deletes only pending), gate-versions copies, checkpoints grow (1.4 MB + 3.9 MB WAL locally already). Change: retention policy per tenant/plan, lifecycle rules on the bucket, checkpoint TTL. M.
- I2 Backups/DR: nothing. Postgres PITR + versioned bucket covers it once state moves. S after B3/R1.
- I3 Migrations: add `schema_version` to `run.json`/DB rows; Alembic once Postgres exists. S.
- I4 Health/readiness: `/healthz` (process up) vs `/readyz` (DB reachable, checkpointer reachable, model creds present, worker heartbeat recent). S.
- I5 `RECURSION_LIMIT` and `PARK_SECONDS` thread joins (`driver.py:226`) hold request handlers for up to 5 s — fine on one box, a tail-latency source under load. Disappears with the queue.

NICE
- N1 Cartridge validation and command-skill generation at startup are fine; move image builds out.
- N2 Replay driver is a great fixture for load testing the API tier without a model — keep it.

## 5. Target topology

```
                 ┌──────────── CDN / ingress (TLS, auth, WAF) ────────────┐
                 │  sticky not required                                    │
   Browser ──────┤                                                          │
   (AG-UI SSE,   │   ┌──────────────┐   ┌──────────────┐                   │
    run SSE)     └──▶│ API replica  │ … │ API replica  │  stateless       │
                     │ FastAPI      │   │ FastAPI      │  (chat graph,    │
                     │ chat graph   │   │              │   routes, SSE)   │
                     └──┬───┬───┬───┘   └──────────────┘                   │
                        │   │   │
        Postgres ◀──────┘   │   └──────▶ Redis / PG LISTEN (run + list events, pub/sub)
   (tenants, projects,      │
    sessions, runs, steps,  │  claims (SKIP LOCKED) / heartbeats
    events, usage_events,   ▼
    LangGraph checkpoints)  ┌─────────────────────┐    ┌─────────────────────┐
                            │ worker pool (CPU)   │    │ worker pool (GPU)   │
                            │ dsagent worker      │    │ k8s Jobs per step   │
                            │ drives run graph,   │    │ or managed GPU jobs │
                            │ enforces timeouts   │    │ (nvidia.com/gpu:1)  │
                            └─────────┬───────────┘    └──────────┬──────────┘
                                      │ SandboxBackendProtocol      │
                                      ▼                             ▼
                            ┌───────────────────────────────────────────────┐
                            │ sandbox tier: per-tenant namespace, one pod   │
                            │ per (project-session | run); warm pool;       │
                            │ idle TTL; no egress except model API; gVisor │
                            │ or Firecracker for SaaS; kernel env only for  │
                            │ self-host trusted mode                        │
                            └───────────────────────────────────────────────┘
                                      │
                                      ▼
                            Object storage (datasets, run workspaces synced at
                            step end, artifacts, exports, gate-versions) with
                            lifecycle rules; served via signed URLs, never through
                            the API tier's disk
```

Streaming: the run stream stays SSE but reads from the events table + pub/sub, so any replica serves any run; the AG-UI chat stream remains per-request (it is a request, not a session). UI: replace the four polling loops with one project-level SSE channel; keep polling only as fallback.

## 6. Ordered migration plan (what to keep vs change)

0. Config object + data dir env var + Dockerfile for the harness + structured logging + `/healthz`,`/readyz` + SIGTERM handler that closes envs and marks runs `interrupted`. (Keeps everything; makes the appliance deployable.) 1–2 weeks.
1. Postgres for metadata: projects, sessions, runs/steps, events (index = today's line number), usage_events; `RunState.save` → `RunStore` interface with the filesystem implementation kept for CLI/tests. Postgres checkpointer for both graphs. Kills B3, R8, I2/I3; SSE and gate answers become replica-agnostic. 3–4 weeks.
2. Worker tier: `QueueDriver` implementing `Driver`; `dsagent worker` claims runs with lease + heartbeat; reaper re-queues expired leases; enforce `Step.timeout`; stop flag → DB. Kills B1, B4, half of R2. 3–4 weeks.
3. Object storage for datasets and workspaces: seed once per sandbox, sync at step end, signed URLs for files/exports; artifact index written by the worker. Kills R1, R6 rglob, I1. 2–3 weeks.
4. Sandbox tier: `kind: k8s` (or a managed sandbox provider) backend; chat sandbox per project-session with idle TTL and warm pool; workflow sandbox per run; GPU steps as Jobs on a GPU pool; images from a registry. Kills B2, R3. 6–10 weeks.
5. Tenancy + billing: auth principal → `tenant_id` everywhere, opaque ids, per-tenant rate card, usage events from a LangChain callback on every model call (chat included) and from sandbox leases. Kills B5, R5. 4–6 weeks.
6. UI push channel + retention/quotas + SLO dashboards. 2–3 weeks.

Keep: cartridge loader, runner DAG/gate semantics, `run.json`/events schemas (become rows), Replay/FakeAgent test harness, chart validation, export, the AG-UI bridge.

## 7. Summary

(a) Top 5 blockers: B1 in-process run threads (no jobs/leases/retries); B2 execution sandbox = the API process (kernel env, inherit_env); B3 local SQLite/in-memory checkpoints (gates not answerable across replicas or restarts for chat); B4 no disposability (envs and GPU containers leak on shutdown, no auto-resume, unenforced 4h timeout); B5 no tenant model or per-tenant metering.

(b) Phased path:
- Phase 0 (demo / design partner, 2–4 weeks): single container per customer, persistent volume, steps 0 + the SIGTERM/reaper piece of 2; run only kernel + docker-on-the-same-VM; one customer per VM. Sell as "self-host appliance", not SaaS.
- Phase 1 (first paying customer, ~3 months): steps 1–3 and the timeout/resumability work; still one deployment per customer (tenant isolation by deployment, which sidesteps B5 and most of B2); GPU via a dedicated GPU VM per customer with the docker backend; usage events for LLM + GPU-hours so the invoice is real.
- Phase 2 (multi-tenant scale, 6–9 months cumulative): steps 4–6; sandbox tier with strong isolation, shared GPU pool with queueing and quotas, per-tenant rate cards, SLOs (chat p95 first-token, gate-answer latency, run completion ratio, orphaned-sandbox count = 0).

(c) Infra cost shape per tenant (formula; do not read any number below as a price — take rates from your cloud's rate card and `prices.yaml`):
- LLM: Σ over steps and chat turns of (uncached_in × r_in + cached × r_cached + cache_write × r_cw + out × r_out). Sourced datum: one `eda-to-report` run = 560 k input / 23 k output tokens ≈ $0.47 on the default model (`docs/runs/eda-to-report-003.md`, `prices.yaml`). Turn count inside `run_python`-heavy steps is the driver (each call re-sends the transcript), so cost scales with step transcript length, not dataset size.
- Compute: (sandbox-hours × r_cpu_vm) with a warm-pool floor per active tenant (1 idle sandbox = a constant, not zero) + worker-hours shared across tenants.
- GPU: fit_hours × r_gpu per MMM run; the 4 h timeout is the ceiling per run; this dominates any tenant that runs MMM, and it is a per-run line item to bill through, not a fixed cost — which is why R5 (metering GPU-seconds from the lease) is required before Phase 1.
- Storage: datasets_bytes × (1 + runs_per_dataset) today (per-run copies) → datasets_bytes + Σ artifacts after step 3; plus Postgres (small) and events.
- Fixed platform: Postgres + Redis + ingress + control-plane nodes, amortized over tenants.

Files referenced: `src/dsagent/{serve.py,driver.py,api.py,api_projects.py,sessions.py,projects.py,datasets.py,pricing.py,cli.py,runs.py,runner/runner.py,runner/tools.py,runner/dispatch.py,runner/charts.py,envs/kernel.py,envs/docker.py,host/build.py,cartridge/models.py}`, `ui/src/{lib/api.ts,components/ChatCenter.tsx,components/Sidebar.tsx,components/RunDock.tsx}`, `prices.yaml`, `cartridges/ds/{cartridge.yaml,workflows/mmm-meridian/workflow.yaml}`.
