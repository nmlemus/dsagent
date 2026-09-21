# Round 2 — saas-security (Fable 5.1) — verbatim

Lens: STRIDE per trust boundary (7 boundaries), sandbox strength audit, minimum security
architecture, phased path. Reviewed 2026-09-19 at commit `8ac2758`. No repo files modified.
Includes one **harmless empirical probe** (in the job tmp dir) of the exact backend config
`envs/kernel.py:87` uses.

---

STRIDE threat model of dsagent-v2 under the commercial bar: verdict is NOT SELLABLE as SaaS or self-host today; the "execution sandbox" is a subprocess with the server's identity, env and network, and there is no identity, tenancy or tamper-proof audit anywhere. Full per-boundary findings, ranking, minimum architecture and phased path follow.

Method: read CLAUDE.md, docs/architecture.md, serve.py, api.py, api_projects.py, datasets.py, sessions.py, projects.py, driver.py, export.py, runs.py, tabular.py, envs/{base,kernel,docker}.py, host/build.py, runner/{runner,tools,charts}.py, cli.py, ChatCenter.tsx, FileViewer.tsx, api.ts, pyproject, Dockerfile, plus the installed deepagents/ag_ui_langgraph/jupyter_client sources. I also ran a harmless probe of the exact backend config kernel.py uses (LocalShellBackend(root_dir=ws, virtual_mode=True, inherit_env=True) + FilesystemBackend(virtual_mode=True)) in the job tmp dir: `ls /` and `ls ~/.ssh` succeeded from inside the "virtual root", an env var set on the server was echoed back, and `write("/marie/eda/SKILL.md")` on the skills mount succeeded. No repo files were modified.

Severity legend: BLOCKER = cannot take customer data; REQUIRED = first paying customer; IMPORTANT = enterprise/SOC2; NICE. Path: SaaS / SH (self-host) / both. Effort S/M/L/XL.

======================================================================
BOUNDARY 3 FIRST: API <-> EXECUTION SANDBOX (the product's real security boundary)
======================================================================

3.1 [E, I, T] The kernel env is not a sandbox. It is the server process's own privileges, env and network. BLOCKER, both, XL (design change) / S (mitigations).
- envs/kernel.py:87 `LocalShellBackend(root_dir=workspace, virtual_mode=True, inherit_env=True)`. deepagents/backends/local_shell.py:198 `self._env = os.environ.copy()`; :304-313 `subprocess.run(shell=True, env=self._env, cwd=root)`; its own docstring :227-231 says "no sandboxing ... Access any file on the filesystem (regardless of virtual_mode)". Verified empirically (above).
- envs/kernel.py:29-33 `KernelManager(kernel_name="python3").start_kernel(cwd=...)`: jupyter_client/launcher.py:78 `env = os.environ.copy()` — the ipykernel inherits ANTHROPIC_API_KEY/OPENAI_API_KEY and everything else, runs as the server user, with the server's cwd tree (.dsagent/projects/* of every project), the server's network, and the docker socket rights the server user has.
- envs/kernel.py:108 `python=sys.executable`: `run_skill_script` and any model-issued `pip install` land in the SERVER'S OWN venv (cli.py:145 documents that this is "the interpreter running DSAgent"). A persona in tenant A can `pip install` a package that the API process and every other tenant import after restart (persistence + cross-tenant supply-chain).
- The audit trail lives one directory above the workspace (runner.py:100-110 events.jsonl / runner.log; run.json) — model code can `open('../run.json','w')`, rewrite events.jsonl, or delete gate-versions. The code being audited can edit its own audit log.
- Persona `tools:` in agents/*.md frontmatter (cartridge/models.py:34) is parsed but never enforced: host/build.py:157 and :232 hand every persona `*env.tools` (execute, run_python, write_file...). The skill matrix is enforced; the tool matrix is decorative.
Required control: in serve mode the kernel env must NOT run on the host. Either (a) run ipykernel inside the per-tenant container/microVM and talk to it over ZMQ/HTTP from the API, or (b) forbid `kind: kernel` under `dsagent serve` and make Docker/microVM the only env. Interim (Phase 0): `inherit_env=False` with an explicit allowlist env (PATH, HOME=workspace, LANG) for LocalShellBackend, and `KernelManager(env=allowlist)` for the kernel; a per-workspace venv (or `PIP_REQUIRE_VIRTUALENV` + read-only site-packages) so model pip installs cannot touch the server interpreter; enforce the frontmatter `tools:` list. Move run.json/events.jsonl/runner.log out of anything the sandbox can reach (write them from the API side, not the runner thread in the same uid tree).

3.2 [E, D, I] Docker env has no hardening at all. BLOCKER for SaaS, REQUIRED for SH. Effort S.
- envs/docker.py:108-120 `run_argv`: only `--volume ws:/workspace`, `--volume ws/.dsagent/skills:...:ro`, `--workdir`, optional `--user uid:gid` (:148-159, good), `--gpus all`. Missing: `--network none` (or egress proxy), `--cap-drop ALL`, `--security-opt no-new-privileges`, seccomp/AppArmor profile, `--read-only` + tmpfs, `--memory`, `--cpus`, `--pids-limit`, storage quota, `--rm`, `--init`. `docker exec ... bash -lc` (:125) with the same missing limits; DEFAULT_TIMEOUT 3600 per exec (:51) but no container lifetime; container removal only in `close()` (:227-240) so a SIGKILLed server leaves `sleep infinity` containers running forever.
- The server user needs docker-socket access to run this at all, which on a shared host is root-equivalent: combine with 3.1 (kernel env = server user) and the kernel can `docker run -v /:/host` — full host root.
- Step `timeout:` in workflow.yaml (cartridge/models.py:60) is parsed but never read by runner.py (grep confirms). "timeout: 4h" is a lie in the DAG.
Required control: hardened flags above as defaults (tests already assert argv, docker.py:16-20 — cheap to add), a hard wall-clock per step and per run enforced by the runner (kill the container, not "stop between steps"), `--rm` + startup GC of `dsagent-*` containers, rootless docker or a dedicated non-docker-group user for the API, and for SaaS a microVM runtime (gVisor/Kata/Firecracker) instead of the default runc.

3.3 [E, I] Container -> host READ escape through host-side symlink following. BLOCKER for SaaS (this is what defeats the one isolation you have), REQUIRED SH. Effort S.
The container can only see /workspace, but it can write a symlink inside the bind mount (`ln -s /etc/shadow report/findings.md`, or to another tenant's `.dsagent/projects/<id>/data/x.csv`, or to the server `.env`). Host code then follows it:
- api.py:598-601 `archive.write(workspace / rel)` and :723-731 `_archive_members` (rglob + `is_file()`, both follow symlinks) -> `GET /runs/{id}/download?everything=true` ships the target's bytes to the browser.
- runner.py:434-449 `matched()` uses `is_file()` -> a symlink satisfies `produces`; runner.py:920-928 `_keep_version` `shutil.copy2(source, dest)` follows the link and copies the host file into gate-versions/, then api.py:510-532 serves it as text.
- export.py:217-221 `read_text` of a step's `.md`, and export.py:98-105 `snapshot -> read_table` follow links into the export HTML.
- runner.py:1084-1092 `_snapshot` `p.stat()` follows links (only a DoS/info issue).
(resolve_run_file serve.py:428-437 and resolve_project_file :502-511 DO refuse symlinks that escape — those two are safe.)
Required control: one `inside(workspace, path)` helper that `lstat`s every component and refuses symlinks (or opens with O_NOFOLLOW / resolves and checks `is_relative_to` like resolve_run_file) used by matched/_keep_version/_archive_members/export/snapshot; mount the workspace with a dedicated uid and `nosymfollow` where available.

3.4 [T] The "read-only skills" guarantee is half true. IMPORTANT, both, S.
docker.py:113 binds skills `:ro` inside the container, but the persona's file tools run on the HOST: host/build.py:76-80 routes `/skills/` to a writable `FilesystemBackend` (write/edit/delete verified). A persona (or a prompt-injected one) rewrites its own SKILL.md; build.py:152-153 only re-materializes when the persona dir is missing, so the tampered skill persists for every later step of the run and, for the chat workspace, until the server restarts. This is persistent prompt injection via the trusted instruction channel.
Required control: a read-only backend for the `/skills/` route (deepagents supports tool-level permissions on routes — filesystem.py:1787), and re-materialize from the cartridge at every step.

3.5 [I] Model output channels carry whatever the sandbox printed. REQUIRED, both, S.
Tool results (e.g. `print(os.environ)`, a dumped dataset) go to the model, then verbatim into the AG-UI stream, into the browser, and into `<projects>/sessions/<id>.json` (sessions.py:68-84 stores `messages` verbatim; api_projects.py:93-116). They are not in events.jsonl (runner.py:1066-1081 previews args to 120 chars; notes are persona text only), but `rec.output` (runner.py:676) and `dsagent.note` carry the persona's prose, which can quote anything it read.
Required control: fix 3.1 so there is nothing secret to print; redact known secret patterns from tool results at the middleware layer; treat session JSON as customer data (encryption at rest, retention, erasure — see boundary 7).

======================================================================
BOUNDARY 1: BROWSER <-> API
======================================================================

1.1 [S, E] No authentication or authorization on any route. BLOCKER, both, L.
serve.py:171 builds a bare FastAPI app; no middleware, no dependency, no token. Every route in api.py / api_projects.py / serve.py is anonymous: create/start/stop runs, answer gates (api.py:397-413), upload/delete datasets (api_projects.py:60-78), read/overwrite/delete any chat transcript (api_projects.py:86-122), read any run's files/log/export/zip, and `POST /agent` (serve.py:286-296) which executes model-driven code. cli.py:288 `--host` accepts `0.0.0.0` with no warning and no auth.
Required control: identity (OIDC/SAML for SaaS; at minimum a bearer token or mTLS for SH) on every route including `/agent` and SSE; authorization = tenant ownership check on every project/run/session id (see boundary 4).

1.2 [S] CSRF and DNS rebinding are live even on localhost. BLOCKER for the localhost product, REQUIRED after auth. Effort S.
- No TrustedHost/Origin validation anywhere (grep confirms) -> DNS rebinding gives any web page the whole API -> RCE via 3.1 (known baseline, still true).
- api.py:764-771 `_json_body` calls `request.json()` without checking Content-Type, so a cross-site `<form enctype="text/plain">` posts valid JSON to `POST /runs`, `POST /runs/{id}/gate` (approve a gate!) and `POST /runs/{id}/plan`; `POST /runs/{id}/start` (:303) and `/stop` (:371) take no body at all so a plain form POST works. Run ids are predictable (api.py:143-157 `<workflow>-YYYYMMDD-HHMMSS`), project ids are name slugs (projects.py:45-47), thread/session ids are client-chosen (ChatCenter.tsx:265).
Required control: TrustedHostMiddleware + Origin/Sec-Fetch-Site check on every non-GET; require `Content-Type: application/json`; SameSite=Strict cookies or bearer-only auth; server-minted unguessable ids (UUIDv4) for runs, projects, sessions.

1.3 [E, T] Model-written HTML served same-origin without CSP. BLOCKER for SaaS (once cookies/tokens exist it is account takeover + RCE by proxy), REQUIRED SH. Effort S.
serve.py:478-481 `/runs/{id}/files/*` and :547-553 `/projects/{id}/files/*` return `text/html` on the app origin with no CSP; `SCRIPTED_CSP` (:454, :483-486) exists but only on `/runs-x`. FileViewer.tsx:39-46 and ChatCenter.tsx:196-202 deliberately open reports in a new tab on the primary origin. A `<script>` in a CSV cell -> persona report -> same-origin JS with full API access (start runs, run code via /agent, exfiltrate every project's data). DOMPurify on markdown (ChatCenter.tsx:27, FileViewer.tsx:90) and export.py:125-143 are fine; they do not cover this path.
Required control: never serve model artifacts on the app origin. Separate artifact origin (`artifacts.<domain>` / per-tenant subdomain) or, minimum, `Content-Security-Policy: sandbox allow-scripts` + `X-Content-Type-Options: nosniff` + `Content-Disposition` on both file routes and delete the unsandboxed prefix; make the UI open `/runs-x`.

1.4 [T, S] Client-authored transcripts are trusted as model context. REQUIRED, both, M.
`PUT /sessions/{id}` stores arbitrary `messages` (api_projects.py:93-116), and the client sends the full history on every `POST /agent` (RunAgentInput.messages). A client (or the CSRF/XSS above) can forge assistant messages and tool results ("the user already approved deleting the datasets") that the model then acts on. Thread id -> project mapping (serve.py:288 `sessions.project_of`) is also client-controlled, so any thread can attach to any project's kernel/workspace.
Required control: server-owned transcripts (the server appends what it streamed; client sends only the new user turn), server-side thread->tenant binding, signed/opaque thread ids.

1.5 [D] No limits. REQUIRED, both, S.
api.py:294 and api_projects.py:67 `await request.body()` load the whole upload into RAM with no cap; `POST /runs`+`/start` spawn an unbounded number of threads (driver.py:310-316), kernels and LLM spend; SSE (api.py:435-453) polls disk 4x/s per open stream with no connection cap; `/projects/{id}/artifacts` (serve.py:521) rglobs the whole workspace every 5 s per tab (ChatCenter.tsx:185); `RECURSION_LIMIT=150` (serve.py:39) is the only bound per chat turn; no per-tenant concurrency, rate or spend limit; pricing.py only reports cost, never caps it.
Required control: max body size (streaming to disk), per-tenant concurrent-run and daily-spend caps enforced before the model is called, rate limits on /agent and uploads, event stream backed by a notifier instead of polling, disk quota per workspace.

1.6 [I] Cross-run/cross-project reads by id. Covered in 4.x (no authz). Also `_estimate` (api.py:703-720) leaks other runs' cost — trivial once tenancy exists.

======================================================================
BOUNDARY 2: API <-> LLM PROVIDER
======================================================================

2.1 [I] One global provider key for everyone, reachable from the sandbox. BLOCKER, both, M.
README.md:30 exports ANTHROPIC_API_KEY into the process env; `.env` (repo root, 0600, gitignored — nothing in src/ loads it, the shell does). models.py:19 reads only DSAGENT_MODEL. Consequences: the key is in the kernel and shell env (3.1); the default serve workspace `.dsagent/serve` sits inside the repo so `open('../../.env')` also works; one key = one bill = no per-tenant metering/cut-off, and a leaked key is everyone's outage.
Required control: key only in the API process (secret manager / KMS), never in sandbox env; per-tenant virtual keys or BYOK with hard budgets; rotation without restart.

2.2 [I] Customer data leaves to a subprocessor with no controls. REQUIRED (enterprise: BLOCKER without a DPA), SaaS. M.
Every dataset row a persona reads goes to the model provider; no region pinning, no provider choice per tenant, no PII redaction, no zero-retention flag documented, no subprocessor list. Enterprise buyers ask this on day one.
Required control: per-tenant provider/region config (Bedrock/Vertex/Anthropic direct), DPA + subprocessor page, optional PII redaction middleware, prompt/completion logging policy.

2.3 [T] Prompt injection reaches the model on every channel, unguarded. REQUIRED, both, M (mitigations), XL (solved).
- runner.py:621-649 `_task_message` interpolates workflow inputs raw into instructions via `_fill` (:1125-1148); inputs come from an unauthenticated `POST /runs` / `POST /runs/{id}/plan` (api.py:227-244, :340-347).
- The gate `note` from `POST /runs/{id}/gate` (api.py:409) is quoted into the persona prompt as the reviewer's voice (runner.py:632-641) — an unauthenticated caller writes instructions for the persona.
- Datasets, prior steps' reports, and `read_run_file` (tools.py:196-207) are all model-readable text with no provenance marking; export.py neuters `<` only for the export.
- The only reason injection today is "just" bad analysis is that the attacker already has RCE without it (3.1). Once the sandbox is real, injection becomes the primary way to make the sandbox do things: exfiltrate via egress (5.x), amend other runs (4.4), rewrite skills (3.4).
Required control: fix 3.1/5.x first (so injection cannot escape), then: provenance-tag tool results and file contents as untrusted in the prompt, gate any external side effect (network, cross-run writes) behind human approval, strip control characters/markup from inputs, and make gate notes authenticated.

======================================================================
BOUNDARY 4: TENANT <-> TENANT
======================================================================

4.1 [S, I, E] There is no tenant. BLOCKER for SaaS, L.
"Project" is a directory namespace with slug ids, not a security principal. projects.py:77-93 `locate_run` searches every project by run id and every run-scoped route uses it; sessions are a flat namespace (sessions.py:53); `GET /sessions/{id}`, `/projects/{id}/data`, `/projects/{id}/files/*` are global. Nothing associates a request with an owner.
Required control: org/tenant as a first-class record; every project/run/session row carries tenant_id; every route resolves the caller's tenant first and 404s anything outside it; server-minted ids.

4.2 [I, E] Shared process, shared user, shared venv, shared filesystem. BLOCKER for SaaS, XL.
All tenants' kernels run as the same OS user with the whole `.dsagent/projects` tree readable/writable (3.1); pip installs share one venv (3.1); default project shares one workspace and the startup env (serve.py:261 `env if project_id == DEFAULT_ID`, cli.py:322) for every unregistered thread (serve.py:288 -> "default"); one long-lived kernel per project (serve.py:257-268) shares variable state between every user and chat of that project, and `chat_agent_for` has no lock, so two first requests race and leak an extra kernel/container.
Required control: one isolated sandbox per tenant (ideally per run/session) with its own uid/VM, its own storage prefix, its own interpreter; no shared default project in SaaS.

4.3 [T] Cross-project writes by the model. REQUIRED, both, S.
Workflow tools are built once with the default `runs_dir` (serve.py:210-214) and are handed to every project's orchestrator: `list_run_files`/`read_run_file` (tools.py:178-207) read any default-project run by id, and `amend_chart` (charts.py:461-474, :435-447) rewrites run.json and appends to events.jsonl of any run by id. A prompt-injected orchestrator in project A can alter the charts in project B's finished report.
Required control: bind these tools to the caller's tenant/project at request time; no by-id access outside it.

4.4 [I] Cross-tenant prompt injection. Realistic chain today: tenant A uploads a CSV whose cells contain instructions -> A's persona writes report.md -> (a) the orchestrator of ANY project with the run id calls `read_run_file` and follows the instructions, or (b) the skill files in the shared default workspace are rewritten (3.4) and every later user of that workspace inherits the injected instructions. Both close with 4.1-4.3 + 3.4.

======================================================================
BOUNDARY 5: SANDBOX <-> INTERNET / HOST NETWORK
======================================================================

5.1 [I, E] Kernel env: full host network as the server user. BLOCKER, both. (Fixed by 3.1.)
From model code: cloud metadata `169.254.169.254` (IMDSv1 credentials, instance identity), every internal service the host can reach, and the dsagent API itself at 127.0.0.1:8000 — unauthenticated — so a persona can create runs, delete other projects' datasets, overwrite sessions, and answer gates for itself (`curl -XPOST localhost:8000/runs/<id>/gate -d '{"decision":"approve"}'`), defeating the human gate (invariant 4) from inside the step.

5.2 [I, D] Docker env: default bridge network, no egress control. BLOCKER for SaaS, REQUIRED for SH, S/M.
No `--network` flag (docker.py:108-120) -> full egress for exfiltration/C2/mining, IMDS reachable (IMDSv2 with hop-limit 1 blocks it; v1 does not), `host.docker.internal` on Docker Desktop, and if the API is bound to 0.0.0.0 the container can hit it (same self-approval trick as 5.1). `pip install` from the model doubles as an exfil channel and a supply-chain entry.
Required control: `--network none` by default; when packages are needed, an egress proxy with an allowlist (internal PyPI mirror, tenant-approved hosts) and blocked RFC1918/link-local; IMDSv2 required + hop limit 1 on the host; the API never reachable from the sandbox network.

======================================================================
BOUNDARY 6: SUPPLY CHAIN
======================================================================

6.1 [T, E] A cartridge is arbitrary code on the host, unsigned. REQUIRED, both (BLOCKER once third-party cartridges are accepted), M.
`dsagent cartridge install` pip-installs whatever requirement strings the cartridge declares (cli.py:121-160; requirements are "opaque" by design, so `git+https://...` or a typosquat is accepted); skill scripts run with the host interpreter in the kernel env (host/build.py:122-123); auto-gate scripts likewise (runner.py:964-966); `envs/*/Dockerfile` is built and run (docker.py:243-256); `hooks/hooks.json` and `.mcp.json` are planned (architecture.md:53-54), `dsagent cartridge add <git/URL>` too (:282). No signature, hash, allowlist or review step.
Required control: signed cartridge manifests with pinned hashes for scripts/Dockerfiles, requirement allowlist (name==version only, from an internal index), cartridge load = admin action with an audit event, and none of it executed on the API host.

6.2 [T] Dependencies unpinned. REQUIRED, both, S.
pyproject.toml uses floors (`>=`) with no lockfile (no uv.lock/requirements.txt); the Dockerfile pulls `python:3.12-slim` by tag and `google-meridian pandas pyarrow plotly markdown` unpinned; `apt-get install git` unpinned. Two pre-1.0 packages are pinned (ag-ui-langgraph 0.0.45, copilotkit 0.1.96) — good, but they are young and read-not-documented. ui/package-lock.json exists (good). Kernel env requirements (cartridge.yaml `pandas>=2`...) are installed into the server venv.
Required control: uv/pip-tools lockfile with hashes, image digests, Dependabot/Renovate + SCA (pip-audit, npm audit, trivy) in CI, SBOM for the SH build.

6.3 [T] Model-fetched packages. Covered: pip inside the kernel hits the server venv (3.1); inside Docker it is unrestricted egress (5.2). Required: per-sandbox site-packages, allowlisted mirror, no network by default.

6.4 [T] Skill materialization is a writable copy (3.4). Also `cartridges/ds/skills/eda/scripts/__pycache__/*.pyc` in the repo shows the cartridge tree itself is writable at runtime by whoever runs scripts — make the cartridge dir read-only to the service user.

======================================================================
BOUNDARY 7: OPERATOR <-> SYSTEM (admin, audit, IR, compliance)
======================================================================

7.1 [R] You cannot reconstruct WHO did what. BLOCKER for enterprise, REQUIRED for first customer, M.
events.jsonl/run.json/runner.log record what the runner did but never an actor: gate answers carry only `note` (runner.py:879-880), uploads, run creation, dataset deletion, session edits and chat turns have no principal, no source IP, no request id. No HTTP access-log configuration (cli.py:339 default uvicorn only), no structured logs, no log shipping. And the record is writable by the sandboxed code it describes (3.1) and by unauthenticated HTTP (`_rewrite_state` api.py:774-787 via `/plan`; `PUT /sessions`).
Required control: append-only audit log written by the API layer with actor/tenant/ip/request-id for every mutating call and every model turn, shipped off-box (SIEM), hash-chained or WORM; run records owned by the API, not the sandbox uid.

7.2 [I, R] GDPR erasure is impossible by design; data is copied everywhere. REQUIRED for EU customers, M.
api.py:246-268 refuses to delete a started run ("a button that can erase that is not a button this API offers"); datasets are copied into every run (datasets.py:53-66 `seed_run`, also on every chat turn serve.py:271), into gate-versions (runner.py:905-933), into export snapshots (export.py:28, 5000 rows per chart), into session transcripts, and live in kernel memory; `DELETE /projects/{id}/data/{name}` (datasets.py:45-50) removes one copy. No retention policy, no encryption at rest, no backup story.
Required control: tenant-scoped erasure job that walks runs/gate-versions/sessions/workspaces and kills kernels; data export endpoint per tenant; retention config; encryption at rest; document the copies.

7.3 [D, E] Incident response tooling is absent. REQUIRED, both, S/M.
`stop` only stops between steps (api.py:371-395, runner.py:583-604) — a runaway `run_python` (600 s per call, unlimited calls) or a mining loop in a container cannot be killed from the API; no "kill tenant", no key rotation without restart, orphaned containers/kernels after a crash (docker.py has no `--rm`; kernels close only via Env.close), `/agent/health` (serve.py:298-300) is a constant. No metrics/tracing, no alerting on spend.
Required control: hard kill (container rm / kernel SIGKILL) endpoint, per-tenant suspend, startup GC, real health/readiness, metrics (runs, spend, sandbox count), alerts.

7.4 [I] Operator-facing SOC2 gaps a buyer will ask on day one: SSO/SCIM, RBAC (viewer/approver/admin; today anyone can approve a gate), encryption at rest and in transit (no TLS config; assumes a proxy), backups/DR, vulnerability management (6.2), change management (branch protection, signed releases), logging retention, pen test, subprocessor list/DPA (2.2), data residency (2.2), incident response plan (7.3), secrets management (2.1). IMPORTANT unless enterprise = REQUIRED.

Re-verified as SAFE (do not re-spend effort): path traversal on the file routes (serve.py:419-437, :502-511), `_safe_name`/`safe_name`/`_safe` id sanitization (api.py:790-802, datasets.py:20-25, sessions.py:21-22), run-id-as-path checks (tools.py:219-230, charts.py:463), markdown rendering through DOMPurify, export markdown neutering + export CSP/attachment (api.py:574-582), `/runs-x` scripted CSP, gate-version path check (api.py:521-528), parquet preview uses pandas not pickle, host uid for containers.

======================================================================
RANKED FINDINGS
======================================================================
1. BLOCKER  3.1 kernel env = server privileges/env/network/venv; audit log writable by the sandbox (both) — XL design / S interim
2. BLOCKER  1.1+4.1 no auth, no tenancy, client-chosen ids (both) — L
3. BLOCKER  2.1+3.5 provider key in sandbox env, no per-tenant secrets/budgets (both) — M
4. BLOCKER  3.2+5.2 Docker unhardened, full egress/IMDS, no limits/timeouts (SaaS; REQUIRED SH) — S/M
5. BLOCKER  3.3 container->host read via symlink following in zip/export/keep_version/matched (SaaS; REQUIRED SH) — S
6. BLOCKER  1.3 model HTML same-origin without CSP (SaaS; REQUIRED SH) — S
7. BLOCKER  1.2 DNS rebinding + text/plain CSRF on gate/run routes (localhost product today) — S
8. REQUIRED 4.2 shared process/uid/venv/kernel across tenants and chats — XL (subsumed by real sandbox)
9. REQUIRED 4.3 model tools read/amend any run by id across projects — S
10. REQUIRED 7.1 no actor in any record; records editable via HTTP and sandbox — M
11. REQUIRED 1.4 client-authored transcripts trusted as context — M
12. REQUIRED 1.5 no body/rate/concurrency/spend/disk limits — S
13. REQUIRED 2.3 prompt injection via inputs, gate notes, datasets, prior reports — M
14. REQUIRED 7.2 erasure impossible, copies everywhere, no encryption at rest — M
15. REQUIRED 7.3 no kill, orphaned sandboxes, no metrics — S/M
16. REQUIRED 6.1 cartridges = unsigned host code incl. pip-install of opaque strings — M
17. REQUIRED 6.2 no lockfile/digests/SCA — S
18. IMPORTANT 3.4 skills mount writable from host-side file tools — S
19. IMPORTANT 3.1b persona `tools:` frontmatter not enforced — S
20. IMPORTANT 2.2 subprocessor/DPA/residency/redaction — M
21. IMPORTANT 7.4 SOC2 control set — L (process + tooling)
22. NICE     step `timeout:` silently ignored; `chat_agent_for` race; `.pyc` in cartridge tree; `_estimate` cross-run cost leak

======================================================================
(a) TOP 5 BLOCKERS
======================================================================
1. The sandbox is not a sandbox: kernel env runs model code as the server user, with the server's env (API keys), filesystem (every tenant), network (IMDS, internal, the unauthenticated API itself) and venv (persistent cross-tenant pip). envs/kernel.py:87,29-33,108.
2. No identity, no tenant, no ownership check on any route or model tool; ids are predictable/client-chosen; localhost is CSRF/DNS-rebinding exposed. serve.py:171, api.py:764, projects.py:77, sessions.py:53, tools.py:178, charts.py:461.
3. Secrets: one global provider key in process env -> sandbox -> transcripts; no per-tenant key/budget/rotation. README:30, models.py:19, sessions.py:68.
4. Docker env: no network/caps/seccomp/limits/timeouts/`--rm`, plus host-side symlink following gives the container read access to the host (api.py:598, runner.py:928, export.py:221). envs/docker.py:108-125.
5. Model-written HTML served on the app origin with no CSP, and the audit trail (run.json/events.jsonl) is writable by the code it audits and by anonymous HTTP. serve.py:478-553, api.py:774-787.

======================================================================
(b) MINIMUM SECURITY ARCHITECTURE TO HOLD A CUSTOMER'S DATA
======================================================================
Sandbox: every `execute`/`run_python`/skill script/auto-gate runs in a per-tenant (ideally per-run) isolated unit — microVM (Firecracker/Kata) for SaaS, at minimum gVisor/rootless container for SH — with: no host mounts except the tenant workspace volume; non-server uid; read-only rootfs + tmpfs; cap-drop ALL, no-new-privileges, default seccomp; cpu/mem/pids/disk quotas; hard wall-clock per step and per run enforced by kill; `--network none` by default, optional egress only through an allowlisting proxy (internal PyPI mirror), IMDS unreachable, the API unreachable; ipykernel lives INSIDE the unit (the host only speaks the kernel protocol to it); pip installs stay inside the unit; skills mounted read-only from a content-addressed cache; all host-side reads of the workspace refuse symlinks. The kernel-on-host env is a `dsagent chat`/dev-only mode, refused by `dsagent serve` unless `--insecure-local`.
Identity & tenancy: OIDC/SAML login; tenant (org) -> projects -> runs/sessions with tenant_id on every record; every route and every model-callable tool resolves the caller's tenant and refuses everything else; server-minted UUIDs; RBAC with an explicit "approver" role for gates; signed gate decisions with actor.
Secrets: provider keys only in the API process from a secret manager; per-tenant virtual keys or BYOK; hard daily/monthly spend caps enforced before invocation; nothing secret in any sandbox env; secret-pattern redaction on tool results.
Data: per-tenant storage prefix (object store or per-tenant uid dir), encryption at rest, run records and audit written by the API (not the sandbox), append-only audit with actor/ip/request-id shipped off-box, retention + tenant erasure job + export.
Web: model artifacts on a separate origin with `sandbox` CSP; strict CSP on the app; Origin/Host/Content-Type enforcement; body/rate/concurrency limits; server-owned transcripts.
Supply chain: lockfiles with hashes, image digests, SCA in CI, signed cartridges with pinned script hashes, requirement allowlist, cartridge load as an audited admin action.
Ops: kill/suspend per run/tenant, sandbox GC, metrics + spend alerts, TLS, backups.

======================================================================
(c) PHASED PATH
======================================================================
Phase 0 — demo-safe (days, all S): refuse `--host` != 127.0.0.1 without `--insecure`; TrustedHost + Origin + Content-Type middleware; a shared bearer token (`DSAGENT_TOKEN`) on every route incl. /agent and SSE; `inherit_env=False` + allowlist env for LocalShellBackend and `KernelManager(env=...)`; move the serve workspace outside the repo; CSP `sandbox` + nosniff on both file routes (UI opens /runs-x); symlink-refusing helper in matched/_keep_version/_archive_members/export/snapshot; Docker flags (`--network none`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`+tmpfs, `--memory/--cpus/--pids-limit`, `--rm`, `--init`); enforce step `timeout`; body-size cap; per-workspace venv or blocked pip for the kernel env; read-only `/skills/` route; enforce frontmatter `tools:`; pin deps + lockfile; TLS via reverse proxy in the docs.
Phase 1 — first paying customer (weeks, M/L): OIDC auth + tenant/project ownership on every route and tool (4.1, 4.3); server-minted ids and server-owned transcripts; kernel env forbidden under serve — ipykernel inside the container, one container per tenant-project (or per run) with its own uid and egress proxy; per-tenant provider key + spend cap + kill switch; API-owned append-only audit log with actor shipped off-box; run records outside the sandbox-writable tree; kill/suspend endpoints + sandbox GC + metrics; erasure/export jobs, retention, encryption at rest; rate limits; SCA in CI; DPA + subprocessor list; a written IR runbook. Self-host ships as a hardened compose/helm with these on by default.
Phase 2 — enterprise (quarters, L/XL): microVM isolation per run; BYOK/KMS, region pinning and provider choice per tenant; SSO/SCIM, RBAC incl. approver role, signed gate decisions; prompt-injection defenses (provenance-tagged untrusted content, human gate on any external effect, quarantined dataset text); signed cartridge registry and allowlisted package mirror; SOC2 Type I->II program (change mgmt, vuln mgmt, logging retention, backups/DR, pen test, vendor review); data-residency deployments.

One-line summary for the owner: today the product's security boundary is "trust the model and trust everyone who can reach port 8000". Until the sandbox is a real isolation unit with no secrets, no egress and no host reach, and every request carries an identity and a tenant, it cannot hold a customer's data — hosted or self-hosted.
