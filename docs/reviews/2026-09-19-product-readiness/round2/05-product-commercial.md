# Round 2 — product-commercial (Fable 5.1) — verbatim (head + parts 1–4)

Lens: "what must be true to charge money" for (A) SaaS and (B) self-host; (C) license
compliance of every dependency; (D) product coherence; (E) moat; (F) engineering maturity.
Reviewed 2026-09-19 at commit `8ac2758`. Test suite run: **369 passed, 12 skipped, 22.7s**
(model-free). No files modified.

---

## 0. Headline

The repo is a well-engineered single-user, single-process, checkout-run dev harness with a strong core idea (declared DAG + gates + `produces` verified on disk + Claude-Code-plugin-compatible cartridges). It is **not installable, configurable, secured, metered or licensed as a product today**, and its shipped UI contradicts the "domain-agnostic" pitch. Neither business path can charge money without a packaging/config/auth layer that does not exist. Path B (self-host) is closer; path A (SaaS) requires a tenancy model the architecture explicitly disclaims (`docs/architecture.md:24` "No multi-tenant auth").

## A. SaaS — what must be true to bill

| # | Finding | Evidence | Sev | Change | Effort |
|---|---|---|---|---|---|
| A1 | **No identity, no tenant, no auth anywhere.** Every route is anonymous; no CORS, no API key, no session user. A "project" is a directory. | grep for `auth|bearer|Depends|CORS` in `src/dsagent/*.py` → 0 hits; `serve.py:171` `FastAPI(title="DSAgent")` no middleware; `api_projects.py:21-118` routes take no principal; `cli.py:288` binds `127.0.0.1` by default | BLOCKER | Auth middleware (OIDC/API key), `tenant_id` on Project/Session/Run, per-tenant roots, RBAC on routes | L |
| A2 | **Token usage is captured for workflow *steps* only, not for chat turns.** Runner records per-step `input_tokens/output_tokens` + cache sub-counts from `usage_metadata` and prices them; run summary aggregates. The `POST /agent` chat path streams through `LangGraphAGUIAgent` with no usage capture, and the orchestrator's own tokens during a run aren't recorded either. | `runner/runner.py:1034-1055` `_record_telemetry`; `runner.py:734-735` `cost_usd = prices.cost(...)`; `runs.py:112-145` aggregate; `serve.py:286-296` chat endpoint — no usage; `sessions.py`/`api_projects.py` grep `usage|cost` → 0 | BLOCKER | Wrap the served graph with a usage-collecting callback/middleware keyed by tenant+session+thread; persist to a ledger (not `run.json`) | M |
| A3 | **Pricing table is a single hand-maintained YAML with one model.** Unknown model → `None` cost (shown as "—"). Billing needs authoritative rates per provider/model and a margin rule. | `prices.yaml` (one entry `anthropic:claude-sonnet-5`); `pricing.py:20,123-130` searched upward from cwd | REQUIRED | Rate table as versioned config (or provider price API), markup policy, currency | S |
| A4 | **No compute/GPU/storage metering.** Wall time exists per run (`started_at/finished_at`, gate wait seconds) but kernel CPU-seconds, container time, GPU hours, workspace bytes are not recorded. The architecture lists GPU pricing as an *open question*. | `runs.py:47-56`; `architecture.md:303` | BLOCKER (if compute is billed) / REQUIRED | Env lifecycle events (`dsagent.env` is planned, `architecture.md:226`) → duration ledger; du of workspace per run | M |
| A5 | **No plans/quotas/limits.** Nothing caps runs, concurrent kernels, upload size, storage, tokens. Uploads read raw body with no size limit; each project spawns a kernel and keeps it forever. | `api.py:271`, `api_projects.py:61` `await request.body()`; `serve.py:257-268` per-project kernel cached in dict, closed only at shutdown | BLOCKER | Quota service + enforcement middleware; idle-kernel reaper; upload limits | M |
| A6 | **Single process, in-memory state, SQLite checkpointer, threads per run.** Not horizontally scalable; a restart loses in-flight chat interrupts (accepted by design). | `driver.py:145` `self._threads`; `serve.py:254-255` dicts; `serve.py:393-403` SQLite; `serve.py:233-251` | BLOCKER for multi-tenant | Job queue + worker pool; Postgres checkpointer; object storage for workspaces | XL |
| A7 | **Code execution is on the server host, as the server user.** Chat and every kernel step run a local Jupyter kernel in the server's process tree; in SaaS this is tenant-to-host RCE. | `envs/kernel.py:29-30` `KernelManager(kernel_name="python3").start_kernel(cwd=...)`; `serve.py:261` `make_env(env_spec, ws)` per project | BLOCKER | All tenant code in Docker/microVM — DockerBackend exists (`envs/docker.py`); make it the only allowed env in hosted mode | L |
| A8 | **ToS/DPA capabilities partial.** Export: HTML report + zip per run (`api.py:543, :584`). Deletion: runs deletable only while `pending` — by design "the record is not deletable" (`api.py:246-268`); no `DELETE /projects/{id}` (`api_projects.py` routes 21-118); sessions and datasets do delete. Audit: `events.jsonl` + `runner.log` per run, but no actor/user on any event. | as cited | REQUIRED | Tenant-level export, hard-delete with retention policy, actor on every event | M |
| A9 | **Admin surface: none** — no tenant list, usage dashboard, per-tenant model/key config, kill switch (`api.py`, `api_projects.py` route inventory). | as cited | REQUIRED | — | M |
| A10 | **Provider keys are process-global env vars**: one `ANTHROPIC_API_KEY` (`.env` at repo root, gitignored), `DSAGENT_MODEL` read in `models.py:19`. No per-tenant BYOK. | as cited | REQUIRED | Per-tenant key vault/BYOK; verify provider commercial terms before reselling model output (not verifiable from the repo — do not assume) | M |

Also relevant to SaaS billing: single process, in-memory dicts, threads per run, SQLite checkpointer (`driver.py:145` `self._threads`; `serve.py:254-255`, `:393-403`) — no horizontal scale; covered as A6.

## B. Self-host product

B1 | **No packaging for the harness.** No root Dockerfile, no compose, no Helm; the only Dockerfile is the cartridge's Meridian env (`cartridges/ds/envs/meridian/Dockerfile`). Install is clone + `pip install -e ".[ui,anthropic]"` + `cd ui && npm install && npm run build` (README.md "Serving the UI", ui/README.md). BLOCKER (self-host). Change: container image (API + built UI), compose for single node, optional Helm; wheel that includes the SPA. Effort M.

B2 | **Works only from a repo-checkout cwd.** `DEFAULT_CARTRIDGE = Path("cartridges/ds")` and `RUNS_DIR = Path(".dsagent/runs")` (`cli.py:42-43`), workspaces `.dsagent/chat` / `.dsagent/serve` (`cli.py:246, :286`), `prices.yaml` searched upward from cwd (`pricing.py:123-130`), UI dist located as `Path(__file__).resolve().parents[2]/"ui"/"dist"` (`serve.py:337-344`) — wrong once installed into site-packages (`DSAGENT_UI_DIST` override exists, `serve.py:325`). The wheel excludes `ui/` and `cartridges/`: `pyproject.toml` `[tool.hatch.build.targets.wheel] packages = ["src/dsagent"]`. BLOCKER. Change: settings object with `DSAGENT_HOME` / cartridge paths / data dir / prices path; ship `ui/dist` as package data. Effort M.

B3 | **Configuration and secrets surface undocumented and ad hoc.** Env vars found: `DSAGENT_MODEL` (`models.py:19`), `DSAGENT_UI_DIST` (`serve.py:325`), `ANTHROPIC_API_KEY`; `DSAGENT_DOCKER` / `DSAGENT_INTEGRATION` for tests. No dotenv loading (no python-dotenv dependency; grep `dotenv` in src → 0) so the root `.env` is read by nothing in the harness. No `--config`. REQUIRED. Change: single documented settings model (pydantic-settings), precedence CLI > env > file. Effort S.

B4 | **Loader writes into the cartridge directory on every load.** `generate_command_skills()` writes `skills/<name>-<wf>/SKILL.md` on each `load_cartridge` (`loader.py:199-248, :281-282`; `cli.py:72` always generates). A read-only install (site-packages, `:ro` volume, immutable image) crashes. REQUIRED. Change: generate into a cache dir or at install time; only `cartridge validate` writes. Effort S.

B5 | **Upgrade / data-migration path: none.** `run.json`, `project.json`, session JSON carry no schema version (`runs.py:85` skips a bad run, no version check); migration so far is "default project adopts the legacy runs dir in place" (`projects.py:13, :55-78`; `serve.py:176-177`) plus a manual folder `.dsagent/legacy-serve-data-backup-20260918`. Gate interrupts live in LangGraph's `checkpoints.sqlite` whose format is theirs. REQUIRED. Change: `schema_version` on every on-disk record + `dsagent migrate`; document checkpoint-DB compatibility per release. Effort M.

B6 | **The cartridge SDK is real but under-documented and drifting.** Validation is good: matrix mirror (`loader.py:168-196`), missing files, cycles (`models.py:116-130`), `sees` (`loader.py:125-129`), env requirement syntax (`loader.py:151-165`). But the spec lives only in `architecture.md §3` plus one README paragraph; the documented `cartridge.yaml` (`architecture.md:83-109`) shows `personas.*.skills` and `extends_plugin`, which the loader never reads — it reads only `role` and `workflows` from the matrix (`loader.py:92-102`); no JSON Schema for `cartridge.yaml` / `workflow.yaml` (Pydantic only, `models.py`); no format version distinct from the cartridge's own `version`; no authoring guide, template, or `dsagent cartridge new`; `dsagent cartridge add <git-url>` is unbuilt (`ROADMAP.md:107`). REQUIRED — this is the extensibility product. Change: versioned spec (`cartridge_format: 1`), JSON Schemas, scaffold command, "write your first cartridge" guide, and a second non-DS reference cartridge. Effort M.

B7 | **Domain-agnosticism is unproven and leaky.** No second cartridge exists (`ROADMAP.md:104-107`, M3 unchecked). Loader contains the literal `"ds-"` prefix (`loader.py:79` `not name.startswith("ds-")`). Multi-cartridge serve/chat uses only `cartridges[0].envs["default"]` (`serve.py:252`; `cli.py:254, :322`). IMPORTANT. Change: ship `cartridges/sdlc`, remove the literal, per-cartridge chat env. Effort M.

B8 | **Customer-facing docs are stale.** Root README still describes a Next.js UI on :3000 with `NEXT_PUBLIC_DSAGENT_ORIGIN`; `ui/README.md` references `@copilotkit/a2ui-renderer` and RunList/Canvas/Chat components that do not exist in `ui/src/components` or `ui/package.json` (the Vite SPA replaced Next.js in commit `63f325f`). IMPORTANT. Effort S.

B9 | **Telemetry / phone-home: none** — only outbound traffic is to the model provider and pip/Docker at build. Good for on-prem. NICE: optional opt-in usage ping later. Effort S.

B10 | **Support surface: no versioned releases, no CHANGELOG, no SECURITY.md.** Git tags v0.6.1…v0.9.1 are v1's; v2 is `2.0.0a1` (`src/dsagent/__init__.py:8`) on branch `ui-harness`; `ROADMAP.md:288` says v2 merges to main as 2.0.0 on the existing `datascience-agent` PyPI line. REQUIRED. Change: semver releases, CHANGELOG, support matrix (Python ≥3.11, Node, Docker). Effort S.

## C. License compliance

THIS REPO: `pyproject.toml` line 9 declares `license = "MIT"`; `cartridges/ds/.claude-plugin/plugin.json` declares `"license": "MIT"`. But there is NO LICENSE file in the repo (`ls LICENSE* NOTICE*` → none). Two problems: (1) package metadata asserts a license whose text is not shipped; (2) if the intent is to SELL the harness, MIT in the metadata is the wrong declaration and needs a deliberate decision (proprietary / BSL / open-core). `ROADMAP.md:288`: v2 is meant to merge into nmlemus/dsagent main and ship as `datascience-agent` 2.0 on PyPI — the existing v1 PyPI line is already public, so an open-core split (MIT harness on PyPI; commercial cartridges/UI/enterprise features) is the least-friction path. BLOCKER (decision), effort S. Both paths.

PYTHON DEPS (installed versions in .venv, read from dist metadata):
- deepagents 0.7.14 — MIT
- langgraph 1.2.11 — MIT; langgraph-checkpoint 4.2.0 — MIT; langgraph-checkpoint-sqlite 3.1.1 — MIT
- langchain 1.4.0 — MIT; langchain-core 1.6.3 — MIT; langchain-anthropic 1.7.2 — MIT; langchain-google-genai 4.4.0 — MIT (langchain-openai not installed)
- ag-ui-langgraph 0.0.45 — MIT; ag-ui-protocol 0.1.22 — MIT
- copilotkit 0.1.96 — MIT
- fastapi 0.141.1 — MIT; starlette 1.6.0 — BSD-3; uvicorn 0.53.0 — BSD-3
- jupyter-client 8.10.0 — BSD-3; ipykernel 7.3.0 — BSD-3
- typer 0.27.2 — MIT; pydantic 2.13.5 — MIT; pyyaml 6.0.3 — MIT; rich 15.0.0 — MIT
- altair 5.5.0 — BSD-3; vl-convert-python 1.9.0 — BSD-3; jsonschema 4.26.0 — MIT; markdown 3.10.3 — BSD-3
- anthropic 1.6.0 — MIT; httpx 0.28.1 — BSD-3; partialjson 1.2.0 — MIT
- cartridge kernel env: pandas 3.0.5 BSD-3; numpy 2.5.3 BSD-3 (+0BSD/MIT/Zlib/CC0); matplotlib 3.11.2 PSF-style
No copyleft, no source-available, no non-commercial. Nothing restricts SaaS or redistribution.

NPM DEPS (from `ui/package.json` and `node_modules/*/package.json`). Runtime:
- @ag-ui/client, @ag-ui/core, @ag-ui/encoder, @ag-ui/proto 0.0.59 — MIT
- @assistant-ui/react 0.15.20 — MIT; @assistant-ui/react-ag-ui 0.0.59 — MIT; @assistant-ui/tap 0.9.17 — MIT
- react / react-dom 18.3.1 — MIT
- marked 18.0.13 — MIT
- dompurify 3.4.15 — (MPL-2.0 OR Apache-2.0): dual; elect Apache-2.0 in your notices so no MPL file-level copyleft question arises
- vega 5.33.1, vega-lite 5.23.0, vega-embed 6.29.0 and vega-* subpackages — BSD-3; d3-* — ISC
- transitive: rxjs 7.8.1 Apache-2.0; @bufbuild/protobuf 2.15.0 (Apache-2.0 AND BSD-3); zod 4.6.5 MIT; radix-ui MIT; uuid MIT; fast-json-patch MIT; secure-json-parse BSD-3; node-fetch MIT; tslib 0BSD; robust-predicates Unlicense
Build-only (not shipped): vite 5.4.21 MIT; typescript 5.9.3 Apache-2.0; @vitejs/plugin-react MIT; caniuse-lite CC-BY-4.0 (browserslist data, dev-time).
Tally across all 258 packages in node_modules: MIT 177, BSD-3 39, ISC 31, Apache-2.0 4, (Apache AND BSD) 1, (MPL OR Apache) 1, BSD-2 1, 0BSD 1, Unlicense 1, CC-BY-4.0 1 (dev). Zero AGPL/SSPL/BSL/non-commercial/unknown.

OBLIGATIONS NOT MET TODAY: MIT/BSD/ISC/Apache all require reproducing the license and copyright in binary distributions; Apache-2.0 additionally requires propagating any NOTICE file (rxjs, typescript, @bufbuild/protobuf). There is no THIRD_PARTY_LICENSES or NOTICE bundled with the SPA (`ui/dist`) or the wheel. REQUIRED for self-host distribution; IMPORTANT for SaaS (network use of permissive code carries no notice duty, but it is expected hygiene). Effort S — generate at build with pip-licenses + license-checker and ship the file.

CARTRIDGE CONTENT: skills prose is original (`cartridges/ds/skills/*/SKILL.md`); `skills/mmm/references/input_schema.md` describes Meridian's API mapping in your own words, not copied docs; `skills/reports/SKILL.md` embeds only the Vega-Lite `$schema` URL; `envs/meridian/Dockerfile` pip-installs `google-meridian` at the customer's build time, so you do not redistribute it. If you ever ship a PREBUILT Meridian image, verify google-meridian's license first (I did not verify it from the repo — do not assume). No third-party content problem found.

MODEL PROVIDER TERMS: not a repo artifact. If SaaS runs on your own keys, verify Anthropic/OpenAI/Google commercial terms for resale of outputs; BYOK (customer keys) sidesteps most of it and is what a self-host customer expects anyway.

## D. Product coherence: the shipped UI is DS-specific and Spanish
- `ui/src/components/ChatCenter.tsx:127-131` `SUGGESTIONS` = three hardcoded Spanish DS prompts ("¿Qué workflows puedes ejecutar y qué hace cada persona?", "Inventa ventas mensuales y muéstralas como gráfica de barras y tabla", "Genera un dataset sintético pequeño y grafica una relación causal").
- `ChatCenter.tsx:363-364` empty-state copy: "delegate to Marie / Noel / Ana / Pablo" — persona names hardcoded in the harness UI. The server already exposes them at `GET /cartridges` (`api.py:178`); the UI simply does not read it.
- `ChatCenter.tsx:392` composer placeholder "Message the DS team…"; `:382` "el equipo está trabajando…"; `:395` "Historial guardado en el servidor · el orquestador puede iniciar runs…". Mixed English/Spanish.
- `ui/src/components/Launcher.tsx:146` "dataset del proyecto", `:153` "— elegir dataset —", `:166` "↥ Subir dataset al proyecto". Spanish and "dataset" vocabulary (`docs/projects-sessions.md:24-25` says the harness calls these "project inputs"; `src/dsagent/datasets.py` and `api_projects.py:54-77` use "dataset" anyway — mild invariant-1 drift).
- `ui/index.html:6` `<title>DSAgent</title>`; `ui/src/App.tsx:140` `<Spine label="DSAgent" …>` — product name hardcoded.
- Theming: only CSS tokens `--accent` / `--accent-dim` for light/dark (`ui/src/styles.css:14-15, :37-38`) and a theme toggle in `Settings.tsx` (localStorage). No logo slot, no brand tokens from config, no i18n layer of any kind, no per-cartridge UI strings.
Verdict: a non-DS buyer cannot deploy this without editing TSX. BLOCKER for the "domain-agnostic harness" positioning; not a blocker if the SKU is "DS team in a box". Fix: a `ui`/`branding` block in `cartridge.yaml` (display name, suggestions, composer placeholder, locale, palette/logo) served through `/cartridges`, an i18n dictionary (en/es), and the empty-state built from the persona roster. Effort M. Both paths.

## E. Differentiation / moat (candid)
Genuinely reusable, not a week's assembly of LangGraph + CopilotKit + assistant-ui:
1. The runner (`src/dsagent/runner/runner.py`, ~1,100 lines): declared DAG over `needs`, per-step persona + env switch, `produces` verified on disk with globs, gates as interrupts kept stable across resume/re-entry, auto-gate scripts materialized into the step's env, per-step token/cost/tool/file telemetry and streamed events, resume from run.json. Opinionated in a way the frameworks are not.
2. The cartridge contract: "valid Claude Code plugin + cartridge.yaml", with skill-scope mirror validation (`loader.py:168-196`) and per-env pip requirements verified at provisioning. A real distribution story into Claude Code — but unproven (M3 unchecked).
3. Server-owned runs + replay tooling (`driver.py`, `replay.py`, `tools/make_replay_fixture.py`): UI development without a model. Good engineering, modest commercial value.
4. The DS cartridge itself — `architecture.md:18` says "This is the product." The MMM/Meridian workflow with data gates and an exportable audited report is the saleable solution.
Glue with low moat: `serve.py` AG-UI bridge (a copied endpoint body), the assistant-ui chat, projects/sessions as JSON directories, the Vega chart pipeline.
Honest read: the bare harness competes with LangGraph Platform, Deep Agents' own tooling and Claude Code plugins. The defensible thing is the combination "verified-deliverable workflows + a domain cartridge that yields audited, exportable reports". Sell the solution; open the harness.

## F. Engineering maturity
- Backend tests: 369 passed, 12 skipped, 22.7s, model-free. Serve/API paths covered: `tests/test_serve.py` 19 tests, `test_runs_api.py` 35, `test_serve_ui.py` 4, `test_projects.py` 9, `test_sessions.py` 6, `test_serve_recursion.py` 3. Good.
- Frontend tests: zero (no `*.test.*` under `ui/src`; README.md: "there are no frontend tests yet"). REQUIRED.
- CI: none (no `.github/` directory). REQUIRED, effort S.
- Python lockfile: none (no `uv.lock` / `requirements.txt`). `pyproject.toml` exact-pins only the two pre-1.0 packages (`ag-ui-langgraph==0.0.45`, `copilotkit==0.1.96`), floors everything else; npm has `package-lock.json`. REQUIRED, effort S.
- Fragile coupling to third-party internals: `serve.py:281-296` is the copied body of `ag_ui_langgraph.add_langgraph_fastapi_endpoint`; `serve.py:278-279` `globals()["RunAgentInput"]`/`globals()["Request"]` hack for FastAPI annotation resolution; gate payload depends on the bridge lifting `reason/message/response_schema` and stashing the rest under `metadata.langgraph.raw` (`serve.py:77-94`); `altair<6` must move in lockstep with vega-lite 5 (`pyproject.toml` ui extra comment; `tests/test_charts.py` guards it); `ROADMAP.md:239-254` documents further traps. IMPORTANT — pin, plus contract tests on the bridge, plus a plan to upstream or vendor.
- Versioning / release: `2.0.0a1` (`src/dsagent/__init__.py:8`); no CHANGELOG; tags v0.6.1–v0.9.1 belong to v1; no PyPI release of v2; recent work committed directly to branch `ui-harness` after a PR-based run (#60–#79). REQUIRED.
- Security disclosure: no SECURITY.md; no auth; host code execution (A1, A7). REQUIRED.
- Docs drift: README and ui/README describe the replaced Next.js UI (B8). IMPORTANT.
- Data hygiene: `.dsagent/` in the working tree holds real run data and a legacy backup with synthetic datasets; `.env` with a live key present (gitignored). NICE — keep out of any packaged image.

## (a) Top 5 blockers per path
SaaS (A):
1. No identity/tenancy/auth on any route (A1; `serve.py:171`, `api_projects.py:21-118`, zero auth/CORS hits in src).
2. Tenant code executes on the host as the server user (A7; `envs/kernel.py:29-30`, `serve.py:261`).
3. Chat-turn and orchestrator tokens are not metered; no ledger, single-model `prices.yaml` (A2/A3; `serve.py:286-296` vs `runner.py:1034-1055`).
4. Single-process, in-memory, thread-per-run, SQLite-checkpoint architecture (A6; `driver.py:145`, `serve.py:254-255, :393-403`).
5. No quotas/limits and no admin surface (A5/A9; unbounded `await request.body()` uploads at `api.py:271` / `api_projects.py:61`; per-project kernels never reaped, `serve.py:257-268`).
Plus the license decision (C) — it gates both.

Self-host (B):
1. No container/wheel packaging; every path is checkout-relative and `ui/dist` is located via `__file__` (B1/B2; `cli.py:42-43`, `serve.py:337-344`, pyproject wheel packages).
2. License decision, missing LICENSE file, no third-party notices (C; `pyproject.toml:9` says MIT, no LICENSE/NOTICE on disk).
3. UI hardcodes DS personas and Spanish copy; no whitelabel/i18n (D; `ChatCenter.tsx:127-131, :363-364, :392`; `Launcher.tsx:146-166`).
4. Cartridge SDK unversioned/under-documented, loader writes into the cartridge dir on load, no second cartridge, literal "ds-" in loader (B4/B6/B7; `loader.py:79, :199-248`, `architecture.md:83-109` vs `loader.py:92-102`).
5. No release process, migrations, CI, or frontend tests (B5/B10/F).

## (b) Which path first
Self-host (B), sold as a solution rather than a platform. Reasons: the SaaS blockers are architectural (tenancy, isolation, horizontal scale) and each is L–XL; the self-host blockers are packaging/config/docs/notices and mostly S–M. The code's actual strengths — a run is a self-contained directory record, Docker env per run, full event/log audit trail, no phone-home — are exactly what a VPC/on-prem buyer values and exactly what a SaaS would have to redesign around object storage and a job queue. The realistic first SKU is "DS/MMM team in a box": harness + DS cartridge + UI deployed in the customer's VPC with the customer's model keys (BYOK removes the provider-resale question). The "domain-agnostic harness" story should not be sold until a second cartridge exists (M3) and the UI is driven from `/cartridges` rather than TSX literals; until then it is a claim, not a product.

## (c) Phased path to first revenue
Phase 0 — design partner (4–6 weeks; pilot fee or services, no license fee yet).
- Decide the license: recommend open-core (MIT harness on the existing `datascience-agent` PyPI line; commercial license for `cartridges/ds`, the UI and enterprise features). Add LICENSE and a generated THIRD_PARTY_LICENSES (pip-licenses + license-checker at build).
- Dockerfile (API + built SPA in one image, Docker socket or DinD for envs) + docker-compose; a settings model replacing cwd-relative paths (`DSAGENT_HOME`, cartridge paths, data dir, prices path); ship `ui/dist` as package data; stop writing generated command skills into the cartridge dir on load.
- CI: pytest + `npm run build` + license check. Rewrite README/ui README as install docs (current ones describe the deleted Next.js UI).
- Run one paid pilot on the partner's data with their keys. Keeping the DS/Spanish UI is fine here — it is the pilot's domain and language.

Phase 1 — first paying self-host customer (following 6–10 weeks).
- Minimal auth: shared API key or reverse-proxy OIDC header, and a role gate on destructive routes (delete dataset/session, gate answer, run start).
- UI strings, persona roster, suggestions, placeholder, branding and locale served from `cartridge.yaml` via `/cartridges`; i18n dictionary (en/es); empty state built from the roster.
- `schema_version` on `run.json` / `project.json` / session JSON + `dsagent migrate`; document checkpoint-DB compatibility per release.
- Usage ledger covering chat turns and orchestrator tokens (a usage-collecting callback on the served graph) — the customer needs internal chargeback even if you do not bill by token.
- Versioned cartridge spec (`cartridge_format: 1`), JSON Schemas for `cartridge.yaml`/`workflow.yaml`, `dsagent cartridge new`, an authoring guide, and `cartridges/sdlc` as the second reference cartridge (proves M3 and removes the "ds-" literal).
- Semver releases, CHANGELOG, SECURITY.md, support matrix. Pricing: annual license per instance + cartridge add-ons + services for custom cartridges.

Phase 2 — scale / optional SaaS (only after 2–3 paying self-host customers).
- Tenant model on Project/Session/Run; Docker-only execution in hosted mode; Postgres checkpointer; job queue + worker pool replacing thread-per-run; quotas (runs, kernels, GPU hours, storage, tokens) with an idle-kernel reaper and upload limits; per-tenant BYOK; admin API/UI; metering → billing (Stripe) on top of the Phase-1 ledger; tenant-level export and hard-delete with retention.
- The runner and the cartridge format should not need to change for this — that is the test of whether the Phase-1 abstractions were right.
