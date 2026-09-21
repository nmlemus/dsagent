# DSAgent startup readiness review

Reviewed September 18, 2026, at commit `8ac2758`. Scope: this repository, its installed dependencies where relevant, its desktop UI, and an isolated replay instance. Assessment by product stage, rather than a single averaged score.

Follow-up added September 19: [prioritized implementation backlog in Spanish](STARTUP-BACKLOG.md), with task IDs, dependencies, acceptance criteria, and execution tracking. This review preserves the original findings; task completion is tracked in the backlog.

**Verdict: credible startup prototype; conditional candidate for supervised customer pilots; not ready for an unattended, shared hosted launch.**

The strongest product idea is a repeatable analytical workflow with explicit review points and evidence that survives the conversation. The architecture already supports much of that idea. The main gaps are execution isolation, consistent state ownership, analytical validation, and a user journey that exposes those capabilities reliably. Commercial demand and willingness to pay remain unproven by the evidence available here.

| Proposed use | Assessment |
|---|---|
| Founder demo, accelerator application, technical feasibility pitch | Yes, with clear alpha positioning and honest limitations |
| Supervised pilot on synthetic or approved non-sensitive data | Conditional: address the critical defects below and define the supported workflow |
| Paid single-user local product | Plausible direction, after safer execution, packaging, recovery, and onboarding work |
| Shared hosted SaaS with customer datasets | No-go in the current implementation |
| Autonomous marketing-budget/modeling decisions | No-go until independent validation and representative evaluations are demonstrated |

**What was verified**

- `.venv/bin/python -m pytest -q`: **369 passed, 12 skipped**, in 24.17 seconds. Skips include real-model and Docker-dependent integration checks. This is meaningful harness coverage, not evidence of general analytical accuracy.
- `.venv/bin/pytest -q`: collection failed in 12 modules because `tests.fakes` could not be imported. The module invocation above resolves the path issue. The documented command is therefore unreliable in this environment.
- `npm run build`: TypeScript and production build passed. Main JavaScript: approximately **923 KB / 258 KB gzip**; lazy chart chunk: **852 KB / 293 KB gzip**. Vite reported large chunks.
- Ruff: seven findings, primarily import organization and broad exception handling. These are modest hygiene issues compared with the functional defects below.
- Desktop visual inspection in Chrome, including the existing workbench and a fresh isolated replay. Selected a public sample dataset, launched EDA, reached its gate, approved, and observed completion and report links. The replay recorded seven chart events.
- Targeted temporary probes reproduced cross-project run aliasing, project preview failure, unsafe HTML response headers, ZIP symlink leakage, out-of-order session overwrite, misleading sample-based uniqueness wording, and permissive modeling diagnostics.
- No fresh paid model calls, real Docker fit, load test, independent statistical benchmark, dependency vulnerability audit, or customer interviews were performed. Browser execution of a malicious HTML payload was not needed or attempted; response behavior and the same-origin launch path were verified.
- Application source was not changed as part of this review. Replay and probe data were created under `/private/tmp`.

**Architecture: a sound foundation with an incomplete trust boundary**

The separation between cartridge, agent host, workflow runner, and execution environment is sensible. Cartridge validation catches persona/skill inconsistencies and DAG problems. Keeping workflow order and approval gates outside model planning is a valuable design choice. The file-backed run record, event stream, artifact checks, chart schema/compiler validation, and injected fake environments make behavior inspectable and testable. React/Vite plus FastAPI is a reasonable stack for this stage; a framework rewrite would not address the principal risks.

The implementation has two execution paths with different guarantees: launcher-driven runs use a background driver and SQLite checkpoints, while chat graphs use per-project environments and in-memory checkpoints. Both need explicit lifecycle contracts. A gate surviving a restart on one path does not establish the same guarantee for the other. The shared project kernel also means concurrent chats can interact through persistent variables and files without a session-level execution lock.

Launcher requests already identify the workflow and inputs, yet `GraphDriver` asks an LLM to call the workflow tool. The code explicitly handles the case where the orchestrator never starts it. This adds latency, cost, and a failure mode to a decision the user has already made. Prefer a direct deterministic runner entry point for structured launches; retain model routing for open-ended chat. See [driver.py](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/driver.py:88).

The harness abstraction needs contract tests before expanding the cartridge ecosystem. Examples of drift include DS-specific naming in the loader, MMM wording in the orchestrator prompt, persona `tools` metadata that is loaded but not applied as a tool restriction, and raw workflow names used as global dictionary keys. Treat cartridges as trusted executable packages today, not independently isolated plugins.

**Priority findings**

Severity is relative to the intended deployment. P0 blocks shared hosted use or exposure of valuable customer data; P1 blocks a dependable pilot or its affected workflow; P2 should follow before broader distribution.

| Priority | Finding and evidence | Customer impact and required outcome |
|---|---|---|
| P0 for hosted use | **No authentication or resource authorization.** Run creation succeeded without credentials; projects, sessions, files, gates, and execution endpoints have no user/tenant ownership checks. | Anyone able to reach the instance can use its capabilities. Default loopback binding is appropriate for a local alpha, but a hosted service needs identity, per-resource authorization, and tenant boundaries before exposure. |
| P0 for sensitive use | **Default execution is the host process environment.** `LocalShellBackend(..., inherit_env=True)` and Jupyter run code locally. The installed backend explicitly states virtual paths provide no sandboxing. | Model-generated code can access host resources and inherited provider secrets. Move execution into disposable constrained workers, allowlist environment variables, and control network egress. Docker support is a useful seam, but its current run arguments do not enforce memory/CPU/PID/network limits. |
| P0 | **Generated HTML is served on the application origin without sandbox headers.** Ordinary `/runs/.../files/report.html` returned `text/html` with no CSP or attachment disposition. The UI opens that URL in a new tab. Project HTML uses the same unsafe pattern. | Generated or uploaded active content can act with the application origin. Apply protection on every active-content route, ideally a separate artifact origin; preserve CSP sandboxing for interactive reports. The protected `/runs-x` and exported-report routes do not protect the ordinary links. |
| P0 | **ZIP export bypasses file containment.** A temporary symlink to a marker outside the workspace returned 404 through the raw-file endpoint but its contents appeared in `download?everything=true`. | Exports can include host-readable data outside the run workspace. Resolve and validate each archive member against the same policy as raw file serving; test symlinks and internal files consistently. |
| P1 | **Run IDs collide across projects.** With the timestamp held fixed, a default-project run and a second-project run received the same ID. Reading the second returned the default run's question. IDs are generated within each project, then resolved globally with default first. | Opening, starting, stopping, or approving a run can target another project's record. Use globally unique IDs and explicit project ownership; make allocation atomic. |
| P1 | **Non-default project table preview returns 404.** A unique project run's CSV returned 200 through `/files` and 404 through `/preview`. The preview resolver omits `store`. | A core project workflow breaks even when the artifact exists. Centralize project-aware run resolution and cover each API surface with non-default-project cases. |
| P1 | **Chat persistence can lose newer data and report success on failed writes.** Saving timestamp 200 followed by timestamp 100 replaced the newer conversation. Writes are non-atomic, lack revision checks, and suppress `OSError`. | Multiple tabs, delayed saves, or disk failures can lose transcripts without warning. Use transactional revisions or compare-and-swap, atomic writes, and actionable error responses. |
| P1 for modeling | **The MMM automatic gate accepts incomplete and non-finite diagnostics.** Both `{"rhat_max":1.0}` and a payload containing NaN passed. Missing divergence and parameter data defaults to success. | A model may pass review without valid convergence evidence. Require a strict diagnostics schema, finite values, required fields, and checks computed independently from actual model output. |
| P1 | **Declared step timeout is not enforced; Stop waits for the current step.** `timeout: 4h` is modeled but unused by the runner. The UI only says “Stop.” Jupyter's timeout stops waiting without interrupting the executing kernel. | Long or runaway work can continue consuming resources after a user expects it to stop. Enforce deadlines, add interrupt/termination semantics, and distinguish “stop after this step” from immediate cancellation. |
| P1 | **Workflow charts are emitted but not rendered by the run UI.** Replay completed with seven `dsagent.chart` events. `useRunStream` handles step/file/note/tool events and ignores charts; RunDock has no chart event view. | A defining product capability is absent from the normal workflow view. Restore evidence-backed chart/table rendering and verify it through the browser. Inline chat chart fences are a separate path with weaker provenance. |
| P1 | **Plan, recovery, and failure UX are incomplete.** Launcher creates and immediately starts; it does not call `/plan`. RunDock does not expose retry/resume or render the returned step/driver errors. Rejection requires continuation that is not offered there. | Users cannot inspect expected work/cost before launch or recover clearly from failure. Provide a preflight view and complete rejection, retry, stale-run, and stopped-run journeys. |
| P1 for development | **The Vite proxy omits `/projects` and `/sessions`.** The SPA now depends on both for boot and persistence. | The documented two-process development flow cannot forward these requests to FastAPI correctly. Add the current API surfaces and verify the same journey in development and production modes. |
| P2 | **Mobile CSS targets an obsolete root.** The responsive rule targets `.app`; App renders `.shell`. Modals lack complete dialog/focus management, and form labels are not associated with controls. | The desktop layout has no effective corresponding narrow-screen switch. Correct responsive selectors and validate keyboard, screen-reader, focus, and narrow-width behavior. This review did not establish WCAG conformance. |
| P2 | **Sampled uniqueness is described as a whole-file fact.** A profile of 500 sampled unique values in a 1,000-row file produces “unique across all 1000 rows.” | The preflight API can overstate data quality. State the sample size or calculate the full-column property before making the claim. |

Primary implementation references: [kernel.py](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/envs/kernel.py:84), [docker.py](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/envs/docker.py:105), [HTML routes](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/serve.py:479), [FileViewer](/Users/nmlemus/projects/dsagent/dsagent-v2/ui/src/components/FileViewer.tsx:34), [archive members](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/api.py:723), [run ID allocation](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/api.py:143), [run lookup](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/projects.py:77), [preview](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/api.py:486), [session writes](/Users/nmlemus/projects/dsagent/dsagent-v2/src/dsagent/sessions.py:68), [MMM gate](/Users/nmlemus/projects/dsagent/dsagent-v2/cartridges/ds/workflows/mmm-meridian/scripts/check_rhat.py:17), [event reducer](/Users/nmlemus/projects/dsagent/dsagent-v2/ui/src/lib/useRunStream.ts:42), [launcher](/Users/nmlemus/projects/dsagent/dsagent-v2/ui/src/components/Launcher.tsx:71), [Vite proxy](/Users/nmlemus/projects/dsagent/dsagent-v2/ui/vite.config.ts:9), [responsive CSS](/Users/nmlemus/projects/dsagent/dsagent-v2/ui/src/styles.css:1462).

**Code quality, durability, and operations**

The code shows deliberate engineering: dependency injection, small environment interfaces, useful comments describing observed failures, atomic run-state replacement, chart grammar alignment, and tests for gate/replay behavior. Seven lint findings are not the reason to delay launch.

The larger maintainability issue is duplicated ownership. APIs, driver threads, chart tools, and the workflow runner update related state; project-aware lookup is applied inconsistently. `runner.py` combines execution, telemetry, file watching, gates, persistence, and chart integration in 1,171 lines; `api.py` carries 804 lines of routes and helpers. Refactor around transactional run state and lifecycle operations when fixing the defects, with behavior tests protecting the boundaries.

Current scaling is bounded by an in-process thread per run, locally cached project environments, filesystem scans, and a local SQLite checkpoint database. `read_events` rereads the log from its beginning even when a cursor skips old events; SSE calls it every 250 ms per subscriber. Several UI panels also poll independently. Uploads buffer whole files in memory, ZIP exports buffer the whole archive, Parquet previews load the full table, and each run copies every project dataset. These choices can work for a small local pilot, but need explicit size/concurrency limits and measurement.

A shared hosted version should separate the API from constrained workers behind a durable queue, establish transactional metadata, and move artifacts to storage with enforced ownership. Add admission control and per-run resource budgets before adding more workers. Horizontal replication of the current process would not make its locks, memory graphs, or ownership global. There is no need to introduce many microservices to do this.

No checked-in CI pipeline or Python dependency lock was found; the frontend does have a lockfile. The Python dependency floors allow significant upstream change. Establish reproducible dependency sets, automated backend/build checks, a small set of browser journey tests, clean-install/package checks, vulnerability scanning, and backup/restore verification. The wheel configuration packages `src/dsagent`; the CLI defaults to a checkout-relative cartridge and the UI looks for a checkout-relative build, so distribution needs a deliberate packaging story.

Version run schemas and record dataset hashes, cartridge/workflow/prompt versions, model configuration, execution image digest, and code artifacts. Current names, files, and telemetry support inspection but do not fully reproduce a past analytical result after dependencies or cartridges change. Logs are useful operational evidence, not an immutable approval audit with identified reviewers.

**UI/UX: a usable analyst workbench with unfinished product decisions**

The three-pane organization makes chat, projects, and run evidence discoverable. Visual styling is consistent; live state, a highlighted approval card, artifact access, and expandable file views are useful. The replay's happy path worked.

First use still requires understanding “orchestrator,” persona names, `eda-to-report`, `key_column`, and `data_path`. The workflow description even calls it a “harness smoke test.” English and Spanish are mixed throughout. Start with the user's intended output: choose a dataset, choose an outcome, confirm the plan, review the evidence, and export a result. Keep persona and environment details available as secondary information.

The gate needs to put the evidence beside the decision. During the replay it showed the approval card while the selected artifact was still `data-profile.json`; the actual gate report became the selected artifact after approval. Its listed deliverables are inert chips. Automatically focus the gate report and make each relevant artifact navigable before approval.

Reports should occupy the main workspace when they are the user's focus. A narrow run dock full of scratch files competes with the deliverable, while the central pane can remain an empty chat. Separate deliverables from working files more clearly and use readable titles instead of timestamp IDs. Add stable navigation to a project/session/run so reload, bookmarking, and support references preserve context.

Settings says chat history is browser-only, while the composer says server-stored and the backend persists it. Correct that statement before asking customers to trust the product with data. Explain where execution and model processing happen, what leaves the machine, retention, and deletion behavior in the actual supported deployment.

**Analytical trust and economics**

Checking that a required file exists is a valuable orchestration contract, but it does not establish that its contents are correct, current, non-empty, or supported by the dataset. The integration suite contains stronger checks on a clean weather dataset, and historical run notes include manual checks, which is a good start. It is not sufficient coverage for arbitrary business data or marketing modeling.

Create a representative evaluation set with known answers and deliberately difficult inputs: duplicates, missing periods, ambiguous units, inconsistent identifiers, dirty CSVs, leakage, small samples, conflicting metrics, and cases where the correct answer is to refuse or request clarification. Test numerical results and trace claims to computed tables. Include malicious instructions embedded in data in the execution-boundary tests.

Measure repeated-run correctness and critical-error frequency, not just completion. Track human corrections and time spent reviewing. Evaluate the final narrative and chart values independently from the model that generated them. Human gates help only when a reviewer can inspect reliable evidence and understand what approval means.

The displayed run cost is not a complete cost ledger. It sums recorded step costs; orchestrator/chat overhead is outside that sum. Step telemetry is replaced on another attempt, and exceptions can skip final telemetry capture. Partial pricing can also look like a complete total when only some steps have known rates. Historical notes report approximately $0.47 for one EDA run, but that is a repository-reported example, not a validated production unit-cost benchmark or provider invoice.

Before pricing, measure total provider charges plus execution/storage costs, retries, failed attempts, and human support per successful accepted deliverable. Record usage incrementally and cumulatively. Establish budget ceilings and unknown-price indicators. A cheap happy-path run does not establish the economics of an account that needs repeated corrections.

**Market position and a testable business hypothesis**

General conversational analysis is already served by products with substantial surrounding workflows. Hex documents agentic analytics connected to inspectable notebooks; Deepnote documents an agent that works across notebook analyses; Julius documents shared notebook collaboration. These are vendor descriptions, not comparative product tests. Sources checked during this review: [Hex AI](https://hex.tech/capability/ai/), [Deepnote Agent](https://deepnote.com/docs/deepnote-agent), [Julius collaboration](https://julius.ai/docs/collaboration/overview).

My inference is that “chat with data using several personas” is weak differentiation. A more defensible hypothesis is **reviewable, repeatable delivery of one high-value analytical service**, where the cartridge captures accepted methodology, data contracts, reviewer decisions, and reusable evidence. Customer-specific integrations, an evaluation corpus, demonstrated reliability, and adoption in recurring work could create defensibility; prompts and role names alone are easy to copy.

One candidate audience is small analytics consultancies or marketing measurement teams producing recurring client deliverables. That fits the existing artifacts, gates, and MMM direction, but it must be tested. Start with an EDA/data-readiness report whose numerical outputs can be validated reliably. Introduce the modeling workflow only after its separate technical and statistical criteria are met. Broadening into SDLC cartridges now would expand the market story faster than the evidence.

This repository does not establish customer traction, pricing acceptance, distribution advantage, retention, revenue, or gross margin. That is an evidence limitation, not a claim that the founder has none outside the repository. A proposal should distinguish implemented capabilities, verified behavior, hypotheses, and roadmap items explicitly.

**Recommended sequence and release criteria**

| Stage | Work | Evidence required before advancing |
|---|---|---|
| Containment and identity | Fix active-content serving, archive containment, globally unique run ownership, project resolution, and session durability; select a safe execution mode | Reproductions above become passing regression tests; no access outside the intended execution/file boundary |
| One complete customer journey | Fix proxies, preflight, charts, gate evidence, reject/revise/resume, errors, stop semantics, and onboarding language | Browser tests cover fresh launch, reload, approval, rejection, retry, disconnected/restarted server, non-default projects, and export |
| Trustworthy analytical pilot | Add strict output/diagnostic checks, provenance, representative evaluations, and full usage accounting | Repeated accepted results on a defined task set; independently verified headline numbers; known failure modes and bounded cost |
| Commercial validation | Work with a small number of design partners on recurring real tasks | Users return without founder prompting, pay for the outcome, and save measurable time after review/corrections |
| Hosted release, if demand supports it | Identity/authorization, isolated workers, quotas, durable queue/metadata, retention controls, monitoring and recovery | Tenant isolation and restore tests, realistic concurrency tests, operational ownership, and documented service limits |

For an initial pilot, proposed targets could include three to five design partners, at least 20 representative tasks, repeated runs on the critical cases, zero unresolved cross-project or execution-boundary failures, at least 95% completion on the explicitly supported task set, and independently checked correctness of every headline result in the acceptance set. These are suggested acceptance criteria, not measured results or industry standards. Customer time saved and repeat willingness to pay should decide expansion.

**Investment of engineering effort:** retain the cartridge/runner architecture and current web stack; fix the trust and lifecycle boundaries; complete one reliable workflow; then validate a narrow paying audience. This project has enough substance to support a serious startup proposal, provided the proposal presents it as an alpha with a concrete validation plan rather than a production-ready autonomous data team.
