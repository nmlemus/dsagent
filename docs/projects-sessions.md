# Projects & Sessions — the object hierarchy

**Date:** 2026-09-18 · **Status:** design, approved forks · Author: Noel + Claude Code.
Companion to `docs/architecture.md` (harness) and `docs/ui-product.md` (runs belong to
the server). This defines how chats and runs stop being two disconnected worlds.

## 1. Problem

Today there are two species that never meet: **chats** (client-only, `localStorage`) and
**runs** (server-owned, persistent). A chat that starts a run isn't linked to it, and there
is no durable container ("project") grouping the work around a dataset/goal. "Connect
everything" = define the objects and how they nest.

## 2. Entities (harness stays domain-agnostic)

- **Cartridge** (`ds`) — the installed capability: personas, skills, workflows, envs.
  Transversal; projects *use* it, they don't own it.
- **Project** — a body of work: a goal + a **workspace**. The durable container. Generic;
  the DS framing ("dataset", "analysis") is the cartridge's, not the harness's.
- **Session** — a **unit of work**: a conversation with the team that may launch runs. The
  navigation unit.
- **Run** — a workflow execution: DAG, gates, artifacts, cost. Server object.
- **Artifact** — what a run (or the chat) produces: profiles, findings, report, figures.
- **Input/dataset** — files in the project workspace, reusable across sessions and runs
  (the harness calls them project inputs; the `ds` cartridge calls them datasets).

## 3. Hierarchy (approved)

A **Session contains its chat *and* the runs it launched** — one unit of work, not two.

```
Instance (one dsagent serve)
 ├─ Cartridge: ds  ── personas · skills · workflows · envs
 └─ Project (goal + workspace)
     ├─ Inputs/Datasets        (project workspace, reusable)
     ├─ Session                (unit of work)
     │    ├─ Chat              (messages · inline charts/tables)
     │    └─ Runs              (launched here) → DAG · gates · artifacts · cost
     ├─ (view) Runs            — aggregate across the project's sessions
     └─ (view) Artifacts       — aggregate deliverables
```

**Relationships:** `Project 1—* Session` · `Session 1—* Run` · `Run *—1 Workflow` ·
`Run.session_id → Session` · inputs are project-scoped.

**Approved decisions (2026-09-18):**
- **Run nested in Session** (not a separate linked list). A run appears as a *card in the
  session's thread* and belongs to that session. A run started by the launcher/CLI creates a
  minimal session (never orphaned).
- **Backend-real projects** (not client-only grouping). The server becomes project-aware so
  runs, workspace and inputs are scoped per project and robust across tabs, reloads and CLI.

## 4. Backend design

**Project = a workspace directory** under a projects root, plus metadata.

```
<projects_root>/
  <project-id>/
    project.json          # {id, name, goal, cartridge, created_at}
    data/                 # project inputs/datasets (reusable)
    runs/<run-id>/…       # runs scoped to the project (today's run dir, namespaced)
    sessions/<sess-id>.json  # session metadata (see §4.2)
```

### 4.1 Runs scoped to a project
`runs_dir` becomes `<projects_root>/<project>/runs`. The runs API is project-scoped:
`GET /projects`, `POST /projects`, `GET /projects/{p}/runs`, `POST /projects/{p}/runs`, and the
existing `/runs/{id}/…` resolve within the active project (or carry the project in the path).
A **`default` project** preserves back-compat and adopts today's flat `runs_dir` on migration.

### 4.2 Sessions: metadata server-side, messages client-side (hybrid)
A **session** is server-side *metadata* — `{id, project, kind: "chat"|"run", title,
created_at, run_ids: […]}` — so a session and its runs live together and survive restarts,
across tabs and the CLI. The **chat messages stay client-side** (`localStorage`, keyed by
session id): assistant-ui is already the source of truth for a thread (it replays messages to
`/agent` each turn; the server checkpointer only spans a turn's interrupt). This is the cheap,
robust split — no server message store, but the *structure* (project → session → runs) is
durable and shared.

- The chat's AG-UI `thread_id` **is** the session id.
- When the orchestrator starts a run in a chat, the runner records `session_id` on the run
  (from graph config), and the session's `run_ids` gains it. That is the link.
- A launcher/CLI run with no session gets a minimal `kind:"run"` session.

### 4.3 Invariants kept
Projects/sessions are generic harness concepts (no DS vocabulary in `src/dsagent/`). Runs,
gates, `produces` verification, resumability — unchanged; they just resolve under a project's
runs dir. Tests stay model-free (`FakeAgent`/stub env).

## 5. UI mapping (on the existing shell)
- **Sidebar top:** Project switcher (active project scopes everything below).
- **Sidebar:** Sessions list (each shows if it has runs / a pending gate); accesses to
  **Inputs**, **Runs** (aggregate), **Artifacts**.
- **Center:** the active session's chat (home), with **run cards inline**.
- **Right dock:** the run view (as today) when a run is opened.
- **Bottom-left:** user / Settings (+ project settings: cartridge, model).

## 6. Phased plan (one task per branch/PR)
1. **Backend — Project model.** Projects root, `project.json`, `runs_dir` per project, a
   `default` project + migration, `GET/POST /projects`, project-scoped runs routes. Tests
   (no model): create project, run resolves under it, default back-compat.
2. **Backend — Sessions + run↔session link.** Session metadata store, `run.session_id`
   recorded by the runner/driver, `/projects/{p}/sessions`. Test: a run started with a
   session id lands in that session's `run_ids`.
3. **Frontend — the unified shell.** Project switcher, sessions scoped to the project, **run
   cards in the chat**, Inputs and Artifacts views, run dock unchanged.

## 7. Premortem (mitigations baked in)
- *Still disconnected* → the run-card-in-chat (phase 3) is the must-have; projects alone
  don't fix it. Ship the link.
- *Client-only projects break with CLI* → chosen backend-real, so CLI/other tabs fit.
- *Ceremony for one user* → a `default` project always exists; new project = one field.
- *Data duplicated per run* → project inputs are reused across runs.
- *Over-nesting* → session is the unit; runs/artifacts show inside + aggregate views.
