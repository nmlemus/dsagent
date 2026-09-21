# Round 1 — synthesis (P0–P3), local single-user bar

Five agents (Opus 4.8) reviewed the code on 2026-09-18 at commit `8ac2758` judged as a
**local single-user dev tool**. Deduplicated and prioritized; ⚠️ = a regression introduced in the
2026-09-18 session; `[…]` = which agents corroborated it. Full reports in this folder. Round 2
re-judged everything under the commercial bar — see `../00-consolidated-plan.md`.

Honest note: the claim documented in `serve.py:231-251` ("single source of truth… always
converges via stable IDs") rests on an atomic, single-writer store. `SessionStore` is neither
(arch H2, premortem cross-note).

## P0 — fix before trusting it with real work (data loss, security, broken core function)

1. **SessionStore non-atomic write + swallowed error + 200-on-failure** — `sessions.py:81` uses bare `write_text` (not temp+`os.replace` like `RunState.save`); `except OSError: pass`; route still returns 200. Reload/poll/crash mid-write → truncated JSON → chat empty or gone. [premortem#1, arch H1] — *the unanimous "fix first"*.
2. ⚠️ **`getSession` wipes an existing chat to empty on any non-404 error** — `ChatCenter.tsx:232-240`: the catch treats a 500/network blip/restart like "new chat" and PUTs `messages:[]` over the real transcript. [ui C1]. Fix: register-empty only on a real 404.
3. ⚠️ **Run-reading/chart-amend/preview tools ignore `store`** — `list_run_files`/`read_run_file`/`_run_workspace` (`runner/tools.py:180-249`), `amend_tools` (`runner/charts.py:436-487`), `preview` (`api.py:486`) resolve `runs_dir/run_id`; chat-started runs in any non-default project live in `projects/<id>/runs/` → "unknown run" on every follow-up. [arch C1+M1].
4. ⚠️ **Model-written HTML served same-origin with no CSP** — `/projects/{id}/files` via bare `FileResponse`; UI opens in a tab at the app origin → LLM `<script>` can read/delete transcripts and drive `/agent`. The runs canvas already sandboxes (`SCRIPTED_CSP`); this route didn't inherit it. [security#2, ui H2]. Fix: `sandbox` CSP + `nosniff` (or attachment).
5. **DNS rebinding → RCE** — no Host validation/CORS/auth; a malicious page rebinds to 127.0.0.1 and drives the API incl. `POST /agent` → `run_python`. [security#1, premortem#9]. Fix: `TrustedHostMiddleware`.
6. ⚠️ **`App.tsx:59-76` sessions-load effect has no `live` guard** — root cause of both manual-test bugs (dropped typing; "New chat" doesn't switch on first click): a late `setActive` clobbers what the user just did and remounts `ChatThread`. [ui H1]. Fix: the standard `live` cleanup guard.

## P1 — breaks under real/extended use

7. **Per-project kernels + `InMemorySaver` never evicted** → OOM over weeks (one resident kernel per project ever chatted in, ~250MB–1GB each). [scale#1, premortem#3].
8. **Two tabs / stale history = last-write-wins, no version guard** → lost turns. [arch H1/H2, premortem#2, ui L1].
9. ⚠️ **New-chat registration fire-and-forget** → first turn runs in the `default` project (wrong workspace/datasets). [arch H3].
10. **`chat_agent_for` blocks the event loop** (env build up to 60s; `seed_run` every turn) → freezes every client. [scale#3].
11. **`seed_run` re-copies all datasets every turn** → IO/latency; torn reads with the kernel. [scale#6, premortem#4].
12. **keepalive save silently drops the last turn of long chats (>64KB)** and misfires on plain chat-switch. [ui H3].
13. **`.clone()` shares one kernel** → concurrent same-project turns corrupt output / 600s timeout. [scale#4].

## P2 — robustness / operability / debt

14. **No observability**: `driver.log` no-op, swallowed catches, no error boundary, no user feedback on failed writes. [premortem#7, ui M2/M6].
15. **No retention/GC** → disk fills (runs, `events.jsonl`, gate-versions, sessions). [scale#10, premortem#4].
16. **`list_project_artifacts` rglobs the whole tree every 5s**; `list_for` parses every transcript for the sidebar. [scale#5/#7].
17. **Kernels orphaned on crash**; random container names unreapable. [scale#2].
18. **A dep/format bump bricks old runs/chats** (no `schema_version`/migration; `globals()` hack; copied bridge internals; positional interrupt matching). [premortem#5, arch M2].
19. **`run.json` lost-update race** for plan/inputs edits. [premortem#6].
20. **Chat gate lost on restart** (documented). Note: a durable async saver would reintroduce a second durable copy — a design decision, not a free fix. [premortem#8].
21. **Default project shares one kernel between chat & driver, unlocked.** [arch M3].

## P3 — polish

`globals()` hack → manual `request.json()` parse [arch M2] · UI hardcodes DS personas/prompts (brushes invariant #1) [arch L1] · ChartView re-embeds every render / huge spec freezes tab [ui M3/M4] · chat rows not keyboard-accessible, no mobile layout, white chart bg in dark mode [ui M1/L3/L4] · upload size cap [security#4] · session id truncated to 128 chars → collision [arch L3] · gate-resume polling bound [arch M4].

## Confirmed fine

Invariants #4/#5/#6 hold; path traversal, dataset write path and id injection are safe; the runs-canvas iframe sandbox is well designed.
