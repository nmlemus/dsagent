# BACKLOG — product readiness (task-ready)

Cada ítem está pensado para convertirse en una tarea/ticket tal cual. Trabajar **de arriba a
abajo**: `DEC-*` (decisiones del owner) → `N-*` (ya, sin esperar) → `F0-*` → `F1-*` → `F2-*`.

Columnas: **Esf.** S <1 sem · M 1–3 sem · L 1–2 meses · XL >2 meses. **Ruta** SaaS / SH
(self-host) / ambas. **Fuente** = reportes en `round2/` y `round1/` con la evidencia. **Dep.** =
ids de los que depende.

---

## DEC — decisiones del owner (bloquean el resto)

| id | Decisión | Recomendación | Estado |
|---|---|---|---|
| DEC-01 | Ruta de negocio primero | Self-host como solución ("DS/MMM team in a box", BYOK); SaaS = Fase 2 tras 2–3 clientes | pendiente |
| DEC-02 | Licencia del repo | Open-core (harness MIT en PyPI `datascience-agent`; comercial para `cartridges/ds`, UI, enterprise). Decidir **antes** del primer contrato: `pyproject.toml:9` ya declara MIT sin archivo `LICENSE` | pendiente |
| DEC-03 | ¿Vender "harness domain-agnostic"? | No hasta `F1-16` (segundo cartridge) + `F1-14` (UI desde cartridge) | pendiente |
| DEC-04 | Sandbox runtime objetivo | Fase 1: ipykernel dentro de contenedor Docker endurecido por proyecto/run; Fase 2: microVM (Firecracker/Kata/gVisor) o proveedor hosted (Modal/E2B/Daytona) | pendiente |
| DEC-05 | Verificaciones legales externas | Términos de reventa de output del proveedor de modelo (BYOK lo esquiva); licencia de `google-meridian` si se envía imagen precompilada | pendiente |

---

## N — ya, sin esperar a vender (todo S; ⚠️ = regresión de la sesión 2026-09-18)

| id | Tarea | Archivos | Esf. | Ruta | Fuente | Dep. |
|---|---|---|---|---|---|---|
| N-01 | `inherit_env=False` + allowlist explícita (PATH, HOME=workspace, LANG) en `LocalShellBackend`; `KernelManager(env=allowlist)` para el kernel; la API key nunca entra al sandbox | `envs/kernel.py:87,29-33` | S | ambas | r2 security 3.1/2.1, tenancy (secrets), premortem #1 | — |
| N-02 | `SessionStore.save` atómico (temp + fsync + `os.replace`, como `RunState.save` `runner.py:283-292`) y **que falle** → la ruta responde 500, no 200; lo mismo para `project.json` y `store_dataset` | `sessions.py:68-83`, `projects.py:174-177`, `datasets.py:28-32`, `api_projects.py:116` | S | ambas | r2 durability §2.1/2.3, premortem #11; r1 premortem#1, arch H1 | — |
| N-03 ⚠️ | Cliente: registrar chat vacío **solo con 404 real**; nunca `PUT []` ante error de red/500; `api.getSession` lanza error tipado con `status` | `ui/src/components/ChatCenter.tsx:232-240`, `ui/src/lib/api.ts:13-18` | S | ambas | r1 ui C1; r2 durability §2.1 | — |
| N-04 ⚠️ | Pasar `store` a `list_run_files`/`read_run_file`/`_run_workspace`, a `amend_tools` y a `preview`; resolver vía `store.locate_run` con fallback | `runner/tools.py:178-249`, `runner/charts.py:411-510`, `api.py:486` | S | ambas | r1 arch C1+M1; r2 tenancy trace #4, security 4.3 | — |
| N-05 ⚠️ | El driver del launcher usa env/workspace **por proyecto**, no el de `default` para todos los runs | `serve.py:305-308,261` | S | ambas | r2 tenancy trace #7, premortem #7 | N-04 |
| N-06 ⚠️ | CSP `sandbox allow-scripts; base-uri 'none'; form-action 'none'` + `X-Content-Type-Options: nosniff` (+ `Content-Disposition` para `.html`) en `/runs/{id}/files` y `/projects/{id}/files`; la UI abre `/runs-x`; eliminar el prefijo sin sandbox | `serve.py:478-481,547-553`, `ui/src/components/FileViewer.tsx:39-46`, `ChatCenter.tsx:196-202`, `ui/src/lib/api.ts:97-98` | S | ambas | r2 security 1.3, premortem #2; r1 security#2, ui H2 | — |
| N-07 | `TrustedHostMiddleware` + check de `Origin`/`Sec-Fetch-Site` en toda mutación + exigir `Content-Type: application/json`; rechazar `--host` ≠ loopback sin `--insecure` | `serve.py:171`, `api.py:764-771`, `cli.py:288` | S | ambas | r2 security 1.2; r1 security#1 | — |
| N-08 ⚠️ | Guard `live`/cleanup en el efecto de carga de sesiones (causa raíz del tipeo perdido y "New chat" que no cambia); fetch de sesiones solo con el proyecto ya asentado | `ui/src/App.tsx:59-76,38-49` | S | ambas | r1 ui H1 | — |
| N-09 | Añadir `LICENSE` (según DEC-02) + `THIRD_PARTY_LICENSES`/NOTICE generado en build (`pip-licenses` + `license-checker`); elegir Apache-2.0 para dompurify | raíz del repo, `pyproject.toml:9`, `ui/package.json` | S | ambas | r2 commercial C | DEC-02 |
| N-10 | Bearer token compartido (`DSAGENT_TOKEN`) en **todas** las rutas incl. `/agent` y SSE | `serve.py`, `api.py`, `api_projects.py`, `ui/src/lib/api.ts` | S | ambas | r2 security 1.1 (interino), premortem #3 | — |
| N-11 | Helper `inside(workspace, path)` que rechaza symlinks (lstat por componente / `O_NOFOLLOW`) en `matched`, `_keep_version`, `_archive_members`, `export`, `_snapshot` | `runner/runner.py:434-449,920-928,1084-1092`, `api.py:598-601,723-731`, `export.py:98-105,217-221` | S | ambas | r2 security 3.3 | — |
| N-12 | Hardening Docker por defecto: `--network none` (o proxy), `--cap-drop ALL`, `--security-opt no-new-privileges`, seccomp, `--read-only`+tmpfs, `--memory/--cpus/--pids-limit`, `--rm`, `--init`; GPU por índice; GC de `dsagent-*` al arrancar | `envs/docker.py:108-125,227-240` (tests de argv en `:16-20`) | S | ambas | r2 security 3.2/5.2, scale #2, premortem #5 | — |
| N-13 | Aplicar `Step.timeout` (matar contenedor/kernel; step failed; re-queue por política) y un wall-clock por run | `runner/runner.py`, `cartridge/models.py:60` | S | ambas | r2 security 3.2, scale R2, premortem #9 | — |
| N-14 | Tope de tamaño de body en uploads (stream a disco) y en `PUT /sessions` | `api.py:294`, `api_projects.py:66-67` | S | ambas | r2 security 1.5; r1 security#4 | — |
| N-15 | Eliminar el fallback "hilo no registrado → `default`": `POST /agent` responde 400/401; el cliente **espera** el registro antes de habilitar send (o declara `project` en el RunAgentInput) | `serve.py:288`, `sessions.py:64-66`, `ChatCenter.tsx:236-240` | S | ambas | r1 arch H3; r2 tenancy trace #2, premortem #7 | N-10 |
| N-16 | keepalive **solo** en `pagehide`; el flush por cambio de chat usa fetch normal; detectar >64KB | `ui/src/components/ChatCenter.tsx:299-350` | S | ambas | r1 ui H3, scale #8 | — |
| N-17 | Reescribir el comentario de `serve.py:231-251` (la afirmación de convergencia no se sostiene: el store no es atómico ni single-writer) | `serve.py:231-251` | S | ambas | r1 arch H2, premortem cross-note | N-02 |

---

## F0 — Fase 0: demo-safe / design partner (2–6 semanas)

| id | Tarea | Archivos | Esf. | Ruta | Fuente | Dep. |
|---|---|---|---|---|---|---|
| F0-01 | Modelo de settings (pydantic-settings): `DSAGENT_HOME`/data dir absoluto, rutas de cartridge, `prices.yaml`, UI dist, `PORT`; precedencia CLI > env > archivo; cargar `.env` | `cli.py:42-43,246,286`, `serve.py:178,325-344`, `pricing.py:123-130`, `models.py:19` | S | ambas | r2 commercial B2/B3, scale (12-factor), durability §5 | — |
| F0-02 | Dockerfile del harness (API + SPA construida) + `docker-compose` single-node; `ui/dist` y `cartridges/` como package data en el wheel | `pyproject.toml` (`[tool.hatch.build.targets.wheel]`), nuevo `Dockerfile`, `compose.yaml` | M | SH | r2 commercial B1/B2, scale (build/release) | F0-01 |
| F0-03 | El loader **no escribe** en el dir del cartridge: `generate_command_skills` a un cache dir o en install; solo `cartridge validate` escribe; cartridge dir read-only para el usuario del servicio | `cartridge/loader.py:199-248,281-282`, `cli.py:72` | S | ambas | r2 commercial B4, security 6.4 | — |
| F0-04 | CI (pytest + `npm run build` + license check + `pip-audit`/`npm audit`); lockfile Python con hashes (uv/pip-tools); digests de imagen | nuevo `.github/workflows/`, `pyproject.toml` | S | ambas | r2 commercial F, security 6.2 | — |
| F0-05 | Versión en el cliente + `If-Match`/412 en `PUT /sessions`; `updated_at` lo pone el servidor | `sessions.py:68-83`, `api_projects.py:93-116`, `ChatCenter.tsx:299-317` | S | ambas | r2 durability §2.2/§7; r1 arch H1 | N-02 |
| F0-06 | `schema_version` en `run.json`, session JSON, `project.json`, chart records; loaders tolerantes (`extra="ignore"`, defaults); rechazar versión más nueva; fixture por versión en tests | `runner/runner.py:267-271,298`, `runner/charts.py:457,493`, `sessions.py`, `projects.py` | S | ambas | r2 durability §2.5/§7, premortem #12 | — |
| F0-07 | `dsagent migrate` idempotente con `--dry-run`: migración 0001 importa el layout `.dsagent/` actual incl. la adopción de `default`/`serve`; elimina `projects/sessions.json` huérfano | nuevo `cli.py` comando | M | ambas | r2 durability §7, commercial B5 | F0-01, F0-06 |
| F0-08 | Checkpoints **fuera** de cualquier workspace; el workspace de chat de `default` ≠ workspace del serve; `large_tool_results` como tmp por turno y prefijo denegado en rutas de archivos | `serve.py:364-405,189-192,489-511` | S | ambas | r2 durability §1/§8, security 3.1 | F0-01 |
| F0-09 | ULIDs para run/project/session (nunca elegidos por el cliente ni adivinables); `new_run_id` atómico | `api.py:143-157`, `projects.py:45-47`, `ui/src/lib/conversations.ts:15` | S | ambas | r2 security 1.2, tenancy (authz), premortem #10 | — |
| F0-10 | `DELETE /projects/{id}` con cascade; runs terminados borrables (trash + purge, auditado — reemplaza el 409 de `api.py:246-268`); borrado de dataset purga las copias sembradas; manifest de datasets con sha256/bytes; `seed_run` solo si el manifest cambia; quitar `plan.profile.preview` | `api_projects.py`, `api.py:246-268,636-643`, `datasets.py:45-66`, `serve.py:271` | M | ambas | r2 durability §5/§8, premortem #8, scale #6 | F0-06 |
| F0-11 | Lock alrededor de `RunState.save` (1 objeto, 2 hilos); mailbox de comandos para mutaciones post-start (stop/gate/amend) en lugar de load-modify-save desde HTTP | `runner/runner.py:504-507,283-292`, `api.py:774-787`, `driver.py:340-364` | M | ambas | r2 durability §2.6/2.7, premortem #6 | — |
| F0-12 | Logging estructurado JSON a stdout con `tenant/project/session/run/step/request_id`; `driver.log` real bajo serve; `SessionStore` loggea en vez de `pass`; errores persistidos en `run.json` | `driver.py:133`, `serve.py:305-308`, `sessions.py:33-34,61-62,82-83,102-103`, `runner.py:372` | S | ambas | r2 scale R4, premortem #13; r1 premortem#7 | — |
| F0-13 | `/healthz` (proceso) y `/readyz` (DB/checkpointer/creds de modelo/docker/heartbeat de worker) | `serve.py:298-300` | S | ambas | r2 scale I4, security 7.3 | — |
| F0-14 | Handler SIGTERM: dejar de aceptar runs, marcar en curso `interrupted` con el step a rehacer, cerrar envs y contenedores, salir antes del grace period; reaper de leases expirados; `timeout_graceful_shutdown` en uvicorn | `serve.py:158-169`, `cli.py:339`, `driver.py`, `runner.py:516-519` | M | ambas | r2 scale B4/#2 | N-12 |
| F0-15 | LRU/idle-TTL para `chat_envs`/`chat_agents` con `close()` en desalojo; `chat_agent_for` en threadpool con lock por proyecto; single-flight por proyecto alrededor de `kernel.run` | `serve.py:254-272,287-296`, `envs/kernel.py:98-101` | M | ambas | r2 scale #1/#3/#4; r1 scale#1/#3/#4 | — |
| F0-16 | `list_project_artifacts` con `os.walk` podando skip-dirs + cache por mtime; `list_for` con índice de cabeceras separado de mensajes; polls pausados con `document.hidden` y con backoff | `serve.py:514-533`, `sessions.py:85-104`, `ChatCenter.tsx:145-190`, `Sidebar.tsx:50-60`, `RunDock.tsx:25-33` | S | ambas | r2 scale #5/#7/#9 | — |
| F0-17 | Venv por workspace (o `PIP_REQUIRE_VIRTUALENV` + site-packages read-only) para que un `pip install` del modelo no toque el intérprete del servidor; aplicar `tools:` del frontmatter de personas; ruta `/skills/` read-only + re-materializar en cada step | `envs/kernel.py:108`, `cli.py:126-151`, `host/build.py:76-80,152-173,232` | S | ambas | r2 security 3.1/3.4/3.1b | N-01 |
| F0-18 | Redacción de patrones de secreto en tool results (middleware); tratar el JSON de sesión como dato de cliente | `host/build.py`, `sessions.py` | S | ambas | r2 security 3.5 | N-01 |
| F0-19 | Reescribir `README.md` y `ui/README.md` como docs de instalación (hoy describen la UI Next.js borrada); `SECURITY.md`; `CHANGELOG.md`; matriz de soporte | raíz, `ui/` | S | ambas | r2 commercial B8/B10/F | F0-02 |
| F0-20 | Error boundary en la SPA; feedback visible al usuario en fallos de guardado/borrado/creación/subida (hoy `.catch(()=>{})`) | `ui/src/main.tsx`, `App.tsx:106`, `ChatCenter.tsx:314`, `ProjectSwitcher.tsx:37`, `DatasetsSection.tsx:24,34` | S | ambas | r1 ui M2/M6 | — |
| F0-21 | Backup nocturno documentado de `.dsagent/` y restore ensayado (interino hasta F1-05) | docs / compose | S | SH | r2 durability §4 | F0-01 |
| F0-22 | Piloto pagado con un design partner (VM dedicada tras oauth2-proxy + TLS, BYOK con tope de gasto, "un dominio de confianza por VM" documentado) | — | — | SH | r2 commercial (c), tenancy Fase 0 | N-*, F0-01..04 |

---

## F1 — Fase 1: primer cliente self-host que paga (6–10 semanas)

| id | Tarea | Archivos | Esf. | Ruta | Fuente | Dep. |
|---|---|---|---|---|---|---|
| F1-01 | Auth OIDC (o cabecera de reverse-proxy) → `Principal{tenant_id, user_id, roles}` inyectado en cada ruta y en `configurable`; rol *approver* para gates; propiedad tenant/proyecto verificada en **cada** ruta y tool; `locate_run(tenant, id)`; listados scoped | `serve.py`, `api.py:165-176`, `api_projects.py`, `projects.py:77-93`, `runner/tools.py` | L | ambas | r2 security 1.1/4.1, tenancy (identity/authz) | N-10, F0-09 |
| F1-02 | Transcripts **del servidor**: el cliente envía solo el turno nuevo; el servidor añade lo que streameó; `session_events` append-only con `seq`/`version`; 409 con eventos faltantes para pestañas stale; thread→tenant vinculado en servidor; ids opacos | `serve.py:286-296`, `api_projects.py:93-116`, `sessions.py`, `ChatCenter.tsx` | M | ambas | r2 security 1.4, durability §7 | F1-01, F1-04 |
| F1-03 | **`kind: kernel` prohibido bajo `serve`** (solo con `--insecure-local`): ipykernel *dentro* del contenedor endurecido, uno por proyecto-sesión (chat) y por run (workflow), con uid propio, egress solo por proxy allowlist, IMDSv2+hop-limit, API inalcanzable desde la red del sandbox; `run.json`/`events.jsonl` escritos desde el lado API, fuera del árbol escribible por el sandbox | `envs/kernel.py`, `envs/docker.py`, `envs/base.py` (`SandboxBackendProtocol`), `runner/runner.py:100-110` | L | ambas | r2 security 3.1/5.x/(b), tenancy (compute), scale B2, premortem #1 | N-01, N-12, DEC-04 |
| F1-04 | Postgres para metadata (tenants, users, projects, datasets, sessions+session_events, runs, run_steps, run_events, charts, artifacts, audit_log, deletion_jobs) con `tenant_id` y `schema_version`; SQLite tras los mismos modelos SQLAlchemy para single-node; Alembic | nuevo `src/dsagent/db/`, `sessions.py`, `projects.py`, `runs.py` | L | ambas | r2 durability §7, scale (state inventory), tenancy | F0-06, F0-07 |
| F1-05 | Object storage (S3/minio) con prefijo por tenant y KMS por tenant: datasets content-addressed (inmutables; los runs referencian por id, nunca copian), workspaces sincronizados al fin de cada step, artifacts por sesión, gate-versions; signed URLs; artifacts en **origen separado** | `datasets.py`, `serve.py:408-437,496-511`, `api.py:543-610` | M | ambas | r2 durability §7, scale R1/#3, security 1.3 | F1-04 |
| F1-06 | `PostgresSaver`/`AsyncPostgresSaver` para ambos grafos (agentes construidos en lifespan); podar hilos de checkpoint al terminar; TTL de checkpoints de chat | `serve.py:364-405,265`, `driver.py` | M | ambas | r2 scale B3, durability §7; nota: reintroduce copia durable → se resuelve con F1-02 (transcript del servidor) | F1-04 |
| F1-07 | **Tier de workers**: `dsagent worker` reclama runs (Postgres `SKIP LOCKED` en Fase 1), heartbeat/lease, llama a `WorkflowRunner` directo (camino CLI) con `on_event`→stream y `ask_human` = persistir gate + `GatePause`; `POST /runs/{id}/gate` escribe la decisión y encola resume; stop = flag en DB; **borrar** el envoltorio de grafos de `driver.py` y `_walk_past_decided_gates`; `run_workflow` del chat solo encola; `QueueDriver` tras el protocolo `Driver` | `driver.py`, `runner/runner.py` (`_stopped`, hooks de storage, `ask_human`), `runner/tools.py:149-164`, `api.py:371-413` | L | ambas | r2 scale B1/#6, tenancy (jobs), premortem #9 | F1-04, F1-05 |
| F1-08 | Eventos y control fuera del filesystem: `run_events` en DB con SSE por offset (o Redis Streams); un canal `/projects/{p}/stream` para cambios de lista; UI deja de hacer polling (fallback) | `api.py:419-465`, `runs.py:162-184`, `ui/src/lib/api.ts:133-152` | M | ambas | r2 scale R6, tenancy (streaming) | F1-04, F1-07 |
| F1-09 | Ledger de uso: callback LangChain en **cada** llamada al modelo (chat, subagentes y runs, incl. steps matados) + cpu/gpu-segundos y bytes desde el lease del sandbox, por tenant/proyecto/run/step; tabla de precios versionada multi-modelo con margen; tope de gasto por tenant **antes** de invocar | `serve.py:286-296`, `runner.py:725-735,1034-1055`, `pricing.py`, `prices.yaml` | M | ambas | r2 commercial A2/A3/A4, scale R5, premortem #6 | F1-04 |
| F1-10 | Cuotas y límites: runs concurrentes, kernels, GPU-horas, storage, tokens/$, rate en `/agent` y uploads; admission control antes de encolar | nuevo middleware | M | ambas | r2 commercial A5, security 1.5, tenancy (quotas) | F1-09 |
| F1-11 | Audit log append-only escrito por la capa API con actor/tenant/ip/request-id para cada mutación y cada turno; gates firmados con actor; enviado fuera de la caja (SIEM); hash-chained | `api.py:397-413`, `runner.py:879-880`, nuevo módulo | M | ambas | r2 security 7.1, commercial A8 | F1-01, F1-04 |
| F1-12 | Erasure y export por tenant/proyecto con job y recibo (recorre runs, gate-versions, sessions, workspaces, checkpoints; mata kernels); retención configurable (defaults en durability §7); cifrado en reposo; `dsagent export/import project` | nuevo, `api_projects.py` | M | ambas | r2 durability §5/§7, security 7.2, premortem #8 | F1-04, F1-05 |
| F1-13 | Kill/suspend por run y por tenant (matar contenedor/kernel en curso, no solo entre steps); GC de sandboxes al arrancar; métricas Prometheus (runs por estado, gate wait, duración de step, tokens por modelo, sandboxes vivos) + alertas de gasto; OpenTelemetry | `api.py:371-395`, `runner.py:583-604`, `serve.py` | M | ambas | r2 security 7.3, scale R4 | F1-07 |
| F1-14 | UI desde el cartridge: bloque `ui:`/`branding` en `cartridge.yaml` (nombre, roster, sugerencias, placeholder, locale, paleta/logo) servido por `/cartridges`; diccionario i18n en/es; empty state desde el roster; quitar la paleta "Aiuda" hardcodeada en `export.py` | `ChatCenter.tsx:126-131,363-364,392`, `Launcher.tsx:146-166`, `App.tsx:140`, `ui/index.html:6`, `export.py:37-72`, `cartridge/models.py`, `api.py:178-188` | M | ambas | r2 commercial D, premortem #14; r1 arch L1 | DEC-03 |
| F1-15 | Spec de cartridge versionada (`cartridge_format: 1`), JSON Schemas para `cartridge.yaml`/`workflow.yaml`, `dsagent cartridge new`, guía "write your first cartridge"; alinear `architecture.md:83-109` con lo que `loader.py:92-102` lee; resolver cartridge+env desde `Project.cartridge` (versionado, inmutable); modelo allowlist | `cartridge/loader.py`, `cartridge/models.py`, `projects.py:38`, `serve.py:252`, `docs/architecture.md` | M | ambas | r2 commercial B6/B7, tenancy (configuration) | — |
| F1-16 | Segundo cartridge de referencia (`cartridges/sdlc`) — prueba M3 y elimina el literal `"ds-"` de `loader.py:79`; chat env por cartridge (no solo `cartridges[0]`) | `cartridges/sdlc/`, `cartridge/loader.py:79`, `serve.py:252` | M | ambas | r2 commercial B7, ROADMAP M3 | F1-15 |
| F1-17 | Cartridges firmados: manifest con hashes de scripts/Dockerfiles, allowlist de requirements (`name==version` desde índice interno), carga de cartridge = acción admin auditada; nada se ejecuta en el host API | `cli.py:121-160`, `host/build.py:122-123`, `runner.py:964-966`, `docker.py:243-256` | M | ambas | r2 security 6.1 | F1-03, F1-11 |
| F1-18 | Imágenes de cartridge construidas en CI y pulled de un registry — nunca en el request/run path | `envs/docker.py:243-256` | S | ambas | r2 scale (build/release), tenancy (cartridge) | F0-04 |
| F1-19 | Tests de frontend (hoy cero) y tests de contrato sobre el bridge AG-UI (`add_langgraph_fastapi_endpoint` copiado, `globals()` hack → reemplazar por parseo manual de `request.json()`; payload de gate `metadata.langgraph.raw`) | `ui/`, `serve.py:278-296,77-94` | M | ambas | r2 commercial F; r1 arch M2 | F0-04 |
| F1-20 | Releases semver, CHANGELOG, `SECURITY.md`, política de disclosure, protección de rama, releases firmados; DPA + lista de subprocesadores; runbook de IR | repo | S | ambas | r2 commercial B10/F, security 2.2/7.4 | — |
| F1-21 | Compose/Helm endurecido (api + worker + postgres + minio + redis) con los controles anteriores activados por defecto; TLS por reverse proxy documentado | `compose.yaml`, `charts/` | M | SH | r2 tenancy (self-host variant), scale (topology) | F1-03..08 |
| F1-22 | Prompt injection (mitigaciones): provenance-tag de contenido no confiable (datasets, reportes previos, `read_run_file`); gate humano en cualquier efecto externo; notas de gate autenticadas; quitar control chars/markup de inputs; los números del data-gate los calcula el script, no la persona | `runner.py:621-649,632-641,1125-1148`, `steps/02-data-gate.md`, `api.py:409` | M | ambas | r2 security 2.3, premortem #4 | F1-01, F1-03 |
| F1-23 | Resumabilidad de steps largos: el cartridge checkpointea su progreso en el workspace (traces de Meridian) y la persona reanuda; sync periódico del workspace durante steps largos; endpoint kill del step en curso; id de contenedor en `run.json` para re-attach tras reinicio | `cartridges/ds/workflows/mmm-meridian/`, `runner.py:557-567`, `docker.py:169` | M | ambas | r2 scale R2, premortem #9 | F1-07 |
| F1-24 | Precio y contrato: licencia anual por instancia + add-ons de cartridge + servicios; declaración RPO/RTO (metadata ≤5 min / 4 h; datasets ≤1 h / 4 h; borrado ≤24 h con recibo) | docs comerciales | — | SH | r2 commercial (c), durability §7 | F1-04, F1-05, F1-12 |

---

## F2 — Fase 2: SaaS / escala (trimestres; solo tras 2–3 clientes self-host)

| id | Tarea | Esf. | Fuente | Dep. |
|---|---|---|---|---|
| F2-01 | `tenant_id` en cada fila, ruta, métrica y prefijo de storage; sin proyecto `default` en SaaS; provisioning de tenants/onboarding; admin API/UI (lista de tenants, uso, kill switch) | L | r2 tenancy (target), commercial A9 | F1-01, F1-04 |
| F2-02 | Pool de sandboxes microVM (Firecracker/Kata/gVisor) o proveedor hosted; warm start; TTL idle; caps por tenant; `SandboxBackendProtocol` (`kind: k8s`/`modal`) | XL | r2 security (b), scale step 4, tenancy | F1-03, DEC-04 |
| F2-03 | Metering → billing (Stripe) sobre el ledger; planes; BYOK/KMS; proveedor y región por tenant; rate card por tenant | M | r2 commercial A, scale R5 | F1-09, F1-10 |
| F2-04 | SSO/SCIM; RBAC completo (viewer/approver/admin); gates firmados; residencia de datos; legal hold | L | r2 security 7.4 | F1-01, F1-11 |
| F2-05 | Fan-out de eventos entre réplicas (DB notify / Redis); API stateless N réplicas; lane GPU con scheduling (`nvidia.com/gpu: 1`) y cola; imágenes desde registry | L | r2 scale (topology), premortem #10 | F1-07, F1-08 |
| F2-06 | Registry de cartridges firmado, content-addressed; builds remotos (Kaniko/BuildKit) con límites; cartridges de tenant solo dentro de su sandbox | M | r2 tenancy (cartridge), security 6.1 | F1-17, F2-02 |
| F2-07 | Materialización de workspaces por reflink/CSI en vez de copias; compactación de checkpoints; tier frío; cuotas de storage y atribución de coste; drill de restore automatizado que falla CI si se incumple RTO | M | r2 durability Fase 2 | F1-05 |
| F2-08 | Defensas de prompt injection nivel 2 (texto de datasets en cuarentena, políticas por tenant); SOC 2 Type I→II (gestión de cambios, vulnerabilidades, retención de logs, pen test, vendor review) | L | r2 security Fase 2 | F1-22, F1-20 |

---

## Fuera de alcance / notas

- **No** reintroducir un checkpointer durable para el chat sin `F1-02` (transcript del servidor): con el store actual crearía una segunda copia durable que puede divergir (r1 arch H2, premortem). Con `F1-02` + `F1-06` la copia autoritativa pasa a ser el servidor y el problema desaparece.
- Las verificaciones "seguras" (path traversal, sanitización de ids, DOMPurify, export CSP) no necesitan trabajo; no re-gastar esfuerzo.
- El documento `docs/startup-readiness-review-2026-09-18.md` (independiente, mismo commit) llega a conclusiones compatibles con esta ruta; consultarlo para el framing por etapa (demo / piloto / paid local / SaaS).
