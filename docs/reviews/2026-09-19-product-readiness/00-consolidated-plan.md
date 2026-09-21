# Plan consolidado — product readiness (2026-09-19)

Síntesis única y deduplicada de los 6 agentes de la ronda 2 (listón comercial) más los 5 de
la ronda 1 (listón local). "Quién" = cuántos de los 6 agentes de la ronda 2 señalaron el
mismo punto de forma independiente. La evidencia completa (`file:line`) está en `round2/` y
`round1/`.

---

## Veredicto (unánime, 6/6)

**Hoy no es vendible ni como SaaS ni como self-host.** Es un appliance de un solo nodo, bien
construido en su núcleo, cuya frontera de seguridad es literalmente *"confía en el modelo y en
cualquiera que alcance el puerto 8000"*.

**La causa raíz de casi todo** (la "suposición más peligrosa del diseño"): el env de
ejecución **no es un sandbox**. Es un subproceso del servidor API con su uid, su intérprete,
sus variables (**API key incluida**), su red (IMDS, servicios internos y **la propia API sin
auth** — el modelo puede aprobarse su propio gate con un `curl` a `localhost:8000/runs/{id}/gate`,
rompiendo el invariante #4 desde dentro), su venv (un `pip install` del modelo **envenena el
servidor para todos**) y su socket de Docker (→ **root en el host**).

**Verificado empíricamente** por `saas-security` con una sonda inofensiva usando la
configuración exacta de `envs/kernel.py:87`: desde dentro del "virtual root", `ls /` y
`ls ~/.ssh` funcionaron, una variable de entorno del servidor se devolvió, y escribir en el
mount "read-only" de skills tuvo éxito.

## Decisión estratégica recomendada (convergente en 3 agentes)

**Self-host primero, vendido como solución, no como plataforma.**

- SKU inicial: **"DS/MMM team in a box"** — harness + cartridge DS + UI en el VPC del cliente,
  **con las keys del cliente (BYOK)**. Los blockers de self-host son packaging/config/docs
  (S–M); los de SaaS son arquitectónicos (L–XL). Lo que el código hace bien (run = directorio
  autocontenido, Docker por run, traza de auditoría, cero phone-home) es exactamente lo que
  valora un comprador on-prem y exactamente lo que un SaaS tendría que rediseñar.
- **No vender "harness domain-agnostic"** hasta tener un segundo cartridge y la UI alimentada
  por `/cartridges`. Hoy la UI hardcodea Marie/Noel/Ana/Pablo y copy en español; un comprador
  no-DS **no puede desplegar sin editar TSX**. *"Hasta entonces es una afirmación, no un
  producto."*
- **Licencia — decisión urgente**: las **258** dependencias (Python + npm) son permisivas
  (MIT/BSD/ISC/Apache; dompurify MPL-2.0 OR Apache-2.0 → elegir Apache; cero AGPL/SSPL/BSL).
  Nada bloquea SaaS ni reventa. Pero **no hay archivo `LICENSE`** mientras `pyproject.toml:9`
  declara `MIT` — legal de cualquier comprador enterprise se frena ahí, y MIT-en-metadata ya es
  discutiblemente una concesión. Recomendación: **open-core**. Faltan los NOTICE /
  `THIRD_PARTY_LICENSES` obligatorios. Verificar fuera del repo: términos de reventa de output
  del proveedor de modelo (BYOK lo esquiva) y la licencia de `google-meridian` si algún día se
  envía una imagen precompilada.
- **SaaS = Fase 2, solo tras 2–3 clientes self-host pagando.** Test de diseño: *el runner y el
  formato de cartridge no deberían cambiar para llegar a Fase 2.*

## Bloqueantes innegociables (consolidados)

| # | Bloqueante | Evidencia clave | Quién |
|---|---|---|---|
| 1 | **El sandbox no es sandbox** — kernel = servidor (`inherit_env=True`, ipykernel hereda `os.environ`, `python=sys.executable`); API key en env; socket de Docker → root; pip al venv del server; **el log de auditoría (`run.json`/`events.jsonl`) es escribible por el código auditado**; Docker sin `--network/--cap-drop/--memory/--pids/--rm`; **escape de lectura host→contenedor vía symlinks** que siguen `zip/export/_keep_version/matched`; skills mount escribible desde las tools host-side; `tools:` del frontmatter de personas no se aplica; `Step.timeout` nunca se lee | `envs/kernel.py:87,29-33,108`, `envs/docker.py:108-125`, `api.py:598`, `runner.py:928`, `host/build.py:76-80,157,232`, `cartridge/models.py:60` | 6/6 |
| 2 | **Sin identidad ni tenant**; el cliente elige `thread/project/run` id; run ids adivinables; `locate_run` cruza todos los proyectos; **todos los runs del launcher corren por el orquestador del proyecto `default`** (fuga de cómputo entre proyectos hoy); tools de runs/amend atadas al `runs_dir` default; hilo no registrado → `default`; DNS rebinding sin TrustedHost; **CSRF por `<form enctype=text/plain>`** al gate (`request.json()` ignora Content-Type); cualquier cliente puede re-`PUT` cualquier session en cualquier proyecto | `serve.py:171,288,305-308`, `api.py:764,151,165-176`, `projects.py:77-93`, `tools.py:178-207`, `charts.py:461`, `api_projects.py:93-116` | 6/6 |
| 3 | **Secretos**: una key global → sandbox → tool results → transcripts; sin key/presupuesto/rotación por tenant | `README:30`, `models.py:19`, `sessions.py:68` | 4/6 |
| 4 | **HTML del modelo en el origen de la app sin CSP** en `/runs/{id}/files` y `/projects/{id}/files`; el prefijo sandboxeado `/runs-x` **existe pero la UI nunca lo usa** (`grep runs-x ui/src` vacío) | `serve.py:478-553`, `FileViewer.tsx:39-46`, `ChatCenter.tsx:196-202` | 4/6 |
| 5 | **Pérdida/corrupción silenciosa**: `SessionStore.save` no atómico + traga `OSError` + responde 200; el cliente hace `PUT []` ante cualquier error no-404 (cadena: torn → 404 → wipe permanente); last-write-wins sin versión (dos pestañas pierden turnos); `run.json` con 3 escritores load-modify-save + 1 objeto en 2 hilos; `RunState.load` = `cls(**d)` frágil en ambos sentidos; **dataset truncado con nombre correcto sembrado en todos los runs y analizado sin error**; `project.json` torn → proyecto desaparece con sus runs; `events.jsonl` sin fsync | `sessions.py:79-83`, `ChatCenter.tsx:232-240`, `datasets.py:28-32,53-66`, `runner.py:271,504-507`, `api.py:774-787` | 4/6 |
| 6 | **GDPR imposible**: de ~12 copias de un dataset, borrar limpia **1**; no existe `DELETE /projects`; runs iniciados imborrables (409); `large_tool_results/` con filas crudas **se sirve por HTTP** y nunca se limpia; `checkpoints.sqlite` vive **dentro del cwd del kernel**; `plan.profile.preview` = 20 filas crudas en `run.json`; `args_preview` en `events.jsonl` | `api.py:246-268,636-643`, `serve.py:393,489-511`, `runner.py:1066-1081` | 3/6 |
| 7 | **Runs dentro del proceso API** (hilos daemon); los del chat **dentro del request SSE** (cerrar la pestaña deja el workflow colgando); sin cola/lease/retry/timeout; un fit GPU muerto a las 3h50 se **re-paga entero** (personas con `checkpointer=False`); contenedores `sleep infinity` huérfanos en crash con nombres aleatorios (no se pueden reapear); `stop` solo entre steps | `driver.py:310-316`, `tools.py:164`, `host/build.py:172`, `docker.py:169,227-240` | 4/6 |
| 8 | **No se puede facturar**: tokens solo por step de workflow, **cero en chat y en subagentes**; step matado = 0 tokens registrados; `prices.yaml` con un modelo; sin GPU/cómputo/storage; sin cuotas ni tope de subida; sin dueño en `Project` | `runner.py:725-735,1034-1055` vs `serve.py:286-296`, `pricing.py:98-108`, `projects.py:34-39` | 4/6 |
| 9 | **No instalable**: sin Dockerfile/compose/Helm del harness; **el wheel excluye `ui/` y `cartridges/`**; rutas relativas al cwd (arrancar desde otro dir **crea un segundo store vacío**); `ui/dist` localizado por `__file__`; el loader **escribe en el dir del cartridge en cada carga** (crash en read-only); sin modelo de settings; `.env` no lo carga nada del harness | `pyproject.toml`, `cli.py:42-43,286`, `serve.py:178,337-344`, `pricing.py:123`, `loader.py:199-248` | 2/6 |
| 10 | **Muro de escalado**: kernels + `InMemorySaver` en proceso y nunca desalojados (OOM); SQLite local; `chat_agent_for` bloquea el event loop (build de env 60s, `seed_run` por turno); un kernel compartido por turnos concurrentes; 2 réplicas rompen chat/gates/streams; `new_run_id` no atómico entre procesos | `serve.py:254-272,393`, `kernel.py:98-101`, `api.py:151-157` | 3/6 |
| 11 | **Cero observabilidad ni actor**: `driver.log` no-op bajo serve; catches tragados en todas partes; sin request id; sin *quién* en ningún evento (ni en respuestas de gate); health = constante; sin kill switch | `driver.py:133`, `runner.py:372,879`, `serve.py:298-300` | 5/6 |
| 12 | **Madurez**: 0 tests de frontend; sin CI; sin lockfile Python; sin CHANGELOG/SECURITY.md; READMEs describen la UI Next.js borrada; SDK de cartridge sin versión y **derivando** (`architecture.md:83-109` documenta `personas.*.skills`/`extends_plugin` que `loader.py:92-102` no lee); literal `"ds-"` en `loader.py:79`; no existe segundo cartridge; `globals()` hack + cuerpo copiado de `add_langgraph_fastapi_endpoint` | `loader.py:79`, `serve.py:278-296` | 3/6 |

**Crecimiento cuantificado** (1 cliente: 10 runs/día, 30 turnos/día, dataset 50 MB):
~2.500 copias idénticas/año → **125 GB/año** (1,25 TB con 500 MB); `seattle-weather.csv` ya
existe **20 veces** en `.dsagent/`; cada `POST /agent` recopia todo (1,5 GB/día de I/O por
proyecto); transcripts con `data.values` inlineado **reenviados enteros al modelo cada turno**
(storage y factura escalan juntos); `InMemorySaver` snapshotea el estado completo en cada
super-step → OOM en meses.

## Lo que está bien — y se conserva

- **El runner** (`runner/runner.py`): DAG declarado sobre `needs`, gates como interrupts
  estables entre resume/re-entrada, `produces` verificado en disco con globs, auto-gate
  scripts materializados, telemetría por step, resume desde `run.json` atómico. *Es el moat
  real.* Compatible con workers + object storage con **3 cambios pequeños** (`_stopped`,
  hooks de storage alrededor de `env_for`/fin de step, un `ask_human` que pausa); el gate se
  persiste *antes* de preguntar. Invariantes #4/#5/#6 intactos tras los cambios recientes.
- **El contrato de cartridge** (plugin de Claude Code + `cartridge.yaml`, matriz de skills
  validada) — el producto de extensibilidad, aún sin versionar.
- SSE hace tail de archivo (no de memoria) → ya es agnóstico a réplicas con storage
  compartido. `Driver` es un protocolo → un `QueueDriver` encaja sin tocar la API. La
  telemetría por step atribuible a run→proyecto es la semilla correcta del medidor de billing.
- Verificados **seguros** (no re-gastar esfuerzo): path traversal en `resolve_run_file` /
  `resolve_project_file` (resolve-then-`is_relative_to`, symlinks y `%2e%2e` cubiertos),
  `safe_name`/`_safe_id`/`_safe`, DOMPurify en markdown, export con CSP + attachment,
  gate-version path check, uid de contenedores, parquet preview con pandas.

## Ruta a la primera venta

### Fase 0 — demo-safe / design partner (2–6 semanas, casi todo S)

Una VM por partner detrás de oauth2-proxy/Cloudflare Access con TLS; BYOK con tope de gasto;
decirle claro que sus usuarios comparten un dominio de confianza y que el código del modelo
corre en esa VM. Backup nocturno de `.dsagent/`. **Un piloto pagado** (fee de piloto o
servicios, sin fee de licencia aún).

Código (todo en `BACKLOG.md` como `N-*` y `F0-*`): env limpio para shell y kernel; venv por
workspace; bearer token en todas las rutas incl. `/agent` y SSE; rechazar `--host` no-loopback
sin `--insecure`; TrustedHost + Origin + Content-Type; CSP `sandbox` + `nosniff` en ambas rutas
de archivos y UI → `/runs-x`; helper anti-symlink; hardening Docker; aplicar `Step.timeout`;
tope de body; escrituras atómicas + `save` que falla → 500; cliente registra solo con 404 real;
`If-Match`/versión; `schema_version` + loaders tolerantes; `DSAGENT_HOME`; checkpoints fuera
del workspace; ULIDs; `DELETE /projects` + runs terminados borrables + borrado de dataset que
purga copias + manifest sha256 + seed solo si cambia; `store` en tools/preview y env por
proyecto en el driver; guard `live` en `App.tsx`; LICENSE + THIRD_PARTY_LICENSES + decisión;
Dockerfile (API+SPA) + compose; settings model; `ui/dist` en el paquete; no escribir en el
cartridge; CI + lockfile; READMEs; logging estructurado + `/healthz`,`/readyz`; SIGTERM que
cierra envs y marca runs `interrupted`.

### Fase 1 — primer cliente self-host que paga (6–10 semanas más)

OIDC + propiedad tenant/proyecto en cada ruta y tool; ids del servidor y transcripts *del
servidor*; **`kind: kernel` prohibido bajo `serve`** → ipykernel *dentro* del contenedor, uno
por proyecto/run, egress por proxy allowlist, IMDS inalcanzable; Postgres para metadata (SQLite
tras los mismos modelos en single-node) + object storage con datasets content-addressed;
`PostgresSaver`/`AsyncPostgresSaver` en ambos grafos (agentes construidos en lifespan); **tier
de workers** (`dsagent worker` llama a `WorkflowRunner` directo por el camino del CLI, pausa en
gate, **borra** el envoltorio de grafos de `driver.py` y `_walk_past_decided_gates`); eventos y
stop fuera del filesystem; ledger de uso incl. chat y subagentes; UI desde cartridge (roster,
sugerencias, placeholder, marca, i18n en/es) vía un bloque `ui:` en `cartridge.yaml`;
`cartridge_format: 1` + JSON Schemas + `dsagent cartridge new` + guía + **segundo cartridge**
(`sdlc`); audit log append-only con actor/ip/request-id enviado fuera de la caja; erasure/export
con recibo; retención; cifrado en reposo; kill/suspend + GC de sandboxes + métricas + alertas de
gasto; releases semver, CHANGELOG, SECURITY.md, matriz de soporte; compose/Helm endurecido;
DPA + lista de subprocesadores; runbook de IR. Precio: licencia anual por instancia + add-ons
de cartridge + servicios.

### Fase 2 — SaaS / escala (trimestres; solo tras 2–3 clientes self-host)

`tenant_id` en cada fila, ruta y métrica; pool de sandboxes microVM (Firecracker/Kata/gVisor)
con warm start, TTL idle y caps por tenant; cuotas → metering → billing (Stripe) sobre el ledger
de Fase 1; BYOK/KMS, residencia y proveedor por tenant; SSO/SCIM, RBAC con rol *approver* y
gates firmados; defensas de prompt injection (provenance de contenido no confiable, gate humano
en cualquier efecto externo); registry de cartridges firmado con builds remotos; fan-out de
eventos entre réplicas; lane GPU con scheduling; SOC 2 Type I→II; regiones de residencia.

## Lo que se haría ya, sin esperar a vender nada

Todos son S, y cuatro son regresiones de la sesión del 2026-09-18: `inherit_env=False` +
kernel con env limpio; escrituras atómicas + no `PUT []`; `store` en las tools + env por
proyecto en el driver; CSP/`nosniff` + UI a `/runs-x`; TrustedHost/Origin/Content-Type +
bearer; guard `live` en `App.tsx`; `LICENSE`. Es el núcleo de la Fase 0 y cierra las heridas
abiertas hoy. Ver `BACKLOG.md` → `N-01`…`N-09`.

## Síntesis de la ronda 1 (listón local) — para referencia

La ronda 1 juzgó el código como herramienta local. Sus hallazgos siguen siendo válidos y en su
mayoría fueron absorbidos y re-clasificados arriba; se conservan por su detalle de causa raíz
(sobre todo UI):

- **P0**: `SessionStore` no atómico + error tragado (premortem#1, arch H1); `getSession` borra
  chat existente ante error no-404 (ui C1, regresión); tools de runs ignoran `store` (arch
  C1+M1, regresión); HTML del modelo mismo-origen sin CSP (security#2, ui H2, regresión); DNS
  rebinding (security#1); carrera en `App.tsx:59-76` sin guard `live` — causa raíz del tipeo
  perdido y "New chat" que no cambia (ui H1, regresión).
- **P1**: kernels nunca desalojados (scale#1); dos pestañas last-write-wins (arch H1/H2);
  registro de chat nuevo fire-and-forget → primer turno en `default` (arch H3); `chat_agent_for`
  bloquea el loop (scale#3); `seed_run` por turno (scale#6); keepalive pierde el último turno de
  chats >64KB (ui H3); `.clone()` comparte kernel (scale#4).
- **P2**: sin observabilidad; sin retención/GC; `rglob` cada 5s; kernels huérfanos en crash;
  bump de deps brickea runs/chats (sin `schema_version`); race de `run.json`; gate de chat
  perdido al reiniciar; env `default` compartido chat/driver.
- **P3**: hack `globals()`; UI hardcodea DS; ChartView re-embebe; a11y; móvil; cap de subida;
  truncado de session id a 128.

Nota honesta: la afirmación documentada en `serve.py:231-251` ("fuente única de verdad… siempre
converge por IDs estables") fue **refutada** por arch H2 y premortem: se apoya en un store
atómico y de un solo escritor, y `SessionStore` no es ninguna de las dos cosas. El comentario
debe reescribirse cuando se implemente `N-02`/`F0-*`.
