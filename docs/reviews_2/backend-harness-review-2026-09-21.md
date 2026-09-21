# Revisión independiente del backend y harness

Fecha: 2026-09-21. Código revisado: `ae4afc4`; durante la revisión HEAD avanzó a `516728c`, que solo añade la investigación documental de Jev. Evaluación solicitada después del cartridge de marketing y de la UI configurable.

## Veredicto

**El desarrollo ha sido bueno como harness experimental: hay decisiones de arquitectura acertadas, pruebas sustanciales y aprendizaje real de ejecuciones. Todavía no constituye un backend listo para operar con clientes sin supervisión.** Conservaría Deep Agents, los cartridges, el runner declarativo y los entornos; invertiría la siguiente etapa en contratos, aislamiento e integridad antes de ampliar capacidades.

La deuda principal no es la elección de framework. Es la distancia entre contratos anunciados —alcance de skills, proyecto propietario, persistencia, límites de ejecución— y lo que el servidor impone en todos los caminos. La UI y el segundo dominio avanzaron más que esas garantías.

No recomiendo una reescritura ni introducir Kubernetes, Redis y múltiples servicios como primer arreglo. Un único servicio y un worker pueden ofrecer un piloto robusto si hay transiciones transaccionales, límites reales y un entorno de ejecución aislado.

## Alcance y evidencia

Leídos `CLAUDE.md`, `docs/architecture.md`, la reconciliación anterior y las implementaciones de loader, host, runner, driver, API, proyectos, sesiones, entornos, recursos y gráficos. Se revisaron las pruebas relevantes y los cambios de presentación/cartridge.

Checks ejecutados sobre el checkout:

| Check | Resultado |
|---|---|
| `.venv/bin/python -m pytest -q` | **385 passed, 12 skipped**, 7 warnings, 27,52 s |
| `.venv/bin/pytest -q` | **Falla la recolección:** 12 errores `ModuleNotFoundError: No module named 'tests'` |
| `.venv/bin/ruff check src tests` | **7 hallazgos:** imports y excepciones amplias/silenciadas |
| Sondas de API con `TestClient`, almacenamiento temporal y driver sin grafo | Confirman colisión de IDs, HTML sin CSP, ZIP fuera del workspace y falsos éxitos de guardado |
| Sondas con `FakeBackend` y agentes sintéticos | Confirman traversal del script, timeout ignorado, telemetría perdida y sustitución de pasos duplicados |

No hubo llamadas a modelos, servicios pagos, Docker real ni acceso a datos de usuarios. El único script ejecutado por la sonda de scope imprimía una marca sintética dentro de un directorio temporal. No se modificó código de producto. Las pruebas omitidas y los dobles no demuestran calidad de respuestas reales, aislamiento del sistema operativo, recuperación frente a SIGKILL ni rendimiento bajo carga.

## Qué está bien construido

- **DAG y agente tienen dueños diferentes.** `WorkflowRunner.run` decide qué paso toca y Deep Agents ejecuta el trabajo dentro del paso. Es una separación valiosa: añadir un dominio no exige inventar otro bucle de herramientas.
- **El cartridge ya cambia la experiencia real.** `Experience` tiene defaults genéricos y rechaza campos desconocidos; marketing declara personas, skills, instrucciones, inputs y gates sin crear un backend paralelo. Las pruebas `test_experience.py` y `test_marketing_cartridge.py` cubren esta parte. Es evidencia de extensibilidad, aunque no de independencia completa.
- **Las revisiones humanas afectan al trabajo.** Un rechazo conserva una versión de artefactos, transmite la nota y obliga a reejecutar el paso; hay regresiones que comprueban el orden de interrupts y evitan que el segundo gate reciba la respuesta del primero.
- **Se han resuelto fallos difíciles del framework.** El `checkpointer=False` de las personas impide heredar accidentalmente el checkpoint de otro paso. El driver tiene un lock al comenzar y distingue errores externos al runner. No son envoltorios superficiales: son integraciones probadas con casos concretos.
- **El registro es útil para depurar.** Eventos, notas atribuidas a pasos, archivos, charts y logs permiten inspeccionar una ejecución y volver a verla. `run.json` usa reemplazo atómico con temporal por proceso/hilo, lo que evita lecturas truncadas en esa ruta.
- **Los entornos son sustituibles y comprobables.** Construcción de comandos Docker separada de ejecución, workspace compartido y auto-gates ejecutados dentro del entorno del paso. Las pruebas pueden verificar contratos sin pagar un modelo.

## Hallazgos prioritarios

### BH-01 — El alcance de skills no es una frontera de ejecución

**Prioridad P0 antes de ejecutar contenido no confiable. Reproducido y confirmado por lectura.**

En `src/dsagent/host/build.py:109` se valida el nombre del skill, pero el parámetro `script` se une a una ruta sin normalización/contención (`:114–123`). Una persona con `reports` puede solicitar `../../../noel/ml/scripts/probe.py` y ejecutar un script situado en la carpeta de un skill no concedido. La sonda devolvió `exit code: 0` y `OUTSIDE_GRANTED_SKILL`.

Además, `_backend` monta la raíz de skills de **todas** las personas (`host/build.py:76`), y el scope de descubrimiento no impide por sí mismo leer rutas de otra persona. `Persona.tools` existe en el modelo (`cartridge/models.py:34`), pero la construcción de agentes usa `env.tools` y herramientas añadidas sin aplicar esa lista (`host/build.py:154`, `:225`). No hay que presentar esos campos como permisos efectivos.

El problema más amplio sigue presente: `LocalShellBackend(... inherit_env=True)` (`envs/kernel.py:87`) y Jupyter (`:30`) ejecutan con los privilegios del proceso local. El root virtual de las herramientas de archivos no es un sandbox de shell/Python. Docker mejora la separación, pero `run_argv` (`envs/docker.py:99`) no declara límites CPU/memoria/PIDs, política de red, reducción de capabilities ni `no-new-privileges`; su mount de skills es de solo lectura, pero contiene todas las personas. No se ejecutó una prueba de escape de Docker en esta revisión.

**Mejora:** separar contexto visible de autorización; resolver/canonicalizar scripts y exigir contención; aplicar una allowlist real de herramientas; definir e implementar el límite de ejecución. La política y registros de control deben permanecer fuera del alcance de código generado.

### BH-02 — HTML y exportaciones mantienen rutas que evaden las protecciones

**Prioridad P0 antes de compartir la instancia. Reproducido.**

`/runs/{id}/files/...` devuelve HTML como `text/html` sin CSP (`serve.py:477–491`). La ruta alternativa `/runs-x` sí impone sandbox, pero no protege la primera URL. La nueva ruta `/projects/{id}/files/...` reproduce el mismo patrón (`serve.py:570–576`). Un enlace directo a un artefacto activo queda en el origen de la aplicación aunque el viewer use un iframe seguro.

La descarga ZIP no reutiliza la contención de `resolve_run_file`: `_archive_members` acepta cualquier `is_file()` (`api.py:759–767`) y `archive.write` sigue symlinks (`:637`). La sonda creó un symlink a **un archivo sintético temporal fuera del workspace**; el ZIP incluyó sus bytes. No se leyeron secretos reales.

Las rutas están montadas sin una capa de identidad/autorización en este app; `TestClient` pudo crear proyectos, leer archivos y guardar sesiones sin credenciales. Una restricción de red externa puede acotar exposición, pero no convierte el proyecto en una unidad autorizada por usuario.

**Límite de evidencia:** se comprobaron cabeceras y bytes con `TestClient`; no se ejecutó JavaScript del artefacto en un navegador ni una explotación del origen/API. La posibilidad de ejecutar contenido activo en ese origen es el riesgo inferido del tipo de respuesta y la ausencia de sandbox en esa ruta.

**Mejora:** una única política de recursos para servir, exportar, previsualizar y leer por herramientas; aislar contenido activo en origen/sandbox apropiado; autenticar y comprobar ownership antes de uso compartido. Probar también SVG y las rutas de chat/proyecto.

### BH-03 — Los proyectos no tienen una identidad de run global y uniforme

**Prioridad P1 de integridad; P0 si hay usuarios/proyectos que deben estar aislados. Reproducido.**

`new_run_id` consulta únicamente el directorio del proyecto y no reserva atómicamente (`api.py:168–182`, `:263–271`). `ProjectStore.locate_run`, en cambio, busca globalmente y devuelve la primera coincidencia (`projects.py:77–93`). Dos proyectos que crean el mismo workflow en el mismo segundo reciben el mismo ID.

Sonda: con reloj fijado, A y B recibieron `eda-to-report-20260921-010101`. `GET /runs/{id}` del run creado para B devolvió `inputs.question = "A"`. También comparten la clave del driver y del checkpoint (`driver.py:293–296`). Es una mezcla funcional de trabajo, incluso sin atacantes ni concurrencia real.

`start`, respuesta de gate y `delete` usan también `run_dir_of`. **Se infiere** que una colisión puede dirigir esas acciones al run equivocado; la sonda no inició, aprobó ni borró un run para demostrarlo. No se presenta ese escenario como una acción observada.

Las rutas de archivos/preview ya aceptan `ProjectStore`, pero las herramientas de conversación `list_run_files` y `read_run_file` siguen usando solo `runs_dir` (`runner/tools.py:177–210`), igual que `read_chart`/`amend_chart` (`runner/charts.py:435–487`). La sonda confirmó que la herramienta declara desconocido un run que la API sí encuentra en un proyecto.

**Mejora:** ID opaco global reservado atómicamente, metadata con project ID y un repositorio/resolvedor único para HTTP, CLI, driver y herramientas. El lookup debe incluir el contexto autorizado, no buscar el primer archivo de nombre parecido.

### BH-04 — La persistencia aún puede reconocer escrituras inexistentes o perder trabajo

**Prioridad P1. Reproducido; las carreras reales bajo carga no se midieron.**

`SessionStore.save` sobrescribe el JSON y silencia `OSError` (`sessions.py:68–83`); el endpoint devuelve éxito igualmente (`api_projects.py:108–116`). Al simular fallo de disco se obtuvo **PUT 200 y GET 404**. Un guardado viejo con `updated_at=10` reemplazó uno nuevo de `20`, borró mensajes y cambió el proyecto de A a B. El timestamp no se utiliza como control de concurrencia.

La transcripción durable sigue llegando del cliente, y `/agent` ejecuta las `messages` suministradas por él (`serve.py:297–306`). El comentario que llama al store «single source of truth» no significa que el servidor sea dueño del historial ni impide que dos pestañas lo sustituyan. Un hilo no registrado cae en `default` (`sessions.py:64–66`).

`RunState.save` sí es atómico, pero no transaccional entre lectores/escritores independientes (`runner/runner.py:273–292`). El runner, los endpoints de plan/input y las enmiendas de charts escriben documentos completos. Una sonda con dos snapshots mostró que el guardado tardío de uno revirtió el estado `done` guardado por el otro. Los proyectos (`projects.py:174`) y datasets (`datasets.py:28`) todavía usan escritura directa.

**Mejora:** servidor propietario de mensajes y binding de sesión; revisiones/versiones y conflicto explícito; repositorio con transacciones o compare-and-swap, un dueño de transiciones por run y errores de persistencia visibles. Reemplazo atómico evita truncamiento, pero no basta para evitar actualizaciones perdidas.

### BH-05 — El lanzamiento estructurado aún depende de una decisión de modelo y los límites son incompletos

**Prioridad P1. Confirmado por código y agente sintético.**

El launcher ya eligió workflow e inputs, pero `GraphDriver.start` vuelve a pedir a un orquestador que llame a `run_workflow` (`driver.py:91–108`, `:196–205`, `:318`). Se cobra y espera una decisión innecesaria; el propio driver tiene un caso de fallo para cuando el modelo no inicia el workflow (`:326–335`). El grafo del driver usa todos los cartridges y el workspace compartido del servidor (`serve.py:315–318`), a diferencia del chat vinculado al proyecto.

`Step.timeout` se acepta como texto (`cartridge/models.py:63`) y no lo aplica `_drive` (`runner/runner.py:737–763`). Un paso sintético declarado `timeout: 1ms` durmió 50 ms y terminó `done` a los 57 ms. No hay validación de sintaxis de ese timeout. La parada es explícitamente entre pasos (`:583–595`); no corta una herramienta en ejecución. El timeout de Jupyter es espera entre mensajes, no deadline total, y salir del bucle no interrumpe el kernel (`envs/kernel.py:36–59`). El timeout del CLI Docker tampoco prueba por sí solo que el proceso dentro del contenedor murió (`envs/docker.py:188–195`).

El driver usa threads daemon locales y no ofrece ownership/leases entre procesos (`driver.py:310`). El checkpointer SQLite ayuda con gates del launcher; el chat usa memoria y pierde interrupts al reinicio (`serve.py:243–261`). Esto no equivale a recuperación de un paso largo a mitad de ejecución.

**Mejora:** comando determinista `StartRun` → runner para la UI; reservar el LLM para interpretar solicitudes de chat. Deadline propagado a herramientas/entorno, cancelación con confirmación de terminación, cupos y recuperación explícita. Empezar con un worker y propiedad durable; distribuir después si la carga lo requiere.

### BH-06 — El contrato multidominio valida presentación mejor que ejecución e historial

**Prioridad P1 antes de ofrecer cartridges intercambiables. Reproducido en parte; resto confirmado por código.**

El avance de `Experience` y marketing es real, pero quedan estos huecos:

- Dos pasos con el mismo ID no se rechazan. `Workflow.ordered_steps` devuelve `self.step(i)`, que siempre encuentra el primero (`cartridge/models.py:134–154`). Una sonda con instrucciones `first.md` y `second.md` produjo `['first.md', 'first.md']`.
- `load_cartridges` detecta personas duplicadas, no workflows duplicados (`cartridge/loader.py:305–316`). API, driver y tools construyen diccionarios globales por nombre de workflow, de modo que el último gana silenciosamente (`api.py:188`, `driver.py:149`, `runner/tools.py:122`).
- `WorkflowInput.type` es texto y `_resolve_inputs` comprueba presencia y options, pero no valida tipos ni rechaza campos extra (`cartridge/models.py:78`, `runner/runner.py:606–619`). La UI no sustituye esa validación del servidor.
- Se puede crear un proyecto con cualquier nombre de cartridge (`api_projects.py:36`). Si su cartridge ya no está instalado, `cartridges_for` devuelve **todos** los disponibles (`serve.py:227–230`), cambiando el equipo del proyecto silenciosamente.
- Proyecto y run guardan el nombre, pero no la versión/digest ni un snapshot completo del workflow. El detalle del run reconstruye su forma desde el cartridge instalado hoy (`api.py:221–226`); reanudar lee los pasos actuales (`runner/runner.py:529–557`). Una actualización puede modificar el significado del historial o hacer incompatible el resume.
- El prompt global conserva ejemplos específicos del primer dominio: `"Ana"` y `"build an MMM"` (`host/build.py:192–195`), contrarios al invariante 1 de `CLAUDE.md`.

**Mejora:** contrato versionado estricto de ejecución, IDs con namespace y unicidad, validación de inputs centralizada, cartridge ausente en estado no ejecutable, snapshot/digest del contrato por run y migraciones explícitas. La retirada de un cartridge debe preservar lectura y nunca sustituir el dominio sin decisión del usuario.

### BH-07 — La evidencia de calidad y el costo están incompletos justamente cuando falla un trabajo

**Prioridad P1 para compromisos de calidad/costo. Reproducido y confirmado por lectura.**

La telemetría y precio se registran después de que `_drive` devuelve (`runner/runner.py:675–679`). Si el stream entrega mensajes con usage y después falla, no se conservan esos tokens. Una sonda emitió 100 tokens de entrada y 10 de salida antes de lanzar un error; el paso terminó con `usage={}` y `cost_usd=None`. Los retries sustituyen el mismo `StepRecord`; falta un ledger de intentos. El modelo registrado se infiere de la persona/default (`:725–735`), no de cada respuesta efectivamente usada.

El contrato `produces` comprueba archivos existentes (`:434–449`, `:680`), no su validez, vigencia o evidencia. Un artefacto viejo puede satisfacer un retry ordinario; la comprobación de reescritura existe únicamente para un rechazo de gate. Un informe vacío o una pieza de marketing que inventa capacidades puede cumplir el contrato técnico. Esto no exige que el harness entienda marketing: cada cartridge puede declarar validadores/rúbricas y fixtures de aceptación.

**Mejora:** eventos de consumo por intento durante el stream, ledger append-only y presupuestos; provenance/hash de artefactos por intento; validadores estructurales en el harness y evaluaciones de contenido en cartridges. Medir también coste total por trabajo aceptado y revisión humana, no solo por paso que terminó bien.

### BH-08 — La suite aporta confianza, pero el proceso de entrega y operación no sostiene aún esa confianza

**Prioridad P1 antes de una distribución soportada. Verificado localmente.**

Hay 385 tests verdes con `python -m pytest`, pero el comando `pytest` documentado en `CLAUDE.md` no recoge la suite correctamente en esta instalación. Ruff mantiene siete hallazgos. No se encontró lock de dependencias Python ni pipeline `.github` en el checkout; `pyproject.toml` usa límites inferiores amplios para Deep Agents/LangChain. Los warnings del bridge ya indican uso de una API de LangGraph deprecada. No es prueba de que una actualización rompa hoy, sí evidencia de que la reproducibilidad depende del entorno instalado.

Los costes operativos crecen por copias completas de recursos en cada run y turno (`datasets.py:53`, `serve.py:281`), cargas completas en memoria (`api_projects.py:66`, `api.py:327`) y ZIP en memoria (`api.py:634`). El chat comparte entorno/kernel por proyecto (`serve.py:262–282`) sin una cola visible que serialice todas las conversaciones que lo usan; `JupyterKernel.run` consume un canal compartido sin lock. No se probó una carrera real de kernel ni se midió capacidad máxima.

El shutdown cierra entornos de chat, pero no drena los threads del driver ni gestiona huérfanos tras caída abrupta (`serve.py:158–169`). El health del agente es estático (`:308–310`). Hay logs útiles por run, pero no una garantía medida de restore, shutdown o disponibilidad.

**Mejora:** CI reproducible, pruebas de contrato del bridge y restore, límites de upload/recursos/concurrencia, cola por entorno compartido o entornos por sesión, readiness y apagado drenado. Separar endpoints, repositorios y control de ejecución antes de que continúe creciendo la duplicación; el backend ya suma unas 7.200 líneas y `runner.py` 1.171, bastante más que la intención original de un host mínimo.

## Qué cambió respecto a las revisiones anteriores

| Asunto | Estado que esta revisión sostiene |
|---|---|
| UI acoplada por completo a DS | Mejoró sustancialmente: existe Experience, etiquetas de workflow y cartridge marketing. Quedan invariantes del backend y ciclo de vida por cerrar. |
| Preview de runs no-default | **Corregido en esa ruta:** `api.py:522` pasa `store`. No cerrar SR-005 entero: herramientas de lectura/chart siguen sin el resolver. |
| Auto-gate siempre en host | La implementación actual lo materializa y ejecuta en el entorno del paso. No conservar el diagnóstico antiguo como fallo vigente. |
| `run.json` se trunca al escribir | Tiene reemplazo atómico y temporal por hilo/proceso. Persisten los conflictos entre snapshots, que son otro defecto. |
| Gate no sobrevive nunca al proceso | Hay SQLite y pruebas de persistencia para el launcher. El chat y pasos en vuelo tienen límites distintos. |
| Inicio duplicado por dos clicks | El driver tiene lock local; no es coordinación entre workers ni resuelve IDs duplicados entre proyectos. |
| HTML/ZIP, sesiones, scope, IDs, deadlines | Siguen abiertos y esta revisión añade reproducciones locales sobre el código actual. |

No se modifican checkboxes de `BACKLOG.md` con esta revisión. Los arreglos parciales no cierran sus criterios completos.

### Matriz de proyecto/default

| Superficie | Proyecto no-default | Observación |
|---|---|---|
| Archivos de run por HTTP | Usa `store` | Resuelto el lookup simple; sigue dependiendo de IDs globalmente únicos. |
| Preview y perfil del plan | Usa `store` | Corregida la omisión anterior. |
| Equipo del chat con cartridge instalado | Usa el cartridge vinculado | Avance real; si el cartridge falta, el fallback amplía el equipo a todos. |
| `list_run_files` / `read_run_file` | Solo directorio default | Pendiente; reproducido como `unknown run`. |
| `read_chart` / `amend_chart` | Solo directorio default | Pendiente por lectura de la implementación. |
| Hilo de chat no registrado | Se asigna a default | Pendiente una respuesta explícita de registro/ownership. |
| Driver del launcher | Grafo global, localización por run ID | No adopta el grafo por proyecto que usa el chat. |

## Reproducciones mínimas seguras

Ejecutar desde la raíz con `.venv/bin/python`. Todo lo que se crea está dentro de `TemporaryDirectory`; no se requiere API en ejecución, modelo ni Docker. Los fragmentos ilustran las sondas ejecutadas, con nombres sintéticos.

### IDs de dos proyectos y guardado de sesión viejo

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from dsagent.serve import resolve_run_file  # carga primero el adaptador HTTP
from dsagent.api import new_run_id
from dsagent.projects import ProjectStore
from dsagent.sessions import SessionStore

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    store = ProjectStore(root / "projects", root / "runs")
    a, b = store.create("A"), store.create("B")
    with patch("dsagent.api.time.strftime", return_value="20260921-010101"):
        first = new_run_id(store.runs_dir(a.id), "review")
        second = new_run_id(store.runs_dir(b.id), "review")
    assert first == second
    for project in (a, b):
        run = store.runs_dir(project.id) / first
        run.mkdir()
        (run / "run.json").write_text("{}")
    candidates = [store.runs_dir(project.id) / first for project in (a, b)]
    assert len(set(candidates)) == 2
    assert store.locate_run(second) in candidates  # solo uno puede resolverse por ese ID

    sessions = SessionStore(root / "sessions")
    sessions.save("s", project=a.id, title="Nuevo", messages=["nuevo"], updated_at=20)
    sessions.save("s", project=b.id, title="Viejo", messages=[], updated_at=10)
    assert sessions.get("s")["messages"] == []
    assert sessions.get("s")["project"] == b.id
```

El `TestClient` de la sonda completa añadió la comprobación HTTP de que el GET del segundo ID devolvía los inputs del primero. El ejemplo de `locate_run` depende del orden de iteración para elegir A o B; lo que debe permanecer cierto para detectar el defecto es que **uno de los dos runs resulta inaccesible por ese ID**.

### Script fuera del scope concedido

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from dsagent.cartridge import load_cartridge
from dsagent.host.build import _skill_script_tool
from tests.fakes import stub_env

cart = load_cartridge("cartridges/ds", generate_commands=False)
with TemporaryDirectory() as tmp:
    workspace = Path(tmp)
    allowed = workspace / ".dsagent/skills/marie/reports/scripts"
    denied = workspace / ".dsagent/skills/noel/ml/scripts"
    allowed.mkdir(parents=True)
    denied.mkdir(parents=True)
    (denied / "probe.py").write_text('print("OUTSIDE_GRANTED_SKILL")')
    tool = _skill_script_tool(cart, "marie", stub_env(cart.envs["default"], workspace), workspace)
    result = tool.invoke({"skill": "reports", "script": "../../../noel/ml/scripts/probe.py"})
    assert "exit code: 0" in result and "OUTSIDE_GRANTED_SKILL" in result
```

### Duplicación de paso

```python
from pathlib import Path
from dsagent.cartridge.models import Workflow, Step

wf = Workflow(name="probe", path=Path("."), steps=[
    Step(id="same", persona="a", instructions="first.md"),
    Step(id="same", persona="b", instructions="second.md"),
])
assert [step.instructions for step in wf.ordered_steps()] == ["first.md", "first.md"]
```

## Trabajo propuesto, listo para convertir en tareas

IDs `BH-T*` son propuestas de esta revisión, no un segundo backlog autoritativo. Integrarlas con los SR existentes al planificar para evitar duplicados.

| Orden | Tarea | Relación con backlog | Criterio de aceptación |
|---|---|---|---|
| 1 | BH-T01: unificar contención de archivos y aislar contenido activo | SR-002/003/045 | Las rutas raw/proyecto/ZIP/export y herramientas rechazan escapes y symlinks externos; contenido activo no accede al origen/API del producto. Pruebas con sentinels temporales. |
| 2 | BH-T02: hacer real el scope de scripts/herramientas | SR-032/046/048 | Traversal y symlinks fuera del skill fallan antes de ejecutar; herramientas no concedidas no se registran; distinguir y probar discovery frente a permisos. |
| 3 | BH-T03: IDs globales y resolución por ownership | SR-004/005/037 | Creaciones simultáneas en distintos proyectos nunca colisionan; todos los endpoints/tools resuelven el mismo run; casos negativos entre proyectos. |
| 4 | BH-T04: persistencia de sesiones sin falsos éxitos | SR-006/042/043/049 | Fallos de disco devuelven error; un cliente viejo recibe conflicto sin borrar el historial; el proyecto de la sesión es estable; hilo desconocido no cae silenciosamente en default. |
| 5 | BH-T05: repositorio transaccional de runs | SR-012/021/047 | Estado, inputs, plan, charts y decisiones no pierden actualizaciones con dos escritores; versión/esquema explícitos y pruebas de reinicio. Evaluar SQLite antes de requerir infraestructura nueva. |
| 6 | BH-T06: lanzamiento determinista | SR-015 | Una orden estructurada inicia exactamente el workflow preparado sin llamada de routing a LLM; chat sigue pudiendo interpretar lenguaje; gates/replay/UI conservan contrato. |
| 7 | BH-T07: entorno aislado, deadlines, cancelación y cupos | SR-009/010/013/014/018 | Código generado no accede al host/control/secretos; deadline absoluto termina también hijos; cancelar deja estado verificable; saturación rechaza/encola según política. |
| 8 | BH-T08: validación y versionado de cartridges | SR-021/032/050 | Duplicados y tipos inválidos fallan al validar; proyecto con cartridge ausente conserva lectura y rechaza ejecución; run conserva snapshot/digest y no cambia significado al actualizar. |
| 9 | BH-T09: consumo y artefactos por intento | SR-017/020/034 | Un stream que falla conserva usage emitido; retry acumula costo; artefactos registran intento/hash; fixtures DS y marketing detectan resultados técnicamente presentes pero inválidos. |
| 10 | BH-T10: serializar uso de entornos y controlar su vida | SR-011/018/053 | Dos chats del proyecto no roban mensajes ni pisan el kernel; política explícita de cola/aislamiento/TTL; shutdown drena o marca interrupción y limpia solo recursos propios. |
| 11 | BH-T11: límite de recursos y exportaciones | SR-019/033/052 | Upload y preview tienen límites medidos; ZIP evita copia completa en memoria; un recurso grande no bloquea otros proyectos; política de copia y retención documentada. |
| 12 | BH-T12: CI y prueba de fallo/recuperación | SR-001/016/031/040/055 | Instalación reproducible; comando oficial de tests funciona; lint pasa; matriz de contrato AG-UI/Deep Agents y prueba de caída/reinicio con datos sintéticos, sin LLM por defecto. |

BH-T01–05 y la definición del aislamiento son el bloque inicial recomendado. BH-T06–08 convierten la experiencia multidominio en un contrato operacional. No hace falta implementar selección de modelos con Jev antes de cerrar esto: la investigación queda como investigación.

## Criterio para considerar bueno el siguiente incremento

Una demo de marketing funcionando es una prueba útil de extensibilidad. La siguiente prueba más valiosa es crear **dos proyectos de dominios distintos**, ejecutar trabajos con herramientas sintéticas, provocar fallo de disco/stream y reinicio, y comprobar que cada uno recupera su historial, gate, artefactos y costo sin mezcla ni pérdida silenciosa. Después verificar el entorno aislado y un pequeño conjunto de resultados reales por dominio con presupuesto autorizado.

Eso fortalecería precisamente lo que hace valioso al producto: que un equipo de agentes pueda producir trabajo revisable y recuperable bajo control del usuario.
