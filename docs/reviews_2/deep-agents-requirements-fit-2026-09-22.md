# Requisitos y encaje arquitectónico sobre Deep Agents

2026-09-22 · Código DSAgent: `3038ffc` · Evaluación de diseño, sin implementación ni benchmark.

## Conclusión ejecutiva y corrección de la evaluación anterior

**No se ha identificado un requisito funcional de los tres dominios que obligue a sustituir Deep Agents. Sí hay requisitos que exigen cambiar la arquitectura de aplicación que hemos construido encima.** La recomendación de comparar runtimes antes de completar este análisis fue prematura. Esta evaluación reemplaza ese orden recomendado en la investigación anterior; no invalida las capacidades documentadas de Hermes, OpenClaw o NemoClaw.

La dirección recomendada es conservar Deep Agents como ejecutor de trabajo cognitivo y herramientas, utilizar LangGraph para procesos que necesiten estado e iteración, y asignar a DSAgent la responsabilidad de proyectos, autorización, trabajos, recursos, aprobaciones y operación. El runner actual puede servir para recorridos simples, pero no debe convertirse por acumulación de excepciones en el controlador universal de los tres dominios.

Esto no es «todo se puede programar». El encaje se fundamenta en puntos de extensión ya presentes en el SDK instalado, responsabilidades que pertenecen a servicios externos y limitaciones concretas de nuestro código. La matriz indica dónde configurar, integrar o rediseñar; no atribuye al SDK una plataforma empresarial que no proporciona.

## Método y evidencia

Base de requisitos derivada del alcance solicitado: ciencia de datos general, agencias de marketing y SDLC. Es una base arquitectónica revisable de 48 requisitos; no una afirmación de haber capturado todas las necesidades comerciales, regulatorias o de cada cliente. No se inventan objetivos de concurrencia, disponibilidad o latencia aún no definidos.

Se inspeccionaron los paquetes instalados: **Deep Agents 0.7.14, LangGraph 1.2.11, LangChain 1.4.0, langgraph-checkpoint 4.2.0 y langgraph-checkpoint-sqlite 3.1.1**. `langchain-mcp-adapters` no está instalado. Estas versiones describen este entorno; los rangos de `pyproject.toml` no fijan por sí solos una instalación reproducible.

En `deepagents/graph.py:271`, `create_deep_agent` expone `tools`, `middleware`, `subagents`, `skills`, `memory`, `permissions`, `backend`, `interrupt_on`, `response_format`, `state_schema`, `context_schema`, `checkpointer` y `store`; devuelve un `CompiledStateGraph`. Se inspeccionaron también `BackendProtocol`, `SandboxBackendProtocol`, `StoreBackend` y las primitivas `StateGraph`, `Send` y `Command` de LangGraph. No fue necesario ejecutar agentes para constatar esos contratos.

Fuentes oficiales que complementan la inspección local:

- [S1 — Personalización de Deep Agents](https://docs.langchain.com/oss/python/deepagents/customization): herramientas y composición del agente.
- [S2 — Backends](https://docs.langchain.com/oss/python/deepagents/backends): almacenamiento intercambiable y namespaces.
- [S3 — API de grafos de LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api): transiciones, ramificación y distribución de trabajo. Las primitivas relevantes se confirmaron en el paquete Python local.
- [S4 — Persistencia](https://docs.langchain.com/oss/python/langgraph/persistence): diferencia entre checkpoints del hilo y almacenamiento entre hilos.
- [S5 — Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts): pausa/reanudación y reejecución del nodo interrumpido.
- [S6 — Permisos](https://docs.langchain.com/oss/python/deepagents/permissions): límites de las reglas sobre herramientas de archivos.
- [S7 — Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes): conexión con entornos de ejecución y su ciclo de vida.
- [S8 — MCP](https://docs.langchain.com/oss/python/langchain/mcp): superficie de integración con herramientas externas.
- [S9 — Revisión local previa](harness-extensibility-assessment-2026-09-22.md): contrato actual de cartridges, scripts, recursos y UI.

La documentación pública puede evolucionar. La viabilidad de diseño no certifica rendimiento, calidad del modelo, seguridad efectiva o recuperación ya implementada.

## Cómo leer la matriz

**V** = viable sobre Deep Agents/LangGraph; **VC** = viable condicionado por un sistema externo o una garantía operativa explícita. Ninguna fila equivale a «ya implementado».

Cambios: **Configurar** = cablear una capacidad disponible; **Extender** = incorporar un contrato/adaptador acotado; **Rediseñar** = cambiar una responsabilidad o ciclo de vida de DSAgent; **Integrar** = delegar a un servicio especializado. Las referencias E1–E8 identifican evidencia local al final.

### Plataforma compartida: 20 requisitos

| ID | Requisito y criterio de aceptación arquitectónico | Encaje / responsable | Estado actual y cambio necesario |
|---|---|---|---|
| C01 | Roles, modelos y skills por dominio; agregar un rol no exige una condición por nombre en el core | V · SDK + cartridge | Base presente. Extender versionado y validación de compatibilidad. E1 |
| C02 | Herramientas propias y conectores MCP; declarar capacidad requerida y fallar si falta | V · SDK + registro de DSAgent | Hay inyección Python; falta catálogo/resolución declarativa. Extender e integrar MCP. E1 |
| C03 | Permisos efectivos por usuario/proyecto/rol, incluidos subagentes y shell | VC · autorización DSAgent + entorno | `Persona.tools` no se aplica; filtrar herramientas no contiene shell. Rediseñar frontera de autorización. E1/E6 |
| C04 | Identidad y separación organización/cliente/proyecto en API, archivos, memoria y trabajos | VC · servicio de producto + almacenamiento | Carpetas/proyectos no equivalen a autorización. Rediseñar acceso autenticado en todos los recursos. E7 |
| C05 | Secretos vinculados a una capacidad y ámbito, revocables sin editar prompts | VC · servicio de credenciales + adaptadores | Shell local hereda entorno del host. Integrar credenciales limitadas y evitar su exposición general. E6 |
| C06 | Repositorios, datasets, documentos y servicios como recursos identificados/versionados | V · modelo de recursos DSAgent | Copia plana a `data/`. Extender referencias, snapshots y preparación de workspaces. E5 |
| C07 | Conversación recuperable y separada del estado de trabajos | V · checkpointer + servicio de sesiones | Transcript persistido; ejecución del chat en memoria. Rediseñar persistencia de turnos e interrupciones sin dos autoridades. E3 |
| C08 | Conocimiento persistente con fuente, alcance, edición y eliminación | V · store/retrieval + producto | Archivos/conversaciones no forman aún memoria gobernada. Configurar store e incorporar políticas de procedencia. E1/E7 |
| C09 | Planificación exploratoria y ciclos de revisión limitados por política | V · agente + grafo de proceso | Bucle de agente disponible; workflow propio fijo. Extender composición, sin pedir al LLM decidir permisos. E1/E2 |
| C10 | Dependencias, ramas, iteración por ítem y tareas paralelas con límite | V · LangGraph + admisión DSAgent | Runner topológico secuencial. Rediseñar ejecución de procesos, conservando el caso lineal. E2 |
| C11 | Aprobación ligada a actor, acción y versión exacta del resultado; recuperable | V · proceso + registro de decisiones | Gates presentes; relación con interrupts posicionales compleja. Rediseñar identidad de aprobación y su aplicación idempotente. E2/E3/E4 |
| C12 | Un trabajo explícitamente solicitado queda admitido sin depender de que el LLM vuelva a elegir iniciarlo | V · servicio de trabajos | Launcher pide al orquestador llamar `run_workflow`. Rediseñar arranque directo y compartirlo con el chat. E4 |
| C13 | Reinicio de API/worker no pierde trabajo aceptado ni genera dos ejecutores válidos | VC · trabajos persistentes + checkpoints | Threads en proceso; no propietario durable distribuido. Rediseñar admisión, lease y recuperación. E4 |
| C14 | Horarios/eventos crean trabajos deduplicados y conservan historial | V · scheduler externo o servicio DSAgent | Falta scheduler. Integrar disparadores con el mismo servicio de trabajos. No necesita otro motor de agente. E4 |
| C15 | Tareas largas tienen ID, estado, progreso, resultado y cancelación independientes del turno LLM | VC · servicio de cómputo + tools | Ejecución síncrona en herramientas/entornos. Rediseñar hacia submit/status/cancel/result. E2/E6 |
| C16 | Límites de tiempo, intentos, tokens, costo y recursos; detener trabajo en curso según contrato | VC · worker + middleware + procesos | Stop solo entre etapas; `Step.timeout` sin uso en runner. Rediseñar supervisión; estimación de costo no es límite duro. E2/E4/E6 |
| C17 | Reintentar una publicación o cambio externo no duplica silenciosamente su efecto | VC · adaptador + proveedor externo | No ledger general de efectos. Integrar claves idempotentes o reconciliación y estado «resultado desconocido». E2/E4 |
| C18 | Entorno reproducible, aislado y recuperable, con ciclo de vida independiente del request HTTP | VC · backend + supervisor de entornos | Abstracción existente; cierre al salir del runner y dependencia del filesystem del host. Rediseñar ciclo de vida. E2/E6 |
| C19 | Artefactos tipados y versionados con origen, integridad, permisos y visor adecuado | V · servicio de artefactos + UI | Archivos/gráficos presentes; falta contrato transversal. Extender sin introducir contenido ejecutable arbitrario en la UI. E2/E8 |
| C20 | Instalar/actualizar paquetes de dominio de forma reproducible y reversible, sin alterar runs activos | V · catálogo controlado de producto | Carga desde carpetas; falta snapshot de paquete/entorno por run. Extender contrato de versiones y permisos. E1 |

### Ciencia de datos: 8 requisitos

| ID | Requisito y aceptación | Encaje / responsable | Estado actual y cambio necesario |
|---|---|---|---|
| D01 | Ingesta SQL/archivos/objetos con esquema, origen y tamaño controlados | VC · conectores y motor de datos | Upload de archivos. Integrar fuentes; usar referencias/consultas para datos grandes. C02/C06 |
| D02 | Calidad, limpieza, transformaciones y trazabilidad hasta los datos originales | V · skills + ejecución + recursos | Flujos DS existentes; extender contratos de validación y linaje. C06/C19 |
| D03 | Exploración y notebooks; restauración desde código/artefactos y límites explícitos del kernel | V · Jupyter + workspace | Kernel persistente durante su vida, no memoria RAM recuperable tras fallo. Separar sesión de cómputo y persistir código/resultados. C15/C18 |
| D04 | Estadística, inferencia causal, experimentación y ML con criterios metodológicos verificables | VC · skills + librerías + validadores/revisión | No exige otro motor; declarar métodos soportados y evidencia. Un prompt no garantiza corrección científica. C01/C19 |
| D05 | Experimentos comparables: datos, código, parámetros, métricas, semillas y entorno | V · registro de experimentos | Artefactos no bastan como registro. Integrar o implementar modelo de experimento. C06/C19 |
| D06 | Entrenamiento largo, GPU o cómputo remoto sin bloquear un turno ni compartir estado accidental | VC · infraestructura de cómputo | Docker admite GPU, pero falta gestión durable de jobs. Integrar supervisor/servicio. C15/C16/C18 |
| D07 | Evaluación y selección de modelos con evidencia y aprobación para promoción | V · proceso + validadores | Gates útiles; métricas y umbrales deben ser resultados estructurados. C10/C11/C19 |
| D08 | Entrega/serving, reentrenamiento y seguimiento de degradación con rollback | VC · plataforma de modelos + eventos | No es responsabilidad del bucle LLM. Integrar registro, despliegue y monitorización. C14/C17 |

### Agencias de marketing: 7 requisitos

| ID | Requisito y aceptación | Encaje / responsable | Estado actual y cambio necesario |
|---|---|---|---|
| M01 | Separar clientes/marcas, cuentas, materiales y memoria dentro de una agencia | VC · identidad + política de recursos | Proyecto/cartridge no garantiza separación. Requiere C03/C04/C08 incluso con una organización por instalación |
| M02 | Investigación, brief, estrategia y copy referenciados al contexto de marca | V · skills + herramientas | Cartridge documental existente. Extender fuentes, referencias y validación. C01/C02 |
| M03 | Producir y revisar imágenes/audio/video con versiones y especificaciones por canal | VC · proveedores de medios + jobs + visores | No se resuelve cambiando etiquetas. Integrar herramientas y vistas. C15/C19 |
| M04 | Revisiones por responsable y cliente sobre una versión concreta, con historial | V · aprobaciones + UI | Gate simple como base; extender actores, cambios y versión aprobada. C11 |
| M05 | Calendario, horario local, programación y reprogramación verificables | V · scheduler + modelo de campaña | Sin scheduler ni calendario específico. Integrar eventos y vista de dominio. C14 |
| M06 | Entrega/publicación, permisos por cuenta y límites para cambios de presupuesto | VC · conectores + autorización + efecto durable | Falta integración operacional. Acceso a API no equivale a aprobación comercial. C03/C05/C11/C17 |
| M07 | Métricas de campaña normalizadas y recomendaciones sin actuar fuera del mandato | VC · datos + skills + reglas de acción | Extender conectores/definiciones métricas y limitar acciones; no delegar el permiso al modelo. C02/C16 |

### SDLC: 7 requisitos

| ID | Requisito y aceptación | Encaje / responsable | Estado actual y cambio necesario |
|---|---|---|---|
| S01 | Preparar revisión de repo/worktree aislada y registrar origen del cambio | V · recurso repo + backend | Shell puede usar Git; falta contrato de checkout y propiedad. C06/C18 |
| S02 | BMAD/Spec Kit u otros procedimientos adaptados y versionados por proyecto | V · skills + proceso | No hay incompatibilidad fundamental; mapear scripts, rutas, interacciones y métodos de una versión concreta. C01/C02/C20 |
| S03 | Backlog, dependencias e iteraciones de implementación/revisión por tarea | V · LangGraph + modelo de trabajo | Supera el DAG actual, no la capacidad de LangGraph. C09/C10 |
| S04 | Edición/compilación/pruebas/lint con runtimes y dependencias declarados | VC · sandbox + herramientas | Infraestructura parcial; el helper de scripts presupone Python. Extender ejecutores y entornos. C02/C18 |
| S05 | Reparación limitada tras fallo; resultados de test distinguen ejecución, fallo y omisión | V · grafo + agente + check tools | El agente puede iterar; falta contrato durable entre trabajo y evidencia. C09/C16/C19 |
| S06 | Revisión de diff y cambios concurrentes con protección de revisión base | V · Git + aprobaciones + UI | No hay flujo gestionado de diff/base/merge. Extender; no necesita cambiar el SDK. C06/C11/C19 |
| S07 | Crear PR/integrar/desplegar/revertir según políticas y estado del proveedor | VC · adaptadores Git/CI/CD | Externo al motor; permisos y recuperación de efectos explícitos. C03/C05/C17 |

### Operación y UX: 6 requisitos

| ID | Requisito y aceptación | Encaje / responsable | Estado actual y cambio necesario |
|---|---|---|---|
| O01 | Web desacoplada: cerrar/reabrir conserva trabajo y recupera eventos | V · API/eventos + UI | SSE y runs del servidor ya aportan base. Extender replay/cursor y unificar trabajos iniciados por chat/launcher. E3/E4/E8 |
| O02 | Colaboración con roles, conflictos de edición y auditoría atribuible | VC · producto + almacenamiento | Sesiones detectan ciertos conflictos; falta contrato multiusuario completo. C04/C11 |
| O03 | Concurrencia limitada, backpressure, cuotas y equidad entre proyectos | VC · scheduler/worker + BD | Threads locales no ofrecen estas garantías. Rediseñar propietario/admisión; capacidad objetivo aún sin definir. C13/C16 |
| O04 | Trazas/costos/latencia vinculados a modelo, herramienta, intento y versión | V · callbacks/eventos + telemetría | Telemetría parcial existente; extender correlación y cobertura. Medir no equivale a controlar. C16/C19 |
| O05 | Copias, recuperación, retención/borrado, migraciones y despliegues compatibles con runs pendientes | VC · operación + datos + versionado | No hay contrato completo. Definir RPO/RTO, autoridad y compatibilidad antes de automatizar. C07/C13/C20 |
| O06 | Experiencia accesible con tablas/experimentos, piezas/calendario y diffs/pruebas | V · frontend + artefactos | UI común aprovechable; estructura fija no satisface todas las vistas. Extender tres módulos mantenidos por el producto. C19 |

## Los desafíos arquitectónicos decisivos

### 1. Separar la conversación del control del trabajo

Hoy incluso una selección explícita en el launcher pasa por una instrucción al orquestador para que llame la herramienta correcta. El driver tiene una rama de error para el caso en que no lo haga (`driver.py:88,310`). Esa incertidumbre no aporta valor cuando la decisión ya está tomada.

**Recomendación:** API y chat invocan el mismo servicio de trabajos. El chat puede proponer o solicitar un trabajo; el servicio valida, admite y devuelve un ID. El proceso continúa sin mantener toda su vida dentro de una llamada del agente conversacional. Un trabajo interactivo y corto puede esperar el resultado; la operación durable sigue teniendo identidad propia.

### 2. Tener una autoridad clara para cada estado

La aplicación combina `run.json`, checkpoints, threads del driver y transcripciones. No es malo disponer de varios almacenes; el problema aparece si varios deciden independientemente si la misma etapa está terminada o aprobada. `_walk_past_decided_gates` y el tratamiento posicional de interrupts muestran el costo de la coordinación actual.

**Recomendación:** asignar admisión/propiedad/intentos al servicio de trabajos; estado de etapas y espera al grafo de proceso; transcripciones a la conversación; artefactos a su catálogo. La UI consume una proyección. Las respuestas humanas entran como comandos identificados y validados, con entrega reintentable; no se consideran aplicadas solo porque el navegador recibió éxito. Documentar dueño y reconciliación de cada campo antes de elegir tablas o colas.

Un grafo LangGraph de proceso que compone Deep Agents es compatible con el stack. No hay razón arquitectónica para mantener todo el workflow en una herramienta monolítica. Los casos lineales pueden seguir declarándose como plantillas; los procesos iterativos requieren un contrato adicional, no necesariamente un lenguaje YAML universal.

### 3. Elegir fronteras de recuperación, no solo activar un checkpointer

LangGraph vuelve a ejecutar el nodo interrumpido al reanudar. Sus checkpoints no revierten publicaciones ni reconstruyen la RAM de un kernel. La documentación obliga a considerar los efectos previos a un interrupt. [S5](https://docs.langchain.com/oss/python/langgraph/interrupts).

Los agentes de etapa actuales se construyen con `checkpointer=False` para evitar herencia/replay incorrectos. Fue una corrección válida al diseño actual; también significa que la continuidad fina de una etapa no está habilitada. No basta cambiarlo a `True`: cada ejecución hija necesita identidad estable, namespace, política de persistencia y recuperación del entorno.

**Recomendación:** separar nodos por operaciones significativas; persistir IDs de jobs externos; mantener efectos reintentables; distinguir «falló» de «no sabemos si el proveedor lo aplicó». Usar claves idempotentes donde el proveedor las soporte y reconciliación donde no. No prometer exactly-once universal.

### 4. Separar vida del agente, del entorno y del cómputo

El runner cierra entornos en `finally`; el kernel guarda variables únicamente mientras existe. El backend Docker supone un bind mount compartido con el host y la API lee los entregables de ese filesystem. Eso requiere cambio para cómputo remoto, trabajos largos y artefactos grandes.

**Recomendación:** supervisor de entornos con identificadores, TTL y limpieza explícita; herramientas de job `submit/status/cancel/result`; almacenamiento de artefactos referenciado. Restaurar código y archivos no equivale a restaurar un proceso vivo. La decisión de pausar/reanudar entrenamiento pertenece al framework de cómputo y sus checkpoints, no al checkpoint del agente.

### 5. La autorización se aplica fuera del razonamiento del modelo

Los namespaces de memoria permiten organizar ámbitos; no autentican usuarios. Las reglas de filesystem de Deep Agents no cubren automáticamente herramientas personalizadas, MCP ni ejecución arbitraria de shell. [S2](https://docs.langchain.com/oss/python/deepagents/backends), [S6](https://docs.langchain.com/oss/python/deepagents/permissions).

**Recomendación:** identidad autenticada propagada por contexto inmutable, herramientas resueltas para ese ámbito, autorización en el servicio que realiza el efecto y aislamiento del proceso. Diferenciar aprobación comercial de permiso técnico. Los subagentes reciben permisos explícitos o restringidos; cambiar instrucciones no puede ampliar autoridad.

### 6. Extender cartridges con límites definidos

El contrato actual describe personas, skills, workflows y entornos. No describe toda la operación de una agencia, un laboratorio o un equipo de ingeniería. La solución no es convertir cada necesidad en una clave especial del core.

**Recomendación:** paquetes versionados que seleccionan capacidades y procesos soportados, con adaptadores de recursos/herramientas mantenidos explícitamente. El core puede conocer «job», «recurso», «aprobación» o «artefacto»; no necesita conocer «BMAD» o «campaña de Instagram». Tres módulos de UI específicos son compatibles con una base compartida.

## Decisiones recomendadas a partir de los requisitos

| Decisión | Recomendación | Motivo / compromiso |
|---|---|---|
| Motor de agente | Mantener Deep Agents | Sus extensiones cubren las necesidades cognitivas; no hay incompatibilidad identificada que justifique migrar |
| Orquestación de proceso | Componer sobre LangGraph y sacar el control del workflow de la conversación | Admite iteración/ramas y explicita fronteras de estado; obliga a diseñar versiones y reanudación |
| Runner/driver actuales | Rediseñar su responsabilidad; reutilizar reglas útiles | El costo principal está en control y replay, no en los prompts |
| Plataforma de producto | Mantener servicio propio de trabajos, recursos, identidad y artefactos | Cambiar de motor no elimina estas responsabilidades |
| Herramientas/entornos | Contratos declarados con adaptadores y políticas | Separa capacidades de instrucciones y permite recursos locales/remotos |
| Infraestructura | Comenzar con despliegue acotado y workers limitados | Ni microservicios ni Kubernetes se deducen de estos requisitos; RPO/RTO y carga aún deben fijarse |
| Otros runtimes | Conservar investigación como referencia; no priorizar migración ni comparativa ejecutable ahora | Primero cerrar decisiones de arquitectura que ya pueden tomarse por análisis |

El costo de evolución es **significativo y concentrado** en control de ejecución, persistencia y recursos/permisos; no es una simple ampliación de YAML. No se estiman semanas ni porcentajes sin descomponer esas responsabilidades. Tampoco se concluye que todo el sistema sea incorregible.

## Qué no podemos prometer con esta ni con otra arquitectura

- Corrección científica, creativa o de código garantizada únicamente por instrucciones del modelo.
- Restauración arbitraria de cualquier proceso o kernel desde un checkpoint de conversación.
- Un único efecto externo exactamente una vez si el proveedor no ofrece idempotencia ni consulta suficiente para reconciliar.
- Ejecución segura de plugins arbitrarios con secretos del host y sin una frontera efectiva de aislamiento.
- Rendimiento/SLA concretos sin carga objetivo, límites de recursos y validación posterior.

Estos son límites de la garantía solicitada, no argumentos para sustituir Deep Agents por otro agente.

## Próximo entregable de diseño, antes de implementar

Derivar de esta matriz un ADR de control de ejecución que fije: modelo de trabajo/etapa/intento/aprobación/recurso/efecto; autoridad de cada estado; transiciones y fallos; comandos admitidos; identidades y permisos; compatibilidad de versiones y migración de los runs actuales. Prioridad: C03–C07, C10–C18 y C20, según dependencias. Los IDs de esta matriz son requisitos, no otro backlog de implementación.

La implementación y las pruebas posteriores verificarán propiedades que el análisis no puede garantizar —latencia, consumo, calidad y recuperación real—. **No son un prerrequisito para concluir ahora que Deep Agents encaja y que nuestra capa de control necesita rediseño.**

## Evidencia local

- **E1:** `src/dsagent/cartridge/models.py:26,153,170,265`, `cartridge/loader.py:276`, `host/build.py:150,169,232`: contrato, ejecutor Python y herramientas/modelos.
- **E2:** `src/dsagent/runner/runner.py:516,529,554,583,807,956`: vida del entorno, secuencia, stop y gates. `Step.timeout` se declara en models pero no se consume en este runner.
- **E3:** `src/dsagent/serve.py:239,265,378`: chat en memoria, agente por proyecto y checkpoints SQLite para la ruta síncrona con fallback.
- **E4:** `src/dsagent/driver.py:88,123,203,239,310`: launcher, driver, aprobación, replay de gates y threads.
- **E5:** `src/dsagent/datasets.py:53`: copia de archivos planos.
- **E6:** `src/dsagent/envs/kernel.py:85`, `envs/docker.py:1,99`, `envs/base.py:57`: shell/kernel, bind mount y fábrica de entornos.
- **E7:** `src/dsagent/projects.py`, `sessions.py`, `api_projects.py`: almacenamiento y rutas; no equivalen a un modelo completo de identidad/autorización.
- **E8:** `ui/src/components/Launcher.tsx`, `FileViewer.tsx`, `src/dsagent/runner/charts.py`: inputs, visores y artefactos actuales.

Validación de esta entrega: inspección de código y firmas del SDK instalado, fuentes oficiales y enlaces documentales. Sin cambios de runtime, instalación de dependencias ni ejecución de modelos.
