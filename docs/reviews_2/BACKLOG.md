# Backlog reconciliado de producto

Fecha base: 2026-09-19. **Actualizado 2026-09-24:** UI-001 implementada;
SR-032.a y SR-046.a cerradas con evidencia. Los padres SR conservan sus criterios
restantes. Otras entregas recientes del harness están documentadas en el roadmap;
su reconciliación completa con los criterios SR sigue pendiente.

Esta es la fuente de estado vigente para preparación de producto. Los backlogs anteriores se conservan como fuentes históricas; sus checkboxes no se sincronizan. Se conservan SR-001–041 y se añaden SR-042–057. Véanse [síntesis](00-RECONCILIACION.md), [decisiones pendientes](DECISIONES.md) y [mapeo completo](TRAZABILIDAD.md).

## Forma de trabajo y orden

Una tarea técnica en curso, una rama/PR por tarea. Antes de modificar: reproducir el hallazgo contra el HEAD actual, confirmar alcance y asignar responsable. Al cerrar: commit/PR, validación y evidencia durable sin datos privados. Estados: `[ ]` pendiente, `[~]` en curso, `[x]` terminada. Las tareas grandes son unidades de planificación: desglosarlas en subtareas SR-xxx.a/b antes de implementar; cerrar el padre solo cuando se cumpla todo su criterio. No interpretar una fila como una estimación de una semana.

El orden numérico conserva trazabilidad, no impone secuencia. **Prioridad del usuario: UI-001, entregada; después de su revisión, retomar SR-001.** Después, escoger el primer ítem desbloqueado de este orden orientativo: SR-042, 044, 002, 003, 004, 005, 043, 006, 047, 007, 045, 046, 009, 010, 048, 049, 011, 012, 013, 014, 015, 016. Continuar por las dependencias de la puerta elegida. SR-035, SR-050 e inventario SR-051 pueden prepararse sin esperar decisiones comerciales. Si se incluye MMM, adelantar SR-008 antes de usar sus resultados.

P0: límite de confianza/acceso. P1: fiabilidad u operación necesaria. P2: experiencia/eficiencia/expansión según alcance. V: validación de negocio. Compartido significa varias personas o tenants, también en self-host. Las dependencias de la tabla son técnicas; las condiciones de oferta/decisión y las puertas de salida también son obligatorias. DEC-* no bloquea correcciones neutrales.

Conservar [invariantes](../../CLAUDE.md), actualizar [arquitectura](../architecture.md) si cambia y registrar decisiones en [roadmap](../ROADMAP.md). Los criterios describen trabajo pendiente, no autorizan gasto, contactos, publicación ni cambio de licencia.

## Lista única de estado

| Estado | ID | Prioridad | Tarea | Depende de |
|---|---|---|---|---|
| [x] | UI-001 | P1 UI | Rediseñar el workspace para demostraciones empresariales | Prioridad explícita del usuario; [entrega y verificación](../ui-workspace-redesign-2026-09-19.md) |
| [ ] | SR-001 | P1 | Fijar línea base reproducible y checks de CI | — |
| [ ] | SR-002 | P0 | Proteger HTML y contenido activo en todas las rutas | — |
| [ ] | SR-003 | P0 | Aplicar contención de archivos a ZIP y exportaciones | — |
| [ ] | SR-004 | P1 | Hacer globales y atómicos los IDs de runs | — |
| [ ] | SR-005 | P1 | Unificar resolución de runs dentro de proyectos | SR-004 |
| [ ] | SR-006 | P1 | Evitar pérdidas y falsos éxitos al guardar sesiones | — |
| [ ] | SR-007 | P1 | Reparar el proxy de desarrollo de la UI | — |
| [ ] | SR-008 | P1 MMM | Rechazar diagnósticos incompletos o no finitos | — |
| [ ] | SR-009 | P0 | Definir contrato de aislamiento para el piloto | — |
| [ ] | SR-010 | P0 | Implementar el entorno aislado y sus límites | SR-009 |
| [ ] | SR-011 | P1 | Controlar concurrencia y ciclo de vida de chats | SR-006, SR-010 |
| [ ] | SR-012 | P1 | Centralizar transiciones y escrituras del estado del run | SR-004, SR-005 |
| [ ] | SR-013 | P1 | Aplicar deadlines reales a pasos y herramientas | SR-010, SR-012 |
| [ ] | SR-014 | P1 | Implementar parada y cancelación con semántica explícita | SR-013 |
| [ ] | SR-015 | P1 | Lanzar workflows estructurados sin decisión del LLM | SR-012 |
| [ ] | SR-016 | P1 | Recuperar runs y gates tras reiniciar el servidor | SR-011, SR-015, SR-049 |
| [ ] | SR-017 | P1 | Registrar consumo completo por intento y ejecución | SR-012, SR-015 |
| [ ] | SR-018 | P1 | Aplicar presupuestos y límites de concurrencia | SR-014, SR-017 |
| [ ] | SR-019 | P1 | Acotar carga, lectura y copia de datasets | SR-005, SR-010 |
| [ ] | SR-020 | P1 | Validar contenido y evidencia de los entregables | SR-012; SR-008 si incluye MMM |
| [ ] | SR-021 | P1 | Versionar runs y registrar procedencia reproducible | SR-012, SR-020 |
| [ ] | SR-022 | P1 | Corregir inferencias y validación del preflight | SR-005, SR-019 |
| [ ] | SR-023 | P1 | Recuperar el flujo de plan → confirmación → inicio | SR-015, SR-017, SR-022 |
| [ ] | SR-024 | P1 | Renderizar eventos de gráficos y tablas del workflow | SR-005 |
| [ ] | SR-025 | P1 | Mostrar la evidencia correcta antes de aprobar un gate | SR-002, SR-024 |
| [ ] | SR-026 | P1 | Completar errores, rechazo, revisión y reanudación en UI | SR-014, SR-016, SR-025 |
| [ ] | SR-027 | P2 | Añadir navegación persistente a proyecto, chat y run | SR-005, SR-006 |
| [ ] | SR-028 | P1 | Alinear onboarding, idioma y documentación con el producto real | SR-023, SR-026 |
| [ ] | SR-029 | P2 | Reparar layout adaptable y lectura de entregables | SR-024 |
| [ ] | SR-030 | P2 | Corregir accesibilidad de formularios, diálogos y estados | SR-025, SR-026, SR-029 |
| [ ] | SR-031 | P1 | Preparar instalación, actualización y recuperación local | SR-001, SR-010, SR-021, SR-050 |
| [~] | SR-032 | P2 | Endurecer el contrato de cartridges y herramientas | SR-010, SR-020 |
| [ ] | SR-033 | P2 | Medir y corregir cuellos de botella de eventos y UI | SR-018, SR-019, SR-024 |
| [ ] | SR-034 | P1 | Construir evaluación analítica y aceptación del piloto | SR-020, SR-021, SR-026 |
| [ ] | SR-035 | V | Definir cliente objetivo, problema y oferta inicial | — |
| [ ] | SR-036 | V | Ejecutar piloto y medir uso, valor y economía real | SR-018, SR-023, SR-026, SR-028, SR-031, SR-034, SR-035, SR-043, SR-044, SR-045, SR-046, SR-048, SR-049, SR-051, SR-052, SR-053 |
| [ ] | SR-037 | P0 compartido | Añadir identidad y autorización por recurso | SR-004, SR-005, SR-006, SR-045 |
| [ ] | SR-038 | P1 SaaS | Separar API y workers mediante jobs durables | SR-010, SR-012, SR-016, SR-018, SR-037 |
| [ ] | SR-039 | P0 SaaS | Aislar almacenamiento y definir retención/borrado | SR-003, SR-021, SR-037 |
| [ ] | SR-040 | P1 SaaS | Instrumentar operación, recuperación y pruebas de carga | SR-038, SR-039 |
| [ ] | SR-041 | V | Decidir lanzamiento SaaS con evidencia técnica y comercial | SR-036, SR-037, SR-038, SR-039, SR-040 |
| [ ] | SR-042 | P1 | Recuperar sesiones sin sobrescribirlas al fallar GET | — |
| [ ] | SR-043 | P1 | Registrar el chat antes del primer envío | SR-005, SR-042 |
| [ ] | SR-044 | P1 | Impedir que respuestas tardías cambien el chat activo | — |
| [ ] | SR-045 | P0 | Proteger acceso HTTP y peticiones desde otros sitios | — |
| [~] | SR-046 | P0 | Limitar variables heredadas por shell y Jupyter | — |
| [ ] | SR-047 | P1 | Persistir proyectos y datasets de forma atómica | — |
| [ ] | SR-048 | P0 | Separar registros de control de archivos del modelo | SR-009, SR-010 |
| [ ] | SR-049 | P1 | Hacer que el servidor sea dueño de la transcripción | SR-006, SR-042, SR-043 |
| [ ] | SR-050 | P1 | Centralizar configuración y soportar cartridges de solo lectura | — |
| [ ] | SR-051 | P1 comercial | Inventariar derechos, licencias y términos de distribución | — |
| [ ] | SR-052 | P1 | Definir exportación, retención y borrado local verificables | SR-003, SR-021, SR-047 |
| [ ] | SR-053 | P1 | Hacer operable la instalación local | SR-011, SR-012, SR-014 |
| [ ] | SR-054 | P1 MMM | Reanudar pasos analíticos largos desde checkpoints | SR-013, SR-016, SR-021 |
| [ ] | SR-055 | P1 comercial | Preparar releases y cadena de suministro | SR-001, SR-031, SR-032, SR-051 |
| [ ] | SR-056 | V / SaaS | Diseñar y verificar cobro y credenciales por cliente | SR-017, SR-018, SR-036, SR-037 |
| [ ] | SR-057 | P2 condicional | Desglosar requisitos enterprise respaldados por demanda | SR-035, SR-037, SR-040 |

## Especificaciones

<a id="sr-001"></a>

### SR-001 — Línea base y CI

**Alcance:** comandos de desarrollo, configuración pytest, lock de Python, checks backend/build y análisis de dependencias. Archivos: `pyproject.toml`, configuración CI y documentación de instalación.

**Aceptación:** instalación limpia con versiones reproducibles; comando documentado sin errores de importación de `tests.fakes`; suite sin llamadas pagadas por defecto; lint y build en CI; pruebas de modelo/Docker identificadas como opt-in y sus skips visibles. Evaluar vulnerabilidades reales y registrar su resolución o disposición; no asumirlas por la edad de una dependencia. Base histórica: 369 passed, 12 skipped, build correcto y siete hallazgos Ruff.

<a id="sr-002"></a>

### SR-002 — Contenido activo

**Alcance:** `serve.py`, `api.py`, `FileViewer.tsx`, enlaces de archivos del chat y exportaciones. Definir una política común para HTML/SVG y demás formatos activos, incluyendo acceso directo a la URL.

**Aceptación:** un documento de prueba puede mostrar interactividad permitida pero no leer ni modificar la API bajo el origen de la aplicación; todas las rutas de runs/proyectos aplican la política; no se depende de `target=_blank` como aislamiento. Verificar en navegador encabezados y comportamiento, sin afectar datos reales.

<a id="sr-003"></a>

### SR-003 — Contención en exportaciones

**Alcance:** `_archive_members`, creación de ZIP y lectores de archivos de exportación.

**Aceptación:** enlaces simbólicos fuera del workspace, rutas escapadas y archivos internos no entran en ningún modo de exportación; la misma política se cumple al servir archivos individuales. Reproducir con un marcador temporal fuera del workspace; mantener exportables los entregables legítimos.

**Reconciliación:** Cubrir también matched, keep_version, snapshot y todos los lectores del host, no solo ZIP. La comprobación y apertura deben evitar carreras de sustitución de enlaces.

<a id="sr-004"></a>

### SR-004 — Identidad global de runs

**Alcance:** asignación de IDs, creación de directorios, lookup y compatibilidad con runs existentes.

**Aceptación:** dos proyectos lanzando el mismo workflow en el mismo instante obtienen identidades distintas; creación concurrente no sobrescribe runs; start/stop/gate/read operan sobre el propietario correcto; colisiones heredadas se detectan explícitamente y nunca se resuelven por un “primer resultado” ambiguo.

<a id="sr-005"></a>

### SR-005 — Resolución por proyecto

**Alcance:** centralizar resolución validada y reutilizarla en preview, archivos, eventos, gráficos, versiones, exportaciones y acciones.

**Aceptación:** una matriz de pruebas compara proyecto default y no-default; el CSV que devolvía 200 en files y 404 en preview funciona en ambos; IDs desconocidos/rutas inválidas fallan de forma consistente; no hay fallback silencioso a otro proyecto.

**Reconciliación:** Incluir herramientas list/read/amend, charts y estimaciones. El launcher debe usar el proyecto correcto mientras SR-015 sustituye su envoltorio LLM; no conservar un entorno global por defecto.

<a id="sr-006"></a>

### SR-006 — Persistencia de sesiones

**Alcance:** `sessions.py`, API de sesiones y guardado/recuperación en `ChatCenter`.

**Aceptación:** escritura atómica y control de versión del servidor; una petición vieja no sobrescribe una nueva; conflicto de dos pestañas se maneja explícitamente; fallo de disco devuelve error y la UI conserva/reintenta datos pendientes. No confiar solo en timestamps suministrados por el cliente. Probar reload y cierre después de una respuesta larga.

**Reconciliación:** El backend distingue registro corrupto de 404 real. SR-042 elimina la inicialización destructiva; SR-049 incorpora transcripción autoritativa. Atomicidad de un archivo no resuelve escrituras concurrentes.

<a id="sr-007"></a>

### SR-007 — Proxy de desarrollo

**Alcance:** `ui/vite.config.ts` y contratos de rutas que usa la SPA.

**Aceptación:** `/projects`, `/sessions` y demás superficies utilizadas llegan a FastAPI en desarrollo; carga inicial, registro/lectura de chat, datasets y runs funcionan con Vite y con el build servido por FastAPI.

<a id="sr-008"></a>

### SR-008 — Gate MMM estricto

**Alcance:** `cartridges/ds/workflows/mmm-meridian/scripts/check_rhat.py` y productor de diagnósticos.

**Aceptación:** rechazar campos obligatorios ausentes, tipos erróneos, NaN, infinitos y valores inválidos; tests positivos y negativos; conservar evidencia de qué comprobación falló. La validación independiente contra el posterior real se completa en SR-020. Corregir el parser no convierte por sí solo MMM en un producto validado.

<a id="sr-009"></a>

### SR-009 — Diseño del aislamiento

**Entregable:** decisión de arquitectura sobre el entorno soportado del piloto, sin cambiar aún el proveedor por conveniencia.

**Aceptación:** documentar filesystem accesible, secretos, red saliente, CPU/RAM/PIDs, persistencia, limpieza, sistema operativo soportado y límites de confianza. Decidir cómo mantener Python persistente dentro del aislamiento y separar credenciales del orquestador de procesos de análisis. Explicar qué cubre Docker y qué garantías adicionales exige un servicio multiusuario.

**Reconciliación:** Resolver DEC-04 con un modelo de amenazas. Una VM por cliente reduce mezcla entre clientes, pero no protege el plano de control frente al código generado dentro de esa VM.

<a id="sr-010"></a>

### SR-010 — Entorno aislado

**Alcance:** implementar SR-009 en `envs/`, preservando el protocolo y las invariantes del runner.

**Aceptación:** pruebas reales del entorno muestran que código generado no puede leer un marcador del host ni secretos no concedidos, ni acceder a destinos de red prohibidos; límites de recursos aplicados; limpieza tras error/cancelación; EDA puede generar sus entregables. No usar `inherit_env=True` ni un fallback local silencioso para el modo seguro. Tests Docker/entorno separados de la suite rápida.

**Reconciliación:** Aplicar también a chat/ipykernel: red por denegación inicial y excepciones explícitas, sin Docker socket/IMDS/API interna, límites CPU/RAM/PIDs/GPU, capabilities mínimas, no-new-privileges, filesystem de sistema inmutable y temporales acotados. Probar ambos backends, no asumir aislamiento por implementar un protocolo llamado sandbox.

<a id="sr-011"></a>

### SR-011 — Concurrencia de chats

**Alcance:** ownership de kernels/workspaces y caché de agentes por proyecto/sesión.

**Aceptación:** dos conversaciones concurrentes no mezclan respuestas ni variables involuntariamente; política de compartir datos explícita; exclusión o aislamiento según esa política; entornos liberados al cerrar/reiniciar. Probar cambio rápido de proyecto y sesión, solicitudes retrasadas y materialización concurrente de skills.

**Reconciliación:** Añadir TTL/LRU, máximo de entornos y single-flight de creación; evitar trabajo síncrono pesado en el event loop. Kernel persistente por conversación o política explícita de compartir, nunca accidental.

<a id="sr-012"></a>

### SR-012 — Estado del run

**Alcance:** API interna de transiciones/persistencia compartida por runner, driver, API y herramientas de gráficos. Refactor acotado a ownership, sin reescribir el bucle del agente.

**Aceptación:** estados y transiciones válidos definidos; escrituras concurrentes no pierden gráficos/gates; start/stop/aprobación idempotentes; errores/reintentos no conservan información engañosa del intento anterior. Versionar el contrato de persistencia y preservar lectura de runs existentes.

<a id="sr-013"></a>

### SR-013 — Deadlines

**Alcance:** parsear y aplicar `Step.timeout`; coordinar timeout del paso y de herramientas con el entorno.

**Aceptación:** un paso que supera su plazo termina con estado y motivo explícitos; timeout de Jupyter interrumpe la ejecución real; comandos de contenedor no siguen consumiendo recursos sin control tras vencer el plazo del cliente. Probar una tarea deliberadamente lenta sin modelo.

**Reconciliación:** Cubrir wall-clock del run además del paso. Cancelar un Future o vencer una espera no demuestra que el proceso de análisis terminó.

<a id="sr-014"></a>

### SR-014 — Parada y cancelación

**Alcance:** API, runner/driver, entorno y control de UI.

**Aceptación:** distinguir “detener después del paso” de “cancelar ahora”; limpieza y resultado parcial identificados; un stop no comienza pasos nuevos; cancelar un gate no queda esperando indefinidamente; duración de un run detenido deja de crecer y su stream finaliza correctamente. Verificar reanudación solo cuando tenga semántica válida.

<a id="sr-015"></a>

### SR-015 — Inicio determinista

**Alcance:** el launcher invoca directamente el workflow elegido, con sus inputs registrados; el chat conserva el routing del modelo.

**Aceptación:** comenzar desde un formulario no requiere que el LLM elija llamar a `run_workflow`; no ejecuta dos veces con doble clic/reintento; identidad, eventos, gates, artefactos y consumo conservan el contrato. Proteger con pruebas de ambas vías de inicio.

<a id="sr-016"></a>

### SR-016 — Recuperación tras reinicio

**Alcance:** coordinar transcripciones, checkpoints y estado persistido de launcher y chat.

**Aceptación:** reiniciar antes/después de un gate mantiene la pregunta y su run; una respuesta ya aceptada no se reasigna a otro gate; pasos completos no vuelven a cobrarse/ejecutarse; trabajos interrumpidos quedan visibles como recuperables o fallidos. No introducir dos copias de transcripción que diverjan sin reconciliación.

<a id="sr-017"></a>

### SR-017 — Consumo completo

**Alcance:** libro de consumo incremental por llamada/intento/run y resumen de UI.

**Aceptación:** contabilizar orquestación, chat relacionado, pasos, reintentos y fallos; preservar consumo de intentos previos; distinguir subtotal conocido de total con precios desconocidos. Registrar versión de tarifas y reconciliar una muestra con uso del proveedor cuando se haga una ejecución autorizada. El costo de infraestructura/soporte se medirá en SR-036.

**Reconciliación:** El ledger debe cubrir subagentes, llamadas fallidas, compute/GPU y almacenamiento cuando existan. No exigir contadores finales antes de stream: registrar progresivamente y reconciliar al cerrar/fallar.

<a id="sr-018"></a>

### SR-018 — Presupuestos y admisión

**Alcance:** límites configurables de gasto/tokens, trabajos simultáneos y recursos.

**Aceptación:** rechazar o poner en espera trabajo por encima del límite antes de aprovisionarlo; impedir nuevas llamadas cuando se agota presupuesto; mostrar motivo y consumo pendiente. Documentar posible exceso de una llamada en vuelo; no prometer un límite monetario exacto si el proveedor no permite imponerlo.

<a id="sr-019"></a>

### SR-019 — Datasets acotados

**Alcance:** upload, lectura de preview/perfil, exportaciones grandes y copia a runs.

**Aceptación:** tamaños máximos y formatos soportados visibles; upload por streaming o lectura acotada; Parquet no se carga completo para ver pocas filas; datos malformados producen errores comprensibles. Evitar copiar indiscriminadamente datasets no utilizados; sobrescribir un nombre debe tener una semántica explícita y no alterar la entrada de un run iniciado.

<a id="sr-020"></a>

### SR-020 — Contratos de salida y evidencia

**Alcance:** validadores declarados por cartridge; el harness ejecuta el contrato sin asumir reglas de DS.

**Condición de alcance:** para el piloto EDA, cerrar su contrato y evaluación. SR-008 y los criterios del posterior son obligatorios al incluir MMM; registrar esa parte como subtarea pendiente si se difiere el workflow, sin declararla validada.

**Aceptación:** archivos vacíos, obsoletos o con esquema incorrecto no cuentan como entrega correcta; resultados principales se contrastan con tablas calculadas; validadores de MMM derivan/comparan diagnósticos con el modelo real. Separar “archivo producido”, “validación ejecutada” y “resultado aprobado”. Incluir inputs adversariales y conclusiones sin soporte.

**Reconciliación:** Añadir casos de prompt injection en archivos/datos externos: separar instrucciones confiables, procedencia y permisos. Gates de acciones externas requieren actor autorizado y no pueden autoaprobarse desde el sandbox; filtrar texto por patrones no es una defensa suficiente.

<a id="sr-021"></a>

### SR-021 — Procedencia y versiones

**Alcance:** manifiesto de run y lectura/migración de esquemas persistidos.

**Aceptación:** guardar hashes de inputs, versión/contenido relevante de workflow y prompts/skills, configuración del modelo, dependencias o digest del entorno, código generado e intentos; una reanudación detecta incompatibilidades. Un tercero puede identificar qué produjo un resultado; no prometer determinismo del LLM por disponer de hashes.

**Reconciliación:** Versionar también proyectos, sesiones y gráficos. Migraciones explícitas con dry-run, fixtures previas, idempotencia y respaldo; rechazar versiones futuras incompatibles de forma visible. No ignorar indiscriminadamente campos desconocidos ni ocultar registros corruptos.

<a id="sr-022"></a>

### SR-022 — Preflight correcto

**Alcance:** `_profile_inputs`, `_guesses`, tipos/opciones requeridas y validación de inputs antes de iniciar.

**Aceptación:** 500 filas muestreadas no se describen como prueba de unicidad global en 1.000; casos con duplicados fuera de la muestra; CSV irregular no genera un 500; defaults, campos obligatorios, opciones y referencias a archivos se validan antes de gastar recursos. No ejecutar un modelo para describir datos que se pueden comprobar directamente.

<a id="sr-023"></a>

### SR-023 — Plan antes de iniciar

**Alcance:** launcher consume `/plan` y muestra datos, inputs, pasos, gates y estimación con sus limitaciones.

**Aceptación:** usuario puede editar y revisar antes de iniciar; cancelar no deja runs vacíos huérfanos; costo desconocido se indica como desconocido; inputs confirmados son exactamente los del run. Una prueba de navegador recorre upload/selección → plan → editar → iniciar.

<a id="sr-024"></a>

### SR-024 — Gráficos y tablas de workflows

**Alcance:** reducer `useRunStream`, contratos de eventos y vista del run.

**Aceptación:** los siete eventos `dsagent.chart` del replay aparecen con datos por referencia; reload/reconexión no duplica objetos; nueva versión reemplaza correctamente la anterior sin perder historial; errores de lectura se muestran. Alinear gramática y política de carga de datos de gráficos de chat, archivos y workflow; conservar exportación interactiva.

<a id="sr-025"></a>

### SR-025 — Evidencia del gate

**Alcance:** `GateCard`, selección/foco de artefactos y versiones rechazadas.

**Aceptación:** antes de aprobar, la UI abre el documento objeto de revisión y sus enlaces funcionan; indica qué acción se autoriza y qué trabajo sigue; rechazo conserva nota/evidencia; siguiente revisión permite consultar la versión anterior. No seleccionar el informe solo después de aprobarlo.

<a id="sr-026"></a>

### SR-026 — Recuperación y errores en UI

**Alcance:** RunDock, estado de conexión, errores API y acciones de recuperación.

**Aceptación:** error identifica paso y motivo; reintentar continúa desde el punto correcto; rechazar → corregir → volver a revisar → aprobar funciona; recargar/reiniciar durante el flujo conserva contexto recuperable. Verificar también sin credenciales, desconexión, run detenido, doble clic y error de guardado; evitar capturas silenciosas que aparentan éxito.

**Reconciliación:** Añadir ErrorBoundary y errores visibles de mutaciones; no tragar fallos de red, del servidor o de persistencia. SR-042/044 cubren carreras específicas por separado.

<a id="sr-027"></a>

### SR-027 — Navegación persistente

**Alcance:** URLs/estado de navegación de proyecto, sesión y run.

**Aceptación:** reload y enlace directo restauran la selección; atrás/adelante funciona; ID inválido muestra estado claro; cambios rápidos de proyecto no aplican resultados de peticiones viejas. Enlaces locales no se presentan como enlaces públicos para compartir.

<a id="sr-028"></a>

### SR-028 — Onboarding e información veraz

**Alcance:** lenguaje de UI, descriptions de workflows y README raíz/UI.

**Aceptación:** elegir idioma inicial consistente y dejar explícita la decisión; campos orientados a resultados con detalles técnicos secundarios; eliminar “harness smoke test” del flujo del cliente; describir almacenamiento real de chats, procesamiento por proveedores, límites y borrado. Documentación refleja Vite/React y las rutas vigentes; eliminar instrucciones obsoletas de Next/CopilotKit/A2UI que ya no apliquen. Verificar onboarding con alguien que no conozca el harness.

**Reconciliación:** Si DEC-03 exige plataforma general, roster, sugerencias, marca y exportación dependen del cartridge; para piloto DS basta idioma coherente e información veraz. White-label no es requisito universal.

<a id="sr-029"></a>

### SR-029 — Layout adaptable y entregables

**Alcance:** `.shell`, paneles, selección de artefactos y espacio principal de lectura.

**Aceptación:** comprobar al menos 390, 768 y 1440 px; navegación y acción principal accesibles sin depender de la vieja clase `.app`; informe maximizable y legible; separar entregables de archivos de trabajo; una conversación vacía no desplaza la tarea activa a un panel estrecho.

<a id="sr-030"></a>

### SR-030 — Accesibilidad

**Alcance:** nombres/labels, foco, diálogos, teclado, estados y gráficos.

**Aceptación:** labels asociados, nombre accesible de iconos, entrada/retorno de foco y cierre de modal, navegación solo con teclado y avisos comprensibles de progreso/error; contraste y movimiento revisados; gráficos con alternativa tabular cuando corresponda. Adjuntar comprobaciones automáticas y revisión manual; no declarar conformidad global sin auditarla.

<a id="sr-031"></a>

### SR-031 — Distribución y recuperación local

**Alcance:** instalación fuera del checkout, descubrimiento de cartridges/UI, comandos y backup/restore.

**Aceptación:** instalar artefacto en entorno limpio y ejecutar fuera del repositorio; `--version` funciona; requisitos y extras correctos; una copia de seguridad restaura proyectos, sesiones, runs y evidencia en instancia vacía; actualización compatible o migración documentada. No incluir credenciales ni datasets privados en paquetes.

**Reconciliación:** Backup automático configurado y restauración ensayada en instancia vacía. Migración del layout y recuperación de huérfanos con dry-run; no eliminar datos automáticamente por no reconocerlos. Compose/imagen con UI y datos persistentes; Helm solo si el despliegue lo exige.

<a id="sr-032"></a>

### SR-032 — Contrato de cartridges

- [x] SR-032.a (2026-09-24): permisos efectivos de herramientas en workflows/chat,
  delegación restringida y validación de grants; [contrato y evidencia](../tool-permissions.md).
  El padre conserva los criterios restantes de esta sección.

**Alcance:** loader, namespaces, herramientas concedidas, visibilidad y generación de skills.

**Aceptación:** cartridge ficticio de otro dominio funciona; nombres repetidos de persona/workflow entre cartridges no se sobreescriben silenciosamente; metadatos `tools` se aplican o se rechazan/documentan explícitamente si no soportados. Probar límites de rutas en scripts y coherencia de skills; distinguir alcance de herramientas de aislamiento del sistema operativo. Mantener compatibilidad con el contrato de plugin declarado.

**Reconciliación:** Aplicar selección real del cartridge por proyecto, allowlist de modelos y contrato documentado/scaffold. Otro dominio valida la afirmación de plataforma general (DEC-03); no es requisito para demostrar valor EDA.

<a id="sr-033"></a>

### SR-033 — Rendimiento medido

**Alcance:** SSE/log cursor, polling, memoria de sesiones y chunks frontend.

**Aceptación:** benchmark reproducible con volúmenes/concurrencia representativos y límites publicados; no releer todo el log por suscriptor cada 250 ms; detener polling/streams terminales; recortar carga inicial donde los datos lo justifiquen. Comparar latencia, RAM y transferencia antes/después; la optimización no pierde ni duplica eventos.

**Reconciliación:** Medir recorrido de artefactos con poda, índices de cabeceras de sesiones, copias por turno y polling según visibilidad/backoff. Un cursor SSE sobre archivos no prueba distribución entre réplicas; esta requiere SR-038/039.

<a id="sr-034"></a>

### SR-034 — Evaluación analítica y puerta del piloto

**Entregable:** corpus versionado, evaluador y resultados de aceptación de EDA; MMM tiene una matriz separada.

**Aceptación:** al menos 20 tareas representativas propuestas, con respuestas verificables, repeticiones en casos críticos, datos sucios, duplicados, huecos temporales, unidades ambiguas, muestras pequeñas, leakage y casos que exigen preguntar o rechazar. Medir corrección numérica/narrativa, errores graves, finalización y tiempo/costo de revisión. Verificar de manera independiente cada resultado principal del conjunto de aceptación; los tests rápidos no gastan modelos por defecto.

<a id="sr-035"></a>

### SR-035 — Cliente y oferta

**Entregable:** `docs/product/pilot-hypothesis.md`, a crear durante esta tarea.

**Aceptación:** elegir un segmento y un trabajo recurrente, comprador/usuario, alternativa actual, resultado entregable, propuesta de precio y mecanismo de adquisición. Distinguir hipótesis de evidencia de conversaciones/datos. Evaluar consultorías analíticas y equipos de medición como candidatos, no como mercado ya validado. Preparar preguntas y criterios para decidir; contactar personas requiere instrucción explícita del usuario.

<a id="sr-036"></a>

### SR-036 — Piloto comercial

**Entregable:** protocolo, registro por tarea/cliente y readout de continuar/cambiar/detener.

**Aceptación:** seleccionar 3–5 colaboradores como objetivo propuesto; entorno y datasets autorizados; medir éxito, correcciones, tiempo ahorrado frente al método anterior, repetición de uso y disposición/pago real. Economía incluye proveedores, infraestructura, reintentos y soporte humano. No deducir margen de un run histórico de $0.47. Antes de comenzar, cerrar los bloqueos del checklist de piloto de este archivo; registrar aplazamientos P2 con motivo.

**Reconciliación:** Cerrar además la puerta correspondiente al despliegue del apartado final. Precio anual, 2–3 clientes de pago antes de SaaS y 3–5 colaboradores son hipótesis a decidir, no demanda probada. No comprometer RPO/RTO, eliminación en 24 h ni soporte sin medición y responsable.

<a id="sr-037"></a>

### SR-037 — Identidad y permisos en instalaciones compartidas

**Alcance:** usuarios/organizaciones, sesiones autenticadas y autorización del lado servidor.

**Aceptación:** matriz de permisos cubre proyectos, datasets, chats, runs, gates, archivos/exportaciones y streams; pruebas negativas entre tenants; cambiar un ID no concede acceso; identidad del revisor queda registrada. La política de acceso se verifica también en jobs y URLs de archivos. No basta con añadir una pantalla de login.

**Reconciliación:** Obligatorio antes de acceso de varias personas, incluso self-host. Principal y roles/approver viajan a API, herramientas y jobs; auditoría de actor fuera del workspace y pruebas negativas por usuario/proyecto, y por tenant si corresponde. OIDC es una opción, no una compra decidida. Un token único no cumple permisos por persona.

<a id="sr-038"></a>

### SR-038 — Jobs y workers durables

**Alcance:** decisión e implementación de cola, persistencia transaccional y ownership de jobs.

**Aceptación:** dos procesos API/workers no ejecutan dos veces un job por una carrera; caída y reintento tienen idempotencia definida; leases, heartbeats y recuperación observables; gates sobreviven; límites globales y por tenant. Elegir servicios después de medir requisitos; no crear microservicios adicionales sin necesidad.

**Reconciliación:** Antes de réplicas, elegir store transaccional para metadata/eventos/checkpoints con migraciones; un Postgres es candidato, no decisión tomada. Checkpoints durables de chat requieren SR-049, no solo migrar el saver. Events con offsets durables/fanout y leases de sandbox. CPU/GPU con admisión/asignación explícitas e imágenes preconstruidas; sticky sessions no resuelven durabilidad.

<a id="sr-039"></a>

### SR-039 — Almacenamiento y retención

**Alcance:** archivos de tenants, transferencias autorizadas, cuotas, ciclo de vida y borrado.

**Aceptación:** objetos inaccesibles entre tenants; URLs temporales respetan permisos/vencimiento; integridad de inputs/versiones; política de retención y borrado comprobable en datos, derivados y backups según política publicada. Sandbox de contenido activo también se aplica al almacenamiento remoto.

**Reconciliación:** Separar referencias inmutables a datasets de copias de workspace; minimizar previews persistidos, preservar versiones y servir contenido activo desde un límite de origen seguro. Cifrado/claves, cuotas, compactación de checkpoints, exportación y recibos de borrado acordes al modelo desplegado.

<a id="sr-040"></a>

### SR-040 — Operación y recuperación SaaS

**Alcance:** logs/metrics/traces, alertas, despliegue/rollback, backup/restore y carga.

**Aceptación:** seguir un run sin exponer secretos/datasets en logs; detectar jobs atascados, errores, agotamiento de recursos y gasto anómalo; restauración ensayada; objetivos de recuperación y límites de servicio definidos y medidos. Pruebas con concurrencia realista y fallo de worker/almacenamiento. Documentar responsables operativos.

**Reconciliación:** Medir fallos de API, worker, almacenamiento y restauración automática; permitir suspensión por run/tenant. RPO/RTO/SLO se fijan según evidencia, no se adoptan los números de la fuente como promesas. Evaluar pools de sandboxes/GPU y almacenamiento por tiers solo con demanda y costo medidos.

<a id="sr-041"></a>

### SR-041 — Decisión de lanzamiento

**Entregable:** evaluación go/no-go con evidencia actual, riesgos aceptados y alcance comercial.

**Aceptación:** pasar checklist SaaS; demanda y economía respaldadas por SR-036; ninguna afirmación de autonomía, privacidad, fiabilidad o soporte excede lo probado. Separar habilitación técnica de publicación: desplegar/abrir acceso es una acción posterior, no consecuencia automática de cerrar este documento.

**Reconciliación:** DEC-01/02/05 resueltas para la oferta; SR-055 antes de distribuir un producto soportado, SR-056 si hay cobro automatizado y subtareas SR-057 solo si se prometen. La preparación técnica no demuestra mercado ni autoriza publicar.

<a id="sr-042"></a>

### SR-042 — Recuperar sesiones sin sobrescribirlas al fallar GET

**Alcance/archivos:** `ui/src/components/ChatCenter.tsx`, `ui/src/lib/api.ts`, `sessions.py`, `api_projects.py`.

**Aceptación:** Distinguir ausencia real, registro corrupto, 401/403, 500 y fallo de red. Solo una ausencia real permite ofrecer creación; nunca guardar mensajes vacíos tras un error. Conservar datos recuperables y mostrar reintento. Probar cada respuesta y verificar que no se emite un PUT destructivo.

<a id="sr-043"></a>

### SR-043 — Registrar el chat antes del primer envío

**Alcance/archivos:** `serve.py`, `sessions.py`, `ChatCenter.tsx`.

**Aceptación:** Esperar confirmación de registro antes de enviar; rechazar hilos desconocidos sin fallback a default. Crear IDs de sesión/proyecto sin colisiones y vincularlos en servidor; IDs opacos no sustituyen permisos. Probar envío inmediato y registro lento en proyecto no-default.

<a id="sr-044"></a>

### SR-044 — Impedir que respuestas tardías cambien el chat activo

**Alcance/archivos:** `ui/src/App.tsx`, `ChatCenter.tsx`.

**Aceptación:** Invalidar respuestas antiguas por identidad de petición/proyecto. Cambiar proyecto, crear chat o escribir mientras llega la lista no debe reemplazar selección ni borrador. Verificar con respuestas deliberadamente fuera de orden.

<a id="sr-045"></a>

### SR-045 — Proteger acceso HTTP y peticiones desde otros sitios

**Alcance/archivos:** `serve.py`, routers API, cliente HTTP/SSE, configuración de despliegue.

**Aceptación:** Definir modos local y remoto. Validar Host y peticiones mutantes según Origin/Fetch Metadata/content-type; probar peticiones desde un sitio ajeno y clientes CLI autorizados. Proteger todas las rutas remotas, incluidos streams y archivos, con autenticación de despliegue y TLS en el ingreso. Probar sesión expirada en UI/SSE. Un bearer compartido no identifica usuarios: una instalación compartida requiere SR-037. Resolver transporte SSE sin tokens en URLs; CORS y TrustedHost aislados no bastan.

<a id="sr-046"></a>

### SR-046 — Limitar variables heredadas por shell y Jupyter

- [x] SR-046.a (2026-09-24): entorno mínimo de shell/Jupyter; secretos marcador y
  persistencia de kernel verificados; [contrato y evidencia](../tool-permissions.md).
- [ ] SR-046.b: directorio de usuario efectivamente controlado dentro del sandbox
  y contrato de credenciales por herramienta. Omitir HOME no cumple ese aislamiento.

**Alcance/archivos:** `envs/kernel.py`, `envs/docker.py`.

**Aceptación:** Aplicar allowlist a ambos procesos, con directorio HOME controlado para la ejecución. Un secreto marcador del servidor no aparece en shell/kernel. Revisar credenciales entregadas por herramienta. Es una reducción de exposición: no cierra el acceso a archivos, red ni procesos del host; SR-009/010 siguen siendo obligatorias.

<a id="sr-047"></a>

### SR-047 — Persistir proyectos y datasets de forma atómica

**Alcance/archivos:** `projects.py`, `datasets.py`, `api_projects.py`.

**Aceptación:** Aplicar escritura temporal/reemplazo, durabilidad definida y exclusión o versión para actualizaciones concurrentes de manifests. No convertir corrupción/fallo de disco en ausencia. Probar fallo intermedio y escritores concurrentes; preservar registro anterior y devolver error accionable. Sesiones quedan en SR-006.

<a id="sr-048"></a>

### SR-048 — Separar registros de control de archivos del modelo

**Alcance/archivos:** `serve.py`, `host/build.py`, checkpointers, lectores/exportadores de archivos.

**Aceptación:** Checkpoints, run state, eventos de control, auditoría y large_tool_results no son modificables ni descargables por el modelo o por las rutas de artefactos. Skills y paquetes del servidor son de solo lectura para la ejecución; ningún pip modifica el venv del servicio. Registrar actor de gates fuera del workspace. Probar manipulación, acceso HTTP desde sandbox y fugas por todos los lectores. Mover un SQLite de carpeta sin aislamiento no cumple.

<a id="sr-049"></a>

### SR-049 — Hacer que el servidor sea dueño de la transcripción

**Alcance/archivos:** `sessions.py`, `serve.py`, `ChatCenter.tsx`, puente AG-UI.

**Aceptación:** Persistir turnos/eventos con secuencia y versión del servidor; aceptar del cliente el nuevo turno, no roles assistant/tool arbitrarios ni sustitución del historial. ACK, reintento idempotente y conflicto de dos pestañas. En la transición, flush normal al cambiar chat; keepalive solo al salir y nunca única garantía para cuerpos grandes. Topes de payload y prueba >64 KB. Esquema propio y fixtures del puente protegen upgrades. Actualizar el comentario de convergencia solo cuando la garantía exista.

<a id="sr-050"></a>

### SR-050 — Centralizar configuración y soportar cartridges de solo lectura

**Alcance/archivos:** `cli.py`, `serve.py`, loader/generador, precios y rutas de UI.

**Aceptación:** Definir raíz absoluta de datos y precedencia CLI > entorno > archivo; funcionar desde otro cwd. Separar generación explícita de skills durante instalación/build de validación/lectura al arrancar. Arrancar con cartridge montado de solo lectura sin escribir en site-packages; resolver UI/precios/cartridges desde artefactos instalados. Puerto/modelo/rutas documentados y secretos excluidos del paquete.

<a id="sr-051"></a>

### SR-051 — Inventariar derechos, licencias y términos de distribución

**Alcance/archivos:** `pyproject.toml`, manifiesto de plugin, dependencias Python/npm/imágenes, futuros LICENSE y THIRD_PARTY_LICENSES.

**Aceptación:** Producir inventario versionado con versión, origen, licencia/texto/avisos y elementos desconocidos; incluir transitivas, imágenes y modelos. Verificar titularidad y permisos antes de proponer licencia por componente. Registrar cuestiones de proveedores y Meridian para revisión según el uso real. Inventario puede empezar ya; publicar textos o cambiar metadata depende de DEC-02/05. No concluir compatibilidad por una lista de nombres ni que BYOK elimina obligaciones.

<a id="sr-052"></a>

### SR-052 — Definir exportación, retención y borrado local verificables

**Alcance/archivos:** `projects.py`, `datasets.py`, sesiones, checkpoints, exportación.

**Aceptación:** Inventariar originales, copias por run, artefactos, versiones de gates, previews, logs, spill y backups. Definir borrado de runs terminados y proyecto completo con previsualización, recibo y plazo de purga; respetar referencias de otros runs. Importación/exportación portable conserva manifiestos e integridad sin secretos. Probar restauración y borrado en datos temporales; los backups tienen política propia, no prometer borrado instantáneo universal.

<a id="sr-053"></a>

### SR-053 — Hacer operable la instalación local

**Alcance/archivos:** `serve.py`, `driver.py`, `envs/`, configuración de logs.

**Aceptación:** Logs estructurados con correlación y redacción; errores persistidos y visibles. Health de proceso y readiness de capacidades sin gastar modelos. SIGTERM deja de admitir, drena o marca interrumpidos y libera entornos; tras caída abrupta reaper usa etiquetas de instancia/ownership y leases para limpiar huérfanos. Probar reinicio y no borrar contenedores de otra instancia. atexit/--rm no cubren por sí solos SIGKILL.

<a id="sr-054"></a>

### SR-054 — Reanudar pasos analíticos largos desde checkpoints

**Alcance/archivos:** cartridge MMM, entorno y sincronización del workspace.

**Aceptación:** El cartridge guarda progreso válido y compatible durante el fit; el harness conserva/sincroniza ese progreso y muestra avance. Matar y reanudar una ejecución de prueba recupera solo trabajo verificable y registra costo parcial. Definir límites cuando librería/modelo no permiten reanudar. No confundir reanudación entre pasos con recuperación dentro de un fit.

<a id="sr-055"></a>

### SR-055 — Preparar releases y cadena de suministro

**Alcance/archivos:** CI, imágenes, loader, documentos de soporte/release.

**Aceptación:** Versiones y digests fijados; imágenes construidas fuera de peticiones; manifests de scripts/Dockerfiles/dependencias verificables. Solo administrador instala cartridges confiables; ejecutar contenido de un registro ajeno requiere aislamiento y revisión adicionales. Publicar matriz de soporte, CHANGELOG, SECURITY, actualización/rollback y canal de reporte; definir respuesta a incidentes y revisar documentación de tratamiento de datos/subprocesadores aplicable con SR-051. Separar firma/registro remoto en subtareas si DEC-03 lo requiere, incluyendo límites del build remoto. No ejecutar scripts de instalación de terceros en el host por defecto.

<a id="sr-056"></a>

### SR-056 — Diseñar y verificar cobro y credenciales por cliente

**Alcance/archivos:** pricing, ledger, integración de cobro y secretos por tenant.

**Aceptación:** Definir tarifa/modelo BYOK o gestionado con DEC-01/05; contabilizar también GPU/CPU/almacenamiento. Si se requiere cobro automatizado, probar eventos idempotentes, conciliación, límites y rotación/revocación de claves con datos de prueba. Proveedor de pagos/KMS por decidir; ninguna prueba cobra a clientes. Registrar costos de soporte y margen medido.

<a id="sr-057"></a>

### SR-057 — Desglosar requisitos enterprise respaldados por demanda

**Alcance/archivos:** documento de requisitos y tareas derivadas.

**Aceptación:** Solo con requisito validado, dividir SSO/SCIM, roles de aprobación, gates firmados, residencia, retención especial, despliegue Helm, auditoría externa y programa de seguridad en tickets independientes. Cada uno con comprador, criterio, costo y dependencia; no declarar certificación ni cumplimiento por escribir políticas. Esta tarea termina con el desglose y una decisión de alcance; no representa haber implementado esas capacidades.

## Puertas de salida

- **Presentación de la startup:** se puede mostrar una alfa y el problema propuesto, indicando el alcance probado. No afirmar producto comercial seguro, demanda validada, autonomía analítica ni garantías de disponibilidad.
- **Piloto supervisado EDA, instalación dedicada:** SR-001–007, 009–026, 028, 031, 034–035 y 042–053. SR-020 evalúa EDA en esta puerta; SR-008 y la parte MMM de SR-020 son condicionales. SR-051 exige inventario y resolver lo aplicable a la oferta, no una licencia predeterminada. Evaluar SR-027/029/030/033 según audiencia y registrar aplazamientos; un fallo que impide usar el flujo soportado deja de ser P2 aplazable. No admitir fallos abiertos de aislamiento, mezcla de proyectos o pérdida de trabajo.
- **Instalación compartida, incluso self-host:** lo anterior más SR-037 y pruebas de permisos entre usuarios/proyectos. SR-038/039/040 se exigen antes de varios workers/réplicas o de prometer disponibilidad que la instalación simple no soporta; SR-053/031 cubren operación/restore mínimos en una instancia.
- **MMM:** además SR-008, evidencia independiente del posterior en SR-020/034, integración real del entorno y límites de recursos. SR-054 es obligatorio si la oferta promete recuperación dentro del fit; de otro modo explicitar pérdida de trabajo y costo tras interrupción. Aprobación humana no demuestra validez causal.
- **Distribución comercial soportada:** las puertas aplicables, DEC-01/02/05, SR-051/055, instalación limpia y plan de soporte. Acuerdo y precio definidos en SR-035/036; no basta una VM y un token.
- **SaaS compartido:** todas las condiciones aplicables, SR-036–041, SR-055 y decisiones resueltas. SR-056 si hay cobro automatizado; capacidades enterprise de SR-057 solo tras desglosarlas y verificarlas. Multi-tenant exige aislamiento y autorización por tenant, no solamente por proyecto.

Objetivos propuestos para el conjunto soportado: al menos 95% de finalización, cero fallos graves de seguridad o integridad, resultados principales verificados independientemente y costo/revisión humana medidos. Confirmar población, umbrales y método antes de evaluar; el porcentaje no autoriza errores graves ni prueba generalización. No hay fecha de lanzamiento comprometida.

## Registro de ejecución

| ID | Responsable | Inicio | Cierre | Commit/PR | Validación y evidencia | Desviaciones |
|---|---|---|---|---|---|---|
| UI-001 | Codex | 2026-09-19 | 2026-09-19 | Cambios locales, sin commit/PR | [Build y navegador](../ui-workspace-redesign-2026-09-19.md) | Priorización del usuario; no cierra los padres SR |
| SR-032.a / SR-046.a | Codex | 2026-09-24 | 2026-09-24 | `harness: enforce tool grants and minimize compute credentials [ADR]` | [Contrato, pruebas de dispatch y kernel real](../tool-permissions.md) | Los padres siguen parciales; sandbox, HOME controlado y credenciales por herramienta pendientes |

Evidencias futuras: `docs/runs/startup-readiness/SR-xxx/`. El cierre de documentos de planificación no cierra ninguna tarea de implementación.
