# Backlog de preparación de DSAgent como producto

Creado: 2026-09-19. Estado: planificación; ninguna tarea de implementación iniciada.

Fuente: [evaluación técnica y de producto completa](startup-readiness-review-2026-09-18.md), realizada sobre `8ac2758`. El código puede cambiar: reproducir cada hallazgo antes de modificarlo. Los resultados de aquella revisión son una línea base histórica, no una certificación de versiones posteriores.

**Objetivo:** convertir el prototipo en un producto confiable para un piloto supervisado y validar una necesidad recurrente por la que un cliente pague. La plataforma compartida se aborda después, si el modelo comercial la requiere. Se conservan React/Vite, FastAPI, Deep Agents y la separación cartridge → host → runner → entorno.

El diagnóstico es favorable para presentar una startup en fase alfa. Los bloqueos principales son aislamiento de ejecución, archivos activos y exportaciones, identidad de runs, persistencia, confianza analítica y recuperación desde la UI. No se ha demostrado todavía demanda, retención ni rentabilidad. Los tests del harness no equivalen a validar las conclusiones de un análisis.

## Cómo trabajar esta lista

- Este archivo es la fuente de estado de las tareas `SR-*`; [ROADMAP.md](ROADMAP.md) mantiene los hitos históricos y enlaza aquí. No duplicar sus checkboxes en otros documentos.
- Estados: `[ ]` pendiente, `[~]` en curso, `[x]` terminada. Marcar una sola tarea técnica en curso. Orden recomendado: el de la tabla, respetando dependencias. `SR-035` puede comenzar antes como investigación comercial.
- Una tarea técnica por rama/PR. Si al diseñarla resulta demasiado grande, dividirla en subtareas con IDs derivados antes de implementarla. Las tareas de investigación producen documentos y evidencia, no necesariamente código.
- Al comenzar: reproducir el caso, confirmar alcance, registrar responsable y cambiar el estado. Al cerrar: enlazar commit/PR, validación y evidencia. Una implementación previa no basta si el flujo actual sigue fallando.
- Añadir pruebas de regresión a defectos funcionales o de seguridad. Para cambios de texto/documentación basta revisión. Las pruebas de UI deben recorrer el comportamiento relevante, no replicar la implementación.
- Mantener las invariantes de [CLAUDE.md](../CLAUDE.md); registrar cambios de arquitectura en [architecture.md](architecture.md) y decisiones en el roadmap. Deep Agents sigue siendo dueño del bucle del agente.
- Los criterios de aceptación aquí son trabajo propuesto. No implican que las decisiones técnicas pendientes ya estén tomadas ni autorizan despliegues, gasto de modelos o contacto con clientes.

Prioridades: **P0** corrige un límite de confianza o acceso a datos; **P1** afecta la fiabilidad del flujo soportado; **P2** mejora distribución, experiencia o eficiencia; **V** valida producto/negocio. **P0 SaaS** es obligatorio antes de un servicio compartido, aunque se planifique después del piloto local.

## Lista de ejecución

| Estado | ID | Prioridad | Tarea | Depende de |
|---|---|---|---|---|
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
| [ ] | SR-016 | P1 | Recuperar runs y gates tras reiniciar el servidor | SR-011, SR-015 |
| [ ] | SR-017 | P1 | Registrar consumo completo por intento y ejecución | SR-012, SR-015 |
| [ ] | SR-018 | P1 | Aplicar presupuestos y límites de concurrencia | SR-014, SR-017 |
| [ ] | SR-019 | P1 | Acotar carga, lectura y copia de datasets | SR-005, SR-010 |
| [ ] | SR-020 | P1 | Validar contenido y evidencia de los entregables | SR-008, SR-012 |
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
| [ ] | SR-031 | P1 | Preparar instalación, actualización y recuperación local | SR-001, SR-010, SR-021 |
| [ ] | SR-032 | P2 | Endurecer el contrato de cartridges y herramientas | SR-010, SR-020 |
| [ ] | SR-033 | P2 | Medir y corregir cuellos de botella de eventos y UI | SR-018, SR-019, SR-024 |
| [ ] | SR-034 | P1 | Construir evaluación analítica y aceptación del piloto | SR-020, SR-021, SR-026 |
| [ ] | SR-035 | V | Definir cliente objetivo, problema y oferta inicial | — |
| [ ] | SR-036 | V | Ejecutar piloto y medir uso, valor y economía real | SR-018, SR-023, SR-026, SR-028, SR-031, SR-034, SR-035 |
| [ ] | SR-037 | P0 SaaS | Añadir identidad y autorización por recurso | SR-004, SR-005, SR-006, SR-036 |
| [ ] | SR-038 | P1 SaaS | Separar API y workers mediante jobs durables | SR-010, SR-012, SR-016, SR-018, SR-037 |
| [ ] | SR-039 | P0 SaaS | Aislar almacenamiento y definir retención/borrado | SR-003, SR-021, SR-037 |
| [ ] | SR-040 | P1 SaaS | Instrumentar operación, recuperación y pruebas de carga | SR-038, SR-039 |
| [ ] | SR-041 | V | Decidir lanzamiento SaaS con evidencia técnica y comercial | SR-036, SR-037, SR-038, SR-039, SR-040 |

## Especificaciones de las tareas

### SR-001 — Línea base y CI

**Alcance:** comandos de desarrollo, configuración pytest, lock de Python, checks backend/build y análisis de dependencias. Archivos: `pyproject.toml`, configuración CI y documentación de instalación.

**Aceptación:** instalación limpia con versiones reproducibles; comando documentado sin errores de importación de `tests.fakes`; suite sin llamadas pagadas por defecto; lint y build en CI; pruebas de modelo/Docker identificadas como opt-in y sus skips visibles. Evaluar vulnerabilidades reales y registrar su resolución o disposición; no asumirlas por la edad de una dependencia. Base histórica: 369 passed, 12 skipped, build correcto y siete hallazgos Ruff.

### SR-002 — Contenido activo

**Alcance:** `serve.py`, `api.py`, `FileViewer.tsx`, enlaces de archivos del chat y exportaciones. Definir una política común para HTML/SVG y demás formatos activos, incluyendo acceso directo a la URL.

**Aceptación:** un documento de prueba puede mostrar interactividad permitida pero no leer ni modificar la API bajo el origen de la aplicación; todas las rutas de runs/proyectos aplican la política; no se depende de `target=_blank` como aislamiento. Verificar en navegador encabezados y comportamiento, sin afectar datos reales.

### SR-003 — Contención en exportaciones

**Alcance:** `_archive_members`, creación de ZIP y lectores de archivos de exportación.

**Aceptación:** enlaces simbólicos fuera del workspace, rutas escapadas y archivos internos no entran en ningún modo de exportación; la misma política se cumple al servir archivos individuales. Reproducir con un marcador temporal fuera del workspace; mantener exportables los entregables legítimos.

### SR-004 — Identidad global de runs

**Alcance:** asignación de IDs, creación de directorios, lookup y compatibilidad con runs existentes.

**Aceptación:** dos proyectos lanzando el mismo workflow en el mismo instante obtienen identidades distintas; creación concurrente no sobrescribe runs; start/stop/gate/read operan sobre el propietario correcto; colisiones heredadas se detectan explícitamente y nunca se resuelven por un “primer resultado” ambiguo.

### SR-005 — Resolución por proyecto

**Alcance:** centralizar resolución validada y reutilizarla en preview, archivos, eventos, gráficos, versiones, exportaciones y acciones.

**Aceptación:** una matriz de pruebas compara proyecto default y no-default; el CSV que devolvía 200 en files y 404 en preview funciona en ambos; IDs desconocidos/rutas inválidas fallan de forma consistente; no hay fallback silencioso a otro proyecto.

### SR-006 — Persistencia de sesiones

**Alcance:** `sessions.py`, API de sesiones y guardado/recuperación en `ChatCenter`.

**Aceptación:** escritura atómica y control de versión del servidor; una petición vieja no sobrescribe una nueva; conflicto de dos pestañas se maneja explícitamente; fallo de disco devuelve error y la UI conserva/reintenta datos pendientes. No confiar solo en timestamps suministrados por el cliente. Probar reload y cierre después de una respuesta larga.

### SR-007 — Proxy de desarrollo

**Alcance:** `ui/vite.config.ts` y contratos de rutas que usa la SPA.

**Aceptación:** `/projects`, `/sessions` y demás superficies utilizadas llegan a FastAPI en desarrollo; carga inicial, registro/lectura de chat, datasets y runs funcionan con Vite y con el build servido por FastAPI.

### SR-008 — Gate MMM estricto

**Alcance:** `cartridges/ds/workflows/mmm-meridian/scripts/check_rhat.py` y productor de diagnósticos.

**Aceptación:** rechazar campos obligatorios ausentes, tipos erróneos, NaN, infinitos y valores inválidos; tests positivos y negativos; conservar evidencia de qué comprobación falló. La validación independiente contra el posterior real se completa en SR-020. Corregir el parser no convierte por sí solo MMM en un producto validado.

### SR-009 — Diseño del aislamiento

**Entregable:** decisión de arquitectura sobre el entorno soportado del piloto, sin cambiar aún el proveedor por conveniencia.

**Aceptación:** documentar filesystem accesible, secretos, red saliente, CPU/RAM/PIDs, persistencia, limpieza, sistema operativo soportado y límites de confianza. Decidir cómo mantener Python persistente dentro del aislamiento y separar credenciales del orquestador de procesos de análisis. Explicar qué cubre Docker y qué garantías adicionales exige un servicio multiusuario.

### SR-010 — Entorno aislado

**Alcance:** implementar SR-009 en `envs/`, preservando el protocolo y las invariantes del runner.

**Aceptación:** pruebas reales del entorno muestran que código generado no puede leer un marcador del host ni secretos no concedidos, ni acceder a destinos de red prohibidos; límites de recursos aplicados; limpieza tras error/cancelación; EDA puede generar sus entregables. No usar `inherit_env=True` ni un fallback local silencioso para el modo seguro. Tests Docker/entorno separados de la suite rápida.

### SR-011 — Concurrencia de chats

**Alcance:** ownership de kernels/workspaces y caché de agentes por proyecto/sesión.

**Aceptación:** dos conversaciones concurrentes no mezclan respuestas ni variables involuntariamente; política de compartir datos explícita; exclusión o aislamiento según esa política; entornos liberados al cerrar/reiniciar. Probar cambio rápido de proyecto y sesión, solicitudes retrasadas y materialización concurrente de skills.

### SR-012 — Estado del run

**Alcance:** API interna de transiciones/persistencia compartida por runner, driver, API y herramientas de gráficos. Refactor acotado a ownership, sin reescribir el bucle del agente.

**Aceptación:** estados y transiciones válidos definidos; escrituras concurrentes no pierden gráficos/gates; start/stop/aprobación idempotentes; errores/reintentos no conservan información engañosa del intento anterior. Versionar el contrato de persistencia y preservar lectura de runs existentes.

### SR-013 — Deadlines

**Alcance:** parsear y aplicar `Step.timeout`; coordinar timeout del paso y de herramientas con el entorno.

**Aceptación:** un paso que supera su plazo termina con estado y motivo explícitos; timeout de Jupyter interrumpe la ejecución real; comandos de contenedor no siguen consumiendo recursos sin control tras vencer el plazo del cliente. Probar una tarea deliberadamente lenta sin modelo.

### SR-014 — Parada y cancelación

**Alcance:** API, runner/driver, entorno y control de UI.

**Aceptación:** distinguir “detener después del paso” de “cancelar ahora”; limpieza y resultado parcial identificados; un stop no comienza pasos nuevos; cancelar un gate no queda esperando indefinidamente; duración de un run detenido deja de crecer y su stream finaliza correctamente. Verificar reanudación solo cuando tenga semántica válida.

### SR-015 — Inicio determinista

**Alcance:** el launcher invoca directamente el workflow elegido, con sus inputs registrados; el chat conserva el routing del modelo.

**Aceptación:** comenzar desde un formulario no requiere que el LLM elija llamar a `run_workflow`; no ejecuta dos veces con doble clic/reintento; identidad, eventos, gates, artefactos y consumo conservan el contrato. Proteger con pruebas de ambas vías de inicio.

### SR-016 — Recuperación tras reinicio

**Alcance:** coordinar transcripciones, checkpoints y estado persistido de launcher y chat.

**Aceptación:** reiniciar antes/después de un gate mantiene la pregunta y su run; una respuesta ya aceptada no se reasigna a otro gate; pasos completos no vuelven a cobrarse/ejecutarse; trabajos interrumpidos quedan visibles como recuperables o fallidos. No introducir dos copias de transcripción que diverjan sin reconciliación.

### SR-017 — Consumo completo

**Alcance:** libro de consumo incremental por llamada/intento/run y resumen de UI.

**Aceptación:** contabilizar orquestación, chat relacionado, pasos, reintentos y fallos; preservar consumo de intentos previos; distinguir subtotal conocido de total con precios desconocidos. Registrar versión de tarifas y reconciliar una muestra con uso del proveedor cuando se haga una ejecución autorizada. El costo de infraestructura/soporte se medirá en SR-036.

### SR-018 — Presupuestos y admisión

**Alcance:** límites configurables de gasto/tokens, trabajos simultáneos y recursos.

**Aceptación:** rechazar o poner en espera trabajo por encima del límite antes de aprovisionarlo; impedir nuevas llamadas cuando se agota presupuesto; mostrar motivo y consumo pendiente. Documentar posible exceso de una llamada en vuelo; no prometer un límite monetario exacto si el proveedor no permite imponerlo.

### SR-019 — Datasets acotados

**Alcance:** upload, lectura de preview/perfil, exportaciones grandes y copia a runs.

**Aceptación:** tamaños máximos y formatos soportados visibles; upload por streaming o lectura acotada; Parquet no se carga completo para ver pocas filas; datos malformados producen errores comprensibles. Evitar copiar indiscriminadamente datasets no utilizados; sobrescribir un nombre debe tener una semántica explícita y no alterar la entrada de un run iniciado.

### SR-020 — Contratos de salida y evidencia

**Alcance:** validadores declarados por cartridge; el harness ejecuta el contrato sin asumir reglas de DS.

**Aceptación:** archivos vacíos, obsoletos o con esquema incorrecto no cuentan como entrega correcta; resultados principales se contrastan con tablas calculadas; validadores de MMM derivan/comparan diagnósticos con el modelo real. Separar “archivo producido”, “validación ejecutada” y “resultado aprobado”. Incluir inputs adversariales y conclusiones sin soporte.

### SR-021 — Procedencia y versiones

**Alcance:** manifiesto de run y lectura/migración de esquemas persistidos.

**Aceptación:** guardar hashes de inputs, versión/contenido relevante de workflow y prompts/skills, configuración del modelo, dependencias o digest del entorno, código generado e intentos; una reanudación detecta incompatibilidades. Un tercero puede identificar qué produjo un resultado; no prometer determinismo del LLM por disponer de hashes.

### SR-022 — Preflight correcto

**Alcance:** `_profile_inputs`, `_guesses`, tipos/opciones requeridas y validación de inputs antes de iniciar.

**Aceptación:** 500 filas muestreadas no se describen como prueba de unicidad global en 1.000; casos con duplicados fuera de la muestra; CSV irregular no genera un 500; defaults, campos obligatorios, opciones y referencias a archivos se validan antes de gastar recursos. No ejecutar un modelo para describir datos que se pueden comprobar directamente.

### SR-023 — Plan antes de iniciar

**Alcance:** launcher consume `/plan` y muestra datos, inputs, pasos, gates y estimación con sus limitaciones.

**Aceptación:** usuario puede editar y revisar antes de iniciar; cancelar no deja runs vacíos huérfanos; costo desconocido se indica como desconocido; inputs confirmados son exactamente los del run. Una prueba de navegador recorre upload/selección → plan → editar → iniciar.

### SR-024 — Gráficos y tablas de workflows

**Alcance:** reducer `useRunStream`, contratos de eventos y vista del run.

**Aceptación:** los siete eventos `dsagent.chart` del replay aparecen con datos por referencia; reload/reconexión no duplica objetos; nueva versión reemplaza correctamente la anterior sin perder historial; errores de lectura se muestran. Alinear gramática y política de carga de datos de gráficos de chat, archivos y workflow; conservar exportación interactiva.

### SR-025 — Evidencia del gate

**Alcance:** `GateCard`, selección/foco de artefactos y versiones rechazadas.

**Aceptación:** antes de aprobar, la UI abre el documento objeto de revisión y sus enlaces funcionan; indica qué acción se autoriza y qué trabajo sigue; rechazo conserva nota/evidencia; siguiente revisión permite consultar la versión anterior. No seleccionar el informe solo después de aprobarlo.

### SR-026 — Recuperación y errores en UI

**Alcance:** RunDock, estado de conexión, errores API y acciones de recuperación.

**Aceptación:** error identifica paso y motivo; reintentar continúa desde el punto correcto; rechazar → corregir → volver a revisar → aprobar funciona; recargar/reiniciar durante el flujo conserva contexto recuperable. Verificar también sin credenciales, desconexión, run detenido, doble clic y error de guardado; evitar capturas silenciosas que aparentan éxito.

### SR-027 — Navegación persistente

**Alcance:** URLs/estado de navegación de proyecto, sesión y run.

**Aceptación:** reload y enlace directo restauran la selección; atrás/adelante funciona; ID inválido muestra estado claro; cambios rápidos de proyecto no aplican resultados de peticiones viejas. Enlaces locales no se presentan como enlaces públicos para compartir.

### SR-028 — Onboarding e información veraz

**Alcance:** lenguaje de UI, descriptions de workflows y README raíz/UI.

**Aceptación:** elegir idioma inicial consistente y dejar explícita la decisión; campos orientados a resultados con detalles técnicos secundarios; eliminar “harness smoke test” del flujo del cliente; describir almacenamiento real de chats, procesamiento por proveedores, límites y borrado. Documentación refleja Vite/React y las rutas vigentes; eliminar instrucciones obsoletas de Next/CopilotKit/A2UI que ya no apliquen. Verificar onboarding con alguien que no conozca el harness.

### SR-029 — Layout adaptable y entregables

**Alcance:** `.shell`, paneles, selección de artefactos y espacio principal de lectura.

**Aceptación:** comprobar al menos 390, 768 y 1440 px; navegación y acción principal accesibles sin depender de la vieja clase `.app`; informe maximizable y legible; separar entregables de archivos de trabajo; una conversación vacía no desplaza la tarea activa a un panel estrecho.

### SR-030 — Accesibilidad

**Alcance:** nombres/labels, foco, diálogos, teclado, estados y gráficos.

**Aceptación:** labels asociados, nombre accesible de iconos, entrada/retorno de foco y cierre de modal, navegación solo con teclado y avisos comprensibles de progreso/error; contraste y movimiento revisados; gráficos con alternativa tabular cuando corresponda. Adjuntar comprobaciones automáticas y revisión manual; no declarar conformidad global sin auditarla.

### SR-031 — Distribución y recuperación local

**Alcance:** instalación fuera del checkout, descubrimiento de cartridges/UI, comandos y backup/restore.

**Aceptación:** instalar artefacto en entorno limpio y ejecutar fuera del repositorio; `--version` funciona; requisitos y extras correctos; una copia de seguridad restaura proyectos, sesiones, runs y evidencia en instancia vacía; actualización compatible o migración documentada. No incluir credenciales ni datasets privados en paquetes.

### SR-032 — Contrato de cartridges

**Alcance:** loader, namespaces, herramientas concedidas, visibilidad y generación de skills.

**Aceptación:** cartridge ficticio de otro dominio funciona; nombres repetidos de persona/workflow entre cartridges no se sobreescriben silenciosamente; metadatos `tools` se aplican o se rechazan/documentan explícitamente si no soportados. Probar límites de rutas en scripts y coherencia de skills; distinguir alcance de herramientas de aislamiento del sistema operativo. Mantener compatibilidad con el contrato de plugin declarado.

### SR-033 — Rendimiento medido

**Alcance:** SSE/log cursor, polling, memoria de sesiones y chunks frontend.

**Aceptación:** benchmark reproducible con volúmenes/concurrencia representativos y límites publicados; no releer todo el log por suscriptor cada 250 ms; detener polling/streams terminales; recortar carga inicial donde los datos lo justifiquen. Comparar latencia, RAM y transferencia antes/después; la optimización no pierde ni duplica eventos.

### SR-034 — Evaluación analítica y puerta del piloto

**Entregable:** corpus versionado, evaluador y resultados de aceptación de EDA; MMM tiene una matriz separada.

**Aceptación:** al menos 20 tareas representativas propuestas, con respuestas verificables, repeticiones en casos críticos, datos sucios, duplicados, huecos temporales, unidades ambiguas, muestras pequeñas, leakage y casos que exigen preguntar o rechazar. Medir corrección numérica/narrativa, errores graves, finalización y tiempo/costo de revisión. Verificar de manera independiente cada resultado principal del conjunto de aceptación; los tests rápidos no gastan modelos por defecto.

### SR-035 — Cliente y oferta

**Entregable:** `docs/product/pilot-hypothesis.md`, a crear durante esta tarea.

**Aceptación:** elegir un segmento y un trabajo recurrente, comprador/usuario, alternativa actual, resultado entregable, propuesta de precio y mecanismo de adquisición. Distinguir hipótesis de evidencia de conversaciones/datos. Evaluar consultorías analíticas y equipos de medición como candidatos, no como mercado ya validado. Preparar preguntas y criterios para decidir; contactar personas requiere instrucción explícita del usuario.

### SR-036 — Piloto comercial

**Entregable:** protocolo, registro por tarea/cliente y readout de continuar/cambiar/detener.

**Aceptación:** seleccionar 3–5 colaboradores como objetivo propuesto; entorno y datasets autorizados; medir éxito, correcciones, tiempo ahorrado frente al método anterior, repetición de uso y disposición/pago real. Economía incluye proveedores, infraestructura, reintentos y soporte humano. No deducir margen de un run histórico de $0.47. Antes de comenzar, cerrar los bloqueos del checklist de piloto de este archivo; registrar aplazamientos P2 con motivo.

### SR-037 — Identidad y permisos SaaS

**Alcance:** usuarios/organizaciones, sesiones autenticadas y autorización del lado servidor.

**Aceptación:** matriz de permisos cubre proyectos, datasets, chats, runs, gates, archivos/exportaciones y streams; pruebas negativas entre tenants; cambiar un ID no concede acceso; identidad del revisor queda registrada. La política de acceso se verifica también en jobs y URLs de archivos. No basta con añadir una pantalla de login.

### SR-038 — Jobs y workers durables

**Alcance:** decisión e implementación de cola, persistencia transaccional y ownership de jobs.

**Aceptación:** dos procesos API/workers no ejecutan dos veces un job por una carrera; caída y reintento tienen idempotencia definida; leases, heartbeats y recuperación observables; gates sobreviven; límites globales y por tenant. Elegir servicios después de medir requisitos; no crear microservicios adicionales sin necesidad.

### SR-039 — Almacenamiento y retención

**Alcance:** archivos de tenants, transferencias autorizadas, cuotas, ciclo de vida y borrado.

**Aceptación:** objetos inaccesibles entre tenants; URLs temporales respetan permisos/vencimiento; integridad de inputs/versiones; política de retención y borrado comprobable en datos, derivados y backups según política publicada. Sandbox de contenido activo también se aplica al almacenamiento remoto.

### SR-040 — Operación y recuperación SaaS

**Alcance:** logs/metrics/traces, alertas, despliegue/rollback, backup/restore y carga.

**Aceptación:** seguir un run sin exponer secretos/datasets en logs; detectar jobs atascados, errores, agotamiento de recursos y gasto anómalo; restauración ensayada; objetivos de recuperación y límites de servicio definidos y medidos. Pruebas con concurrencia realista y fallo de worker/almacenamiento. Documentar responsables operativos.

### SR-041 — Decisión de lanzamiento

**Entregable:** evaluación go/no-go con evidencia actual, riesgos aceptados y alcance comercial.

**Aceptación:** pasar checklist SaaS; demanda y economía respaldadas por SR-036; ninguna afirmación de autonomía, privacidad, fiabilidad o soporte excede lo probado. Separar habilitación técnica de publicación: desplegar/abrir acceso es una acción posterior, no consecuencia automática de cerrar este documento.

## Puertas de salida

**Piloto local supervisado de EDA:** seguridad de archivos y ejecución resuelta (SR-002, 003, 009, 010); identidad/resolución/persistencia y flujo estable (SR-004–007, 011–016, 022–026); límites y consumo (SR-017–019); validación/procedencia (SR-020–021); información veraz e instalación recuperable (SR-028, 031); baseline automatizada (SR-001); evaluación y cliente definidos (SR-034–035). Las dependencias de SR-036 se interpretan junto a este checklist, no como sustituto de él.

Objetivos de aceptación propuestos: cero fallos abiertos de acceso/contención o confusión entre proyectos; al menos 95% de finalización en el conjunto explícitamente soportado; resultados principales verificados de forma independiente; costo y revisión humana medidos. El 95% no autoriza errores graves ni demuestra generalización fuera del conjunto evaluado. Confirmar el alcance y los umbrales antes de ejecutar la evaluación.

**MMM:** además de la base del piloto, SR-008 y las comprobaciones independientes del posterior de SR-020, integración Docker/modelo y evaluación específica. Mantenerlo identificado como experimental hasta entonces. No presentar aprobación de un gate como validación causal o recomendación comercial autónoma.

**SaaS compartido:** pasar todas las condiciones del piloto y SR-037–040; reevaluar rendimiento/accesibilidad según audiencia y dispositivo objetivo; resolver vulnerabilidades identificadas; decisión explícita SR-041. El orden diferido de estas tareas no reduce su carácter obligatorio antes de compartir una instancia.

## Relación con el roadmap anterior

| Hito existente | Tratamiento en este backlog |
|---|---|
| M2.5/M2.6: UI, charts, plan, gates y recuperación terminados | Conservar historia; verificar y corregir versión actual en SR-023–026. Un checkbox histórico no cierra una regresión. |
| M2.3/M2.4: Docker y ejecución MMM | Reutilizar trabajo; SR-009–010 define aislamiento y SR-008/020/034 valida seguridad/calidad del workflow. |
| M3: expansión a otros dominios/cartridges | Primero SR-032 y validación comercial; expansión no bloquea el piloto EDA. |
| M4: multi-run/reanudación y `--version` | SR-004–005/016/027 verifica multi-run y recuperación; SR-031 incluye distribución y comando de versión. |
| BigQuery, A2UI/MCP Apps, PDF y otras ampliaciones | Permanecen en el roadmap; priorizar solo si el piloto demuestra necesidad. No son requisitos automáticos de la primera oferta. |

## Registro de ejecución

| ID | Responsable | Inicio | Cierre | Commit/PR | Validación y evidencia | Decisiones/desviaciones |
|---|---|---|---|---|---|---|
| — | — | — | — | — | Ninguna tarea iniciada | Próxima recomendada: SR-001 |

Guardar evidencia durable en el repositorio, por ejemplo `docs/runs/startup-readiness/SR-002/`, al ejecutar cada tarea. No enlazar como única evidencia archivos temporales de la revisión inicial. Evitar datasets privados, credenciales y datos personales en evidencias versionadas.
