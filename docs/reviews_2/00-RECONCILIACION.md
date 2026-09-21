# Evaluación reconciliada

Fecha: 2026-09-19. Documentos de origen y nivel de evidencia: [FUENTES.md](FUENTES.md). Estado ejecutable: [BACKLOG.md](BACKLOG.md).

## Veredicto y alcance

**Hay una base técnica para una startup en fase alfa, pero no evidencia suficiente para vender hoy un producto seguro, durable y analíticamente fiable.** Presentar la idea y demostrar un flujo acotado es distinto de abrir una instancia a clientes o comprometer calidad comercial. Así se reconcilia el “favorable para presentar una startup” del primer informe con el “no vendible hoy” de las rondas comerciales.

La propuesta provisional es probar un resultado EDA recurrente en un piloto dedicado y supervisado, una vez cerrados sus bloqueos. Self-host es una alternativa comercial por validar, no una exención de seguridad ni una decisión ya tomada. MMM requiere una puerta de aceptación propia. El salto a SaaS depende de demanda, economía y operación demostradas; un conteo de clientes propuesto no sustituye esas comprobaciones.

Esta es una reconciliación de auditorías sobre `8ac2758`, con lecturas puntuales del código. No se ejecutaron de nuevo toda la suite, la UI ni los experimentos de sandbox; no se han medido mercado, retención o margen. Las conclusiones deben revalidarse contra el commit donde comience cada tarea.

## Qué conservar

React/Vite, FastAPI, Deep Agents como dueño del bucle y la separación cartridge → host → runner → entorno forman una base que puede evolucionar. Conservar DAGs declarados, gates, artefactos, protocolo de entornos y fixtures FakeAgent/replay; reparar sus límites y contratos. No hay evidencia que justifique una reescritura completa ni que pruebe que esta arquitectura sea por sí sola una ventaja competitiva defendible.

Los tests y runs históricos aportan evidencia del harness. No certifican aislamiento del sistema operativo, precisión de informes, escalabilidad o experiencia comercial. La evaluación inicial registró 369 tests aprobados y 12 omitidos usando `python -m pytest`, build de UI correcto y siete hallazgos Ruff; esos resultados siguen siendo históricos.

## Hallazgos consolidados

| Área | Diagnóstico que se conserva | Trabajo resultante |
|---|---|---|
| Arquitectura y ejecución | Kernel/shell local comparte privilegios con el host; parámetros Docker incompletos; código generado puede alcanzar recursos ajenos al análisis | SR-009/010/046/048; modelo de amenazas y pruebas reales |
| Seguridad HTTP y archivos | Rutas raw de HTML quedan fuera de la protección del viewer; ZIP puede incluir symlinks externos; API carece de identidad | SR-002/003/045 y SR-037 para uso compartido |
| Datos e identidad | Colisiones globales de runs y lookup por proyecto inconsistente; hilo desconocido puede caer en default | SR-004/005/043; IDs no sustituyen autorización |
| Persistencia | Guardado no atómico o silencioso, mensajes del cliente como autoridad, escrituras concurrentes y estado/checkpoints divergentes | SR-006/012/016/021/042/047/049 |
| Ejecución durable | Launcher depende de que un LLM llame al workflow; plazos no aplicados y parada incompleta; pasos largos pierden progreso | SR-013–016/054; jobs con leases si la escala lo exige en SR-038 |
| Confianza analítica | Diagnósticos MMM incompletos/no finitos pasan; existencia de archivos no valida contenido; inferencias de muestras se presentan como globales | SR-008/020–022/034; evaluación independiente y procedencia |
| UI/UX | Falta plan revisable; charts del workflow no se muestran; gate no enfoca evidencia a tiempo; recuperación insuficiente y carreras al cargar chats | SR-023–027/042–044/049 |
| Diseño y accesibilidad | Panel principal limita entregables, responsive usa clase obsoleta, navegación/foco incompletos y mensajes bilingües o inexactos | SR-028–030; verificación con usuarios y dispositivos objetivo |
| Código y contratos | Buenas separaciones conviven con estado global, duplicación de resolución, excepciones silenciadas y puente de librería frágil | SR-001/005/011/012/024/032/049/053; mantener invariantes |
| Rendimiento y costos | Copias/lecturas completas, polling repetido, entornos sin política de caducidad, builds en petición y medición parcial de consumo | SR-011/017–019/033/055; optimizar con benchmark |
| Distribución y operación | Dependencia del checkout, configuración dispersa, falta de contrato de migración/retención/restore y operación mínima | SR-021/031/050/052/053/055; SR-038–040 antes de escalar |
| Producto y comercial | Problema, comprador, adquisición, precio y soporte no validados; licencia declarada sin inventario suficiente | SR-035/036/051/056; decisiones DEC-01/02/03/05 |

El detalle y las referencias a archivos permanecen en las fuentes y en cada especificación del backlog. Los casos de HTML, ZIP, colisión de runs, preview no-default, sesiones antiguas y gates MMM tienen reproducción documentada en el informe inicial. Las narrativas de incidentes de los premortems son escenarios, no incidentes reales observados.

## Diferencias y correcciones que prevalecen

| Afirmación de las fuentes | Resolución para trabajar |
|---|---|
| “Presentable como startup” frente a “no vendible” | Son listones distintos. Alfa presentable con alcance explícito; lanzamiento comercial bloqueado. |
| Self-host primero como recomendación general | Mantener como hipótesis en DEC-01. Instalación dedicada, self-host compartido y SaaS tienen requisitos distintos. |
| Auth/tenancy solo después del piloto | Auth por persona/recurso antes de compartir incluso una instalación self-host. Token de despliegue no equivale a roles/tenant. |
| DEC-* “bloquea el resto” | Solo bloquea decisiones dependientes. No impide corregir datos, archivos, carreras de UI o inventariar derechos. |
| Todo el bloque N es pequeño; fases con fechas fijas | No se heredan estimaciones como compromisos. Aislamiento, CSRF, cancelación y sesiones requieren diseño y regresión. Desglosar antes de estimar. |
| Canvas/paths “seguros” en la ronda local | Limitar esa conclusión a la ruta y prueba concretas. Raw HTML y exportaciones contradicen cualquier garantía global. Mantener matriz de todas las rutas. |
| CORS/TrustedHost/bearer como cierre de seguridad | Definir autenticación, permisos y defensa frente a peticiones de otros sitios por separado; probar UI, CLI, SSE y enlaces. Nunca enviar tokens en URLs por comodidad. |
| `make_env` inicia Jupyter y bloquea `/agent` hasta 60 s | En el código leído, Jupyter se crea al primer `run_python`; no en `make_kernel_env`. Copias y builds síncronos sí merecen medición. No conservar una latencia no medida como hecho. |
| Kernel equivale literalmente al proceso API | Shell/Jupyter son procesos locales con privilegios heredados, no un límite de seguridad independiente. Escape vía Docker socket depende de acceso real al daemon; no se afirma que se haya explotado. |
| Aislamiento por VM dedicada basta | Reduce mezcla entre clientes; no impide que análisis modifique servidor, registros, paquetes o secretos dentro de esa VM. |
| Limpiar con atexit/--rm o mover SQLite resuelve control | Caída abrupta necesita GC con ownership/leases. SQLite fuera del workspace solo protege si el proceso de análisis carece de acceso; excluir también spills y archivos internos. |
| Atomic replace o timestamps resuelven convergencia | Evitan algunos daños pero no actualizaciones perdidas. Versiones del servidor, transcripción autoritativa y dueño de transiciones son requisitos distintos. |
| Migrar checkpointer basta para chat durable | Depende también de transcripción autoritativa, ownership, compatibilidad y protocolo de reanudación. F1-06 incorpora SR-049 como dependencia. |
| SSE es independiente de réplica y sticky sessions resuelven escala | Son propuestas de topología. Verificar eventos durables, leases, sandbox persistente y recuperación antes de esa afirmación. |
| Ignorar campos desconocidos resuelve upgrades | Versiones y migraciones explícitas con fixtures; una versión futura incompatible debe fallar de forma visible sin ocultar historia. |
| MIT impide vender / open-core es obligatorio / BYOK evita términos | No se adoptan conclusiones jurídicas. DEC-02/05 y SR-051 verifican derechos y textos según uso real. Ninguna licencia se cambia aquí. |
| “258 dependencias” prueban compatibilidad total | El conteo documentado corresponde al análisis npm de la fuente, no a un inventario auditado de Python, imágenes y transitivas. Pendiente inventario trazable. |
| Sin phone-home implica no hay salida de datos | No demostrado: código generado puede usar red y proveedores reciben datos según configuración. Definir política y comprobarla. |
| Borrado equivale automáticamente a cumplimiento normativo | Verificar originales, derivados, checkpoints, logs y backups según política. No se emite conclusión de cumplimiento ni plazo contractual universal. |
| RPO ≤5 min, RTO 4 h y borrado 24 h | Son objetivos propuestos por una fuente, no garantías medidas. Definirlos con SR-036/040 y responsables de operación. |
| Segundo cartridge, billing, Helm, SCIM o certificación bloquean cualquier piloto | Condicionales al producto prometido y al comprador. El problema subyacente se conserva, pero las extensiones se difieren explícitamente. |

## Secuencia reconciliada

1. **Base y defectos inmediatos:** SR-001 y correcciones de pérdida de chats, HTML/ZIP, IDs/proyectos y acceso HTTP. Añadir mitigación de entorno sin confundirla con sandbox. Preparar cliente objetivo e inventario de licencias en paralelo lógico, sin iniciar varias tareas técnicas a la vez.
2. **Piloto seguro y útil:** aislamiento verificable, persistencia/recuperación, límites/costos, evidencia analítica, plan/gates/entregables, instalación y restore. Cerrar la puerta del backlog según audiencia; no saltarla por asignar una VM dedicada.
3. **Producto soportado:** demanda/pago y costo de soporte medidos; licencias coherentes, distribución/release, retención y permisos para el uso comprometido. La oferta de MMM necesita evaluación adicional.
4. **Escala cuando haga falta:** metadata transaccional, cola/leases, workers y almacenamiento compartido; eventos, permisos por tenant y operación probados. Billing y enterprise siguen el modelo comercial y requisitos reales.

No se ha elegido Postgres, Redis, Kubernetes, proveedor de sandbox, sistema de pagos ni licencia por el hecho de aparecer en un reporte. Son candidatos que las tareas deben evaluar. Tampoco hay autorización implícita para desplegar, gastar modelos o contactar clientes.

## Resultado documental

Se conservan 41 IDs SR y se añaden 16 tareas para evitar que problemas concretos se pierdan dentro de épicas. Las 76 entradas del otro backlog se integran mediante mapeo explícito: cinco decisiones y 71 tareas. El [mapeo](TRAZABILIDAD.md) cubre 117 identificadores de entrada. Los originales permanecen intactos; esta carpeta y su backlog prevalecen para el trabajo futuro.
