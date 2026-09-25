# ADR — Control local de cómputo

Fecha: 2026-09-24. Estado: implementado.

## Decisión

El servicio conserva el bloqueo del run y supervisa un subprocess por ejecución
del workflow. Deep Agents y sus herramientas viven dentro de ese subprocess.
`process.local` es un provider del host: start, poll, cancel y reconcile. No se
intenta matar threads de Python. Los embedders pueden optar explícitamente por
el modo síncrono anterior; ese modo no promete cancelación ni deadlines.

Los providers y factories que cruzan el límite de proceso deben ser funciones
importables de paquetes instalados. El payload es JSON de configuración; no se
deserializa código arbitrario ni se trasladan closures o conexiones vivas.

## Propiedad y estados

Cada invocación tiene un UUID y un registro `.execution/processes/<id>/process.json`.
El registro guarda PID, tiempo de creación, host/boot y causa de terminación.
El proveedor es su único escritor mientras el supervisor está vivo. En recuperación
se obtiene primero el bloqueo del run. Los snapshots del runner siguen en run.json.

Estados de proceso: starting → running → exited; running → cancelling → cancelled
o timed_out. Una caída se registra como interrupted. Si no se confirma limpieza:
cleanup_required. No se permite un nuevo intento mientras haya cómputo anterior
activo o sin reconciliar. PID solo no prueba identidad: se coteja creación y boot.

Un proceso guardián independiente observa la vida del servidor. Si lo pierde,
termina el worker y sus recursos registrados. Al reiniciar, el servicio reconcilia los registros antes
de clasificar el run y nunca repite automáticamente un paso incierto.

## Tiempos y cancelación

`Step.timeout` acepta segundos o valores con s/m/h, positivos y finitos. Incluye
provisionamiento, llamadas al modelo, herramientas y gate automático. El gate
humano no consume ese tiempo. Sin timeout declarado se aplica un límite local
de una hora por paso. Al excederlo se envía TERM y luego KILL tras una gracia
corta. Solo después de verificar la terminación se publica stopped/failed.
El bootstrap tiene 60 segundos y el cierre de entornos 30 segundos. Los tiempos
de espera de Docker durante limpieza son acotados. Un error del worker conserva
su diagnóstico; un timeout deja su causa en el proceso y en el paso.

Cancelar puede dejar archivos parciales y efectos externos ya realizados. El
reintento es explícito. Esto no hace transaccionales las APIs externas.

## Recursos

Los kernels registran identidad de proceso y los contenedores registran un nombre
único y etiqueta de propietario antes de usarse. El supervisor intenta toda la
limpieza aunque una parte falle. Docker inaccesible o identidad dudosa significa
cleanup_required, nunca éxito silencioso. No se eliminan recursos ajenos.
La publicación de recursos comienza con un registro de intención; si el proceso
muere antes de registrar el PID, se bloquea el reintento y se exige inspección
manual. No se pretende identificar por conjetura un proceso sin identidad.

Los comandos de shell son síncronos: sus hijos en segundo plano se eliminan al
terminar el comando. Los trabajos que necesiten sobrevivir a una herramienta
requieren un contrato de jobs explícito; `nohup` no proporciona ese contrato.

`GET /runs/{id}` expone el último registro de proceso. `POST /runs/{id}/stop`
solicita cancelación; la UI muestra «Cancelando» hasta confirmar la terminación.
`POST /runs/{id}/reconcile` verifica limpieza sin ejecutar pasos. La UI ofrece
«Verificar limpieza» para recursos pendientes; después puede reintentarse el paso.

## Alcance

POSIX, un host, disco local. Es control de ciclo de vida, no aislamiento de
seguridad. No soporta programas que escapen deliberadamente del árbol/grupo,
workers remotos, reinicio de entrenamientos desde checkpoints ni quotas globales.
El chat interactivo sigue fuera de este límite; se aplica a workflows del servicio.
Los embedders que utilizan closures o objetos vivos deben usar `isolated=False`
explícitamente; el servidor siempre usa el modo aislado. El provider del proceso
se fija por id/versión en `run.json`, separado del manifiesto del cartridge.

## Aceptación

Procesos reales: tarea larga cancelada, límite temporal, hijos eliminados, caída
del supervisor, reconciliación, negativa ante identidad distinta; runner real
con factories simuladas importables y gates persistentes. Docker: contrato y
errores con daemon simulado; una integración real requiere un daemon disponible.

Pruebas: `tests/test_process_control.py`. La suite completa pasó con 479 tests y
12 integraciones omitidas. Incluye un kernel Jupyter real, muerte del servidor y
del guardián, cancelación de hijos, timeout del runner y aprobación tras reintento.
No se hicieron llamadas pagadas a modelos ni builds de Docker en esta validación.

Referencias de implementación: [identidad y estado de procesos en psutil](https://psutil.readthedocs.io/en/latest/)
y [gestión de subprocesses de Python](https://docs.python.org/3/library/subprocess.html).
