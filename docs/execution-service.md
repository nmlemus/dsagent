# ADR — Servicio de ejecución independiente del chat

Fecha: 2026-09-23. Estado: aceptado e implementado en este bloque.

## Decisión

`ExecutionService` es el dueño de los workflows del servidor. El launcher llama
directamente al servicio; `run_workflow` en el chat envía un trabajo al mismo
servicio y devuelve su identificador. Deep Agents sigue ejecutando las personas;
el chat ya no contiene el ciclo de vida de un workflow dentro de un interrupt.

## Autoridad y almacenamiento

| Dato | Autoridad |
| --- | --- |
| Pasos, intentos, resultados, gate pendiente, snapshots | `run.json` |
| Decisión aceptada para un gate | `.execution/decisions/<gate_id>.json` (inmutable) |
| Solicitud de inicio | `.execution/request.json` |
| Dueño activo | bloqueo exclusivo del SO, `.execution/owner.lock` |
| Conversación | SessionStore; no controla la ejecución |

Una decisión nombra run, paso, intento y gate UUID. Se persiste antes de confirmar
HTTP. Repetir la misma respuesta es idempotente; una respuesta distinta o dirigida
a otro intento se rechaza. El runner consume la decisión por identidad, omite
pasos terminados y gates aprobados, y libera sus entornos al aparcarse.

Los bloqueos son por run y funcionan entre threads y procesos del mismo host.
Un segundo bloqueo corto serializa aceptación de decisiones y salida del worker:
una respuesta durante el cierre de entornos no puede perder el despertar del job.
Las mutaciones administrativas utilizan el bloqueo de ejecución también.

El chat dispone de `get_run_status` para consultar progreso y revisiones; la UI
muestra los jobs asociados a la sesión y sigue sus eventos por `/runs/{id}/events`.
Las aprobaciones HTTP requieren `gate_id`: una misma decisión y nota devuelven
éxito al reintentarse; una contradicción devuelve 409. Un cliente antiguo sin
identidad recibe 400 y debe actualizarse. El modo replay conserva su contrato.

## Recuperación

- `pending` sin solicitud de inicio: conserva su estado.
- `pending` con solicitud persistida: se despacha al arrancar el servicio.
- `awaiting_gate`: conserva pregunta e identidad. Una decisión ya persistida se
  consume al arrancar aunque el proceso muriera antes de despacharla.
- `running` sin dueño: pasa a `interrupted`. No se repite automáticamente ningún
  paso con efectos inciertos. El operador revisa los resultados y pide reintento.
- `done`, `failed`, `stopped` y rechazo ya aplicado: no se relanzan solos.

Un reintento explícito vuelve a ejecutar el paso no terminado y puede repetir
efectos externos. No se promete exactly-once. Una aprobación persistida evita
repetir el trabajo ya terminado; no hace transaccionales las herramientas externas.

## Alcance y alternativas

Despliegue soportado: monolito modular en un host POSIX con disco local durable.
No se presenta `flock` como coordinación distribuida ni se soporta NFS/multihost.
Se conserva el formato existente en vez de introducir otra base de datos que
duplique el estado de ejecución. Una futura cola distribuida deberá implementar
leases con fencing y reconciliación de los procesos remotos.

`GraphDriver` y el modo síncrono de `workflow_tools` quedan como compatibilidad
para embedders/CLI existentes; el servidor deja de utilizarlos. No deben operar
concurrentemente sobre los mismos runs que el servicio.

No incluye cancelación de llamadas en vuelo, deadlines, autorización de tools,
restauración de entornos de cómputo ni instalación de plugins externos.
Los workers son threads del servidor; esta fase no añade una cola con límites de
concurrencia globales ni un gestor de procesos huérfanos. `--memory` se conserva
como opción compatible pero no cambia la durabilidad de las aprobaciones.

## Validación requerida

Launcher sin orquestador; envío desde chat al mismo servicio; dos dueños
concurrentes; gates consecutivos; respuesta duplicada, contradictoria y obsoleta;
respuesta durante aparcado; reinicio esperando aprobación y después de persistir
la respuesta; muerte durante un paso sin repetición automática; reintento explícito.

Pruebas implementadas en `tests/test_execution_service.py`: runner real con
personas simuladas y almacenamiento real, incluyendo exclusión entre procesos,
salida abrupta dentro de un paso y salida abrupta esperando un gate. El segundo
caso se retoma por HTTP, comprobando que el archivo del paso aprobado no cambia
de mtime. Las pruebas no evalúan calidad del modelo, contenedores reales ni
recuperación de efectos remotos.

Validación local del bloque: suite completa de Python (452 passed, 12 skipped)
y, después de añadir cobertura de stop y adopción de gates antiguos, las 17
pruebas del servicio pasan. Ruff pasa en los módulos modificados; las 5 pruebas
de UI y `npm run build` pasan. El build conserva los avisos de bundles grandes.

## Siguiente bloque

1. Definir propiedad, cancelación, deadlines y reconciliación de procesos de
   cómputo, antes de agregar ejecución remota o workers en varios hosts.
2. Aplicar permisos efectivos por persona/herramienta; el registro de providers
   por sí solo no es un límite de autorización.
3. Presupuesto y admisión de trabajos, con límites de concurrencia y métricas por
   intento. El contador de intento de este bloque no sustituye una contabilidad
   histórica completa.
