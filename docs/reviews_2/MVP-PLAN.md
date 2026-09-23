# Propuesta de alcance y ejecución del MVP

2026-09-21 · Base: `65a039b`. Propuesta para decidir y ejecutar; no registra como aceptadas las decisiones comerciales pendientes.

**Actualización del 2026-09-22:** el usuario fija tres dominios de producto —ciencia de datos, agencias de marketing y SDLC— y solicita una pausa para evaluar la arquitectura y la posible sustitución de partes del harness. Ver [agentes persistentes y dirección de producto](persistent-agents-and-three-domain-direction-2026-09-22.md). La recomendación de conservar el harness que sigue es histórica y queda sujeta a esa comparación; el primer recorrido comercial sigue por decidir. No se inicia la implementación de los tres dominios completos como un único MVP.

**Evaluación posterior:** la [matriz de requisitos sobre Deep Agents](deep-agents-requirements-fit-2026-09-22.md) recomienda mantener el SDK y rediseñar el control de trabajos/procesos de DSAgent. La siguiente actividad propuesta es cerrar el diseño y sus responsabilidades, antes de implementar o comparar motores.

## Resultado que buscamos

Un usuario definido completa un trabajo real de forma repetida, revisa el resultado, lo utiliza fuera de DSAgent y obtiene un beneficio que justifica su costo y el soporte de AiudaLabs. La prueba del MVP es ese uso, junto con la capacidad de operarlo y recuperar fallos; una ejecución exitosa o una UI presentable no bastan.

**Recomendación:** conservar el harness genérico y lanzar un flujo concreto mediante una instalación dedicada, gestionada y supervisada por AiudaLabs. Una organización por instalación y acceso restringido a usuarios identificados. Empezar con concurrencia baja y explícitamente limitada. El aislamiento de clientes no sustituye al aislamiento del código generado respecto al servidor.

El primer usuario/dominio está por confirmar. Marketing para AiudaLabs/YoMap es un candidato porque existe `campaign-brief` y trabajo visible en el proyecto. Eso demuestra disponibilidad para una prueba interna, no demanda externa. Si hay un comprador de DS con trabajo recurrente y acceso a datos adecuados, EDA puede ser una mejor primera oferta. No abrir ambos frentes comerciales a la vez.

La decisión DEC-03 conserva su carácter pendiente, pero su justificación histórica de «único dominio desarrollado» ya no describe HEAD: ahora existen DS, marketing y una UI configurable. Esto mejora la evidencia arquitectónica; no valida una plataforma comercial general.

## Alcance de producto propuesto

| Incluido | Límite inicial |
|---|---|
| Proyecto y recursos persistentes | Una organización por instalación; formatos y tamaños soportados declarados |
| Un workflow principal | Inputs explícitos, preparación, inicio determinista, estado y errores comprensibles |
| Revisión humana | Leer evidencia, aprobar o pedir cambios; conservar versiones y decisiones |
| Entregables utilizables | Vista previa, descarga y exportación; validación específica del dominio |
| Continuidad | Cerrar navegador, volver al trabajo, recuperar gates y gestionar reinicios sin éxito ficticio |
| Operación básica | Usuarios autorizados, ejecución aislada, límites, consumo, logs útiles, backup y restore probado |
| Soporte del piloto | Responsable, contacto y procedimiento de recuperación; intervenciones registradas |

Para marketing, el flujo candidato es **brief y guía de marca → análisis → estrategia aprobada → copy por canal → revisión final y exportación**. El cartridge actual ya tiene el gate de estrategia; la aceptación del copy final necesita un mecanismo explícito en el producto/piloto. No confundir generación de copy con publicación, compra de medios ni resultados de campaña.

Para EDA, el equivalente es **datos y pregunta → calidad revisada → análisis → informe aceptado y exportado**. MMM no entra automáticamente por estar disponible en el repositorio.

Quedan fuera del primer alcance: marketplace de cartridges, instalación arbitraria de plugins por clientes, SaaS compartido de autoservicio, billing automatizado, SSO empresarial, publicación automática en redes, agentes autónomos sin revisión, nuevos dominios y selección de modelos con Jev. La investigación de Jev permanece guardada.

## Secuencia por incrementos verificables

Cada incremento agrupa trabajo, no representa una sola tarea ni una estimación de calendario. Implementar subtareas pequeñas con evidencia y cierre en [BACKLOG.md](BACKLOG.md). Las 57 tareas no son una lista que haya que completar íntegra antes de aprender del mercado; los criterios de seguridad e integridad aplicables a la oferta sí deben cumplirse.

| Incremento | Trabajo | Evidencia de salida | Referencias existentes |
|---|---|---|---|
| 0. Contrato del MVP y línea base | Elegir usuario, trabajo recurrente, inputs, entregable aceptable, operación y límites. Actualizar evidencia de reparaciones parciales; fijar dependencias y CI. Definir pronto el límite de ejecución. | Una ficha de oferta, casos representativos y comandos de verificación reproducibles. ADR de ejecución con pruebas requeridas. | SR-001/009/035/050; DEC-01/03/04 |
| 1. Datos y ejecución protegidos | Cerrar rutas de contenido activo; permisos efectivos de herramientas, credenciales acotadas y entorno aislado. Acceso autenticado y autorizado. Persistencia transaccional de runs y sesiones; tratamiento de conflictos en UI. | Pruebas negativas de archivos/permisos, dos escritores, fallo de disco y dos pestañas; sin mezcla, pérdida silenciosa ni código generado con acceso al plano de control. | SR-002–006/010/012/037/042–049; BH-T01–05/07 |
| 2. Un trabajo completo y recuperable | Lanzamiento determinista, validación/versionado del cartridge y sus inputs, deadlines y cancelación reales, serialización del entorno, recuperación y consumo por intento con cupos. Integrar leer → decidir → descargar. | El workflow elegido completa su recorrido; rechazo genera una revisión; reinicio conserva decisiones y entregables. Fallo/cancelación quedan explícitos, sin duplicar efectos. | SR-011/013–023/025–026/032/053; BH-T06–10; UI-R01–03 |
| 3. Release para piloto | Validadores de resultados, evaluación con casos representativos, instalación soportada, backup/restore, límites de recursos, exportación/borrado, inventario de componentes y condiciones aplicables. Recorrido de onboarding y soporte. | Release identificable e instalable; restore ensayado; resultados revisados contra una rúbrica; operación documentada y responsable asignado. | SR-019–021/028–031/034/051–055 |
| 4. Uso recurrente | Primero uso interno cuando haya un usuario interno adecuado; después piloto externo con alcance y costo acordados. Medir cada trabajo y cada intervención. | Usuarios vuelven, usan entregables y el costo total es defendible. Decisión de continuar, ajustar o detener basada en evidencia. | SR-036; DEC-01/02/05 |

El trabajo sobre identidad visual continúa solo donde desbloquea el recorrido. Tipografía y detalles de ID-05 pueden completarse después de las correcciones funcionales de revisión/archivos. No reiniciar el diseño.

**Primer PR técnico recomendado:** SR-002, contenido activo en todas las rutas de runs/proyectos, visor y descargas. Reproducir en HEAD, definir una política única y verificarla en navegador con un archivo sintético que no pueda acceder a la API del producto. En la preparación del incremento 0, subir CI de prioridad respecto al orden histórico del roadmap: no esperar al final del hardening para disponer de una línea base reproducible.

## Condiciones mínimas antes de un piloto externo

- Acceso limitado a usuarios autorizados; aislamiento de ejecución demostrado con pruebas, no inferido por tener un contenedor o una instancia dedicada.
- Sin pérdida silenciosa en los casos de disco lleno, conflicto de guardado, refresh, rechazo, reinicio y cancelación cubiertos por la oferta. Si un paso no puede reanudarse, mostrar interrupción y ofrecer un reinicio seguro sin duplicar efectos.
- Resultados versionados y descargables; una actualización de cartridge no cambia silenciosamente el significado de un run anterior.
- Consumo de intentos exitosos y fallidos registrado. Límites de tiempo, carga y concurrencia aplicados. Un tope monetario debe declarar su margen por llamadas en vuelo y retraso del proveedor, no prometer corte exacto si no existe.
- Copia de seguridad restaurada en una instalación de prueba; procedimiento de recuperación y soporte conocido.
- Rúbrica y casos de aceptación del dominio, incluyendo entradas incompletas o engañosas. Contar archivos existentes no demuestra calidad.
- Oferta, tratamiento de datos y distribución delimitados; resolver las decisiones aplicables sin asumir que una clave del cliente o una licencia declarada resuelve todo.

No se requiere recuperación automática arbitraria, infraestructura distribuida o alta concurrencia para este primer alcance. Sí estados honestos, un camino de recuperación probado y cumplimiento de los límites que se comuniquen.

## Qué medir durante el piloto

| Medida | Definición y utilidad |
|---|---|
| Trabajo terminado y aceptado | Entregable utilizado/aceptado por el usuario sobre solicitudes válidas; reportar también rechazos y errores |
| Tiempo hasta el primer resultado útil | Desde inputs completos hasta primera versión utilizable; separar cómputo, espera humana y correcciones |
| Ahorro neto | Comparación con el proceso anterior, descontando preparación, revisión y corrección; medir, no asumir |
| Costo por entregable aceptado | Modelos de todos los intentos + infraestructura atribuible + tiempo de soporte/retrabajo |
| Repetición | Usuarios que vuelven en su siguiente oportunidad real de uso; no confundir una visita con recurrencia |
| Intervenciones de ingeniería | Veces que alguien debe tocar código/archivos/DB para completar o recuperar un trabajo |

Como muestra inicial, propongo 2–3 usuarios y 10–20 trabajos reales, ajustados a la frecuencia del caso elegido. Son metas de aprendizaje, no prueba estadística de fiabilidad. Acordar antes del piloto los umbrales de calidad, tiempo y costo con el usuario; no inventar un SLA ni un porcentaje universal de aceptación.

Uso interno puede validar utilidad y operación. Para sostener la oferta comercial hace falta evidencia externa de repetición y disposición a pagar; un piloto de pago aporta una señal más fuerte que una demo favorable. SaaS compartido se reconsidera cuando usuarios y economía justifiquen su coste, no porque sea requisito de la palabra MVP.

## Forma de ejecutar

Una subtarea técnica a la vez: reproducción/contrato → implementación → prueba de aceptación → commit → actualización de evidencia. Validación del problema y ejemplos de trabajo pueden avanzar mientras se endurece el backend. No contactar clientes, contratar servicios, publicar ni ejecutar modelos pagos por el mero hecho de figurar en este plan.

El siguiente paso de producto es identificar al primer usuario y fijar su resultado concreto. Eso determina qué cartridge se evalúa en profundidad; no bloquea reparar contenido activo, persistencia o aislamiento comunes al harness.
