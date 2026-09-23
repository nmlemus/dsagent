# DeepSeek Harness: qué adoptar de «everything is a plugin»

2026-09-23 · DSAgent evaluado: `10af959` · Snapshot externo: `00102833dfaee1da9f48a3a8eae9d34005a75218`, commit fechado 2026-09-22T15:25:38Z.

Estado: investigación y recomendación de diseño. Se leyeron documentación y archivos de código de ese snapshot; no se instaló ni ejecutó el proyecto. Complementa la [matriz de 48 requisitos](deep-agents-requirements-fit-2026-09-22.md).

## Veredicto

**Sí adoptaría su principio de composición por capacidades. No hay evidencia de que migrar a este producto sea hoy el camino más sólido o rápido para DSAgent.** Son decisiones diferentes: organización modular, motor de agente y modelo de despliegue.

La idea refuerza nuestra dirección sobre Deep Agents: convertir los cartridges en composiciones de capacidades reales, con contratos y responsabilidades claras. No exige que toda extensión sea dinámica, que cada función sea un plugin o que cualquier paquete pueda reemplazar las políticas del producto.

## Qué significa en su implementación

Cordis proporciona un contexto de servicios: los plugins declaran dependencias, aportan implementaciones y registran eventos y efectos con limpieza asociada. Los consumidores utilizan contratos de servicio en lugar de importar una implementación concreta. Descargar un plugin retira sus registros y recursos asociados; **no revierte automáticamente efectos externos**, como una publicación. [Primer de Cordis](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/cordis-primer.md).

La arquitectura extiende esa composición al adaptador del modelo, registro de herramientas, sesiones y bucle del agente. Perfiles y bundles describen la combinación que se inicia. Distingue tres papeles por capacidad: contrato, proveedor y consumidor. La aplicación web es otra composición sobre la misma base. [Arquitectura](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/architecture.md).

La fortaleza es concreta: un cambio de proveedor no debería obligar a reescribir cada consumidor. No elimina las decisiones de arquitectura; obliga a hacer explícitas las interfaces, dependencias y vidas de los componentes.

## Evidencia que limita la promesa de «sólido y rápido»

1. **Madurez declarada.** El README avisa de developer preview y cambios incompatibles. `SAFETY.md` dice que no ha pasado una auditoría de seguridad y no debe tratarse como listo para producción. Son límites publicados por el proyecto, no una conclusión de que todo su código sea defectuoso. [README](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/README.md), [Safety](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/SAFETY.md).
2. **Trabajos de fondo no equivalen a trabajos durables del producto.** Su contrato de jobs separa identidad/control del productor. El proveedor `LocalJobRegistry` guarda registros en un `Map` del proceso y cancela trabajo al disponer del propietario/servicio. Es útil, pero no demuestra admisión durable ni recuperación distribuida tras reinicio. Un proveedor alternativo sería trabajo adicional. [Contrato de jobs](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/subsystems/jobs.md), [implementación, líneas 128–192 y 617–646](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/packages/jobs/jobs-local/src/index.ts#L128).
3. **Su workflow tiene otra semántica.** Puede ejecutar un script de orquestación escrito por el modelo que crea subagentes. Es una capacidad opcional, con cancelación y limpieza; la documentación del motor PTC declara que no tiene un deadline global transcurrido. Eso no satisface automáticamente nuestras aprobaciones versionadas y procesos recuperables. No se auditó aquí la totalidad de motores/plugins posibles. [Workflow](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/subsystems/workflow.md).
4. **Persistencia de sesión tiene fronteras.** Su log conserva los resultados asentados y proyecta el contexto del modelo; el streaming vivo es transitorio. La arquitectura señala que una pérdida de proceso antes de asentar un intento no deja su stream durable. Esto distingue correctamente hechos persistidos de actividad visible, sin ofrecer recuperación arbitraria de cómputo. [Arquitectura, Session log](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/architecture.md#session-log).
5. **El ciclo de plugins también tiene costo.** Su gestor maneja instalación, activación y aplicación de cambios como estados diferentes. Documenta restauración parcial ante fallos y que reemplazar paquetes requiere reiniciar para cargar una nueva generación de módulos. La modularidad no elimina despliegues, compatibilidad y migraciones. [Gestor de plugins](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/packages/boot/plugin-manager/README.md).

Además, su SDK Python lanza el runtime `dsh` del perfil SDK; no es un sustituto Python directo de `create_deep_agent`. Adoptarlo implicaría integrar otro runtime y sus protocolos, aunque mantuviéramos nuestro frontend. [Arquitectura, Application launch](https://github.com/deepseek-ai/deepseek-harness/blob/00102833dfaee1da9f48a3a8eae9d34005a75218/docs/architecture.md#application-launch).

## Aplicación concreta a nuestra matriz

La valoración de esta tabla es propia: identifica qué resuelve el patrón y qué responsabilidad queda pendiente en DSAgent.

| Requisitos | Qué aporta una arquitectura de plugins | Qué no resuelve por sí sola |
|---|---|---|
| C01/C02/C20: dominios, herramientas y paquetes | Paquetes versionados que declaran y consumen capacidades | Compatibilidad del paquete, permisos concedidos y actualización de runs activos |
| C06/C18: recursos y ejecución | Proveedores de dataset, repositorio, filesystem y cómputo detrás de contratos | Aislamiento efectivo, sincronización de artefactos y recuperación del entorno |
| C09/C10: procesos variables | Plantillas o proveedores de proceso por dominio | Autoridad de estado, semántica de reintento y límite de iteraciones |
| C11/C13/C17: aprobación, durabilidad y efectos | Puntos claros para integrar validación y persistencia | Transacciones, propiedad del trabajo y reconciliación con sistemas externos |
| C03/C04/C05: identidad, permisos y secretos | Inyección de capacidades autorizadas y contexto delimitado | Una frontera de seguridad entre plugins que comparten proceso y privilegios |
| C19/O06: resultados y experiencia | Visores y tipos de artefacto registrables | Accesibilidad, manejo seguro de contenido y UX del dominio |
| D03–D08: notebooks, experimentos y modelos | Adaptadores especializados reutilizables | Plataforma de cómputo, linaje, evaluación científica y operación de modelos |
| M01–M07/S01–S07: agencia y SDLC | Composición de herramientas, recursos, procesos y vistas | Políticas comerciales, separación de clientes y garantías de entrega |

## Arquitectura recomendada para DSAgent

Un **monolito modular** con workers cuando lo necesite la ejecución: paquetes internos componibles, catálogo controlado y despliegue reproducible. Esta recomendación busca reducir componentes operativos inicialmente; plugin no significa microservicio.

### Contratos compartidos

El host posee el contrato de trabajo/estado, contexto autenticado, concesión de capacidades, registro de aprobaciones y catálogo de artefactos. Sus implementaciones pueden reemplazarse técnicamente, pero la composición aprobada del producto debe exigir esas garantías. Un plugin de dominio no puede desactivar autorización ni aprobarse sus propios efectos.

Deep Agents implementa la ejecución del agente. LangGraph puede implementar procesos recuperables. Ambos pueden quedar detrás de límites explícitos sin construir ahora un adaptador universal para cualquier motor.

### Capacidades registrables

Concentraría las extensiones en cinco superficies:

1. **Herramientas/conectores:** SQL, Git, proveedores de contenido, APIs externas.
2. **Recursos/workspaces:** datasets, repositorios y bibliotecas de marca, con identidad y versión.
3. **Ejecución/cómputo:** Jupyter, contenedores y jobs remotos con ciclo de vida declarado.
4. **Procesos de dominio:** definición del estado y reglas de transición que usan agentes como trabajadores.
5. **Artefactos/vistas:** métricas y experimentos; piezas y calendario; diffs y pruebas.

Los plugins de UI iniciales serían módulos revisados y compilados con nuestra aplicación. No propongo cargar código de frontend arbitrario proporcionado por usuarios.

### Tres composiciones de producto

| Paquete de dominio | Composición orientativa |
|---|---|
| Ciencia de datos | Recursos de datos + SQL/Python/Jupyter + experimentos/modelos + validación + vistas analíticas |
| Agencias de marketing | Contexto por marca/cliente + investigación/medios + aprobación + calendario/publicación + métricas |
| SDLC | Repositorio/worktree + herramientas de desarrollo + BMAD/Spec Kit adaptados + pruebas/revisión + entrega |

El cartridge pasa a declarar esa composición y sus instrucciones. No debe contener implementaciones ad hoc que eviten las fronteras de autorización o estado. Tampoco puede prometer que instalar solo Markdown añade todas esas capacidades.

## Qué haría para que sea rápido y sólido

- Definir únicamente contratos que necesitan los 48 requisitos, con un proveedor inicial por capacidad. No crear un ecosistema de plugins antes de tener producto.
- Componer al inicio, validar dependencias y publicar un manifiesto de versiones. Posponer hot reload de código y reemplazos durante una ejecución.
- Congelar la composición usada por cada run; permitir actualizar el despliegue sin reinterpretar silenciosamente trabajo pendiente.
- Distinguir desinstalar una capacidad, migrar sus datos y revertir sus efectos externos: son operaciones diferentes.
- Mantener separadas las políticas obligatorias y las extensiones de dominio; los plugins ejecutables internos son código de confianza, no una sandbox.
- Conservar la UI y procedimientos útiles, y rediseñar el control de ejecución según la matriz anterior. No hace falta portar Cordis ni cambiar de lenguaje para aplicar inversión de dependencias y gestión explícita del ciclo de vida.

**Recomendación final de diseño:** adoptar la composición por capacidades de DeepSeek Harness sobre nuestra base Deep Agents/LangGraph. No elegir su runtime como atajo a producción basándose en el lema. La velocidad vendrá de contratos acotados y componentes reutilizables; la solidez, de autoridad de estado, permisos, recuperación y compatibilidad. No se ha medido el costo de migrar y no se afirma que una opción sea más rápida en ejecución.

## Alcance de evidencia

Lectura de arquitectura, Cordis, jobs, workflow, scope, herramientas, persistencia, gestor de plugins y aviso de seguridad; inspección selectiva de `core/scope/src/index.ts`, `core/tools/src/index.ts` y `jobs/jobs-local/src/index.ts`. Los archivos externos se descargaron como texto a un directorio temporal; no se ejecutó código del proyecto. No es una auditoría completa de seguridad ni un benchmark. La recomendación queda como diseño propuesto, sin cambiar el runtime de DSAgent.
