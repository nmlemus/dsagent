# Flexibilidad real del harness y prueba de arquitectura propuesta

2026-09-22 · Código evaluado: `7a6b805` · Complementa y corrige la [evaluación de SDLC del 21](sdlc-cartridge-feasibility-2026-09-21.md).

Estado: **evaluación y propuesta, no decisión de arquitectura aprobada ni implementación**. Lectura de código y documentación oficial; no se instaló BMAD/Spec Kit, no se ejecutó un cartridge SDLC ni se midió carga. El informe original se conserva íntegro.

## Veredicto

La preocupación es válida. Hoy DSAgent es un host de agentes configurable con workflows de etapas, aprobaciones y artefactos. Tiene componentes reutilizables, pero todavía no demuestra la promesa de convertirse en un equipo operativo de cualquier dominio cambiando únicamente el cartridge.

La [tesis original](../architecture.md#1-thesis) promete que sustituir DS por BMAD convierte el mismo binario en un equipo SDLC. La implementación cubre mejor el cambio de instrucciones y documentos que el cambio de recursos, herramientas y procesos. El cartridge de marketing amplía el vocabulario y el trabajo realizado, pero no demuestra por sí solo integración con repositorios, sistemas externos o procesos persistentes con iteraciones.

No hay evidencia para concluir que hace falta reescribirlo. Sí hay evidencia para revisar el alcance del contrato de extensión antes de vender esa flexibilidad.

## Cuatro dimensiones distintas

| Dimensión | Evidencia actual | Valoración |
|---|---|---|
| Dominio e instrucciones | Personas, modelos por persona, skills, instrucciones, etapas y etiquetas configurables | Buena base para equipos que trabajan sobre archivos y entregables |
| Recursos y capacidades | Entradas centradas en archivos; entornos kernel/docker; herramientas cableadas desde Python | Extensibilidad parcial, sin contrato completo para incorporar nuevos recursos e integraciones desde un cartridge |
| Proceso | Orden de dependencias, gates, repetición de una etapa rechazada; agente puede iterar dentro de una etapa | No equivale a gestionar un backlog, iteraciones entre etapas o trabajos hijos persistentes |
| Escala operativa | Driver con threads del proceso, estado de runs en archivos | No demuestra distribución, recuperación con efectos externos ni capacidad multiusuario bajo carga |

Un DAG describe dependencias; el runner actual recorre su orden topológico secuencialmente. Tener un DAG no implica ejecutar ramas en paralelo. Tampoco una buena separación de cartridges implica capacidad para atender muchos usuarios: eso requiere validación independiente.

## Correcciones al informe anterior

### 1. Reutilizar el material no garantiza compatibilidad del método

`load_cartridge` exige `cartridge.yaml` y carga personas, skills, workflows, entornos y experiencia. No implementa la ejecución de `hooks/hooks.json` ni la conexión de `.mcp.json`, aunque ambos aparecen en el diseño original. `Persona.tools` se carga pero el constructor de agentes no aplica esa lista para restringir herramientas.

Además, `run_skill_script` antepone siempre `env.python`. Un script de shell no se vuelve compatible por estar dentro de un skill. Spec Kit documenta variantes Bash, PowerShell y Python; elegir Python podría facilitar una adaptación concreta, pero habría que validar también rutas, inicialización y estado del proyecto. Esto contradice la seguridad del anterior «no necesita adaptador». [Documentación oficial de Spec Kit](https://github.github.io/spec-kit/quickstart.html).

### 2. La elección Spec Kit frente a BMAD no se sostiene por «lineal frente a iterativo»

La documentación actual de Spec Kit incluye repetir implementación y verificación de convergencia, añadiendo tareas cuando encuentra faltantes. BMAD ofrece una ruta para un cambio pequeño directamente sobre un repositorio existente, sin exigir primero toda una cadena de roles. La comparación anterior simplifica demasiado ambos métodos. Conviene elegir un caso acotado y fijar la versión a adaptar, no asumir compatibilidad con todas sus variantes. [Spec Kit](https://github.github.io/spec-kit/quickstart.html), [BMAD](https://docs.bmad-method.org/start/build-your-first-change/).

### 3. No hay integración de repositorios; eso no significa que Git sea imposible

`seed_run` copia archivos planos a `workspace/data`; no prepara un repositorio ni registra su revisión de origen. Pero el entorno ofrece shell: si Git, acceso y credenciales están disponibles, técnicamente podría utilizarlo. No se probó aquí.

Lo que falta es un ciclo de vida gestionado: obtener una revisión, preparar una copia aislada, controlar qué puede modificarse, conservar el diff y entregar el resultado con trazabilidad. Crear un repositorio desde cero tampoco está demostrado automáticamente por poder escribir archivos. Las afirmaciones absolutas «greenfield funciona» y «brownfield no» exceden la evidencia del informe.

### 4. Hay iteración, pero no un proceso iterativo completo

Deep Agents puede ejecutar ciclos de edición y pruebas dentro de una etapa. El runner también devuelve una etapa rechazada a `pending`, conserva versiones y transmite la observación en el siguiente intento. La ausencia es más específica: no hay contrato declarado para recorrer historias, coordinar trabajos hijos y reanudar ese proceso conservando límites e historial.

Se puede coordinar parte desde el chat o un script. Eso no ofrece automáticamente las mismas garantías que un proceso persistente reconocido por el sistema.

### 5. UI configurable y artefactos obligatorios tienen un alcance concreto

`experience` cambia el contenido de cinco secciones fijas. No añade un selector de repositorios, una revisión de diffs ni una vista de pruebas. El visor genérico puede presentar documentos de SDLC, pero no equivale a una experiencia completa de ingeniería.

`produces` verifica presencia de archivos; además se comprueba que una etapa devuelta haya reescrito al menos un entregable. Ninguna comprobación establece que el código cumpla el requerimiento. Existen gates automáticos para verificaciones específicas, pero hay que implementarlas y conservar su evidencia.

### 6. Ampliar el core no viola por sí mismo la independencia del dominio

Añadir una capacidad genérica para preparar workspaces o ejecutar herramientas declaradas es evolución de plataforma. Añadir reglas como `if cartridge == "bmad"` al runner sí introduciría conocimiento del dominio.

La prueba adecuada no es «nunca tocar el harness»: es comprobar que, una vez incorporada una capacidad general, otros cartridges puedan utilizarla sin otra excepción específica. Hay además residuos literales de DS en el prompt del orquestador (`Ana`, `MMM`), pero eliminarlos no resuelve las limitaciones estructurales.

## Qué conservar y qué ampliar

Conservar separación cartridge/host, personas y skills, gates, entregables, API/UI común y el bucle de herramientas de Deep Agents. No recomiendo reimplementar ese bucle ni convertir el runner en un planificador libre.

Propongo validar estas extensiones, en este orden y solo con alcance justificado por el caso de prueba:

1. **Recursos y workspace.** Contrato para preparar una copia de trabajo desde un recurso identificado y versionado. Un adaptador de repositorio sería una implementación; archivos y datasets continuarían siendo otras.
2. **Capacidades y permisos.** Declaración de herramientas requeridas, resolución mediante adaptadores y aplicación real de permisos. Distinguir el conocimiento del skill del acceso a shell, Git o servicios. Incluir ejecutores de scripts explícitos, sin asumir que todo es Python.
3. **Proceso persistente.** Cuando el caso lo exija, coordinar unidades de trabajo con intentos limitados, estado, aprobaciones y recuperación. Puede envolver los workflows actuales; no exige introducir todos los bucles posibles en YAML.
4. **Resultados y vistas.** Contratos para documentos, diffs y resultados de pruebas, con visores controlados. El cartridge declara qué produce; la UI conoce cómo presentar ese tipo de resultado.

La durabilidad, el aislamiento y los límites de ejecución son requisitos transversales; no deben esperar a terminar las cuatro extensiones. Tampoco resolverlos por sí solos convierte el sistema en una plataforma extensible. Cola de trabajos, workers distribuidos y otras decisiones de escala deben responder a carga y recuperación medidas, no a la existencia de un tercer cartridge.

## Próxima prueba recomendada

Antes de importar BMAD completo o invertir en otra demostración documental, ejecutar **un cambio pequeño sobre un repositorio existente** con pruebas locales. Esto valida la arquitectura; no implica convertir SDLC en el primer mercado del producto.

Secuencia: preparar revisión de origen → especificar cambio → implementar → detectar fallo de prueba y corregir con límite de intentos → presentar diff y evidencia → pausar para revisión → reiniciar el servicio → retomar y exportar un patch.

| Criterio de aceptación propuesto | Qué evita dar por supuesto |
|---|---|
| Origen identificado y copia de trabajo aislada; escritura fuera del área denegada | Que una carpeta virtual sea una frontera suficiente |
| Pruebas ejecutadas y resultados registrados, incluido un fallo controlado | Que generar código equivalga a completarlo |
| Reintentos limitados y efectos sin duplicación al recuperar | Que reejecutar un agente sea reanudar un proceso |
| Aprobación recuperable después de reiniciar | Que persistir archivos preserve toda la ejecución |
| Segunda tarea sobre el proyecto con revisión de partida explícita | Que un único run demuestre continuidad del trabajo |
| DS y marketing siguen funcionando; sin condiciones por nombre de dominio en el core | Que la nueva capacidad sea realmente general |

Esta es una propuesta de aceptación, **no una prueba ya superada**. Primero implementar la mínima integración de recursos y herramientas necesaria; ampliar el proceso solo si ese recorrido demuestra que hace falta. Incorporar tareas aprobadas al backlog vigente, sin crear una segunda lista de estados paralela.

## Evidencia local y límites

Anclas referidas al commit evaluado; los archivos pueden cambiar posteriormente:

- `src/dsagent/cartridge/loader.py:276`: contrato real de carga.
- `src/dsagent/cartridge/models.py:26,153,170,215`: herramientas declaradas, orden de pasos, entornos y navegación fija.
- `src/dsagent/host/build.py:150,185,216`: ejecutor Python, composición de herramientas y ejemplos de DS en el orquestador.
- `src/dsagent/datasets.py:53`: copia plana de entradas.
- `src/dsagent/envs/kernel.py:87`: shell y kernel perezoso; no demuestra aislamiento del sistema anfitrión.
- `src/dsagent/runner/runner.py:554,680,897,956`: ejecución secuencial, entregables, rechazo y gates automáticos.
- `src/dsagent/driver.py:311`: lanzamiento de trabajo en threads del proceso.
- `ui/src/components/Launcher.tsx:41`: formulario y selección de archivos de entrada.

Fuentes externas consultadas el 2026-09-22, sin fijar una release. Sirven para corregir la caracterización de los métodos, no para certificar compatibilidad. No se estiman capacidad concurrente, coste de integración ni calidad del modelo sin ejecutar la prueba. No se modificó el runtime ni se ejecutaron tests de aplicación para este cambio documental.
