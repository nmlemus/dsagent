# ADR — Permisos de herramientas y entorno de cómputo

Fecha: 2026-09-24. Estado: implementado en el host local.
Alcance del backlog: SR-032.a y SR-046.a. Los padres conservan criterios abiertos.

## Problema y decisión

`Persona.tools` se cargaba pero no se usaba al construir agentes. Filtrar solo
`env.tools` tampoco era suficiente: Deep Agents añade herramientas de filesystem,
shell y delegación. Su subagente general-purpose no hereda automáticamente nuestro
middleware. Una restricción escrita en el prompt no impedía una llamada forzada.

El host aplica un `ToolPolicy` inmutable al construir cada persona, tanto en
workflows como en el chat. Filtra los esquemas presentados al modelo y comprueba
de nuevo cada llamada antes del dispatch, en las rutas síncrona y asíncrona.
Una llamada denegada devuelve un ToolMessage de error sin ejecutar la herramienta.
Deep Agents conserva el bucle, las herramientas y el manejo de contexto.

## Configuración y compatibilidad

Los nombres de herramientas de DSAgent se declaran en `cartridge.yaml`, sin
introducir campos específicos del runtime en el frontmatter de Claude Code:

```yaml
personas:
  editor:
    role: Editor
    tools: [ls, read_file, write_file, edit_file, glob, grep, write_todos]
```

Precedencia: `personas.<name>.tools` en cartridge.yaml, luego `tools` del
frontmatter. Se aceptan listas de nombres exactos, sin comodines ni duplicados;
no se traducen alias de otros runtimes. `[]` no concede herramientas.
Ausencia o `null` usa un conjunto finito: ls, read_file, write_file, edit_file,
glob, grep, write_todos, show_chart y show_table. Los dos últimos solo se ofrecen
cuando el host suministra esas herramientas.

Python, shell y scripts requieren concesión explícita. Herramientas nuevas de un
entorno o plugin no se heredan por defecto. Los nombres desconocidos fallan antes
de crear el agente; los nombres reservados de cómputo pueden estar declarados
aunque un entorno concreto no aporte esa herramienta, por ejemplo run_python en
`filesystem`. Una concesión no instala un proveedor ni amplía el scope de skills.

El cartridge de ciencia de datos conserva sus herramientas de cómputo mediante
declaraciones explícitas. Marketing concede solo herramientas de archivos y
planificación. Un cartridge de terceros que dependía de ejecución implícita debe
declararla al migrar. Las factorías inyectadas por embedders que reemplazan por
completo los builders son código confiable y deben aplicar su propia política.

Las concesiones efectivas y la versión de política entran en la huella de
ejecución. Reanudar con permisos/configuración distintos exige restaurar la
versión anterior o crear un run nuevo; no se migran runs históricos silenciosamente.

## Delegación y autoridad

Las personas no reciben `task`; declararlo se rechaza. El router del host puede
delegar únicamente a las personas cargadas en su roster. Se rechazan destinos
desconocidos y el general-purpose implícito antes de invocarlos. El router recibe
las herramientas de control de workflows suministradas por el host y las
herramientas básicas de archivos, pero no las herramientas de cómputo del entorno.

Estas concesiones son parte de una composición administrada de código confiable.
No identifican usuarios ni autorizan recursos entre tenants. Tampoco hacen que
una API externa sea idempotente o que una aprobación humana pueda sustituirse por
una decisión del plugin. Esos contratos siguen pendientes.

## Variables de shell y Jupyter

El cliente del modelo conserva las credenciales del servidor. Shell y Jupyter
reciben un diccionario construido desde cero: PATH del intérprete más las rutas
base del sistema, UTF-8, desactivación de paquetes del usuario y directorios de
configuración/caché de IPython, Jupyter y Matplotlib dentro del workspace.
No se heredan claves de proveedores, credenciales cloud, PYTHONPATH, hooks de
inicio de shell ni la ruta de control del supervisor.

Jupyter usa el intérprete de DSAgent y descarta `argv`/`env` inyectados por el
kernelspec local. Las herramientas que necesiten credenciales deberán recibirlas
mediante un contrato explícito futuro. Docker recibe solo variables fijas de
locale, HOME y caché; la configuración o credenciales incluidas en una imagen
siguen siendo responsabilidad del administrador.

## Límite de seguridad y trabajo siguiente

Conceder `run_python`, `execute` o `run_skill_script` permite ejecutar código
arbitrario con las facultades del entorno. Por ejemplo, permitir Python y denegar
shell no impide que Python invoque comandos. El control es sobre herramientas,
no una clasificación de todos los efectos posibles de un programa.

El backend kernel sigue usando el usuario y filesystem del host y solo está
permitido en desarrollo confiable. Serve usa ahora Docker endurecido para personas
con cómputo y `filesystem` para las que solo editan archivos; HOME, red, rootfs,
recursos y `.dsagent` están controlados según el
[ADR de aislamiento](sandbox-isolation.md). Esto sigue sin separar usuarios o
tenants ni imponer cuota al bind mount.

Siguiente parte del bloque de seguridad:

1. Completar autorización por recurso y credenciales explícitas por herramienta.
2. Añadir cuotas de disco/admisión y decidir rootless Docker o workers remotos para
   una instalación compartida.
3. SR-002: cerrar contenido activo servido desde el origen de la aplicación.

## Evidencia

`tests/test_tool_permissions.py` usa grafos reales de Deep Agents y un modelo
determinista: llamadas denegadas, archivos permitidos, delegación de chat,
subagente genérico, grants vacíos, herramientas nuevas y nombres inválidos.
`tests/test_capability_contracts.py` comprueba que un grant no concede un proveedor
ausente y que cambiar permisos invalida un resume antes de escribir estado.

`tests/test_compute_variables.py` verifica secretos marcador, hooks de inicio,
shell real y kernel persistente real (este último con `DSAGENT_KERNEL=1`).
No se usan credenciales reales ni llamadas pagadas para estas pruebas.

Validación registrada: suite general con 500 pruebas aprobadas y 13 omitidas;
después de añadir casos de permisos y validar el error del loader, 56 pruebas
dirigidas aprobadas con `DSAGENT_KERNEL=1`. Ruff pasa en todos los módulos
modificados. Servidor reiniciado y respuestas HTTP verificadas en el puerto 8799.

La implementación se contrastó con el código instalado de Deep Agents y la
[referencia oficial de middleware de LangChain](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware).
