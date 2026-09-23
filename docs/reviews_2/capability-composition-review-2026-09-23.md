# Revisión de la implementación de composición por capacidades

Fecha: 2026-09-23. Commit revisado: `7b187d04370ea85beb83c9d82ecabc0fc6f19b67`.

## Veredicto

La dirección es aprovechable: registro explícito, resolución de dependencias,
despacho extensible de entornos/scripts y separación del motor Deep Agents.
Esta entrega es una primera capa de resolución de proveedores, pero todavía no
cumple todas las garantías anunciadas de compatibilidad, alcance por cartridge y
composición congelada. Recomiendo corregir los hallazgos siguientes antes de
construir conectores y recursos sobre ella. No recomiendo desechar el runtime.

Revisión local de código y sondas con directorios temporales, agentes falsos y
scripts inocuos. No se invocaron modelos pagos ni Docker. Suite ejecutada:
`PYTHONPATH=. .venv/bin/pytest -q`: **408 passed, 12 skipped, 8 warnings**.
Los casos siguientes no están cubiertos por las siete pruebas nuevas del commit.
Este informe no modifica código de aplicación.

## Hallazgos y tareas de corrección

### CAP-R01 · P1 · La huella no fija la configuración efectiva de ejecución

Evidencia: `src/dsagent/capabilities/core.py:125` y
`src/dsagent/runner/runner.py:552`.

La huella contiene id, kind, versión, dependencias y metadatos de proveedores.
No incluye las instancias `EnvSpec`, imagen Docker, requisitos de paquetes ni
la definición del cartridge/workflow que se reanuda. Cambiar la imagen de
`meridian` y los requisitos del entorno `default` deja exactamente la misma
huella. El guard de resume permite por tanto una ejecución con otra configuración.

Esto no invalida el hash como identificador del catálogo de proveedores. Sí
impide presentarlo como congelación del entorno o reproducibilidad de un run:
las instrucciones y configuración se siguen leyendo desde el cartridge actual.

**Corrección:** distinguir manifiesto de proveedores y snapshot de ejecución.
Fijar configuración normalizada, identidad/versión del cartridge y contenido
relevante del workflow/skills. Para imágenes reproducibles se necesita digest o
una política explícita, pues conservar un tag mutable tampoco fija su contenido.

**Aceptación:** un run pausado no continúa silenciosamente después de modificar
imagen, configuración o instrucciones: usa su snapshot o rechaza el cambio antes
de ejecutar trabajo. La política de adopción de runs antiguos queda explícita.

### CAP-R02 · P2 · La inyección de una composición evita validar el contrato del cartridge

Evidencia: `src/dsagent/runner/runner.py:342`;
`src/dsagent/serve.py:162` repite la aceptación sin validación.

El constructor usa `capabilities or compose_cartridges([cartridge])`. Si el
embedder entrega una composición, no comprueba requisitos ni versiones del
cartridge. Sonda: un cartridge exige `script.python@99`; la composición normal
lo rechaza, pero inyectar una composición con `script.python@1` permite terminar
el mismo dry run como `done`. También pueden faltar proveedores obligatorios.
`workflow_tools` restringe correctamente en su camino particular, pero no
protege los demás puntos de entrada.

**Corrección:** validar y derivar la composición del consumidor en todas las
entradas públicas. La inyección permite elegir implementaciones; no debe omitir
la validación del contrato. Reutilizar una única función de enlace.

**Aceptación:** tanto el runner directo como el servidor y los builders rechazan
versiones incompatibles o requisitos ausentes antes de crear agentes/entornos,
independientemente de si reciben una composición explícita.

### CAP-R03 · P2 · Los chats con varios cartridges comparten ejecutores no declarados

Evidencia: `src/dsagent/host/build.py:150` y `:267`.

El orchestrator construye todas las personas con el mismo `Env`; el tool elige
el ejecutor desde `env.capabilities` sin restringirlo a su cartridge. Esto afecta
al chat CLI con varios cartridges y al proyecto web predeterminado sin binding,
que reciben la unión del servidor. Los proyectos no predeterminados ligados a
un único cartridge sí reciben una composición restringida.

Sonda: Marketing declara `capabilities: []`. Con DS + Marketing en el mismo
entorno, un script `.py` de un skill concedido a Lucía se ejecuta correctamente
usando `script.python`, aportado por DS. La contención del archivo funciona;
el problema es el alcance de la capacidad, no un escape de filesystem.

**Corrección:** enlazar cada consumidor a las capacidades declaradas por su
cartridge al construir el tool, incluso si comparte infraestructura con otros.

**Aceptación:** cargar un segundo cartridge no habilita ni vuelve ambiguos los
ejecutores del primero. Cubrir CLI multicartridge y chat web predeterminado,
además de proyectos con binding.

### CAP-R04 · P2 · Regresión en el ensamblaje público de entorno y agente

Evidencia: `src/dsagent/envs/base.py:70` y
`src/dsagent/host/build.py:150`.

`make_env(spec, workspace)` ahora adjunta una composición que contiene solo el
proveedor del entorno. Al construir después una persona, ese valor impide el
fallback a las capacidades del cartridge. Un script Python válido del cartridge
DS falla con `no enabled 'script-runner' capability handles extension='.py'`.
Los CLI modificados pasan la composición explícita; los consumidores existentes
de la API de dos argumentos quedan rotos. Las pruebas de scripts crean el
entorno con `make_kernel_env`, por lo que no detectan la regresión.

**Corrección:** definir explícitamente quién compone el agente y su entorno,
y conservar una ruta compatible o fallar en el ensamblaje con una migración
documentada. No distinguir implícitamente entre un `Env` creado por factoría
concreta y otro creado por la API pública.

**Aceptación:** `make_env` → `build_persona_agent`/`build_orchestrator` ejecuta un
script legítimo del cartridge DS sin cambios inesperados para el embedder.

### CAP-R05 · P2 · El tipo del proveedor se valida después de ejecutar trabajo

Evidencia: `src/dsagent/capabilities/builtin.py:88`,
`src/dsagent/capabilities/core.py:95` y
`src/dsagent/runner/runner.py:996`.

Las exigencias implícitas de entorno y gate guardan solo un id. La composición
verifica existencia/versión, pero no el tipo que necesita el consumidor.
`gate: {kind: auto, check: check.py, runner: environment.kernel}` se compone sin
error. La sonda ejecutó primero al agente falso; después `provider(...,
kind='script-runner')` lanzó `CapabilityError`. El archivo `run.json` quedó con
`status: running`, pese a que el runner ya había terminado con excepción.

**Corrección:** exigir tipo de contrato durante el enlace y detectar conflictos
de selección de ejecutores antes del trabajo. Registrar un estado terminal
honesto cuando la invocación de un proveedor de gate falla en runtime.

**Aceptación:** el ejemplo anterior falla antes de invocar al agente; un
proveedor válido cuya ejecución lanza una excepción deja un run fallido con
causa explícita y recursos liberados.

## Límites arquitectónicos que no son regresiones nuevas

- `requires` resuelve presencia y orden del grafo; no construye ni inyecta las
  instancias dependientes. La factoría `acme.identity` del ejemplo documental
  nunca se invoca automáticamente. No existe todavía un contenedor con ciclo de
  vida de servicios, rollback de inicialización ni cierre de dependencias.
- `Callable[..., Any]` y `kind` dejan la forma del contrato en convenciones.
  Antes de añadir recursos/conectores conviene definir protocolos tipados por
  superficie y contexto de ejecución explícito.
- El registro no es autorización. El entorno kernel conserva shell y Python;
  omitir `script.python` limita el tool de scripts, no elimina la capacidad
  general de ejecutar Python. La frase documental «una capacidad autoriza» es
  demasiado fuerte mientras no existan políticas y enforcement efectivos.
- El cierre inverso de `Env.closers` es útil, pero una excepción interrumpe los
  cierres restantes y el backend. El runner tampoco garantiza el cierre de los
  demás entornos si uno falla. Este comportamiento ya existía: el commit cambia
  el orden, no resuelve el ciclo de vida completo.

## Orden recomendado

1. Unificar enlace/validación de consumidores: CAP-R02, CAP-R03, CAP-R04 y CAP-R05.
2. Definir e implementar el snapshot efectivo y su compatibilidad: CAP-R01.
3. Precisar los contratos de dependencia, lifecycle y permisos antes de
   extender el catálogo a `tool-provider` y `resource-provider`.

La implementación aporta una base útil. Su estado correcto es «resolución de
proveedores integrada parcialmente», con tareas concretas para cerrar sus
garantías, y no una arquitectura de plugins ya completada.
