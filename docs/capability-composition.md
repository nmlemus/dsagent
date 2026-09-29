# Composición de capacidades

**Estado:** primera capa implementada · **Fecha:** 2026-09-23

Actualización 2026-09-28: el host añade [process.local](process-control.md) para
supervisar workflows y [permisos de herramientas](tool-permissions.md) efectivos
por persona. `environment.filesystem` aporta un backend sin ejecución y
`environment.docker` aplica el [contrato de aislamiento](sandbox-isolation.md) en
serve. Los permisos siguen sin sustituir autorización por usuario.

DSAgent conserva Deep Agents/LangGraph como motor y separa el contenido de un
cartridge de las funciones ejecutables del host. Un skill explica cómo trabajar;
una capacidad selecciona la implementación disponible para ese consumidor.
Esta selección no sustituye autorización ni aislamiento del sistema operativo.

## Contrato implementado

Una `CapabilityProvider` declara:

- un identificador estable, por ejemplo `environment.kernel`;
- una clase de contrato (`environment` o `script-runner` en esta primera capa);
- una versión exacta;
- sus dependencias;
- la factoría confiable que implementa el contrato.

`CapabilityRegistry` se construye al arrancar. Rechaza proveedores duplicados,
dependencias ausentes, ciclos y versiones incompatibles. `compose()` produce un
`CapabilitySet` inmutable con un manifiesto determinista y una huella SHA-256.
No hay carga automática de paquetes, hot reload ni ejecución de plugins
arbitrarios proporcionados por usuarios.

Los proveedores incluidos son:

| ID | Contrato | Versión | Selección |
|---|---|---:|---|
| `environment.kernel` | `environment` | 1 | `env.kind: kernel` |
| `environment.docker` | `environment` | 1 | `env.kind: docker` |
| `environment.filesystem` | `environment` | 1 | `env.kind: filesystem` |
| `process.local` | `process` | 1 | supervisor local de workflows |
| `script.python` | `script-runner` | 1 | archivos `.py` |
| `script.shell` | `script-runner` | 1 | archivos `.sh` |

## Declaración de un cartridge

Las capacidades ejecutables que no sean el entorno se declaran explícitamente:

```yaml
name: ds
version: 0.1.0

capabilities:
  - id: script.python
    version: "1"

envs:
  default: { kind: docker, image: example/analysis@sha256:… }
  heavy: { kind: docker, image: example/worker:1 }
```

Cada entorno implica `environment.<kind>`. Para un identificador que no siga
esa convención puede usarse `provider`:

```yaml
envs:
  batch:
    kind: remote-job
    provider: acme.compute-job
```

Si falta un proveedor o la versión no coincide, la composición falla antes de
crear un agente o ejecutar un paso. `compose_cartridges(cartridges, registry)`
acepta tanto un registro como una composición existente: en ambos casos valida
los requisitos y devuelve el subconjunto del consumidor. También valida el tipo
esperado por entornos/gates y detecta ejecutores ambiguos para una extensión
dentro de cada cartridge. Inyectar una composición no permite omitir estos checks.

Cada tool de scripts captura ese subconjunto al construirse, incluso en un chat
multicartridge. `make_env(spec, workspace)` conserva su forma de dos argumentos:
crea el entorno y deja que el builder enlace después las capacidades del cartridge.
Cuando el caller suministra una composición explícita, se respeta como límite;
no se añaden proveedores implícitamente para satisfacer requisitos ausentes.

Los gates automáticos también declaran su ejecutor. `script.python` es el valor
compatible por defecto; otro lenguaje se selecciona sin cambiar el runner:

```yaml
gate:
  kind: auto
  check: checks/validate.sh
  runner: script.shell
```

## Manifiesto y control de cambios por run

El runner escribe en `run.json`:

```json
{
  "capability_fingerprint": "…",
  "capabilities": [
    {"id": "environment.docker", "kind": "environment", "version": "1", "requires": [], "metadata": {}},
    {"id": "script.python", "kind": "script-runner", "version": "1", "requires": [], "metadata": {"extension": ".py"}}
  ]
}
```

La huella anterior identifica proveedores, no la ejecución completa. Ahora
`run.json` incluye además `execution` (schema 1): hash de configuración efectiva,
modelo por defecto, identidad del cartridge/workflow y hashes de archivos de
personas, skills, workflows y contextos de build. Se excluyen cachés Python y
`.git`; se rechazan symlinks a directorios que no puedan inventariarse. La
comprobación es conservadora y abarca las definiciones del cartridge completo.

El launcher escribe este registro al crear el run; CLI/chat al entrar al runner.
Un resume con cambios de definición o contenido se rechaza antes de modificar
el estado o invocar al agente. Se comprueba de nuevo entre pasos y antes de
gates para detectar ediciones mientras se ejecuta. La política es **restaurar la
versión registrada o iniciar otro run**; no se guarda una copia ejecutable de
los archivos ni se migra un run automáticamente.

`environment_identities` registra el id real de imagen Docker inspeccionado
en el contenedor y, para kernel, la versión Python y las versiones instaladas
de los paquetes declarados. Al reprovisionar, un cambio falla antes de ejecutar
al agente o al script del gate. Un tag Docker que apunta a otra imagen deja de
pasar inadvertido; la inspección ocurre después de arrancar el contenedor y no
es una garantía contra efectos de su entrypoint. Si la imagen no puede
identificarse, se libera el contenedor y se rechaza el entorno.

Los runs antiguos sin `execution` se adoptan una sola vez y se deja constancia
en `runner.log`: no se puede reconstruir retrospectivamente qué fuentes usaron.
La identidad de un entorno se registra en su primera provisión conocida.
Cada run recibe solo las capacidades de su cartridge, aunque el servidor tenga
otros dominios cargados.

Esto detecta cambios de las entradas declaradas, no convierte el entorno en
hermético: el código de proveedores se versiona por disciplina de release,
los paquetes transitivos/datos externos no están bloqueados, y un job remoto
debe aportar su propia identidad. No constituye una frontera de seguridad
frente a modificaciones concurrentes maliciosas de los archivos.

## Añadir un proveedor interno

El host registra código revisado durante su ensamblaje:

```python
from dsagent.capabilities import CapabilityProvider, builtin_registry

registry = builtin_registry()
registry.register(CapabilityProvider(
    id="acme.identity",
    kind="identity-context",
    version="1",
    factory=provide_identity_context,
))
registry.register(CapabilityProvider(
    id="acme.compute-job",
    kind="environment",
    version="1",
    factory=make_remote_job_env,
    requires=("acme.identity",),
))
composition = registry.compose(["acme.compute-job"])
```

La factoría de un entorno recibe `(EnvSpec, workspace)` y devuelve `Env`; su
`close()` sigue siendo la frontera de liberación de kernels, contenedores o jobs.
El cierre de `Env` es idempotente y se intentan todos los cierres aunque uno
falle; el runner registra errores de cleanup sin ocultar el resultado del trabajo.
Una factoría `script-runner` recibe `(Env, ruta_relativa, argv)` y devuelve la
respuesta de ejecución del backend. El host conserva la validación de alcance:
un script debe estar dentro del skill concedido a la persona.

`requires` comprueba presencia y ciclos; no inicializa ni inyecta servicios. En
el ejemplo, `provide_identity_context` requiere wiring explícito de la aplicación.
Un contenedor de servicios con lifecycle de dependencias queda fuera de esta capa.
Una excepción al ejecutar un gate automático queda registrada como fallo de gate,
paso y run; los interrupts de revisión humana conservan su semántica.

## Siguientes contratos

La primera entrega probó entornos y ejecutores de scripts. `process.local` añadió
posteriormente el control local de cómputo. Conectores, recursos y vistas aún no
son capacidades. El orden original de contratos para los tres dominios era:

1. `resource-provider`: dataset, repositorio y biblioteca de marca con identidad
   y versión.
2. `tool-provider`: SQL, Git, búsqueda, medios y APIs, con permisos efectivos.
3. `process-provider`: estado y transiciones de DS, marketing y SDLC fuera del
   tool call monolítico. Implementado para workflows locales mediante
   ExecutionService y process.local; chat interactivo y workers remotos pendientes.
4. `artifact-renderer` y validadores específicos por tipo de resultado.
5. proveedores de cómputo remoto con cancelación, deadlines, reconciliación y
   recuperación explícitas.

Cada contrato debe entrar cuando exista un recorrido vertical que lo consuma.
El registro no sustituye autorización, aislamiento, persistencia ni semántica de
reintentos; esas garantías siguen perteneciendo al host.
