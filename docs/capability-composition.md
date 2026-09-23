# Composición de capacidades

**Estado:** primera capa implementada · **Fecha:** 2026-09-23

DSAgent conserva Deep Agents/LangGraph como motor y separa el contenido de un
cartridge de las funciones ejecutables del host. Un skill explica cómo trabajar;
una capacidad autoriza y suministra el código que puede ejecutar ese trabajo.

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
  default: { kind: kernel }
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
crear un agente o ejecutar un paso.

Los gates automáticos también declaran su ejecutor. `script.python` es el valor
compatible por defecto; otro lenguaje se selecciona sin cambiar el runner:

```yaml
gate:
  kind: auto
  check: checks/validate.sh
  runner: script.shell
```

## Composición congelada por run

El runner escribe en `run.json`:

```json
{
  "capability_fingerprint": "…",
  "capabilities": [
    {"id": "environment.kernel", "kind": "environment", "version": "1", "requires": [], "metadata": {}},
    {"id": "script.python", "kind": "script-runner", "version": "1", "requires": [], "metadata": {"extension": ".py"}}
  ]
}
```

Un resume con otra composición se rechaza antes de modificar el estado. Los runs
anteriores a este contrato se adoptan una sola vez al reanudarse y quedan
fijados desde entonces. Cada run recibe solo las capacidades de su cartridge,
aunque el servidor tenga otros dominios cargados.

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
Una factoría `script-runner` recibe `(Env, ruta_relativa, argv)` y devuelve la
respuesta de ejecución del backend. El host conserva la validación de alcance:
un script debe estar dentro del skill concedido a la persona.

## Siguientes contratos

Esta entrega prueba la composición en las dos superficies que ya ejecutaban
código. No convierte todavía conectores, recursos, procesos ni vistas en
capacidades. El orden recomendado para los tres dominios es:

1. `resource-provider`: dataset, repositorio y biblioteca de marca con identidad
   y versión.
2. `tool-provider`: SQL, Git, búsqueda, medios y APIs, con permisos efectivos.
3. `process-provider`: estado y transiciones de DS, marketing y SDLC fuera del
   tool call monolítico.
4. `artifact-renderer` y validadores específicos por tipo de resultado.
5. proveedores de cómputo remoto con cancelación, deadlines, reconciliación y
   recuperación explícitas.

Cada contrato debe entrar cuando exista un recorrido vertical que lo consuma.
El registro no sustituye autorización, aislamiento, persistencia ni semántica de
reintentos; esas garantías siguen perteneciendo al host.
