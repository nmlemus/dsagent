# ADR — Aislamiento de cómputo para el servidor

Fecha: 2026-09-28. Estado: implementado para la instalación local de un solo host.
Alcance del backlog: SR-009.a, SR-010.a, SR-046.b en parte y SR-048.a.
Los padres conservan los criterios de autenticación, almacenamiento, credenciales y
operación multiusuario que se enumeran al final.

## Problema y decisión

Una herramienta `execute` o `run_python` permite ejecutar código arbitrario. La
allowlist de herramientas reduce quién recibe esa facultad, pero el backend
`kernel` seguía ejecutando el código con el usuario, red y filesystem del servidor.
Eso no es un límite aceptable para una UI que procesa archivos y prompts externos.

El modo `serve` admite ahora dos fronteras por defecto:

- `filesystem`, para personas que solo leen y escriben dentro del workspace y no
  reciben shell ni Python;
- `docker`, para cómputo, con una imagen preconstruida y una política de contenedor
  propiedad del host que el cartridge no puede debilitar.

Un entorno `kernel` local o un Docker que solo declare `build:` se rechaza antes de
provisionar el servidor. `--insecure-local` conserva ambos comportamientos para
desarrollo explícitamente confiable. No es un modo soportado para un piloto.

El cartridge DS usa `dsagent/ds-default:latest` y
`dsagent/ds-meridian:latest`; conserva sus Dockerfiles como fuentes auditables y
parte de la huella, pero las imágenes se construyen fuera del arranque. Marketing
usa `filesystem`, porque sus personas actuales no necesitan ejecutar código.

## Contrato del sandbox Docker

Cada entorno de un run recibe un contenedor reutilizable durante ese run. El modelo
y el cliente del proveedor permanecen en el proceso del host; solo shell, scripts,
gates automáticos y el kernel analítico entran al contenedor.

La política fija:

| Superficie | Garantía local implementada |
|---|---|
| Filesystem | `/workspace` es el único bind escribible. El rootfs es de solo lectura. Todo `/workspace/.dsagent` se superpone como bind de solo lectura. No se monta el socket Docker. |
| Identidad | El contenedor usa el UID:GID POSIX del servicio y se rehúsa a arrancar con UID 0. |
| Secretos | Se construye un entorno mínimo; no se heredan variables del servidor. HOME, cachés, Jupyter, IPython y Matplotlib viven bajo `/tmp`. |
| Red | `--network none`; no hay interfaz con salida, DNS, metadata cloud ni API interna. |
| Privilegios | `--cap-drop ALL` y `no-new-privileges`; el cartridge no agrega capabilities. |
| Recursos | 8 GiB RAM y swap total, 4 CPU, 256 PIDs, 1024 archivos abiertos y `/tmp` tmpfs de 512 MiB con `nosuid,nodev,noexec`. |
| Persistencia | El workspace persiste en el host. El rootfs y `/tmp` se descartan al cerrar el entorno. |
| Python | Un kernel IPython persistente y su cliente corren dentro del mismo contenedor sin red; imports y variables sobreviven entre llamadas `run_python`. |
| Limpieza | `Env.close()` detiene kernel y contenedor. El supervisor registra y elimina contenedores etiquetados tras cancelación o caída. |
| Procedencia | `run.json` registra el image ID real (`sha256:`), además de la huella de definiciones, Dockerfiles y permisos. |

La política se aplica en `DockerBackend`, no en YAML. Un cartridge puede escoger
una imagen aprobada y declarar si requiere GPU; no puede solicitar red, mounts,
variables, capacidades o límites más amplios.

Las imágenes incluidas parten hoy de Linux `python:3.12-slim`. La ruta verificada
es macOS con Docker Desktop y un host POSIX. Un despliegue Linux debe ejecutar el
servicio como usuario no root y pasar las mismas sondas reales antes del piloto.

## Pruebas negativas y funcionales

La prueba real inspecciona la configuración de Docker y ejecuta intentos desde el
contenedor. Comprueba que el proceso no es root, puede escribir entregables en el
workspace y no puede:

- modificar `.dsagent` ni el rootfs;
- observar un archivo marcador fuera del workspace;
- heredar un secreto marcador del servidor;
- abrir una conexión IP;
- acceder al socket Docker.

Otra prueba abre dos llamadas contra el kernel interno y demuestra que conserva
estado. La suite Docker también recorre Meridian, materialización de skills,
scripts de skills, gates automáticos y verificación host de `produces`. Estas
pruebas son opt-in con `DSAGENT_DOCKER=1`; la suite rápida valida los argumentos de
política y que el servidor rechaza kernels/builds inseguros.

## Límites de confianza y siguientes contratos

Este cambio permite un piloto supervisado en una instalación dedicada; no hace
segura una instalación compartida ni convierte Docker en una frontera equivalente
a una VM o microVM. Permanecen abiertos:

- autenticación y autorización por usuario, proyecto, run y gate (SR-037/045);
- cuotas de disco del bind mount, admisión y presupuestos agregados por tenant
  (SR-018/019);
- credenciales por herramienta, conectores y excepciones de red con deny-by-default
  (resto de SR-046);
- rootless Docker o un worker/microVM remoto para una frontera multiusuario, además
  de proteger el daemon que crea los contenedores;
- imágenes publicadas por digest y cadena de suministro/release (SR-055). El image
  ID congela un run iniciado, pero un tag local sigue siendo mutable entre runs;
- separar del workspace todos los registros de control y auditar cada lector HTTP
  (resto de SR-048), y aislar contenido HTML activo (SR-002);
- almacenamiento durable, leases y reconciliación entre múltiples hosts
  (SR-038–040).

No se concede red para instalar paquetes durante un run. Añadir una dependencia
significa construir y revisar una imagen nueva. No existe fallback silencioso al
kernel del host: si Docker o la imagen no están disponibles, el entorno falla.

Referencias del mecanismo: [opciones de `docker run`](https://docs.docker.com/reference/cli/docker/container/run/),
[driver de red `none`](https://docs.docker.com/engine/network/drivers/none/) y
[seguridad del motor Docker](https://docs.docker.com/engine/security/).
