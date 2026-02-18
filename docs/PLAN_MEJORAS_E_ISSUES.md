# Plan: Mejoras e issues para futuras implementaciones

Revisión del proyecto DSAgent (feb 2025): hallazgos de seguridad, consistencia, calidad y mejoras sugeridas. Pendiente de implementación.

---

## Qué es el proyecto

**DSAgent** es un agente de ciencia de datos impulsado por LLMs que:

- Ejecuta código Python en un kernel Jupyter persistente.
- Usa LiteLLM para planificación y generación de código (OpenAI, Anthropic, Groq, Ollama, etc.).
- Expone **CLI** (`dsagent chat`, `dsagent run`) y **API REST + WebSocket** (`dsagent serve`).
- Soporta MCP (Model Context Protocol) y Human-in-the-Loop (HITL).

Arquitectura relevante:

- **`src/dsagent/agents/`**: `PlannerAgent` (planificación) y `ConversationalAgent` (chat).
- **`src/dsagent/core/`**: motor, ejecutor Jupyter, planner, HITL.
- **`src/dsagent/server/`**: FastAPI, rutas (sessions, chat, files, artifacts, kernel, hitl), WebSocket.
- **`src/dsagent/session/`**: sesiones con backends SQLite y JSON.
- **`src/dsagent/config.py`** y **`src/dsagent/server/deps.py`**: configuración y dependencias del servidor.

---

## Seguridad

### 1. Path traversal en descarga/borrado de archivos (prioridad alta)

En **`src/dsagent/server/routes/files.py`** y **`src/dsagent/server/routes/artifacts.py`**:

- `download_file`, `delete_file` y `download_artifact`, `delete_artifact` construyen `file_path = base_path / filename` sin comprobar que el resultado quede dentro de `base_path`.
- Un `filename` como `../../../etc/passwd` o `..\\..\\foo` puede hacer que se resuelva fuera del directorio de la sesión.

**Recomendación:** Tras resolver la ruta (`file_path.resolve()`), comprobar que está bajo `base_path.resolve()` (por ejemplo con `is_relative_to()` en Python 3.9+), y rechazar con 400 si no.

### 2. Path traversal en store de sesiones (backend JSON)

En **`src/dsagent/session/store.py`** (JSON backend):

- `_session_path(session_id)` hace `self.storage_dir / f"{session_id}.json"`.
- Un `session_id` como `../../tmp/evil` (si en algún flujo se aceptara) podría apuntar fuera de `storage_dir`.

**Recomendación:** Validar `session_id` en los puntos de entrada (API): solo permitir caracteres seguros (p. ej. alfanuméricos, guión, underscore) o formato conocido (p. ej. `YYYYMMDD_HHMMSS_xxxxxx`). Opcionalmente en el store: normalizar y comprobar que el path resuelto esté dentro de `storage_dir`.

### 3. API key opcional y CORS por defecto

- En **`src/dsagent/server/deps.py`**: si `DSAGENT_API_KEY` no está definido, `verify_api_key` no exige clave; todas las rutas que dependen de ella quedan abiertas.
- En **`src/dsagent/config.py`** / app: `cors_origins` por defecto es `"*"` con `allow_credentials=True`, lo que en producción puede ser demasiado permisivo.

**Recomendación:** Documentar claramente que en producción debe configurarse `DSAGENT_API_KEY` y orígenes CORS restringidos; opcionalmente en modo "producción" rechazar arranque si no hay API key.

### 4. WebSocket: API key en query string

En **`src/dsagent/server/websocket.py`** la API key se pasa por query (`?api_key=xxx`). Las query strings suelen quedar en logs y en historial del servidor.

**Recomendación:** Documentar el riesgo; a medio plazo valorar un flujo de autenticación que no exponga la clave en la URL (por ejemplo token en el primer mensaje del WebSocket).

### 5. Ejecución de código sin aislamiento

En **`src/dsagent/core/executor.py`** el código generado por el agente se ejecuta en el kernel con el mismo usuario y sin sandbox. Código malicioso o erróneo podría acceder al sistema de archivos, red, etc.

**Recomendación:** Dejar documentado como riesgo conocido; para entornos multi-usuario o no confiables, considerar contenedores efímeros por sesión o límites de recursos (CPU/memoria) a nivel de despliegue.

### 6. Content-Disposition y nombres de archivo

En **`src/dsagent/server/routes/sessions.py`** (export JSON) se usa `session_id` directo en `Content-Disposition: filename="..."`. Si `session_id` contuviera comillas o saltos de línea podría facilitar header injection.

**Recomendación:** Sanitizar o escapar cualquier valor que se inserte en cabeceras (p. ej. quitar o codificar caracteres no seguros en `filename`).

---

## Consistencia y mantenibilidad

### 7. Dos fuentes de configuración (config vs deps)

- **`src/dsagent/config.py`** define `DSAgentSettings` y `get_settings()` (cacheado).
- **`src/dsagent/server/deps.py`** define `ServerSettings` y otro `get_settings()` (también cacheado).

El servidor usa `deps.get_settings()` en el lifespan; la resolución del modelo usa `config.get_default_model()` que internamente usa `config.get_settings()`. Así hay dos "fuentes de verdad" para cosas como `default_model`, `api_key`, `cors_origins`, etc.

**Recomendación:** Unificar en una sola clase de settings (por ejemplo la de `config.py`) y que el servidor dependa de ella, o documentar explícitamente qué lee cada parte y evitar duplicar campos críticos.

### 8. Versión de API desincronizada

En **`src/dsagent/server/app.py`** línea 31: `API_VERSION = "0.6.2"` mientras que en **`pyproject.toml`** y **`src/dsagent/__init__.py`** la versión del paquete es `0.8.4`.

**Recomendación:** Sincronizar `API_VERSION` con la versión del paquete o definir una política clara (p. ej. "API version = major.minor del paquete").

### 9. Dockerfile: versión en LABEL

En **`Dockerfile`** el `LABEL version="0.7.0"` no coincide con la versión actual del proyecto (0.8.4).

**Recomendación:** Usar la misma versión que en `pyproject.toml` o generar el label en el build a partir del mismo origen.

---

## Calidad y buenas prácticas

### 10. Validación de `session_id` en la API

Los endpoints reciben `session_id` como path parameter sin validar formato. Aunque el backend SQLite usa parámetros preparados (sin SQL injection), un `session_id` con `..` o `/` puede ser problemático en el backend JSON y en rutas que construyen paths (archivos, notebooks).

**Recomendación:** Añadir un validador (Pydantic o función) que rechace `session_id` con caracteres no permitidos en todos los routers que lo usan.

### 11. Límites de tamaño en uploads

En **`src/dsagent/server/routes/files.py`** no hay límite explícito en el tamaño del cuerpo o por archivo. Un cliente podría subir archivos muy grandes y afectar disco o memoria.

**Recomendación:** Configurar límites en FastAPI (tamaño máximo de request) o por archivo y documentarlos.

### 12. Timeout y recursos del kernel

En **`src/dsagent/core/executor.py`** el timeout por ejecución es configurable (`code_timeout`), pero no hay límite global de tiempo de vida del kernel ni de memoria. Sesiones largas o código pesado pueden acumular uso.

**Recomendación:** Documentar; en despliegues compartidos considerar políticas de límite de tiempo o reciclado de kernels.

### 13. Health/ready sin auth

**`src/dsagent/server/routes/health.py`** no usa `verify_api_key`, lo cual es razonable para comprobaciones de carga y readiness, pero el endpoint expone versión y estado de componentes.

**Recomendación:** Dejar sin auth por defecto; si en algún entorno se quiere ocultar versión, valorar un health "mínimo" sin detalles o con auth opcional.

---

## Resumen de prioridades

| Prioridad | Tema |
|-----------|------|
| Alta | Path traversal en files y artifacts (descarga/borrado) |
| Alta | Validación de `session_id` y path seguro en JSON store |
| Media | Unificar configuración (config vs deps) |
| Media | Sincronizar versiones (API, Dockerfile, paquete) |
| Media | Límites de upload y documentación de API key/CORS |
| Baja | Content-Disposition seguro, WebSocket auth sin query, documentar riesgos de ejecución de código |
