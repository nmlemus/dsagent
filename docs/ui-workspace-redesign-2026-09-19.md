# UI-001 — Workspace empresarial para demostraciones

2026-09-19 · Implementado y verificado localmente; cambios sin commit/PR todavía.

El usuario priorizó expresamente este rediseño antes de SR-001. El objetivo es que una persona interesada pueda entender la propuesta, preparar un análisis y explorar evidencia desde una interfaz presentable. Este trabajo modifica el frontend existente y mantiene FastAPI, los contratos de datos y el runtime de Deep Agents.

## Resultado

- Nueva portada del proyecto, identidad en verde profundo, superficies claras, iconos SVG propios, tipografía de sistema y temas claro/oscuro/sistema. Sin fuentes ni imágenes remotas, ni una nueva biblioteca de componentes.
- Navegación entre vista general, asistente, análisis, biblioteca de datos y entregables. URLs hash conservan la sección, historial del navegador y búsqueda global con Cmd/Ctrl+K.
- Métricas derivadas de respuestas reales del servidor, con alcance explícito de los últimos 100 análisis. Fallos de conexión muestran aviso y reintento; no se sustituyen por métricas de ejemplo.
- Biblioteca con carga por selector/drag-and-drop, búsqueda, confirmación de borrado y acción Analizar que preselecciona el archivo.
- Creación de análisis en dos pasos: configuración y revisión del recorrido antes de iniciar. Tipos numéricos/booleanos se serializan según el contrato. Se reutiliza el ID creado si falla el inicio y se reintenta en el mismo diálogo.
- Espacio amplio de resultados: pasos, evidencia del gate, entregables, archivos, cinco gráficos y dos tablas de la grabación. Se consumen eventos `dsagent.chart` y se ignoran versiones anteriores de un mismo gráfico.
- Asistente con mensajes orientados al usuario y sugerencias editables sin autoenvío. Historial existente no se sustituye por vacío al fallar una lectura; solo un borrador nuevo con 404 se registra y se espera su confirmación antes de habilitar el chat.
- Diálogos nativos, cierre por Escape, retorno/contención del foco, nombres accesibles, navegación móvil con fondo inerte y reducción de movimiento. Se preserva el borrado de conversaciones con confirmación.
- Chat, vista de run y runtime de gráficos se cargan bajo demanda. El JavaScript inicial pasó de aproximadamente 923 KB a 196 KB minificados (de 258 KB a 62 KB gzip). Los chunks diferidos de chat/Vega siguen siendo grandes.

## Archivos y organización

| Responsabilidad | Archivos |
|---|---|
| Composición, navegación y búsqueda | `ui/src/App.tsx` |
| Portada, listas y bibliotecas | `ui/src/components/WorkspaceViews.tsx` |
| Navegación y proyectos | `Sidebar.tsx`, `ProjectSwitcher.tsx` |
| Tokens y diseño responsive | `ui/src/workspace.css` |
| Diálogos e iconografía | `Dialog.tsx`, `Icon.tsx` |
| Configuración y confirmación de análisis | `Launcher.tsx`, `LauncherModal.tsx` |
| Resultados, gráficos y revisión | `RunDock.tsx`, `FocusOverlay.tsx`, `GateCard.tsx`, `useRunStream.ts` |
| Estado y errores de cliente | `useWorkspace.ts`, `api.ts`, `conversations.ts`, `ChatCenter.tsx` |
| Tema, metadata y proxy | `Settings.tsx`, `ui/index.html`, `ui/public/favicon.svg`, `ui/vite.config.ts` |

`styles.css` conserva estilos de renderizadores existentes; `workspace.css` proporciona el sistema visual y adapta esos componentes. Los textos propios de la UI son españoles; preguntas, prompts de gates y entregables históricos conservan su idioma original. Desde el 2026-09-20 el vocabulario del dominio (nombres de secciones, sustantivos, sugerencias, títulos de workflows, pasos y campos) viene del cartridge a través de `experience:` en `cartridge.yaml` y de los campos de presentación de `workflow.yaml`; ver `docs/architecture.md` §3.5 y `cartridges/marketing` como prueba de otro dominio.

## Verificación ejecutada

Historia verificada: UI → API FastAPI → almacenamiento temporal → SSE → revisión humana → resultados visibles.

| Comprobación | Resultado |
|---|---|
| TypeScript + build de producción | `npm run build --prefix ui` correcto. Aviso esperado por chunks diferidos mayores de 500 KB. |
| Portada conectada | Proyectos, análisis, datasets y artefactos desde API; sin fixtures incrustadas en el frontend. |
| Proyecto nuevo | Creado desde el diálogo con nombre y objetivo; seleccionado correctamente. |
| Carga de CSV | Selector de archivos → PUT → biblioteca: `seattle-weather.csv`, 47.1 KB, en el proyecto de verificación. |
| Buscar datos y analizar | Filtrar por `seattle`; Analizar abre configuración con el archivo seleccionado. |
| Lanzamiento y gate | En proyecto default: revisar → iniciar → `awaiting_gate`; se muestra `data-gate.md` antes de aprobar; comentario + aprobar → `done`, 4/4 pasos. Run de prueba: `eda-to-report-20260919-113009`. |
| Visualizaciones | Grabación `run-eda-charts`: siete eventos visibles, dos tablas con filas y cinco visualizaciones Vega. |
| Entregables | Run completado visible con título de la pregunta; acceso a resultados y enlace de descarga. |
| Asistente | Apertura/registro; sugerencia rellena compositor sin enviar. No se ejecutó una conversación con un modelo. |
| Búsqueda global | Estado sin coincidencias visible; Escape cierra. |
| Teclado móvil | Apertura del menú; Shift+Tab desde primer control vuelve al último; Escape cierra. Navegación cerrada fuera del árbol visible. |
| Responsive | Inspección visual a 390 × 844 y 1440 × 1000; a 768 × 1024 se comprobó ausencia de overflow horizontal. |
| Apariencia | Claro y oscuro inspeccionados; selección de tema aplicada inmediatamente. |
| Consola | Sin errores de JavaScript en los flujos verificados; Vega emitió avisos de sort/scale de las specs históricas. |
| Higiene | Prettier aplicado a los componentes del cambio; `git diff --check` correcto. |

El servidor de verificación usa `build_app` y el replay existente, sin modelos ni Docker. Su modo replay **no admite crear runs en proyectos no-default**: `Replay.create` no acepta el argumento `project` que envía la API. Esa prueba devolvió 400 y mostró el error; se verificó el recorrido de ejecución en default. No se modificó el backend para encubrir esta limitación. La carga/creación del proyecto nuevo se verificó por separado.

No se han certificado todas las rutas de accesibilidad, concurrencia, guardado/versionado o ejecución live. La confirmación es de configuración de UI: todavía no consume el `/plan` persistido ni resuelve por completo SR-023. El reducer mantiene la versión más reciente de cada gráfico; un historial navegable de versiones sigue pendiente en SR-024. Las rutas de archivos y límites de seguridad del backend siguen bajo sus tareas originales.

## Guía breve para presentar

1. Abrir Vista general en el proyecto deseado.
2. Mostrar la biblioteca y elegir Analizar sobre un archivo apropiado.
3. Revisar objetivo, fuente, pasos y revisión humana antes de iniciar.
4. Abrir el análisis, revisar evidencia y avanzar cuando corresponda.
5. Mostrar Gráficos y tablas y la biblioteca de Entregables.
6. Usar Asistente para preguntas abiertas; las sugerencias se editan antes de enviar.

Para una demostración sin consumo de modelos, usar el modo replay existente y explicar que reproduce una ejecución grabada. Para la instancia habitual, recargar la página tras generar `ui/dist`; el servidor sirve el build actualizado. No se inició ningún despliegue externo.

## Relación con el backlog

UI-001 es el encargo visual priorizado por el usuario. Aporta partes de SR-023–030/033/042–044/049 y corrige la configuración proxy de SR-007. Los padres SR permanecen pendientes hasta comprobar todos sus criterios, incluidos los que exceden este rediseño. No se cerraron tareas comerciales o de seguridad por mejorar la presentación.

La gestión de foco de los diálogos sigue el patrón de interacción de [W3C APG](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/). Es una referencia de diseño, no una declaración de conformidad global. Se aplicaron las guías locales `vercel-plugin:react-best-practices` y `vercel-plugin:verification`.
