# Gráficos visibles en Asistente y Análisis

2026-09-22 · Corrección sobre la base `1f1f0d9`.

## Fallos reproducidos

1. **Vega dibujaba un canvas de ancho cero.** En Análisis → Gráficos y tablas, los eventos y los CSV estaban disponibles, pero los cuatro gráficos del análisis inspeccionado producían `<canvas width="0">`. `vega-embed` aplica `display: inline-block` al host; sin un ancho externo, el gráfico con `width: "container"` medía un contenedor que se contraía al tamaño inicial del canvas. Las tablas sí se mostraban.
2. **Las imágenes Markdown usaban la raíz de la aplicación.** En Asistente → `mmm-report.md`, cinco figuras con `src="figs/...png"` tenían `naturalWidth=0`. El informe estaba en `report/`, pero el navegador buscaba las figuras respecto a la URL del workspace, sin el prefijo de archivos del proyecto.

## Cambio

- `ChartView` establece el ancho del host antes de llamar a Vega. Para specs adaptables, toma el ancho disponible y conserva el autosize declarado, o aplica fit horizontal. Un `ResizeObserver` ajusta la vista cuando cambia el espacio disponible; se desconecta y finaliza la vista al desmontar. Los gráficos con tamaños explícitos mantienen su especificación y pueden desplazarse horizontalmente.
- `markdownHtml` conserva DOMPurify y resuelve las imágenes locales según el `FileSource`: junto al documento para informes, desde la raíz del proyecto para mensajes del Asistente. Se aplica tanto a archivos de runs como a archivos del proyecto. Imágenes externas, inline y URLs ya asociadas a un recurso no se reescriben.
- Cinco tests de regresión cubren proyectos distintos, rutas relativas y absolutas del workspace, referencias desde un informe de run, nombres codificados y URLs externas. Usan el compilador TypeScript existente y el runner de Node, sin añadir dependencias.

La documentación oficial de [tamaño adaptable de Vega-Lite](https://vega.github.io/vega-lite/docs/size.html) exige que el contenedor tenga dimensiones determinadas fuera del gráfico. El cambio respeta ese contrato y no modifica los datos ni las especificaciones guardadas por los agentes.

## Verificación

| Comprobación | Resultado |
|---|---|
| Análisis existente, eventos → CSV → Vega | Cuatro gráficos antes vacíos dibujan datos; dos tablas siguen visibles |
| Apertura en escritorio | Canvas con ancho visible de 1044 px en viewport de 1280 px |
| Cambio de escritorio a móvil | Los cuatro canvas pasan a 298 px dentro de un viewport de 390 px |
| Informe Markdown del Asistente | Cinco imágenes cargadas desde `/projects/default/files/report/figs/`, todas con dimensiones naturales positivas; figura de ROI inspeccionada visualmente |
| `npm test` en `ui/` | 5 tests pasan |
| `npm run build` en `ui/` | TypeScript y Vite pasan; continúa el aviso existente de chunks grandes |
| `git diff --check` | Sin errores |

Se usaron trabajos y archivos existentes; no se ejecutó ningún modelo ni se regeneraron resultados. El render del Asistente se verificó en el informe abierto desde la conversación; la resolución de imágenes directamente en mensajes comparte el helper y está cubierta por tests de rutas. No se afirma haber probado todas las gramáticas Vega ni todos los formatos de imagen.

La pestaña abierta conservaba el HTML del build anterior. Se comprobó el bundle servido y se verificó con una URL de consulta nueva; si una pestaña antigua sigue mostrando gráficos vacíos, recargar sin caché. El servidor local continúa en `:8799`.

Esta corrección aporta evidencia parcial a SR-024/SR-029; no cierra automáticamente sus criterios completos ni cambia el estado del backlog.
