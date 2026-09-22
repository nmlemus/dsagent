# Evaluación de la implementación visual de AiudaLabs

2026-09-21 · HEAD revisado: `ffc3629` · Cambio visual principal: `ed7e6f6`.

## Veredicto

**La dirección está bien aplicada y ya es presentable en una demo supervisada.** El parecido con BioSyS disminuyó de manera sustancial: desapareció el hero verde con órbitas, el naranja tiene un papel limitado y la portada prioriza trabajos y revisiones. La separación entre AiudaLabs, DSAgent y experiencia funciona al cambiar de proyecto.

No recomendaría otro rediseño. Recomendaría una pasada de usabilidad sobre revisión y acceso a entregables, y completar la tipografía. El resultado actual transmite una herramienta sobria y profesional; todavía tiene menos personalidad tipográfica que el sitio corporativo, y la interacción central puede aprovechar mejor la pantalla.

Es un juicio de diseño basado en inspección visual y del código, no una medición de conversión ni una atribución del origen humano/IA del diseño.

## Qué se revisó realmente

- Commits `b8ceabd` (nombres de experiencia), `ed7e6f6` (UI), `ffc3629` (informes), y mensajes/alcance de los tres commits recientes de reparación del backend.
- UI servida en `http://127.0.0.1:8799/`: proyectos existentes Mi primer proyecto, Marketing YoMap y demo. No se crearon proyectos ni se ejecutaron/aprobaron trabajos.
- Escritorio de 1440 × 1000: inicio de DS con revisiones pendientes, modal de revisión y lectura de su informe, inicio de marketing, cambio de proyecto, temas claro y oscuro.
- Móvil de 390 × 844: inicio de marketing y formulario Preparar campaña. También se observó la adaptación intermedia alrededor de 736 px.
- Los assets observados en el DOM, `index-DaX3F7OD.js` e `index-uW0XOK2K.css`, coinciden con el build local actual. No se está evaluando la pestaña antigua de `:8000`.
- `npm run build` pasó, incluyendo TypeScript. Vite advierte de chunks de chat/gráficos mayores de 500 kB; no se midió rendimiento de red ni se atribuye un fallo visual a ese aviso.
- No había un proyecto totalmente vacío entre los que se inspeccionaron: su nueva composición se revisó en código, no se da por verificada visualmente. Tampoco se completó una ejecución real de modelo, auditoría integral de accesibilidad ni prueba exhaustiva de teclado.

## Lo que funciona

1. **Marca estable.** DSAgent permanece en la navegación y Marketing/Data Science identifica el dominio del proyecto. La firma de AiudaLabs es discreta. Se comprobó cambiando entre proyectos.
2. **Jerarquía útil.** En DS, las revisiones aparecen antes que el historial. En marketing, la campaña y sus piezas ocupan el centro de la pantalla. Hay mucho menos contenido promocional dentro del trabajo diario.
3. **Paleta controlada.** Crema, tinta y naranja separan el producto de BioSyS. La variante oscura conserva la familia sin invertir literalmente los colores. El código distingue naranja de marca y naranja para texto/controles.
4. **Lectura y vocabulario.** La separación de informes y archivos de trabajo, el encabezado del documento y su descarga ayudan. El cambio a Campañas, Biblioteca de marca y Piezas mantiene una estructura reconocible.
5. **Responsive razonable.** En las vistas móviles inspeccionadas, las secciones se apilan y el formulario permanece utilizable sin un desbordamiento horizontal visible. No equivale a verificar todos los estados o documentos posibles.

## Ajustes recomendados por prioridad

### UI-R01 · La evidencia queda detrás de la decisión

**Prioridad alta para el flujo central.** Al pulsar Revisar en escritorio, el modal muestra cabecera, métricas, una tarjeta grande de comentario/aprobación y la secuencia de pasos. El documento comienza por debajo de la primera pantalla, incluso a 1000 px de alto. El archivo nombrado dentro de la tarjeta de revisión es texto, no un enlace al documento.

Evidencia: `ui/src/components/RunDock.tsx:186` monta `GateCard` antes del índice y el visor; `GateCard.tsx` representa los archivos como elementos de lista. Se comprobó visualmente con `data-gate.md`.

**Propuesta:** documento como superficie principal; panel compacto de decisión al lado en escritorio. En móvil, primero evidencia y después decisión, con acceso claro entre ambos. Mantener comentario y acciones cerca de lo que se está leyendo sin obligar a subir todo el modal. No hace falta imponer una comprobación artificial de que el usuario leyó todo.

**Aceptación:** desde Revisar se ve el documento pertinente y se entiende qué se aprueba; leer y responder no exige buscar de nuevo los controles. Los nombres de archivos llevan a su vista previa.

### UI-R02 · El inicio promete lectura/descarga pero algunas filas no abren nada

**Prioridad alta para una demo.** En Marketing YoMap, bajo «Lo que ya puedes leer y descargar», la campaña es un botón, pero `strategy.md` y `notas-yomap.md` son filas informativas. Comparten casi el mismo lenguaje visual, aunque no tienen acción. Las filas de materiales tampoco son accesos individuales.

Evidencia: `ui/src/components/WorkspaceViews.tsx:389` renderiza artefactos como `span.plain-row`; los materiales hacen lo mismo en `:424`. El árbol accesible confirma que solo el trabajo terminado es un botón.

**Propuesta:** abrir el archivo del proyecto en el visor correspondiente, con descarga disponible. Si una fila es únicamente informativa, hacerlo explícito y ofrecer junto a la sección una ruta clara a los archivos. En la sección de piezas conviene distinguir el trabajo que las produjo del archivo concreto.

**Aceptación:** cualquier archivo presentado como disponible para leer/descargar se abre con ratón y teclado desde ese lugar, sin una búsqueda adicional por otras pantallas.

### UI-R03 · «Tiempo registrado» mezcla ejecución y espera

**Prioridad media, importante para credibilidad.** La revisión inspeccionada mostró `102h 23m` junto a un costo de `$0.166`. El valor incluye el tiempo transcurrido desde el inicio del run, también mientras espera aprobación; no representa 102 horas de cómputo.

Evidencia: `ui/src/components/RunDock.tsx:162` presenta `detail.duration`; `src/dsagent/runs.py:178` calcula fin/inicio o ahora/inicio. La cifra no está inventada, pero el rótulo permite una interpretación equivocada.

**Propuesta:** llamarlo «Tiempo transcurrido» y mostrar «Esperando revisión desde…». Añadir tiempo de ejecución solo cuando exista una medición que excluya pausas; no estimarlo a partir de la apariencia del estado.

**Aceptación:** un trabajo que espera varios días no parece haber consumido esos días ejecutando el modelo.

### UI-R04 · La familia tipográfica aún no está instalada

**Prioridad media para identidad.** El CSS nombra Satoshi, Instrument Serif y JetBrains Mono, pero no incorpora sus archivos ni reglas `@font-face`. Por tanto, el parecido tipográfico con AiudaLabs depende de fuentes instaladas en el equipo y de los fallbacks del sistema. El propio commit declara esta limitación.

Evidencia: `ui/src/workspace.css:15` y stacks desde `:47`; no se encontraron declaraciones de fuentes cargadas en la UI. Un nombre en `font-family` no garantiza que el navegador disponga de esa fuente. No se determinó la fuente efectiva de cada glifo mediante un inspector de fuentes.

**Propuesta:** servir localmente las fuentes de marca tras comprobar sus condiciones de uso, con pesos necesarios y fallback cuidado. Mantener la serif limitada al inicio/editorial.

**Aceptación:** la familia se descarga y renderiza en un navegador sin esas fuentes instaladas; se conserva legibilidad durante la carga. Esta es una mejora de identidad, no un bloqueante funcional.

### UI-R05 · Afinar presentación de una demo y completar las otras pantallas

**Prioridad media/baja.** Los prompts largos se usan como título, los gates existentes muestran instrucciones en inglés y las listas contienen varias ejecuciones similares. Eso es contenido existente, no evidencia de que el nuevo diseño genere duplicados. Aun así, dificulta explicar el producto a interesados.

**Propuesta:** título corto editable del trabajo, pregunta completa como detalle y texto de revisión localizado por cartridge. Preparar un proyecto de demostración con trabajos reales y una historia comprensible; conservar el historial original. Completar ID-05: launcher, chat, biblioteca y listados deben adoptar la misma jerarquía, reduciendo rótulos decorativos cuando repiten el título.

**Aceptación:** en un recorrido breve una persona externa identifica empresa, producto, proyecto, qué se produjo y qué decisión debe tomar. No usar métricas ficticias ni presentar un objetivo como resultado logrado: «500 instalaciones de la app» es el nombre/objetivo de una campaña, no prueba de 500 instalaciones obtenidas.

## Cómo leer ahora el informe anterior del backend

El [informe independiente](backend-harness-review-2026-09-21.md) documenta el código anterior a las reparaciones. No debe usarse para afirmar que todos sus repros siguen abiertos en HEAD.

| Commit | Correcciones introducidas según el cambio | Límite que conserva |
|---|---|---|
| `65e33e4` | Contención de scripts y ZIP; rechazo de IDs de pasos/workflows duplicados | No equivale a aislamiento del proceso ni a resolver todas las rutas de contenido activo |
| `8897804` | IDs con sufijo aleatorio y reserva; resolver compartido por API y herramientas | Resolver por ID no incorpora autorización por usuario/proyecto |
| `463b1e3` | Error visible de guardado, rechazo de guardados antiguos/rebinding y escritura atómica de sesión | Transcripción aún enviada por el cliente; tratamiento del 409 en UI pendiente según el commit |

Esta revisión visual no cierra tareas amplias del backlog ni revalida todo el backend. El trabajo técnico nuevo responde a hallazgos concretos y debe reconocerse por separado del estado de preparación comercial.

Comprobación adicional sobre HEAD: `.venv/bin/python -m pytest -q tests/test_loader.py tests/test_run_identity.py tests/test_runs_api.py tests/test_sessions.py tests/test_skill_scripts.py` terminó con **78 passed**, dos warnings. Se ejecutaron los cinco módulos tocados por esas reparaciones; no se repitió la suite completa ni se realizaron llamadas a modelos.

## Siguiente incremento

Implementar UI-R01 y UI-R02 primero; después UI-R03 y fuentes. Mantener el sistema visual y las funcionalidades existentes. El mayor retorno ahora está en hacer directa la secuencia **abrir → leer → decidir**, no en añadir decoración ni cambiar otra vez la paleta.
