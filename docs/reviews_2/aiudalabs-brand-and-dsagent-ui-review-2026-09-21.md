# Identidad de AiudaLabs y dirección visual para DSAgent

2026-09-21 · Revisión y propuesta, sin cambios en la UI ni en el sitio corporativo.

## Veredicto

**Sí tomaría la identidad de AiudaLabs como base, después de depurar su aplicación.** Tiene un nombre con una asociación humana útil, una paleta cálida y un contraste tipográfico reconocible. Su debilidad es la combinación de una propuesta comercial demasiado amplia con recursos visuales repetidos y poca evidencia visible del trabajo que anuncia.

No hace falta descartar la marca ni producir otra aplicación desde cero. Hace falta distinguir tres niveles: AiudaLabs como empresa, DSAgent como producto y el cartridge como experiencia de dominio. Comparten familia visual, pero cumplen funciones distintas.

La similitud señalada con BioSyS es razonable. Ambos diseños usan verdes profundos, blancos cálidos, bordes suaves, pequeños rótulos en mayúsculas, tarjetas y lenguaje de descubrimiento/evidencia. Esas decisiones juntas hacen que se perciba una misma firma visual. No permiten deducir de manera fiable qué herramienta produjo el diseño; el problema accionable es la diferenciación insuficiente.

## Fuentes y alcance

- Sitio publicado: [AiudaLabs, español](https://www.aiudalabs.com/es), inspeccionado en navegador: portada, jerarquía completa de encabezados, sección oscura y sección de proyectos.
- Repositorio local: `/Users/nmlemus/projects/aiudalabs.com/website`, HEAD `ed58aa1`. Leídos `AGENTS.md`, `CLAUDE.md`, `src/app/globals.css`, logo, hero, botones, footer, composición de la home y documentos de identidad. Los documentos de ese repositorio permanecen intactos.
- DSAgent: HEAD `516728c`, equivalente al backend/UI de `ae4afc4` más el documento de Jev. Se revisaron `ui/src/workspace.css`, `Sidebar.tsx`, `WorkspaceViews.tsx` y el contrato de experiencia.
- BioSyS: comparación del HTML/CSS local de `biosys-cloud/src/biosys_cloud/web/`. También se localizó `biosys2`; el diseño verde comparable es el de biosys-cloud. No se ejecutaron sus procesos ni se alteraron sus archivos.
- La pestaña preexistente de DSAgent en `:8000` conserva una UI antigua y muestra conexión pendiente; no había listener en ese puerto al inspeccionar. Sirve como referencia visual anterior, no como verificación del último build. Las conclusiones sobre la UI nueva se apoyan en el código commiteado.

Es una crítica de diseño y producto, no una prueba de conversión, reconocimiento de marca ni estudio de usuarios. Los juicios de diferenciación son profesionales y cualitativos.

## 1. Qué conservaría de AiudaLabs

| Elemento | Lo que aporta | Cómo lo trasladaría al producto |
|---|---|---|
| Nombre AiudaLabs | Asociaciones de ayuda e inteligencia; cercano a una audiencia hispanohablante | Firma estable de la empresa y vínculo al sitio corporativo |
| Naranja `#E8440A` | Un punto de atención claro y coherente entre piezas | Marca y acentos limitados; tonos ajustados para texto/controles |
| Crema `#FAF8F4` y tinta `#0D0D0F` | Calidez y legibilidad sin depender de un fondo oscuro | Fondo de trabajo discreto y jerarquía por contraste |
| Satoshi | Una voz tipográfica compartida entre empresa y producto | Títulos, navegación y texto de interfaz |
| Instrument Serif | Contraste editorial útil en mensajes de marca | Uso ocasional en portada pública; evitarlo en tablas, formularios y estados |
| JetBrains Mono | Precisión para código y datos técnicos | IDs, tiempos y fragmentos de código; no todos los botones y rótulos |
| Mensaje de método y trabajo verificable | Encaja con workflows y revisión humana | Mostrar proceso, autoría, estados y entregables reales |

La identidad ya contiene suficiente material para construir familia de productos. Añadir más colores, tipografías o efectos no es la carencia principal.

## 2. Qué critico y por qué

### A. El logo comunica desarrollo, pero todavía tiene poca singularidad

El nombre es más propio que los brackets. La combinación `<.../>`, «ai» resaltado y «labs» pequeño comunica tecnología con rapidez; también puede confundirse visualmente con marcas de consultoras y herramientas para programadores. Esta es una apreciación visual, no un estudio de marcas registradas.

Mantendría el nombre y la firma actual durante esta fase. Refinaría espaciado y proporciones, y probaría una marca compacta legible a 16/24 px. No asumiría que el texto completo del favicon escala bien porque el SVG admita ese tamaño. La solución compacta debe probarse en contexto antes de reemplazarla.

### B. El sistema visual está más desarrollado que la diferenciación comercial

La home ofrece equipos, implementación y capacitación; sirve para una empresa de servicios. Como presentación de una startup de producto deja abiertas demasiadas preguntas: quién compra primero, qué obtiene exactamente y qué puede experimentar ahora.

La recomendación no es convertir toda AiudaLabs en una landing de DSAgent. La empresa puede conservar sus líneas de servicio, mientras cada producto tiene una página con problema, flujo, evidencia y acceso propios. Presentar un producto interno no equivale a demostrar demanda o tracción comercial.

### C. La portada repite su decisión en vez de avanzar la historia

El panel del hero y la sección de caminos usan el mismo encabezado «Empieza por el punto de mayor impacto» y presentan las mismas tres vías. Después reaparecen talento, agentes y playbooks como otra capa de explicación.

Un panel inicial de elección puede ser útil. La sección siguiente debería aportar algo nuevo: un trabajo realizado, un resultado verificable, el alcance de una oferta o el proceso de colaboración. La repetición hace que un sitio cuidado se lea como una sucesión de bloques intercambiables.

### D. La sección de evidencia sigue siendo principalmente afirmaciones

`src/components/sections/proof-build.tsx:38` muestra tres tarjetas de texto. Los proyectos se describen y se explica qué demuestran, pero esas tarjetas no enlazan una demo/caso ni muestran capturas, límites o mediciones reproducibles.

Es una oportunidad importante porque ya existen productos. Propondría mostrar uno con más profundidad: problema, interfaz real, qué funciona hoy y enlace al caso. Incluir métricas solo cuando existan mediciones y explicitar si se trata de un prototipo interno, piloto o producto usado por clientes.

### E. Hay demasiados recursos decorativos con la misma función

El sitio combina grano global, varios brillos radiales, grandes paneles oscuros, tarjetas anidadas, badges y animaciones de entrada similares. Cada recurso puede funcionar por separado, pero juntos compiten con la información.

En una aplicación de uso frecuente, el contenido del usuario debe dominar. Reduciría sombras, reservaría los paneles oscuros para un motivo concreto y eliminaría la decoración permanente de la superficie de trabajo. No trasladaría la regla de animar todas las secciones a DSAgent. También falta una estrategia explícita de movimiento reducido en el código inspeccionado; debe verificarse durante implementación. [Referencia W3C sobre movimiento](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html).

### F. La paleta necesita roles accesibles, no sustituciones literales

Calculé contraste a partir de los colores definidos en código, sin redondear para decidir cumplimiento:

| Combinación de tokens | Contraste aproximado | Uso a revisar |
|---|---|---|
| Blanco sobre naranja `#E8440A` | 3.99:1 | Botones actuales de texto pequeño |
| Naranja `#E8440A` sobre crema `#FAF8F4` | 3.76:1 | Etiquetas pequeñas |
| Gris `#8A8A92` sobre crema `#FAF8F4` | 3.23:1 | Texto auxiliar y evidencia |
| Blanco sobre naranja propuesto `#C4380A` | 5.36:1 | Variante candidata para controles |

WCAG AA requiere 4.5:1 para texto normal y 3:1 para texto grande. Estos cálculos de pares de tokens no constituyen una auditoría de todas las composiciones con transparencias/gradientes. El logo tiene una excepción normativa; no la extendería a botones ni etiquetas. [WCAG 1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

### G. La voz mezcla audiencias y necesita edición

En la navegación española aparecen «Proof» e «Inversión»; el cuerpo acumula términos como delivery, pods y enablement. Puede funcionar con compradores técnicos, pero complica la comprensión de otros compradores de operaciones. Preferiría etiquetas concretas: «Casos y productos», «Servicios» y «Precios» según el destino.

La afirmación «AI-first» es una categoría amplia. Para DSAgent propondría describir una conducta observable, por ejemplo: «Coordina especialistas, revisa cada etapa y conserva el resultado». Es una propuesta de copy, no una promesa de seguridad o recuperación que el backend todavía deba demostrar.

También hay una inconsistencia pequeña pero visible: el footer enlaza Servicios a `#services`, aunque la home monta la sección `#caminos`; y Productos apunta a DTM mientras la navegación superior tiene `/productos`. Corregir estos recorridos aporta más confianza que añadir otro efecto.

## 3. Cómo se parece DSAgent a BioSyS

| Dimensión | BioSyS Cloud | DSAgent actual | Cambio recomendado |
|---|---|---|---|
| Familia cromática | Tinta verde `#163e37`, papel `#f5f7f3`, acento lima | Verde `#21634d`, fondo `#f6f7f5`, hero `#123e32` | Tinta neutra, crema y naranja de AiudaLabs |
| Composición | Workspace, resumen numérico, tarjetas de trabajo y resultados | Workspace, hero, indicadores y tarjetas de actividad | Dar prioridad a cola de trabajo/revisión y al entregable abierto |
| Microtipografía | Eyebrows pequeños, tracking, badges | Eyebrows pequeños, tracking, badges | Menos etiquetas decorativas; jerarquía por peso, tamaño y espacio |
| Decoración | Órbitas en estado vacío, marca con punto, tarjetas suaves | Pipeline/órbitas, marca con punto, tarjetas suaves | Identidad propia y visuales derivados del trabajo real |
| Voz | Explorar, evidencia, próximo insight | Perspectiva, decisión, evidencia | Acciones y objetos concretos del proyecto/cartridge |

Los dos productos necesitan navegación y estados, por lo que no eliminaría patrones comunes que funcionan. Cambiaría las decisiones expresivas y qué información recibe el mayor espacio. Solo recolorear las mismas superficies mantendría buena parte del parecido.

## 4. Dirección recomendada para DSAgent

**Una herramienta de trabajo de AiudaLabs, con jerarquía editorial y precisión operativa.** Mantener navegación, búsqueda, diálogos, adaptación a móviles y la separación de dominios ya conseguida. Cambiar la firma visual y la portada del espacio de trabajo.

### Marca estable, dominio configurable

```text
AiudaLabs                  Empresa que desarrolla el producto
  DSAgent                  Producto y navegación reconocibles
    Marketing Studio       Experiencia del cartridge activo
    Data Science           Otra experiencia del mismo producto
```

Hoy `Sidebar.tsx:95` usa `experience.title` como nombre de marca principal. Recomiendo separar la marca de producto de ese título: al cambiar de cartridge el usuario debe seguir reconociendo la aplicación. El cartridge puede modificar vocabulario, ejemplos, recursos y workflows; no necesita reemplazar la identidad global. No propongo renombrar DSAgent en esta revisión; si su nombre limita el posicionamiento multidominio, sería una decisión de producto separada.

### Pantalla principal

En un proyecto vacío, ofrecer un inicio compacto con objetivo, recursos y workflows apropiados. En un proyecto con actividad, sustituir el gran hero promocional por lo que requiere atención:

1. Proyecto y experiencia activos, con una acción principal clara.
2. Revisiones pendientes con nombre del trabajo, etapa y entregable.
3. Trabajos recientes, con responsable/estado y continuidad.
4. Entregables recientes con vistas previas reales cuando el formato lo permita.
5. Recursos y ayuda contextual en un nivel secundario.

Una misma estructura sirve para DS y marketing. La identidad cambia la presentación; la experiencia sigue viniendo del cartridge. Cualquier número o vista previa de una demo debe provenir de datos reales o estar identificado como ejemplo.

### Lenguaje visual

- Crema muy discreto en el fondo y blanco en el área de edición/lectura; tinta neutra para estructura.
- Naranja reservado a marca, selección y acción principal. Verde/ámbar/rojo permanecen como estados, con texto e icono además de color.
- Satoshi como base; mono para datos técnicos; serif ocasional fuera del flujo operativo.
- Radios de 6–10 px como punto inicial para controles/superficies, con consistencia; sombras principalmente para elementos superpuestos.
- Menos cajas anidadas y más listas, divisores, alineación y espacio. Evitar que cada dato se convierta en una tarjeta KPI.
- Movimiento breve para explicar apertura, cambio de estado y progreso; respetar preferencia de movimiento reducido.
- Marca de AiudaLabs discreta y constante. La diferencia frente a BioSyS debe ser visible también en escala de grises y en la composición.

Son decisiones de diseño propuestas; no nuevos requisitos técnicos ni un tema ya implementado.

## 5. Secuencia de trabajo propuesta

Los siguientes IDs son propuestas y no alteran el backlog reconciliado.

| ID | Tarea | Criterio para revisarla |
|---|---|---|
| ID-01 | Separar empresa, producto y experiencia en la UI | DS y marketing conservan la misma marca de producto al cambiar de proyecto |
| ID-02 | Preparar tokens y tipografía de la familia AiudaLabs | Contraste medido en claro/oscuro; roles de marca y estado diferenciados |
| ID-03 | Diseñar inicio vacío y activo | Proyecto con actividad muestra trabajo/revisión antes que un mensaje promocional |
| ID-04 | Rediseñar una revisión y un entregable | Identidad propia en la tarea central; lectura, errores y acciones siguen claros |
| ID-05 | Aplicar el sistema al resto de pantallas | Sin perder búsqueda, responsive, teclado ni configuración por cartridge |
| ID-06 | Editar narrativa y evidencia del sitio corporativo | Una ruta comercial clara, duplicación eliminada y un caso real verificable |
| ID-07 | Revisar firma compacta y consistencia editorial | Marca legible en tamaños pequeños, nombres coherentes y enlaces correctos |
| ID-08 | Validar con personas externas | Recuerdan empresa/producto, entienden el trabajo que pueden realizar y completan una tarea; registrar resultados, no impresiones del autor |

El primer cambio de UI debería cubrir **inicio + revisión + entregable**. Es una muestra suficiente para valorar identidad sin rehacer de golpe todas las pantallas. La reescritura del sitio corporativo es un alcance separado y todavía no se ha ejecutado.

## Decisión recomendada

Conservar AiudaLabs como familia visual, reducir sus recursos decorativos y reforzar evidencia/copy. Para DSAgent, construir una aplicación orientada al trabajo real usando esa familia, manteniendo las funcionalidades nuevas. La diferenciación vendrá de una composición propia, una marca estable y contenido específico; el naranja por sí solo no la conseguirá.
