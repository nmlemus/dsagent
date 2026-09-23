# Viabilidad de un cartridge de SDLC (BMAD, Spec Kit) sin tocar harness ni UI

2026-09-21 · HEAD evaluado: `1f1f0d9` · Rama: `ui-harness` · Pregunta: ¿se puede añadir un método de ciclo de vida de software como cartridge, sin cambiar código en `src/dsagent/` ni en `ui/`?

## Veredicto

**Factible para la mitad de planificación, con configuración y markdown solamente. La mitad de implementación choca con tres límites del harness que no se resuelven desde un cartridge.**

Recomendación: empezar por **Spec Kit**, no por BMAD. Sus fases son lineales y con puntos de aprobación, que es exactamente la forma que el harness ya sabe ejecutar. Cierra además el primer punto de M3 del roadmap sin escribir Python.

Esto es una evaluación de diseño leyendo el código. **No se construyó ningún cartridge ni se ejecutó nada**; los límites señalados están verificados en el código, pero la afirmación de que "un cartridge de SDLC funcionaría" no está demostrada empíricamente hasta que exista uno.

## Lo que ya soporta el harness

### El material coincide con el formato

Un cartridge es un plugin de Claude Code más `cartridge.yaml` (invariante 2). Tanto BMAD como Spec Kit se distribuyen como agentes en markdown, plantillas y scripts — el mismo material del que están hechos los skills. El mapeo no necesita adaptador:

| Del método | Al cartridge |
|---|---|
| Roles (Analyst, PM, Architect, PO, SM, Dev, QA) | `personas:` en `cartridge.yaml`, un `agents/*.md` cada uno |
| Plantillas y tareas | `skills/<nombre>/` con su `SKILL.md` y `scripts/` |
| Secuencia de fases | pasos en `workflows/<nombre>/workflow.yaml` |
| Quién puede usar qué plantilla | matriz `skills: { <skill>: { scope: [...] } }` (invariante 3) |

### Los gates son el encaje exacto

"El PO aprueba el PRD antes de que empiece arquitectura" es un `gate:` declarado en un paso. Es la única cosa que estos métodos piden y que casi ningún andamiaje de agentes ofrece de forma estructural: en BMAD la aprobación es una instrucción en un prompt; aquí el runner para y no continúa. Esto es para lo que se construyó el harness.

### `produces` convierte la disciplina en contrato

Un paso que debe escribir `docs/prd.md` falla si no aparece en disco (invariante 5, verificada por el runner). El rigor documental de estos métodos deja de depender de que el modelo se acuerde.

### El vocabulario completo de la UI es del cartridge

El bloque `experience:` está validado en estricto —`Experience(BaseModel, extra="forbid")`, `models.py:295`, cargado en `loader.py:90`— servido por la API (`api.py:161`) y consumido por la UI (`App.tsx:38,145,296,300`; `LauncherModal` vía `useExperience`). Un cartridge de SDLC cambia título, navegación, sustantivos, copy del home y sugerencias del asistente sin una línea de frontend:

```yaml
experience:
  title: Ingeniería
  nouns:
    run:         { singular: entrega,       plural: entregas }
    resource:    { singular: especificación, plural: especificaciones }
    deliverable: { singular: artefacto,     plural: artefactos }
```

Una clave desconocida es error, no un silencio — así que el riesgo de que un cartridge nuevo caiga en el copy genérico sin que nadie lo note está cubierto.

### El entorno no cuesta nada

`kind: kernel` con `requirements: []` da un shell montado en el workspace vía `LocalShellBackend`. El kernel de Jupyter arranca **en perezoso**, solo si una persona llama `run_python` (`envs/kernel.py:99-100`), así que un cartridge que nunca ejecuta Python no paga por uno. `_verify_requirements` corre al crear el env, de modo que una lista vacía tampoco falla.

### Los entregables markdown se muestran

`FileViewer.tsx` renderiza markdown de verdad (`markdownHtml`, líneas 5 y 120). Un PRD o un `spec.md` se leen en la UI sin trabajo adicional. Los command skills se generan solos (invariante 7), así que `/sdlc-<workflow>` aparece también dentro de Claude Code.

## Los tres límites

### 1. No se puede sembrar un repositorio

`datasets.py:seed_run` copia únicamente ficheros planos:

```python
for f in project_data_dir.iterdir():
    if f.is_file():
        shutil.copy2(f, dest / f.name)
```

Los directorios se saltan, y el destino es `workspace/data/`. No hay camino hoy para "ejecuta esto sobre mi código existente".

**Consecuencia:** *greenfield* funciona — el cartridge produce el spec, el plan, las tareas y el código desde cero. *Brownfield* no. Esto descarta de entrada los flujos de BMAD que empiezan documentando un sistema ya existente.

### 2. El runner no planifica, y un DAG no itera

Invariante 4: los workflows son DAGs declarados y el modelo nunca elige el siguiente paso. El ciclo central de desarrollo de BMAD es un bucle: el SM redacta la historia *n*, Dev la implementa, QA revisa, y se repite por cada historia del backlog. Eso no se expresa como DAG.

La salida practicable es **un run por historia**, iniciados desde el chat, donde el orquestador llama `run_workflow` (`host/build.py:217-223`, disponible al orquestador y a las personas listadas en el workflow). Funciona, pero la iteración vive *fuera* del workflow: en una persona o en el orquestador, no en el contrato declarado.

Dicho sin adornos: **el harness modela un pipeline; BMAD modela un proceso con bucle.** Spec Kit es un pipeline, y por eso encaja.

### 3. No hay git

Nada bajo `src/dsagent/` sabe de ramas, commits ni pull requests. La salida de un run son ficheros en su workspace, recuperables por export o zip. El último metro de cualquier método de SDLC —aterrizar el cambio en la rama— es manual.

Probablemente correcto para un harness que debe seguir siendo agnóstico de dominio, pero conviene decirlo antes de prometer un ciclo completo.

## Recomendación

**Primero Spec Kit.** Cuatro fases lineales (specify → plan → tasks → implement) con aprobación entre ellas: es un DAG con gates casi literal, sus artefactos son documentos —lo que el harness verifica bien— y el cartridge es YAML y markdown. Cierra el segundo punto de M3.

**Después la mitad de planificación de BMAD**, Analyst → PM → Architect → PO, que también es un DAG limpio y aporta personas con más carácter.

**El bucle de desarrollo de BMAD queda fuera** hasta que se decida cambiar el harness. Si se decide: el cambio **no va en el runner** —no se toca la invariante 4— va en el sembrado del workspace y en un `for-each` a nivel de *workflow*. Es una decisión de diseño con su propia evaluación adversarial, no un parche.

## Beneficio lateral: audita la invariante 1

La invariante 1 dice que nada bajo `src/dsagent/` menciona ciencia de datos, personas o skills por nombre. Un cartridge de otro dominio es la única prueba real de eso. **Si construirlo obliga a tocar el harness, la invariante ya estaba rota y no se sabía.** Ese resultado es valioso incluso si el cartridge de SDLC luego se descarta.

El único residuo específico de DS que se vio es `WorkflowInput.guess: unique_column` (`api.py:94`, `api.py:747`), y es opcional: un input que no lo declara no lo usa. No estorba.

## Límites de esta evaluación

- No se construyó ni ejecutó nada. Los límites del harness están leídos en el código; el encaje del método no está probado.
- **La caracterización de BMAD y Spec Kit proviene de conocimiento general, no de leer sus distribuciones actuales.** Antes de comprometer el mapeo de la tabla conviene comprobar su estructura real de ficheros contra sus repositorios, porque ambos proyectos se mueven rápido.
- No se evaluó coste por run de un workflow documental, ni si los prompts de estos métodos rinden con el modelo por defecto del harness.
- No se revisó licencia de BMAD ni de Spec Kit para redistribución dentro de un cartridge, que es una pregunta abierta si el cartridge se publica.
