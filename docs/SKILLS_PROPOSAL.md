# Propuesta: Integración de Skills en DSAgent

## Resumen Ejecutivo

Adoptar el estándar **Agent Skills** para extender las capacidades de DSAgent con paquetes de conocimiento reutilizables, descubribles y portables.

**Principio clave**: Los skills NO vienen incluidos en DSAgent. El usuario los instala según necesite, igual que los MCP servers.

---

## 1. Arquitectura Propuesta

### 1.1 Estructura de Directorios

```
~/.dsagent/
├── .env                    # Configuración
├── mcp.yaml                # MCP servers
├── skills.yaml             # Registro de skills instalados
└── skills/                 # Skills instalados por el usuario
    ├── eda-analysis/
    │   ├── SKILL.md
    │   └── scripts/
    ├── ml-training/
    └── my-company-etl/
```

### 1.2 Nuevo Módulo: `src/dsagent/skills/`

```
src/dsagent/skills/
├── __init__.py
├── loader.py              # SkillLoader - descubre y carga skills
├── registry.py            # SkillRegistry - registro de skills disponibles
├── executor.py            # SkillExecutor - ejecuta scripts de skills
├── installer.py           # SkillInstaller - instala skills desde fuentes
└── validator.py           # Wrapper para skills-ref validation
```

---

## 2. Instalación de Skills

### 2.1 Comando `dsagent skills install`

```bash
# Desde GitHub (caso más común)
dsagent skills install github:anthropics/claude-cookbooks/skills/custom_skills/creating-financial-models

# Forma corta para repos con skill en raíz
dsagent skills install github:dsagent-skills/eda-analysis

# Desde URL directa
dsagent skills install https://github.com/user/repo/tree/main/my-skill

# Desde directorio local
dsagent skills install ./my-local-skill

# Desde registry (futuro)
dsagent skills install @dsagent/eda-analysis
```

### 2.2 Flujo de Instalación

```
┌─────────────────────────────────────────────────────────────┐
│  dsagent skills install github:user/repo/path/to/skill     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 1: Resolver fuente                                    │
│  - Parsear: tipo=github, repo=user/repo, path=path/to/skill │
│  - Determinar URL de descarga                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 2: Descargar                                          │
│  - Clonar/descargar a directorio temporal                   │
│  - Extraer subdirectorio si aplica                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 3: Validar (usando skills-ref)                        │
│  - validate(temp_path) → errores                            │
│  - read_properties(temp_path) → metadata                    │
│  - Verificar no hay conflicto de nombres                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 4: Verificar dependencias                             │
│  - Leer compatibility del SKILL.md                          │
│  - Verificar paquetes Python instalados                     │
│  - Preguntar si instalar faltantes                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 5: Instalar                                           │
│  - Mover a ~/.dsagent/skills/{skill-name}/                  │
│  - Registrar en ~/.dsagent/skills.yaml                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PASO 6: Confirmar                                          │
│  ✓ Skill 'eda-analysis' instalado                           │
│    Descripción: Exploratory data analysis...                │
│    Scripts: basic_eda.py, correlation.py                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Validación con skills-ref

Usar la librería oficial **skills-ref** para validación:

**Repositorio**: https://github.com/agentskills/agentskills/tree/main/skills-ref

```python
# En src/dsagent/skills/validator.py

from skills_ref import validate, read_properties, to_prompt

class SkillValidator:
    """Wrapper para validación de skills usando skills-ref."""

    def validate(self, skill_path: Path) -> List[str]:
        """Valida estructura y formato del skill.

        Returns:
            Lista de errores (vacía si válido)
        """
        return validate(str(skill_path))

    def read_metadata(self, skill_path: Path) -> dict:
        """Extrae metadata del skill.

        Returns:
            {"name": "...", "description": "...", ...}
        """
        return read_properties(str(skill_path))

    def generate_prompt(self, skill_paths: List[Path]) -> str:
        """Genera XML para inyectar en system prompt.

        Returns:
            XML con available_skills
        """
        return to_prompt(*[str(p) for p in skill_paths])
```

**Funciones de skills-ref**:
- `validate(path)` - Verifica estructura y formato SKILL.md
- `read_properties(path)` - Extrae metadata como JSON
- `to_prompt(path1, path2, ...)` - Genera XML para system prompt

### 2.4 Archivo skills.yaml

```yaml
# ~/.dsagent/skills.yaml
skills:
  - name: eda-analysis
    source: github:dsagent-skills/eda-analysis
    version: "1.0.0"
    installed_at: "2026-01-09T14:30:00"

  - name: creating-financial-models
    source: github:anthropics/claude-cookbooks/skills/custom_skills/creating-financial-models
    version: latest
    installed_at: "2026-01-09T15:00:00"

  - name: my-company-etl
    source: local:./my-company-etl
    installed_at: "2026-01-08T10:00:00"
```

---

## 3. Componentes Principales

### 3.1 SkillInstaller

```python
class SkillInstaller:
    """Instala skills desde múltiples fuentes."""

    def __init__(self, skills_dir: Path, validator: SkillValidator):
        self.skills_dir = skills_dir  # ~/.dsagent/skills/
        self.validator = validator

    def install(self, source: str) -> InstallResult:
        """Instala un skill desde una fuente.

        Args:
            source: github:user/repo/path, URL, o path local

        Returns:
            InstallResult con status y metadata
        """
        # 1. Parsear fuente
        source_info = self._parse_source(source)

        # 2. Descargar a temporal
        temp_path = self._download(source_info)

        # 3. Validar
        errors = self.validator.validate(temp_path)
        if errors:
            raise SkillValidationError(errors)

        # 4. Leer metadata
        metadata = self.validator.read_metadata(temp_path)

        # 5. Verificar conflictos
        if self._skill_exists(metadata["name"]):
            raise SkillExistsError(metadata["name"])

        # 6. Verificar dependencias
        missing_deps = self._check_dependencies(metadata)
        if missing_deps:
            self._prompt_install_deps(missing_deps)

        # 7. Mover a destino final
        dest_path = self.skills_dir / metadata["name"]
        shutil.move(temp_path, dest_path)

        # 8. Registrar
        self._register(metadata, source)

        return InstallResult(success=True, metadata=metadata)

    def _parse_source(self, source: str) -> SourceInfo:
        """Parsea fuente: github:, https://, o path local."""
        if source.startswith("github:"):
            return self._parse_github(source[7:])
        elif source.startswith("https://"):
            return self._parse_url(source)
        else:
            return SourceInfo(type="local", path=Path(source))

    def _parse_github(self, ref: str) -> SourceInfo:
        """Parsea referencia GitHub: user/repo o user/repo/path/to/skill."""
        parts = ref.split("/")
        if len(parts) == 2:
            # user/repo - skill en raíz
            return SourceInfo(
                type="github",
                owner=parts[0],
                repo=parts[1],
                path="",
            )
        else:
            # user/repo/path/to/skill
            return SourceInfo(
                type="github",
                owner=parts[0],
                repo=parts[1],
                path="/".join(parts[2:]),
            )
```

### 3.2 SkillLoader

```python
class SkillLoader:
    """Descubre y carga skills instalados."""

    def __init__(self, skills_dir: Path = None):
        self.skills_dir = skills_dir or Path.home() / ".dsagent" / "skills"

    def discover_skills(self) -> List[SkillMetadata]:
        """Escanea directorio y retorna metadata de skills instalados."""
        skills = []
        if not self.skills_dir.exists():
            return skills

        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                metadata = self._load_metadata(skill_dir)
                skills.append(metadata)
        return skills

    def load_skill(self, name: str) -> Skill:
        """Carga un skill completo por nombre."""
        skill_path = self.skills_dir / name
        if not skill_path.exists():
            raise SkillNotFoundError(name)

        # Parsear SKILL.md
        metadata = self._load_metadata(skill_path)
        instructions = self._load_instructions(skill_path)
        scripts = self._list_scripts(skill_path)

        return Skill(
            metadata=metadata,
            instructions=instructions,
            scripts=scripts,
            path=skill_path,
        )
```

### 3.3 SkillRegistry

```python
class SkillRegistry:
    """Registro central de skills disponibles."""

    def __init__(self, loader: SkillLoader):
        self.loader = loader
        self.skills: Dict[str, SkillMetadata] = {}
        self._discover()

    def _discover(self) -> None:
        """Descubre skills al inicio."""
        for metadata in self.loader.discover_skills():
            self.skills[metadata.name] = metadata

    def get_skills_for_prompt(self) -> str:
        """Genera contexto para inyectar en system prompt."""
        if not self.skills:
            return ""

        # Usar skills-ref to_prompt si está disponible
        try:
            from skills_ref import to_prompt
            skill_paths = [self.loader.skills_dir / name for name in self.skills]
            return to_prompt(*[str(p) for p in skill_paths])
        except ImportError:
            # Fallback manual
            lines = ["## Available Skills\n"]
            for name, meta in self.skills.items():
                lines.append(f"- **{name}**: {meta.description}")
            return "\n".join(lines)
```

### 3.4 SkillExecutor

```python
class SkillExecutor:
    """Ejecuta scripts de skills en el kernel."""

    def __init__(self, kernel: JupyterExecutor, registry: SkillRegistry):
        self.kernel = kernel
        self.registry = registry

    async def execute_script(
        self,
        skill_name: str,
        script_name: str,
        parameters: Dict[str, Any],
    ) -> ExecutionResult:
        """Ejecuta un script de skill con parámetros."""
        # 1. Cargar skill
        skill = self.registry.loader.load_skill(skill_name)

        # 2. Encontrar script
        script_path = skill.path / "scripts" / script_name
        if not script_path.exists():
            script_path = skill.path / script_name

        if not script_path.exists():
            raise ScriptNotFoundError(skill_name, script_name)

        # 3. Preparar código
        code = self._prepare_execution(script_path, parameters)

        # 4. Ejecutar en kernel
        result = await self.kernel.execute(code)

        return result

    def _prepare_execution(self, script_path: Path, params: dict) -> str:
        """Prepara código para ejecución con parámetros."""
        script_content = script_path.read_text()

        # Inyectar parámetros como variables
        param_code = "\n".join(f"{k} = {repr(v)}" for k, v in params.items())

        return f"{param_code}\n\n{script_content}"
```

---

## 4. Integración con ConversationalAgent

### 4.1 Modificaciones al System Prompt

```python
def _build_system_prompt(self) -> str:
    prompt = CONVERSATIONAL_SYSTEM_PROMPT

    # Inyectar contexto del kernel
    if self._kernel_snapshot:
        prompt += f"\n\n{self._kernel_snapshot.get_context_summary()}"

    # Inyectar skills disponibles
    if self._skill_registry and self._skill_registry.skills:
        prompt += f"\n\n{self._skill_registry.get_skills_for_prompt()}"

    return prompt
```

### 4.2 Nuevo Tag: `<use_skill>`

El agente puede invocar skills explícitamente:

```xml
<use_skill name="eda-analysis" script="basic_eda.py">
  <param name="dataframe">df</param>
  <param name="target_column">price</param>
</use_skill>
```

El `PlanParser` detecta este tag y delega al `SkillExecutor`.

---

## 5. CLI para Skills

### 5.1 Comandos

```bash
# Listar skills instalados
dsagent skills list

# Ver detalles de un skill
dsagent skills info <name>

# Instalar skill
dsagent skills install <source>
# Ejemplos:
#   dsagent skills install github:dsagent-skills/eda-analysis
#   dsagent skills install github:anthropics/claude-cookbooks/skills/custom_skills/creating-financial-models
#   dsagent skills install ./my-local-skill

# Desinstalar skill
dsagent skills remove <name>

# Actualizar skill
dsagent skills update <name>

# Validar skill (antes de publicar)
dsagent skills validate <path>

# Crear template para nuevo skill
dsagent skills create <name>
```

### 5.2 Comandos Slash en Chat

```
/skills              # Listar skills disponibles
/skill <name>        # Ver detalles del skill
```

---

## 6. Skills Oficiales de DSAgent (Repositorio Separado)

En lugar de built-in, mantener un repositorio separado con skills recomendados:

**Repositorio**: `github:dsagent-skills/`

```
dsagent-skills/
├── eda-analysis/           # Análisis exploratorio
├── ml-training/            # Entrenamiento de modelos
├── data-loading/           # Carga desde múltiples fuentes
├── visualization/          # Gráficos y dashboards
├── feature-engineering/    # Feature engineering
├── model-evaluation/       # Evaluación de modelos
└── reporting/              # Generación de reportes
```

**Instalación por el usuario**:
```bash
dsagent skills install github:dsagent-skills/eda-analysis
dsagent skills install github:dsagent-skills/ml-training
```

---

## 7. Dependencias

### 7.1 Nueva dependencia opcional

```toml
# pyproject.toml
[project.optional-dependencies]
skills = [
    "skills-ref>=0.1.0",    # Validación de skills
    "gitpython>=3.1.0",     # Clonado de repos
]
```

### 7.2 Instalación

```bash
pip install "datascience-agent[skills]"
```

---

## 8. Plan de Implementación

### Fase 7.1: Core Infrastructure
- [ ] Crear `src/dsagent/skills/`
- [ ] Implementar `SkillValidator` (wrapper de skills-ref)
- [ ] Implementar `SkillInstaller` con soporte GitHub/local
- [ ] Implementar `SkillLoader` y `SkillRegistry`
- [ ] Parseo de SKILL.md (usando skills-ref)
- [ ] Tests unitarios

### Fase 7.2: CLI
- [ ] Comando `dsagent skills install`
- [ ] Comando `dsagent skills list`
- [ ] Comando `dsagent skills remove`
- [ ] Comando `dsagent skills info`
- [ ] Manejo de skills.yaml

### Fase 7.3: Integración con Agente
- [ ] Modificar system prompt para incluir skills
- [ ] Implementar `SkillExecutor`
- [ ] Parsear tag `<use_skill>` en respuestas
- [ ] Inyectar scripts en kernel
- [ ] Comandos slash `/skills`, `/skill`

### Fase 7.4: Skills Oficiales (Repo Separado)
- [ ] Crear repositorio `dsagent-skills/`
- [ ] Crear `eda-analysis` skill
- [ ] Crear `ml-training` skill
- [ ] Crear `data-loading` skill
- [ ] Crear `visualization` skill
- [ ] Documentación de cada skill

---

## 9. Flujo de Uso

### 9.1 Instalación de Skills

```bash
$ dsagent skills install github:dsagent-skills/eda-analysis

Downloading skill from github:dsagent-skills/eda-analysis...
Validating skill structure...
✓ SKILL.md valid
✓ Scripts found: basic_eda.py, correlation.py, missing_analysis.py

Checking dependencies...
  pandas ✓
  matplotlib ✓
  seaborn ✓

Installing to ~/.dsagent/skills/eda-analysis/

✓ Skill 'eda-analysis' installed successfully!

  Description: Comprehensive exploratory data analysis for datasets
  Scripts: basic_eda.py, correlation.py, missing_analysis.py

  The agent will automatically use this skill when relevant.
```

### 9.2 Uso en Conversación

```
User: "Hazme un análisis exploratorio del dataset de ventas"

Agent (internamente):
1. Detecta intención: EDA
2. Busca skills relevantes: encuentra "eda-analysis"
3. Carga instrucciones del skill
4. Decide usar basic_eda.py

Agent (respuesta):
"Voy a usar el skill **eda-analysis** para analizar el dataset.

<use_skill name="eda-analysis" script="basic_eda.py">
  <param name="dataframe">df_ventas</param>
</use_skill>

[Ejecuta el script, muestra resultados]

El dataset tiene 50,000 filas y 15 columnas..."
```

---

## 10. Beneficios de Este Enfoque

| Beneficio | Descripción |
|-----------|-------------|
| **Sin bloat** | DSAgent viene ligero, usuario instala lo que necesita |
| **Estándar abierto** | Compatible con Claude Code y otros agentes |
| **Extensible** | Usuarios crean y comparten skills fácilmente |
| **Actualizable** | Skills se actualizan independientemente del core |
| **Validado** | Usa librería oficial skills-ref para validación |
| **Descubrible** | El agente sabe qué skills tiene disponibles |

---

## 11. Próximos Pasos

1. **Crear branch** `feature/skills-integration`
2. **Fase 7.1**: Core infrastructure
3. **Fase 7.2**: CLI commands
4. **Fase 7.3**: Agent integration
5. **Fase 7.4**: Official skills repository
