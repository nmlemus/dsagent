"""Pydantic models for Skills."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillCompatibility(BaseModel):
    """Compatibility requirements for a skill."""

    python: List[str] = Field(default_factory=list)
    """Python package dependencies (e.g., ['pandas>=2.0', 'matplotlib'])."""

    dsagent: Optional[str] = None
    """Minimum DSAgent version required."""


class SkillMetadata(BaseModel):
    """Metadata extracted from SKILL.md frontmatter."""

    name: str
    """Unique identifier for the skill."""

    description: str = ""
    """Short description of what the skill does."""

    version: str = "1.0.0"
    """Skill version."""

    author: Optional[str] = None
    """Skill author."""

    compatibility: SkillCompatibility = Field(default_factory=SkillCompatibility)
    """Compatibility requirements."""

    tags: List[str] = Field(default_factory=list)
    """Tags for categorization."""

    use_cases: List[str] = Field(default_factory=list)
    """When to use this skill (auto-extracted from instructions)."""


class SkillScript(BaseModel):
    """Information about a script in a skill."""

    name: str
    """Script filename."""

    path: Path
    """Full path to the script."""

    description: Optional[str] = None
    """Description of what the script does."""


class Skill(BaseModel):
    """A complete skill with metadata and content."""

    metadata: SkillMetadata
    """Skill metadata from frontmatter."""

    instructions: str
    """Full instructions from SKILL.md (markdown content)."""

    scripts: List[SkillScript] = Field(default_factory=list)
    """Available scripts in the skill."""

    path: Path
    """Path to the skill directory."""

    class Config:
        arbitrary_types_allowed = True

    def get_summary_context(self) -> str:
        """Generate summary context for system prompt (no full instructions).

        This is the default for progressive disclosure - just enough info
        for the LLM to know when to use the skill.

        Returns:
            Formatted string with skill summary.
        """
        # Build use cases string
        if self.metadata.use_cases:
            use_cases_str = "; ".join(self.metadata.use_cases[:3])
            if len(self.metadata.use_cases) > 3:
                use_cases_str += "..."
        else:
            use_cases_str = "see description"

        skill_md_path = self.path / "SKILL.md"

        lines = [
            f"- **{self.metadata.name}**: {self.metadata.description}",
            f"  - Use when: {use_cases_str}",
            f"  - Location: `{skill_md_path}`",
        ]

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Generate full context including all instructions.

        Use this when the skill is activated and full instructions are needed.

        Returns:
            Formatted string with complete skill info for the LLM.
        """
        lines = [
            f"### Skill: {self.metadata.name}",
            f"**Description**: {self.metadata.description}",
            f"**Location**: `{self.path}`",
        ]

        if self.scripts:
            lines.append("\n**Available Scripts**:")
            for script in self.scripts:
                desc = f" - {script.description}" if script.description else ""
                lines.append(f"- `{script.name}`{desc}")
                lines.append(f"  Path: `{script.path}`")

        lines.append("\n**Instructions**:")
        lines.append(self.instructions)

        return "\n".join(lines)

    def get_prompt_context(self) -> str:
        """Generate context to inject into system prompt.

        DEPRECATED: Use get_summary_context() for summaries or
        get_full_context() for complete instructions.

        Returns:
            Formatted string with skill info for the LLM.
        """
        return self.get_full_context()

    def get_script(self, name: str) -> Optional[SkillScript]:
        """Get a script by name.

        Args:
            name: Script filename (with or without .py)

        Returns:
            SkillScript if found, None otherwise.
        """
        # Normalize name
        if not name.endswith(".py"):
            name = f"{name}.py"

        for script in self.scripts:
            if script.name == name:
                return script
        return None


class InstalledSkill(BaseModel):
    """Record of an installed skill in skills.yaml."""

    name: str
    """Skill name."""

    source: str
    """Original source (github:user/repo, local:path, etc.)."""

    version: str = "latest"
    """Installed version."""

    installed_at: datetime = Field(default_factory=datetime.now)
    """Installation timestamp."""

    path: Optional[str] = None
    """Path where skill is installed."""


class SkillsConfig(BaseModel):
    """Configuration file for installed skills (~/.dsagent/skills.yaml)."""

    skills: List[InstalledSkill] = Field(default_factory=list)
    """List of installed skills."""

    def get_skill(self, name: str) -> Optional[InstalledSkill]:
        """Get an installed skill by name."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def add_skill(self, skill: InstalledSkill) -> None:
        """Add a skill to the config."""
        # Remove existing if present
        self.skills = [s for s in self.skills if s.name != skill.name]
        self.skills.append(skill)

    def remove_skill(self, name: str) -> bool:
        """Remove a skill from the config.

        Returns:
            True if skill was removed, False if not found.
        """
        original_len = len(self.skills)
        self.skills = [s for s in self.skills if s.name != name]
        return len(self.skills) < original_len


class InstallResult(BaseModel):
    """Result of a skill installation."""

    success: bool
    """Whether installation succeeded."""

    skill_name: str
    """Name of the skill."""

    message: str = ""
    """Status message."""

    metadata: Optional[SkillMetadata] = None
    """Skill metadata if successful."""

    path: Optional[Path] = None
    """Installation path if successful."""

    class Config:
        arbitrary_types_allowed = True
