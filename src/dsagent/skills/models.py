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

    def get_prompt_context(self) -> str:
        """Generate context to inject into system prompt.

        Returns:
            Formatted string with skill info for the LLM.
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
