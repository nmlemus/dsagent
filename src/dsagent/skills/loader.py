"""Skill loader - discovers and loads skills from disk."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from dsagent.skills.models import (
    Skill,
    SkillCompatibility,
    SkillMetadata,
    SkillScript,
)

logger = logging.getLogger(__name__)

# Regex to extract YAML frontmatter from SKILL.md
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillParseError(Exception):
    """Error parsing a skill."""

    pass


class SkillNotFoundError(Exception):
    """Skill not found."""

    pass


class SkillLoader:
    """Discovers and loads skills from disk.

    Skills are stored in ~/.dsagent/skills/ by default.
    Each skill is a directory containing a SKILL.md file.
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        """Initialize the skill loader.

        Args:
            skills_dir: Directory containing skills. Defaults to ~/.dsagent/skills/
        """
        self.skills_dir = skills_dir or (Path.home() / ".dsagent" / "skills")

    def discover_skills(self) -> List[SkillMetadata]:
        """Discover all installed skills.

        Scans the skills directory for valid skill directories
        (those containing SKILL.md).

        Returns:
            List of SkillMetadata for each discovered skill.
        """
        skills = []

        if not self.skills_dir.exists():
            logger.debug(f"Skills directory does not exist: {self.skills_dir}")
            return skills

        for entry in self.skills_dir.iterdir():
            if not entry.is_dir():
                continue

            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                logger.debug(f"Skipping {entry.name}: no SKILL.md")
                continue

            try:
                metadata = self._parse_metadata(skill_file)
                skills.append(metadata)
            except SkillParseError as e:
                logger.warning(f"Failed to parse skill {entry.name}: {e}")
                continue

        logger.info(f"Discovered {len(skills)} skills")
        return skills

    def load_skill(self, name: str) -> Skill:
        """Load a complete skill by name.

        Args:
            name: Skill name (directory name in skills_dir)

        Returns:
            Complete Skill object with metadata, instructions, and scripts.

        Raises:
            SkillNotFoundError: If skill doesn't exist.
            SkillParseError: If skill is invalid.
        """
        skill_path = self.skills_dir / name

        if not skill_path.exists():
            raise SkillNotFoundError(f"Skill '{name}' not found in {self.skills_dir}")

        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            raise SkillNotFoundError(f"Skill '{name}' has no SKILL.md file")

        # Parse SKILL.md
        metadata, instructions = self._parse_skill_file(skill_file)

        # Override name with directory name for consistency
        metadata.name = name

        # Discover scripts
        scripts = self._discover_scripts(skill_path)

        return Skill(
            metadata=metadata,
            instructions=instructions,
            scripts=scripts,
            path=skill_path,
        )

    def load_skill_from_path(self, path: Path) -> Skill:
        """Load a skill from an arbitrary path.

        Useful for validating skills before installation.

        Args:
            path: Path to skill directory

        Returns:
            Complete Skill object.

        Raises:
            SkillNotFoundError: If path doesn't exist.
            SkillParseError: If skill is invalid.
        """
        if not path.exists():
            raise SkillNotFoundError(f"Path does not exist: {path}")

        skill_file = path / "SKILL.md"
        if not skill_file.exists():
            raise SkillParseError(f"No SKILL.md found in {path}")

        metadata, instructions = self._parse_skill_file(skill_file)

        # Use directory name as skill name if not specified
        if not metadata.name:
            metadata.name = path.name

        scripts = self._discover_scripts(path)

        return Skill(
            metadata=metadata,
            instructions=instructions,
            scripts=scripts,
            path=path,
        )

    def _parse_skill_file(self, skill_file: Path) -> Tuple[SkillMetadata, str]:
        """Parse a SKILL.md file into metadata and instructions.

        Args:
            skill_file: Path to SKILL.md

        Returns:
            Tuple of (SkillMetadata, instructions_markdown)

        Raises:
            SkillParseError: If file is invalid.
        """
        content = skill_file.read_text(encoding="utf-8")

        # Extract frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise SkillParseError(
                f"No YAML frontmatter found in {skill_file}. "
                "SKILL.md must start with ---\\n...\\n---"
            )

        frontmatter_yaml = match.group(1)
        instructions = content[match.end():].strip()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_yaml)
        except yaml.YAMLError as e:
            raise SkillParseError(f"Invalid YAML frontmatter: {e}")

        if not isinstance(frontmatter, dict):
            raise SkillParseError("Frontmatter must be a YAML dictionary")

        # Extract metadata
        metadata = self._extract_metadata(frontmatter)

        # Auto-extract use cases from instructions
        use_cases = self._extract_use_cases(instructions)
        if use_cases:
            metadata.use_cases = use_cases

        return metadata, instructions

    def _extract_use_cases(self, instructions: str) -> List[str]:
        """Extract use cases from 'When to Use' section in instructions.

        Looks for sections like:
        - ## When to Use This Skill
        - ## When to Use
        - ## Use Cases

        Args:
            instructions: Full markdown instructions

        Returns:
            List of use case strings (max 5)
        """
        # Patterns to match "When to Use" type sections
        patterns = [
            r"##\s*When to Use This Skill\s*\n(.*?)(?=\n##|\Z)",
            r"##\s*When to Use\s*\n(.*?)(?=\n##|\Z)",
            r"##\s*Use Cases\s*\n(.*?)(?=\n##|\Z)",
            r"##\s*When Should I Use\s*\n(.*?)(?=\n##|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, instructions, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()

                # Extract bullet points (lines starting with - or *)
                bullets = re.findall(r"^[-*]\s*(.+)$", content, re.MULTILINE)

                if bullets:
                    # Clean up and limit to 5 most relevant
                    use_cases = []
                    for bullet in bullets[:5]:
                        # Clean up the text
                        cleaned = bullet.strip()
                        # Truncate very long use cases
                        if len(cleaned) > 100:
                            cleaned = cleaned[:97] + "..."
                        use_cases.append(cleaned)
                    return use_cases

        return []

    def _parse_metadata(self, skill_file: Path) -> SkillMetadata:
        """Parse only metadata from a SKILL.md file.

        Args:
            skill_file: Path to SKILL.md

        Returns:
            SkillMetadata
        """
        metadata, _ = self._parse_skill_file(skill_file)
        return metadata

    def _extract_metadata(self, frontmatter: dict) -> SkillMetadata:
        """Extract SkillMetadata from parsed frontmatter.

        Args:
            frontmatter: Parsed YAML dictionary

        Returns:
            SkillMetadata
        """
        # Handle compatibility
        compat_data = frontmatter.get("compatibility", {})
        compatibility = SkillCompatibility(
            python=compat_data.get("python", []),
            dsagent=compat_data.get("dsagent"),
        )

        return SkillMetadata(
            name=frontmatter.get("name", ""),
            description=frontmatter.get("description", ""),
            version=str(frontmatter.get("version", "1.0.0")),
            author=frontmatter.get("author"),
            compatibility=compatibility,
            tags=frontmatter.get("tags", []),
        )

    def _discover_scripts(self, skill_path: Path) -> List[SkillScript]:
        """Discover scripts in a skill directory.

        Looks for .py files in:
        - skill_path/scripts/
        - skill_path/*.py (excluding __init__.py)

        Args:
            skill_path: Path to skill directory

        Returns:
            List of SkillScript objects
        """
        scripts = []

        # Check scripts/ subdirectory
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            for py_file in scripts_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                scripts.append(
                    SkillScript(
                        name=py_file.name,
                        path=py_file,
                    )
                )

        # Check root directory for .py files
        for py_file in skill_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            # Don't duplicate if already in scripts/
            if not any(s.name == py_file.name for s in scripts):
                scripts.append(
                    SkillScript(
                        name=py_file.name,
                        path=py_file,
                    )
                )

        return scripts

    def skill_exists(self, name: str) -> bool:
        """Check if a skill is installed.

        Args:
            name: Skill name

        Returns:
            True if skill exists, False otherwise.
        """
        skill_path = self.skills_dir / name
        return skill_path.exists() and (skill_path / "SKILL.md").exists()
