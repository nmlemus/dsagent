"""Skill registry - central registry of available skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from dsagent.skills.loader import SkillLoader, SkillNotFoundError
from dsagent.skills.models import Skill, SkillMetadata

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry of available skills.

    Maintains an in-memory cache of skill metadata and provides
    methods to generate prompt context for the LLM.
    """

    def __init__(self, loader: Optional[SkillLoader] = None):
        """Initialize the registry.

        Args:
            loader: SkillLoader instance. Creates default if not provided.
        """
        self.loader = loader or SkillLoader()
        self._skills: Dict[str, SkillMetadata] = {}
        self._loaded_skills: Dict[str, Skill] = {}

    def discover(self) -> int:
        """Discover and register all installed skills.

        Returns:
            Number of skills discovered.
        """
        self._skills.clear()
        self._loaded_skills.clear()

        for metadata in self.loader.discover_skills():
            self._skills[metadata.name] = metadata
            logger.debug(f"Registered skill: {metadata.name}")

        return len(self._skills)

    def refresh(self) -> int:
        """Refresh the registry (alias for discover).

        Returns:
            Number of skills discovered.
        """
        return self.discover()

    @property
    def skills(self) -> Dict[str, SkillMetadata]:
        """Get all registered skill metadata."""
        return self._skills.copy()

    def get_skill_names(self) -> List[str]:
        """Get list of registered skill names."""
        return list(self._skills.keys())

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """Get metadata for a skill.

        Args:
            name: Skill name

        Returns:
            SkillMetadata if found, None otherwise.
        """
        return self._skills.get(name)

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a complete skill by name.

        Loads the skill if not already cached.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise.
        """
        if name not in self._skills:
            return None

        # Check cache
        if name in self._loaded_skills:
            return self._loaded_skills[name]

        # Load skill
        try:
            skill = self.loader.load_skill(name)
            self._loaded_skills[name] = skill
            return skill
        except SkillNotFoundError:
            return None

    def has_skill(self, name: str) -> bool:
        """Check if a skill is registered.

        Args:
            name: Skill name

        Returns:
            True if skill is registered.
        """
        return name in self._skills

    def register(self, skill: Skill) -> None:
        """Manually register a skill.

        Args:
            skill: Skill to register.
        """
        self._skills[skill.metadata.name] = skill.metadata
        self._loaded_skills[skill.metadata.name] = skill

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if skill was unregistered, False if not found.
        """
        if name not in self._skills:
            return False

        del self._skills[name]
        self._loaded_skills.pop(name, None)
        return True

    def get_prompt_context(self, skill_names: Optional[List[str]] = None) -> str:
        """Generate context for injection into system prompt.

        This is the main method used by ConversationalAgent to get
        skill information for the LLM.

        Args:
            skill_names: Specific skills to include. If None, includes all.

        Returns:
            Formatted markdown/text for the system prompt.
        """
        if not self._skills:
            return ""

        # Determine which skills to include
        names = skill_names if skill_names else list(self._skills.keys())

        # Load all requested skills
        skills_to_include = []
        for name in names:
            skill = self.get_skill(name)
            if skill:
                skills_to_include.append(skill)

        if not skills_to_include:
            return ""

        # Build prompt context
        lines = [
            "## Available Skills",
            "",
            "You have access to the following skills. Each skill provides "
            "specialized capabilities through instructions and scripts.",
            "To use a skill's script, follow its instructions and execute the script using exec().",
            "",
        ]

        for skill in skills_to_include:
            lines.append(skill.get_prompt_context())
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def get_skills_summary(self) -> str:
        """Get a brief summary of available skills.

        Returns:
            Short summary suitable for display.
        """
        if not self._skills:
            return "No skills installed."

        lines = ["Installed skills:"]
        for name, meta in self._skills.items():
            lines.append(f"  - {name}: {meta.description}")

        return "\n".join(lines)

    def find_skills_by_tag(self, tag: str) -> List[SkillMetadata]:
        """Find skills by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching SkillMetadata.
        """
        return [
            meta
            for meta in self._skills.values()
            if tag.lower() in [t.lower() for t in meta.tags]
        ]

    def find_skills_by_keyword(self, keyword: str) -> List[SkillMetadata]:
        """Find skills by keyword in name or description.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching SkillMetadata.
        """
        keyword_lower = keyword.lower()
        return [
            meta
            for meta in self._skills.values()
            if keyword_lower in meta.name.lower()
            or keyword_lower in meta.description.lower()
        ]
