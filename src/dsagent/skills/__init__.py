"""Skills module for DSAgent.

This module provides functionality for discovering, loading, and managing
Agent Skills - reusable knowledge packages that extend the agent's capabilities.

Skills are installed by the user and stored in ~/.dsagent/skills/.
Each skill contains a SKILL.md file with instructions and optional scripts.
"""

from dsagent.skills.models import Skill, SkillMetadata, InstalledSkill
from dsagent.skills.loader import SkillLoader
from dsagent.skills.registry import SkillRegistry
from dsagent.skills.installer import SkillInstaller

__all__ = [
    "Skill",
    "SkillMetadata",
    "InstalledSkill",
    "SkillLoader",
    "SkillRegistry",
    "SkillInstaller",
]
