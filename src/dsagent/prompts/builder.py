"""Prompt builder for DSAgent.

This module provides the PromptBuilder class that composes system prompts
from reusable sections for different agent modes.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from dsagent.prompts import sections


class PromptBuilder:
    """Builds system prompts from reusable sections.

    This class provides methods to construct complete system prompts
    for different agent modes (engine vs conversational) while sharing
    common sections like workspace rules, data rules, and tool priority.

    Example:
        # For ConversationalAgent
        prompt = PromptBuilder.build_conversational_prompt(
            kernel_context="Variables: df (DataFrame)",
            tools=["web_search", "bigquery_query"],
        )

        # For AgentEngine
        prompt = PromptBuilder.build_engine_prompt(
            tools=["web_search"],
        )
    """

    @classmethod
    def build_engine_prompt(
        cls,
        tools: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for AgentEngine (autonomous, plan-focused).

        This prompt is used by the standalone AgentEngine which always
        operates in plan-based autonomous mode.

        Args:
            tools: List of available MCP tool names

        Returns:
            Complete system prompt string
        """
        parts = [
            sections.ENGINE_ROLE,
            sections.ENGINE_RESPONSE_FORMAT,
            sections.ENGINE_CRITICAL_RULES,
            sections.DATA_RULES,
        ]

        # Add tool sections if tools are available
        if tools:
            parts.append(sections.TOOL_PRIORITY_RULES)
            parts.append(sections.TOOL_GUIDANCE)

        parts.extend([
            sections.AVAILABLE_LIBRARIES,
            sections.BASH_LATEX,
            sections.WORKSPACE_STRUCTURE,
            sections.FILE_RULES,
            sections.FILE_EXAMPLES,
        ])

        return "\n\n".join(parts)

    @classmethod
    def build_conversational_prompt(
        cls,
        kernel_context: str = "",
        current_date: Optional[str] = None,
        tools: Optional[List[str]] = None,
        skills_context: Optional[str] = None,
    ) -> str:
        """Build prompt for ConversationalAgent (interactive + autonomous).

        This prompt is used by the ConversationalAgent which supports both
        interactive chat mode and autonomous execution based on intent
        classification.

        Args:
            kernel_context: Current kernel state (variables, dataframes)
            current_date: Current date string (defaults to today)
            tools: List of available MCP tool names
            skills_context: Optional skills/commands context

        Returns:
            Complete system prompt string
        """
        if current_date is None:
            current_date = date.today().strftime("%Y-%m-%d")

        # Build the role section with date
        role_with_date = f"{sections.CONVERSATIONAL_ROLE}\n\n**Current date**: {current_date}"

        # Build kernel context section
        context_section = f"## Current Session Context\n{kernel_context}"

        parts = [
            role_with_date,
            context_section,
            sections.CONVERSATIONAL_RESPONSE_FORMAT,
            sections.CONVERSATIONAL_CRITICAL_RULES,
            sections.DATA_RULES,
        ]

        # Add tool sections if tools are available
        if tools:
            parts.append(sections.TOOL_PRIORITY_RULES)
            parts.append(sections.TOOL_GUIDANCE)

        parts.extend([
            sections.FILE_RULES_SHORT,
            sections.AVAILABLE_LIBRARIES,
            sections.BASH_LATEX,
        ])

        # Add skills context if available
        if skills_context:
            parts.append(skills_context)

        return "\n\n".join(parts)
