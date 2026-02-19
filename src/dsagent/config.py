"""Centralized configuration for DSAgent.

This module provides a single source of truth for all configuration,
supporting environment variables, .env files, and programmatic overrides.

Configuration Priority (highest to lowest):
    1. Explicit parameters (API request, CLI --model flag)
    2. Session-stored model (from database)
    3. DSAGENT_DEFAULT_MODEL environment variable
    4. LLM_MODEL environment variable (legacy/CLI compatibility)
    5. Hardcoded fallback "gpt-4o"

Usage:
    from dsagent.config import get_settings, get_default_model

    settings = get_settings()
    model = get_default_model()

    # With explicit override
    model = get_default_model(explicit="claude-3-5-sonnet")

    # With session model
    model = get_default_model(session_model=session.model)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Hardcoded fallback - only used when no env vars are set
FALLBACK_MODEL = "gpt-4o"


class DSAgentSettings(BaseSettings):
    """Unified settings for DSAgent CLI and Server.

    All settings can be configured via environment variables with the
    DSAGENT_ prefix (e.g., DSAGENT_DEFAULT_MODEL).

    For backward compatibility, LLM_MODEL is also supported as a fallback
    for the default_model setting.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="DSAGENT_",
    )

    # ─── Model Configuration ──────────────────────────────────────────────────
    default_model: Optional[str] = Field(
        default=None,
        description="Default LLM model when not specified in request",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    max_rounds: int = Field(default=30, ge=1)
    code_timeout: int = Field(default=300, ge=1)

    # ─── Workspace ────────────────────────────────────────────────────────────
    workspace: str = Field(default="./workspace")
    sessions_dir: str = Field(default="workspace")
    session_backend: str = Field(default="sqlite")

    # ─── Server Settings ──────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: Optional[str] = Field(
        default=None,
        description="If set, API and WebSocket require X-API-Key. In production, set DSAGENT_API_KEY.",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated origins or '*' for all. In production, set to specific origins.",
    )
    require_api_key: bool = Field(
        default=False,
        description="If True, server refuses to start when api_key is not set (use in production).",
    )
    default_hitl_mode: str = Field(default="none")

    # ─── Upload limits ─────────────────────────────────────────────────────────
    max_upload_mb: float = Field(
        default=50.0,
        ge=0,
        description="Max upload size per file in MB (0 = no limit).",
    )

    # ─── Observability ────────────────────────────────────────────────────────
    observability_enabled: bool = Field(default=False)
    observability_providers: Optional[str] = Field(default=None)

    @field_validator("default_model", mode="before")
    @classmethod
    def resolve_model_from_legacy(cls, v: Optional[str]) -> Optional[str]:
        """Check legacy LLM_MODEL if DSAGENT_DEFAULT_MODEL not set."""
        if v is None:
            legacy = os.getenv("LLM_MODEL")
            if legacy:
                logger.debug(f"Using legacy LLM_MODEL={legacy}")
                return legacy
        return v


@lru_cache
def get_settings() -> DSAgentSettings:
    """Get cached DSAgent settings.

    Returns:
        DSAgentSettings instance with resolved configuration.

    Note:
        Settings are cached for performance. Use clear_settings_cache()
        to reload settings (useful for testing).
    """
    settings = DSAgentSettings()

    # Log configuration source for debugging
    if settings.default_model:
        source = "DSAGENT_DEFAULT_MODEL"
        if os.getenv("LLM_MODEL") and not os.getenv("DSAGENT_DEFAULT_MODEL"):
            source = "LLM_MODEL (legacy)"
        logger.info(f"Configuration: default_model={settings.default_model} (from {source})")
    else:
        logger.info(f"Configuration: no default_model set, will use fallback={FALLBACK_MODEL}")

    return settings


def clear_settings_cache() -> None:
    """Clear cached settings.

    Useful for testing or when environment variables change at runtime.
    """
    get_settings.cache_clear()


def get_default_model(
    explicit: Optional[str] = None,
    session_model: Optional[str] = None,
) -> str:
    """Get the effective default model using the resolution cascade.

    Resolution order (first non-None wins):
        1. explicit - Model passed as parameter (API request, CLI flag)
        2. session_model - Model stored in session (for resumed sessions)
        3. DSAGENT_DEFAULT_MODEL - Primary environment variable
        4. LLM_MODEL - Legacy environment variable (CLI compatibility)
        5. FALLBACK_MODEL - Hardcoded fallback ("gpt-4o")

    Args:
        explicit: Explicitly specified model (highest priority)
        session_model: Model from session storage

    Returns:
        Resolved model name

    Example:
        # Use default from environment
        model = get_default_model()

        # Override with explicit model
        model = get_default_model(explicit="claude-3-5-sonnet")

        # Use session model if available
        model = get_default_model(session_model=session.model)
    """
    if explicit:
        logger.info(f"Model resolution: using explicit={explicit}")
        return explicit

    if session_model:
        logger.info(f"Model resolution: using session_model={session_model}")
        return session_model

    settings = get_settings()
    if settings.default_model:
        logger.info(f"Model resolution: using settings.default_model={settings.default_model}")
        return settings.default_model

    logger.info(f"Model resolution: using fallback={FALLBACK_MODEL}")
    return FALLBACK_MODEL


def log_configuration() -> None:
    """Log current configuration for debugging.

    Useful for troubleshooting deployment issues.
    """
    settings = get_settings()
    logger.info("=== DSAgent Configuration ===")
    logger.info(f"  default_model: {settings.default_model or f'(none, fallback={FALLBACK_MODEL})'}")
    logger.info(f"  workspace: {settings.workspace}")
    logger.info(f"  sessions_dir: {settings.sessions_dir}")
    logger.info(f"  session_backend: {settings.session_backend}")
    logger.info(f"  temperature: {settings.temperature}")
    logger.info(f"  max_tokens: {settings.max_tokens}")
    logger.info(f"  host: {settings.host}")
    logger.info(f"  port: {settings.port}")
    logger.info("=============================")
