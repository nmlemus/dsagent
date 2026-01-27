"""Configuration models for observability."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ObservabilityProvider(str, Enum):
    """Supported observability providers.

    These map directly to LiteLLM callback names.
    See: https://docs.litellm.ai/docs/observability/callbacks
    """

    NONE = "none"

    # Primary providers (well-documented)
    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    LUNARY = "lunary"
    HELICONE = "helicone"
    TRACELOOP = "traceloop"

    # Additional providers
    ARIZE = "arize"
    POSTHOG = "posthog"
    SENTRY = "sentry"
    ATHINA = "athina"
    OPENMETER = "openmeter"
    PROMPTLAYER = "promptlayer"
    DATADOG = "datadog"
    BRAINTRUST = "braintrust"
    LOGFIRE = "logfire"

    # OpenTelemetry
    OTEL = "otel"
    LANGFUSE_OTEL = "langfuse_otel"

    @classmethod
    def from_string(cls, value: str) -> "ObservabilityProvider":
        """Convert string to provider enum."""
        value_lower = value.lower().strip()
        for provider in cls:
            if provider.value == value_lower:
                return provider
        raise ValueError(f"Unknown observability provider: {value}")


# Environment variables required by each provider
PROVIDER_ENV_VARS: Dict[ObservabilityProvider, List[str]] = {
    ObservabilityProvider.LANGFUSE: [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ],
    ObservabilityProvider.LANGSMITH: [
        "LANGCHAIN_API_KEY",
    ],
    ObservabilityProvider.LUNARY: [
        "LUNARY_PUBLIC_KEY",
    ],
    ObservabilityProvider.HELICONE: [
        "HELICONE_API_KEY",
    ],
    ObservabilityProvider.TRACELOOP: [
        "TRACELOOP_API_KEY",
    ],
    ObservabilityProvider.ARIZE: [
        "ARIZE_API_KEY",
        "ARIZE_SPACE_KEY",
    ],
    ObservabilityProvider.POSTHOG: [
        "POSTHOG_API_KEY",
    ],
    ObservabilityProvider.SENTRY: [
        "SENTRY_DSN",
    ],
    ObservabilityProvider.ATHINA: [
        "ATHINA_API_KEY",
    ],
    ObservabilityProvider.DATADOG: [
        "DD_API_KEY",
    ],
    ObservabilityProvider.BRAINTRUST: [
        "BRAINTRUST_API_KEY",
    ],
    ObservabilityProvider.LOGFIRE: [
        "LOGFIRE_TOKEN",
    ],
}


class ObservabilityConfig(BaseModel):
    """Configuration for LLM observability.

    This configures LiteLLM callbacks to send traces to observability platforms.

    Example:
        config = ObservabilityConfig(
            enabled=True,
            providers=["langfuse"],
            session_id="user-session-123",
            tags=["production", "chat"],
        )

    Environment Variables:
        DSAGENT_OBSERVABILITY_ENABLED: Set to "true" to enable
        DSAGENT_OBSERVABILITY_PROVIDERS: Comma-separated list of providers
        DSAGENT_OBSERVABILITY_MASK_INPUT: Set to "true" to redact prompts
        DSAGENT_OBSERVABILITY_MASK_OUTPUT: Set to "true" to redact responses
    """

    enabled: bool = Field(
        default=False,
        description="Enable observability logging",
    )

    providers: List[Union[ObservabilityProvider, str]] = Field(
        default_factory=list,
        description="List of observability providers to use",
    )

    # Trace metadata
    trace_user_id: Optional[str] = Field(
        default=None,
        description="User ID for trace attribution",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping related traces",
    )

    tags: List[str] = Field(
        default_factory=list,
        description="Tags to attach to all traces",
    )

    # Privacy settings
    mask_input: bool = Field(
        default=False,
        description="Redact input prompts from logs",
    )

    mask_output: bool = Field(
        default=False,
        description="Redact output responses from logs",
    )

    # Provider-specific settings
    langfuse_host: Optional[str] = Field(
        default=None,
        description="Custom Langfuse host URL (default: https://cloud.langfuse.com)",
    )

    langsmith_project: Optional[str] = Field(
        default=None,
        description="LangSmith project name",
    )

    @field_validator("providers", mode="before")
    @classmethod
    def parse_providers(cls, v: Any) -> List[ObservabilityProvider]:
        """Parse providers from various formats."""
        if v is None:
            return []

        if isinstance(v, str):
            # Handle comma-separated string
            v = [p.strip() for p in v.split(",") if p.strip()]

        result = []
        for item in v:
            if isinstance(item, ObservabilityProvider):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(ObservabilityProvider.from_string(item))
                except ValueError:
                    # Skip unknown providers with a warning
                    pass
            # Skip None values
        return result

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create config from environment variables.

        Reads:
            DSAGENT_OBSERVABILITY_ENABLED
            DSAGENT_OBSERVABILITY_PROVIDERS
            DSAGENT_OBSERVABILITY_MASK_INPUT
            DSAGENT_OBSERVABILITY_MASK_OUTPUT
            DSAGENT_OBSERVABILITY_SESSION_ID
            DSAGENT_OBSERVABILITY_TAGS
        """
        enabled_str = os.getenv("DSAGENT_OBSERVABILITY_ENABLED", "false")
        enabled = enabled_str.lower() in ("true", "1", "yes")

        providers_str = os.getenv("DSAGENT_OBSERVABILITY_PROVIDERS", "")
        providers = [p.strip() for p in providers_str.split(",") if p.strip()]

        mask_input_str = os.getenv("DSAGENT_OBSERVABILITY_MASK_INPUT", "false")
        mask_input = mask_input_str.lower() in ("true", "1", "yes")

        mask_output_str = os.getenv("DSAGENT_OBSERVABILITY_MASK_OUTPUT", "false")
        mask_output = mask_output_str.lower() in ("true", "1", "yes")

        session_id = os.getenv("DSAGENT_OBSERVABILITY_SESSION_ID")

        tags_str = os.getenv("DSAGENT_OBSERVABILITY_TAGS", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        # Provider-specific
        langfuse_host = os.getenv("LANGFUSE_HOST")
        langsmith_project = os.getenv("LANGCHAIN_PROJECT")

        return cls(
            enabled=enabled,
            providers=providers,
            mask_input=mask_input,
            mask_output=mask_output,
            session_id=session_id,
            tags=tags,
            langfuse_host=langfuse_host,
            langsmith_project=langsmith_project,
        )

    def get_active_providers(self) -> List[ObservabilityProvider]:
        """Get list of active providers (excluding NONE)."""
        return [
            p for p in self.providers
            if isinstance(p, ObservabilityProvider) and p != ObservabilityProvider.NONE
        ]

    def validate_provider_env_vars(self) -> Dict[ObservabilityProvider, List[str]]:
        """Check which required environment variables are missing.

        Returns:
            Dict mapping providers to their missing env vars.
        """
        missing = {}

        for provider in self.get_active_providers():
            required_vars = PROVIDER_ENV_VARS.get(provider, [])
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            if missing_vars:
                missing[provider] = missing_vars

        return missing
