"""Observability module for DSAgent.

This module provides LLM observability integration via LiteLLM callbacks.
Supports multiple providers: Langfuse, LangSmith, Lunary, Helicone, and more.

Example:
    from dsagent.observability import ObservabilityConfig, ObservabilityManager

    config = ObservabilityConfig(
        enabled=True,
        providers=["langfuse"],
    )

    manager = ObservabilityManager(config)
    manager.setup()
"""

from dsagent.observability.config import (
    ObservabilityProvider,
    ObservabilityConfig,
)
from dsagent.observability.manager import ObservabilityManager

__all__ = [
    "ObservabilityProvider",
    "ObservabilityConfig",
    "ObservabilityManager",
]
