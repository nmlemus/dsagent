"""Observability manager for configuring LiteLLM callbacks."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dsagent.observability.config import (
    ObservabilityConfig,
    ObservabilityProvider,
    PROVIDER_ENV_VARS,
)

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Manages LiteLLM observability callbacks.

    This class configures LiteLLM to send traces to one or more
    observability platforms (Langfuse, LangSmith, etc.).

    Example:
        config = ObservabilityConfig(
            enabled=True,
            providers=["langfuse"],
            session_id="user-123",
        )

        manager = ObservabilityManager(config)
        manager.setup()

        # Later, when making LLM calls:
        metadata = manager.get_metadata(generation_name="chat-response")
        response = litellm.completion(..., metadata=metadata)

        # Cleanup when done
        manager.teardown()
    """

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize the observability manager.

        Args:
            config: Observability configuration. If None, loads from env.
        """
        self.config = config or ObservabilityConfig.from_env()
        self._is_setup = False
        self._original_success_callback: List[str] = []
        self._original_failure_callback: List[str] = []

    def setup(self) -> bool:
        """Configure LiteLLM callbacks based on config.

        Returns:
            True if setup was successful, False otherwise.
        """
        if not self.config.enabled:
            logger.debug("Observability is disabled")
            return False

        active_providers = self.config.get_active_providers()
        if not active_providers:
            logger.debug("No observability providers configured")
            return False

        # Validate environment variables
        missing = self.config.validate_provider_env_vars()
        if missing:
            for provider, vars in missing.items():
                logger.warning(
                    f"Observability provider '{provider.value}' missing env vars: {vars}"
                )
            # Filter out providers with missing vars
            active_providers = [p for p in active_providers if p not in missing]

        if not active_providers:
            logger.warning("No observability providers have required credentials")
            return False

        # Import litellm here to avoid import errors if not installed
        try:
            import litellm
        except ImportError:
            logger.error("litellm not installed, cannot setup observability")
            return False

        # Save original callbacks
        self._original_success_callback = getattr(litellm, "success_callback", []) or []
        self._original_failure_callback = getattr(litellm, "failure_callback", []) or []

        # Set provider-specific environment variables
        self._configure_provider_settings()

        # Configure callbacks
        provider_names = [p.value for p in active_providers]
        litellm.success_callback = provider_names
        litellm.failure_callback = provider_names

        self._is_setup = True
        logger.info(f"Observability configured with providers: {provider_names}")

        return True

    def _configure_provider_settings(self) -> None:
        """Configure provider-specific settings via environment variables."""
        # Langfuse
        if self.config.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", self.config.langfuse_host)

        # LangSmith
        if self.config.langsmith_project:
            os.environ.setdefault("LANGCHAIN_PROJECT", self.config.langsmith_project)

    def get_metadata(
        self,
        generation_name: Optional[str] = None,
        trace_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Get metadata dict for LiteLLM calls.

        This metadata is passed to observability providers for trace attribution.

        Args:
            generation_name: Name for this specific generation/completion
            trace_name: Name for the overall trace
            trace_id: Custom trace ID (auto-generated if not provided)
            parent_observation_id: Parent observation for nested traces
            **extra: Additional metadata key-value pairs

        Returns:
            Metadata dictionary to pass to litellm.completion()

        Example:
            metadata = manager.get_metadata(
                generation_name="chat-response",
                trace_name="user-conversation",
            )
            response = litellm.completion(
                model="gpt-4",
                messages=[...],
                metadata=metadata,
            )
        """
        if not self.config.enabled:
            return extra

        metadata: Dict[str, Any] = {}

        # Trace-level metadata
        if self.config.trace_user_id:
            metadata["trace_user_id"] = self.config.trace_user_id

        if self.config.session_id:
            metadata["session_id"] = self.config.session_id

        if self.config.tags:
            metadata["tags"] = self.config.tags

        # Privacy settings
        if self.config.mask_input:
            metadata["mask_input"] = True

        if self.config.mask_output:
            metadata["mask_output"] = True

        # Generation-level metadata
        if generation_name:
            metadata["generation_name"] = generation_name

        if trace_name:
            metadata["trace_name"] = trace_name

        if trace_id:
            metadata["trace_id"] = trace_id

        if parent_observation_id:
            metadata["parent_observation_id"] = parent_observation_id

        # Add any extra metadata
        metadata.update(extra)

        return metadata

    def get_call_metadata(
        self,
        session_id: Optional[str] = None,
        call_type: str = "chat",
    ) -> Dict[str, Any]:
        """Get metadata for a specific LLM call.

        Convenience method that auto-generates trace/generation names.

        Args:
            session_id: Session ID to use (overrides config)
            call_type: Type of call (chat, summarize, etc.)

        Returns:
            Metadata dictionary for the LLM call.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        return self.get_metadata(
            generation_name=f"dsagent-{call_type}",
            trace_name=f"dsagent-{call_type}-{timestamp}",
            session_id=session_id or self.config.session_id,
        )

    def teardown(self) -> None:
        """Remove callbacks and restore original state."""
        if not self._is_setup:
            return

        try:
            import litellm

            litellm.success_callback = self._original_success_callback
            litellm.failure_callback = self._original_failure_callback

            self._is_setup = False
            logger.debug("Observability callbacks removed")
        except ImportError:
            pass

    def is_active(self) -> bool:
        """Check if observability is currently active."""
        return self._is_setup and self.config.enabled

    def get_status(self) -> Dict[str, Any]:
        """Get current observability status.

        Returns:
            Dict with status information.
        """
        active_providers = self.config.get_active_providers()
        missing_vars = self.config.validate_provider_env_vars()

        return {
            "enabled": self.config.enabled,
            "is_setup": self._is_setup,
            "providers": [p.value for p in active_providers],
            "missing_env_vars": {
                p.value: vars for p, vars in missing_vars.items()
            },
            "session_id": self.config.session_id,
            "mask_input": self.config.mask_input,
            "mask_output": self.config.mask_output,
        }

    def __enter__(self) -> "ObservabilityManager":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.teardown()


# Global manager instance for convenience
_global_manager: Optional[ObservabilityManager] = None


def get_observability_manager() -> ObservabilityManager:
    """Get or create the global observability manager.

    Creates a manager from environment variables if not already created.

    Returns:
        The global ObservabilityManager instance.
    """
    global _global_manager

    if _global_manager is None:
        config = ObservabilityConfig.from_env()
        _global_manager = ObservabilityManager(config)

    return _global_manager


def setup_observability(config: Optional[ObservabilityConfig] = None) -> bool:
    """Setup global observability.

    Convenience function to configure observability globally.

    Args:
        config: Optional config. Uses env vars if not provided.

    Returns:
        True if setup was successful.
    """
    global _global_manager

    _global_manager = ObservabilityManager(config)
    return _global_manager.setup()


def teardown_observability() -> None:
    """Teardown global observability."""
    global _global_manager

    if _global_manager:
        _global_manager.teardown()
        _global_manager = None
