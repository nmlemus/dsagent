"""Validation utilities for DSAgent configuration."""

from __future__ import annotations

import os
from typing import Optional, Tuple


class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""

    pass


def is_using_proxy() -> bool:
    """Check if LLM_API_BASE proxy is configured."""
    return bool(os.getenv("LLM_API_BASE"))


def apply_llm_api_base(model: str) -> None:
    """Set API base URLs for LLM providers.

    When LLM_API_BASE is set (e.g., LiteLLM proxy), all models route through
    OPENAI_API_BASE since the proxy provides an OpenAI-compatible API.

    When LLM_API_BASE is NOT set, models use their native SDKs directly.
    For OpenAI models, we explicitly set the default API base to ensure
    compatibility across different LiteLLM versions.
    """
    # When using a proxy, ALL models go through OPENAI_API_BASE
    if is_using_proxy():
        if not os.getenv("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = os.getenv("LLM_API_BASE")
        return

    # Set default API base for OpenAI models (needed for some LiteLLM versions)
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1", "o3")):
        if not os.getenv("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"


def get_proxy_model_name(model: str) -> str:
    """Transform model name for proxy usage.

    When using LLM_API_BASE (a LiteLLM proxy), non-OpenAI models need
    the 'openai/' prefix to force LiteLLM to use the OpenAI SDK instead
    of native provider SDKs.

    Args:
        model: Original model name (e.g., 'claude-sonnet-4-5')

    Returns:
        Transformed model name for proxy (e.g., 'openai/claude-sonnet-4-5')
        or original name if no proxy is configured.
    """
    if not is_using_proxy():
        return model  # No proxy, use native SDKs

    model_lower = model.lower()

    # Already has openai/ prefix - ready for proxy
    if model_lower.startswith("openai/"):
        return model

    # Local models don't go through proxy
    if model_lower.startswith(("ollama/", "ollama_chat/", "local/")):
        return model

    # Azure has its own handling
    if model_lower.startswith("azure/"):
        return model

    # OpenAI models (gpt-*, o1*, o3*) work directly with OPENAI_API_BASE
    if model_lower.startswith(("gpt-", "o1", "o3")):
        return model

    # For all other models (Claude, Gemini, DeepSeek, etc.),
    # prefix with openai/ to force OpenAI SDK through proxy
    # Remove existing provider prefix if present (e.g., anthropic/claude -> claude)
    if "/" in model:
        model = model.split("/", 1)[1]

    return f"openai/{model}"


# Mapping of model prefixes to required environment variables
MODEL_PROVIDER_KEYS = {
    # OpenAI models
    "gpt-": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    # Anthropic models
    "claude": "ANTHROPIC_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    # Google models
    "gemini": "GOOGLE_API_KEY",
    "google/": "GOOGLE_API_KEY",
    # Groq
    "groq/": "GROQ_API_KEY",
    # Together AI
    "together/": "TOGETHER_API_KEY",
    "together_ai/": "TOGETHER_API_KEY",
    # OpenRouter
    "openrouter/": "OPENROUTER_API_KEY",
    # Mistral
    "mistral/": "MISTRAL_API_KEY",
    # DeepSeek
    "deepseek/": "DEEPSEEK_API_KEY",
    # Cohere
    "cohere/": "COHERE_API_KEY",
    "command": "COHERE_API_KEY",
    # Perplexity
    "perplexity/": "PERPLEXITYAI_API_KEY",
    # Fireworks
    "fireworks_ai/": "FIREWORKS_API_KEY",
    # Hugging Face
    "huggingface/": "HUGGINGFACE_API_KEY",
    # Azure OpenAI
    "azure/": "AZURE_API_KEY",
    # Local models (no key required)
    "ollama/": None,
    "ollama_chat/": None,
    "local/": None,
}


def get_provider_for_model(model: str) -> Tuple[str, Optional[str]]:
    """Get the provider name and required API key for a model.

    Args:
        model: The model name/identifier

    Returns:
        Tuple of (provider_name, required_env_var or None if no key needed)
    """
    model_lower = model.lower()

    for prefix, env_var in MODEL_PROVIDER_KEYS.items():
        if model_lower.startswith(prefix):
            provider = prefix.rstrip("/-")
            return provider, env_var

    # Default: assume OpenAI-compatible
    return "openai", "OPENAI_API_KEY"


def validate_api_key(model: str) -> None:
    """Validate that the required API key exists for the given model.

    When using a LiteLLM proxy (LLM_API_BASE is set), all non-local models
    require OPENAI_API_KEY since they route through the OpenAI-compatible proxy.

    When not using a proxy, models require their native provider keys
    (ANTHROPIC_API_KEY for Claude, etc.).

    Args:
        model: The model name/identifier

    Raises:
        ConfigurationError: If the required API key is not set
    """
    model_lower = model.lower()

    # Local models never need API keys
    if model_lower.startswith(("ollama/", "ollama_chat/", "local/")):
        return

    # When using proxy, all models need OPENAI_API_KEY
    if is_using_proxy():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                f"Using LiteLLM proxy (LLM_API_BASE is set) requires OPENAI_API_KEY.\n\n"
                f"Set it using one of these methods:\n"
                f"  1. Environment variable: export OPENAI_API_KEY='your-proxy-key'\n"
                f"  2. .env file: Add OPENAI_API_KEY=your-proxy-key to your .env file\n\n"
                f"This key authenticates with your LiteLLM proxy, not directly with providers."
            )
        return

    # Not using proxy - need native provider key
    provider, env_var = get_provider_for_model(model)

    # No key required for local models
    if env_var is None:
        return

    api_key = os.getenv(env_var)

    if not api_key:
        raise ConfigurationError(
            f"Model '{model}' requires {env_var} to be set.\n\n"
            f"Set it using one of these methods:\n"
            f"  1. Environment variable: export {env_var}='your-api-key'\n"
            f"  2. .env file: Add {env_var}=your-api-key to your .env file\n\n"
            f"Alternatively, set up a LiteLLM proxy and configure LLM_API_BASE.\n"
            f"See .env.example for configuration options."
        )


def validate_model_name(model: str) -> None:
    """Validate that the model name is reasonable.

    Args:
        model: The model name/identifier

    Raises:
        ConfigurationError: If the model name appears invalid
    """
    if not model or not isinstance(model, str):
        raise ConfigurationError("Model name must be a non-empty string")

    # Check for common mistakes
    model_lower = model.lower()

    # GPT-5 doesn't exist yet
    if "gpt-5" in model_lower:
        raise ConfigurationError(
            f"Model '{model}' does not exist. "
            f"Did you mean 'gpt-4o' or 'gpt-4-turbo'?"
        )

    # Check for typos in common models
    typo_corrections = {
        "gpt4": "gpt-4",
        "gpt4o": "gpt-4o",
        "claude3": "claude-3",
        "claude-sonnet": "claude-3-5-sonnet-20241022",
        "claude-opus": "claude-3-opus-20240229",
    }

    for typo, correction in typo_corrections.items():
        if model_lower == typo:
            raise ConfigurationError(
                f"Model '{model}' appears to be a typo. "
                f"Did you mean '{correction}'?"
            )


def validate_configuration(model: Optional[str]) -> None:
    """Validate all configuration for running with the given model.

    Args:
        model: The model name/identifier (must be non-empty; use get_default_model()
               before calling if the value can be None)

    Raises:
        ConfigurationError: If any configuration is invalid
    """
    if not model or not isinstance(model, str):
        raise ConfigurationError(
            "Model name must be set. Use DSAGENT_DEFAULT_MODEL or LLM_MODEL, "
            "or pass --model. Run 'dsagent init' to configure."
        )
    apply_llm_api_base(model)
    # validate_model_name(model)
    validate_api_key(model)
