"""FastAPI Dependencies for DSAgent Server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import Depends, Header, HTTPException, Query, status

from dsagent.config import DSAgentSettings, get_settings as _get_settings

if TYPE_CHECKING:
    from dsagent.server.manager import AgentConnectionManager
    from dsagent.session import SessionManager

# Single source of truth: server uses centralized config from dsagent.config
# Alias for backward compatibility in type hints (e.g. ServerSettings = Depends(get_settings))
ServerSettings = DSAgentSettings


def get_settings() -> DSAgentSettings:
    """Get cached server settings (delegates to config.get_settings)."""
    return _get_settings()


def set_connection_manager(manager: "AgentConnectionManager") -> None:
    """Set the global connection manager (called on startup)."""
    global _connection_manager
    _connection_manager = manager


def set_session_manager(manager: "SessionManager") -> None:
    """Set the global session manager (called on startup)."""
    global _session_manager
    _session_manager = manager


def get_connection_manager() -> "AgentConnectionManager":
    """Get the connection manager.

    Raises:
        HTTPException: If manager not initialized
    """
    if _connection_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not fully initialized",
        )
    return _connection_manager


def get_session_manager() -> "SessionManager":
    """Get the session manager.

    Raises:
        HTTPException: If manager not initialized
    """
    if _session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not fully initialized",
        )
    return _session_manager


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: DSAgentSettings = Depends(get_settings),
) -> Optional[str]:
    """Verify API key if authentication is enabled.

    Args:
        x_api_key: API key from header
        settings: Server settings

    Returns:
        The API key if valid

    Raises:
        HTTPException: If authentication fails
    """
    # If no API key configured, auth is disabled (dev mode)
    if not settings.api_key:
        return None

    # API key required but not provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify API key
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


async def verify_websocket_api_key(
    api_key: Optional[str] = Query(None),
    settings: DSAgentSettings = Depends(get_settings),
) -> Optional[str]:
    """Verify API key for WebSocket connections (via query param).

    Args:
        api_key: API key from query parameter
        settings: Server settings

    Returns:
        The API key if valid

    Raises:
        HTTPException: If authentication fails
    """
    # If no API key configured, auth is disabled (dev mode)
    if not settings.api_key:
        return None

    # API key required but not provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required (pass as ?api_key=xxx)",
        )

    # Verify API key
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key
