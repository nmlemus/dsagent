"""FastAPI Application for DSAgent Server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dsagent.server.deps import (
    ServerSettings,
    get_settings,
    set_connection_manager,
    set_session_manager,
)
from dsagent.server.manager import AgentConnectionManager
from dsagent.session import SessionManager

logger = logging.getLogger(__name__)

# API version
API_VERSION = "0.6.2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown of the server.
    """
    settings = get_settings()
    logger.info(f"Starting DSAgent Server v{API_VERSION}")

    # Create sessions directory
    sessions_dir = Path(settings.sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SessionManager
    session_manager = SessionManager(
        workspace_path=sessions_dir,
        backend=settings.session_backend,
    )
    set_session_manager(session_manager)
    logger.info(f"SessionManager initialized with {settings.session_backend} backend")

    # Initialize ConnectionManager
    connection_manager = AgentConnectionManager(
        session_manager=session_manager,
        default_model=settings.default_model,
        default_hitl_mode=settings.default_hitl_mode,
    )
    set_connection_manager(connection_manager)
    logger.info("ConnectionManager initialized")

    # Store in app state for access in routes
    app.state.session_manager = session_manager
    app.state.connection_manager = connection_manager
    app.state.settings = settings

    yield

    # Shutdown
    logger.info("Shutting down DSAgent Server...")
    await connection_manager.shutdown_all()
    logger.info("Server shutdown complete")


def create_app(
    settings: Optional[ServerSettings] = None,
    include_routes: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override
        include_routes: Whether to include API routes (False for testing)

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="DSAgent API",
        description="WebSocket and REST API for DSAgent Conversational Data Science Agent",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if include_routes:
        # Import and include routers
        from dsagent.server.routes.artifacts import router as artifacts_router
        from dsagent.server.routes.chat import router as chat_router
        from dsagent.server.routes.files import router as files_router
        from dsagent.server.routes.health import router as health_router
        from dsagent.server.routes.hitl import router as hitl_router
        from dsagent.server.routes.kernel import router as kernel_router
        from dsagent.server.routes.sessions import router as sessions_router
        from dsagent.server.websocket import router as websocket_router

        # Health endpoints (no auth)
        app.include_router(health_router, tags=["Health"])

        # API routes
        app.include_router(sessions_router, prefix="/api", tags=["Sessions"])
        app.include_router(chat_router, prefix="/api", tags=["Chat"])
        app.include_router(kernel_router, prefix="/api", tags=["Kernel"])
        app.include_router(files_router, prefix="/api", tags=["Files"])
        app.include_router(artifacts_router, prefix="/api", tags=["Artifacts"])
        app.include_router(hitl_router, prefix="/api", tags=["HITL"])

        # WebSocket route
        app.include_router(websocket_router, tags=["WebSocket"])

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Run the server with uvicorn.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    import uvicorn

    uvicorn.run(
        "dsagent.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
