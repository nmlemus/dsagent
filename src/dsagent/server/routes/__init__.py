"""API Routes for DSAgent Server."""

from dsagent.server.routes.artifacts import router as artifacts_router
from dsagent.server.routes.chat import router as chat_router
from dsagent.server.routes.files import router as files_router
from dsagent.server.routes.health import router as health_router
from dsagent.server.routes.hitl import router as hitl_router
from dsagent.server.routes.kernel import router as kernel_router
from dsagent.server.routes.sessions import router as sessions_router

__all__ = [
    "artifacts_router",
    "chat_router",
    "files_router",
    "health_router",
    "hitl_router",
    "kernel_router",
    "sessions_router",
]
