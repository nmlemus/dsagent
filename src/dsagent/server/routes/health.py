"""Health check endpoints.

These endpoints are intentionally unauthenticated so load balancers and
orchestrators can perform health/readiness checks. They expose version
and component status. If you need to hide version in a given environment,
consider a minimal health endpoint or optional auth.
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from dsagent.server.deps import get_connection_manager, get_session_manager
from dsagent.server.models import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint (no authentication).

    Returns:
        Health status with version and timestamp.
    """
    from dsagent.server.app import API_VERSION

    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness check endpoint.

    Verifies that all required services are initialized and ready.

    Returns:
        Readiness status with individual component checks
    """
    checks = {}

    # Check session manager
    try:
        session_manager = get_session_manager()
        checks["session_manager"] = session_manager is not None
    except Exception:
        checks["session_manager"] = False

    # Check connection manager
    try:
        connection_manager = get_connection_manager()
        checks["connection_manager"] = connection_manager is not None
    except Exception:
        checks["connection_manager"] = False

    # Overall readiness
    ready = all(checks.values())

    return ReadinessResponse(
        ready=ready,
        checks=checks,
    )
