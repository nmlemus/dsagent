"""Session management endpoints."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse

from dsagent.server.deps import (
    get_connection_manager,
    get_session_manager,
    verify_api_key,
)
from dsagent.server.manager import AgentConnectionManager
from dsagent.server.models import (
    CreateSessionRequest,
    ErrorResponse,
    SessionListResponse,
    SessionResponse,
    UpdateSessionRequest,
)
from dsagent.session import Session, SessionManager, SessionStatus

router = APIRouter(dependencies=[Depends(verify_api_key)])


def _session_to_response(session) -> SessionResponse:
    """Convert a Session (object or dict) to SessionResponse."""
    # Handle dict from list_sessions
    if isinstance(session, dict):
        return SessionResponse(
            id=session.get("id", ""),
            name=session.get("name"),
            status=session.get("status", "active"),
            created_at=session.get("created_at", datetime.utcnow()),
            updated_at=session.get("updated_at", datetime.utcnow()),
            message_count=session.get("message_count", 0),
            kernel_variables=0,
            workspace_path=session.get("workspace_path"),
        )

    # Handle Session object
    kernel_vars = 0
    if hasattr(session, "kernel_snapshot") and session.kernel_snapshot:
        kernel_vars = len(session.kernel_snapshot.variables or {})

    return SessionResponse(
        id=session.id,
        name=session.name,
        status=session.status.value if hasattr(session.status, "value") else str(session.status),
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.history.messages) if session.history else 0,
        kernel_variables=kernel_vars,
        workspace_path=str(session.workspace_path) if session.workspace_path else None,
    )


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> SessionResponse:
    """Create a new session.

    Args:
        request: Session creation parameters
        session_manager: Session manager instance
        connection_manager: Connection manager instance

    Returns:
        Created session information
    """
    # Create session
    session = session_manager.create_session(name=request.name)

    # Store agent configuration in session for persistence
    session.model = request.model
    session.hitl_mode = request.hitl_mode or "none"
    session_manager.save_session(session)

    # Create and start agent for this session
    await connection_manager.get_or_create_agent(
        session.id,
        model=request.model,
        hitl_mode=request.hitl_mode,
    )

    return _session_to_response(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=100),
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionListResponse:
    """List all sessions.

    Args:
        status_filter: Optional status filter (active, paused, completed, error, archived)
        limit: Maximum number of sessions to return
        session_manager: Session manager instance

    Returns:
        List of sessions
    """
    # Convert status filter
    session_status = None
    if status_filter:
        try:
            session_status = SessionStatus(status_filter)
        except ValueError:
            pass

    sessions = session_manager.list_sessions(status=session_status, limit=limit)

    return SessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=len(sessions),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Get a specific session.

    Args:
        session_id: Session ID
        session_manager: Session manager instance

    Returns:
        Session information

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return _session_to_response(session)


@router.put(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> SessionResponse:
    """Update a session.

    Args:
        session_id: Session ID
        request: Update parameters
        session_manager: Session manager instance
        connection_manager: Connection manager instance

    Returns:
        Updated session information

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Update fields
    if request.name is not None:
        session.name = request.name

    if request.status is not None:
        try:
            session.status = SessionStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {request.status}",
            )

    # Update HITL mode
    if request.hitl_mode is not None:
        valid_modes = ["none", "plan_only", "plan", "on_error", "plan_and_answer", "full"]
        if request.hitl_mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hitl_mode: {request.hitl_mode}. Valid: {valid_modes}",
            )
        session.hitl_mode = request.hitl_mode
        # Update running agent if exists
        connection_manager.set_agent_hitl_mode(session_id, request.hitl_mode)

    # Save session
    session.touch()
    session_manager.save_session(session)

    return _session_to_response(session)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> None:
    """Delete a session.

    Args:
        session_id: Session ID
        session_manager: Session manager instance
        connection_manager: Connection manager instance

    Raises:
        HTTPException: If session not found
    """
    if session_manager.load_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Shutdown agent if running
    await connection_manager.shutdown_agent(session_id)

    # Delete session
    session_manager.delete_session(session_id)


@router.post(
    "/sessions/{session_id}/archive",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def archive_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> SessionResponse:
    """Archive a session.

    Args:
        session_id: Session ID
        session_manager: Session manager instance
        connection_manager: Connection manager instance

    Returns:
        Archived session information

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Shutdown agent if running
    await connection_manager.shutdown_agent(session_id)

    # Archive session
    session_manager.archive_session(session_id)

    # Reload to get updated status
    session = session_manager.load_session(session_id)
    return _session_to_response(session)


@router.get(
    "/sessions/{session_id}/notebook",
    responses={404: {"model": ErrorResponse}},
)
async def export_notebook(
    session_id: str,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Export session as Jupyter notebook.

    Args:
        session_id: Session ID
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        Jupyter notebook file

    Raises:
        HTTPException: If session not found or notebook not available
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Check if agent has a notebook
    agent = connection_manager.get_agent(session_id)

    if agent:
        # Export notebook from running agent
        import asyncio
        notebook_path = await asyncio.to_thread(agent.export_notebook)
        if notebook_path and notebook_path.exists():
            return FileResponse(
                path=notebook_path,
                media_type="application/x-ipynb+json",
                filename=f"{session_id}.ipynb",
            )

    # Try to find existing notebook in session workspace
    if session.notebooks_path:
        notebooks_dir = Path(session.notebooks_path)
        if notebooks_dir.exists():
            notebooks = list(notebooks_dir.glob("*.ipynb"))
            if notebooks:
                # Return most recent notebook
                notebook = max(notebooks, key=lambda p: p.stat().st_mtime)
                return FileResponse(
                    path=notebook,
                    media_type="application/x-ipynb+json",
                    filename=notebook.name,
                )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No notebook available for this session",
    )


@router.get(
    "/sessions/{session_id}/export",
    responses={404: {"model": ErrorResponse}},
)
async def export_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> JSONResponse:
    """Export full session data as JSON.

    Args:
        session_id: Session ID
        session_manager: Session manager instance

    Returns:
        Session data as JSON

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Export session as JSON
    return JSONResponse(
        content=session.model_dump(mode="json"),
        headers={
            "Content-Disposition": f'attachment; filename="{session_id}.json"'
        },
    )
