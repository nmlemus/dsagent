"""Kernel state endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status

from dsagent.server.deps import (
    get_connection_manager,
    get_session_manager,
    verify_api_key,
)
from dsagent.server.manager import AgentConnectionManager
from dsagent.server.models import (
    DataFrameInfoResponse,
    ErrorResponse,
    ExecuteCodeRequest,
    ExecutionResultResponse,
    KernelStateResponse,
    KernelVariableResponse,
)
from dsagent.session import SessionManager

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get(
    "/sessions/{session_id}/kernel",
    response_model=KernelStateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_kernel_state(
    session_id: str,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> KernelStateResponse:
    """Get the current state of the kernel.

    Args:
        session_id: Session ID
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        Kernel state including variables, dataframes, and imports

    Raises:
        HTTPException: If session not found or kernel not available
    """
    # Check session exists
    session = session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get agent
    agent = connection_manager.get_agent(session_id)

    if agent is None:
        # Return state from session snapshot if no active agent
        if session.kernel_snapshot:
            return _snapshot_to_response(session.kernel_snapshot)
        return KernelStateResponse(is_running=False)

    try:
        # Get current kernel state from agent
        state = await asyncio.to_thread(agent.get_kernel_state)
        if state:
            # Convert dict to KernelSnapshot if needed
            from dsagent.session.models import KernelSnapshot
            if isinstance(state, dict):
                snapshot = KernelSnapshot(
                    variables=state.get("variables", {}),
                    dataframes=state.get("dataframes", {}),
                    imports=state.get("imports", []),
                )
            else:
                snapshot = state
            return _snapshot_to_response(snapshot, is_running=True)
        return KernelStateResponse(is_running=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting kernel state: {str(e)}",
        )


@router.get(
    "/sessions/{session_id}/kernel/variables",
    response_model=list[KernelVariableResponse],
    responses={404: {"model": ErrorResponse}},
)
async def get_kernel_variables(
    session_id: str,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> list[KernelVariableResponse]:
    """Get list of variables in the kernel.

    Args:
        session_id: Session ID
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        List of variables with their types

    Raises:
        HTTPException: If session not found
    """
    state = await get_kernel_state(session_id, connection_manager, session_manager)
    return state.variables


@router.post(
    "/sessions/{session_id}/kernel/execute",
    response_model=ExecutionResultResponse,
    responses={404: {"model": ErrorResponse}},
)
async def execute_code(
    session_id: str,
    request: ExecuteCodeRequest,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> ExecutionResultResponse:
    """Execute code directly in the kernel.

    This bypasses the LLM and executes code directly in the IPython kernel.

    Args:
        session_id: Session ID
        request: Code to execute
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        Execution result

    Raises:
        HTTPException: If session not found or execution error
    """
    # Check session exists
    if session_manager.load_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get or create agent
    agent = await connection_manager.get_or_create_agent(session_id)

    try:
        return await connection_manager.execute_code(session_id, request.code)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/kernel/reset",
    response_model=KernelStateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def reset_kernel(
    session_id: str,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> KernelStateResponse:
    """Reset the kernel to a clean state.

    This clears all variables and imports.

    Args:
        session_id: Session ID
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        New kernel state (should be empty)

    Raises:
        HTTPException: If session not found or reset error
    """
    # Check session exists
    if session_manager.load_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get agent
    agent = connection_manager.get_agent(session_id)

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active agent for this session",
        )

    try:
        # Reset kernel
        await asyncio.to_thread(agent.reset_kernel)

        # Return new state
        return await get_kernel_state(session_id, connection_manager, session_manager)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting kernel: {str(e)}",
        )


def _snapshot_to_response(
    snapshot: "KernelSnapshot",
    is_running: bool = False,
) -> KernelStateResponse:
    """Convert a KernelSnapshot to KernelStateResponse.

    Args:
        snapshot: Kernel snapshot from session
        is_running: Whether the kernel is currently running

    Returns:
        API response model
    """
    from dsagent.session.models import KernelSnapshot

    variables = []
    if snapshot.variables:
        for name, var_type in snapshot.variables.items():
            variables.append(
                KernelVariableResponse(
                    name=name,
                    type=var_type,
                    value_preview=None,  # We don't store previews in snapshot
                )
            )

    dataframes = []
    if snapshot.dataframes:
        for name, info in snapshot.dataframes.items():
            dataframes.append(
                DataFrameInfoResponse(
                    name=name,
                    shape=info.get("shape", [0, 0]),
                    columns=info.get("columns", []),
                    dtypes=info.get("dtypes", {}),
                    memory_mb=info.get("memory_mb", 0.0),
                )
            )

    return KernelStateResponse(
        variables=variables,
        dataframes=dataframes,
        imports=snapshot.imports or [],
        is_running=is_running,
    )
