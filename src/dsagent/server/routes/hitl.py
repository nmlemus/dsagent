"""HITL (Human-in-the-Loop) API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dsagent.server.deps import get_connection_manager, verify_api_key
from dsagent.server.manager import AgentConnectionManager


router = APIRouter(dependencies=[Depends(verify_api_key)])


# =============================================================================
# Request/Response Models
# =============================================================================


class HITLStatusResponse(BaseModel):
    """Status of HITL for a session."""

    enabled: bool = Field(..., description="Whether HITL is enabled")
    mode: Optional[str] = Field(None, description="HITL mode if enabled")
    awaiting_feedback: bool = Field(..., description="Whether agent is waiting for feedback")
    awaiting_type: Optional[str] = Field(
        None, description="Type of feedback awaited (plan, code, error, answer)"
    )
    pending_plan: Optional[dict] = Field(None, description="Plan awaiting approval")
    pending_code: Optional[str] = Field(None, description="Code awaiting approval")
    pending_error: Optional[str] = Field(None, description="Error needing guidance")
    pending_answer: Optional[str] = Field(None, description="Answer awaiting approval")


class HITLActionRequest(BaseModel):
    """Request to perform a HITL action."""

    action: str = Field(
        ...,
        description="Action to perform: approve, reject, modify, retry, skip, feedback",
    )
    message: Optional[str] = Field(None, description="Optional message/feedback")
    modified_plan: Optional[str] = Field(None, description="Modified plan (for modify action)")
    modified_code: Optional[str] = Field(None, description="Modified code (for modify action)")


class HITLActionResponse(BaseModel):
    """Response from a HITL action."""

    success: bool = Field(..., description="Whether the action was successful")
    message: str = Field(..., description="Status message")


# =============================================================================
# Routes
# =============================================================================


@router.get("/sessions/{session_id}/hitl/status", response_model=HITLStatusResponse)
async def get_hitl_status(
    session_id: str,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> HITLStatusResponse:
    """Get HITL status for a session.

    Returns whether HITL is enabled, the mode, and any pending items awaiting feedback.
    """
    agent = connection_manager.get_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No agent found for session {session_id}")

    hitl = agent.hitl
    if not hitl:
        return HITLStatusResponse(
            enabled=False,
            mode=None,
            awaiting_feedback=False,
            awaiting_type=None,
        )

    pending = hitl.get_pending_state()

    # Convert PlanState to dict if present
    pending_plan_dict = None
    if pending.get("pending_plan"):
        plan = pending["pending_plan"]
        pending_plan_dict = {
            "raw_text": getattr(plan, "raw_text", str(plan)),
            "steps": [
                {"number": s.number, "description": s.description, "completed": s.completed}
                for s in getattr(plan, "steps", [])
            ],
        }

    return HITLStatusResponse(
        enabled=hitl.is_enabled,
        mode=hitl.mode.value if hitl.mode else None,
        awaiting_feedback=hitl.is_awaiting_feedback,
        awaiting_type=pending.get("awaiting_type"),
        pending_plan=pending_plan_dict,
        pending_code=pending.get("pending_code"),
        pending_error=pending.get("pending_error"),
        pending_answer=pending.get("pending_answer"),
    )


@router.post("/sessions/{session_id}/hitl/respond", response_model=HITLActionResponse)
async def respond_to_hitl(
    session_id: str,
    request: HITLActionRequest,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> HITLActionResponse:
    """Respond to a HITL request.

    Actions:
    - approve: Approve the pending plan/code/answer
    - reject: Reject and abort the task
    - modify: Provide a modified plan or code
    - retry: Retry the failed operation
    - skip: Skip the current step
    - feedback: Send textual feedback to the agent
    """
    agent = connection_manager.get_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No agent found for session {session_id}")

    hitl = agent.hitl
    if not hitl:
        raise HTTPException(
            status_code=400, detail="HITL is not enabled for this session"
        )

    if not hitl.is_awaiting_feedback:
        raise HTTPException(
            status_code=400, detail="Agent is not awaiting feedback"
        )

    action = request.action.lower()

    if action == "approve":
        hitl.approve(request.message)
        return HITLActionResponse(success=True, message="Approved")

    elif action == "reject":
        hitl.reject(request.message)
        return HITLActionResponse(success=True, message="Rejected - task aborted")

    elif action == "modify":
        if request.modified_plan:
            hitl.modify_plan(request.modified_plan, request.message)
            return HITLActionResponse(success=True, message="Plan modification accepted")
        elif request.modified_code:
            hitl.modify_code(request.modified_code, request.message)
            return HITLActionResponse(success=True, message="Code modification accepted")
        else:
            raise HTTPException(
                status_code=400,
                detail="Modify action requires modified_plan or modified_code",
            )

    elif action == "retry":
        hitl.retry(request.message)
        return HITLActionResponse(success=True, message="Retrying operation")

    elif action == "skip":
        hitl.skip(request.message)
        return HITLActionResponse(success=True, message="Step skipped")

    elif action == "feedback":
        if not request.message:
            raise HTTPException(
                status_code=400, detail="Feedback action requires a message"
            )
        hitl.send_feedback(request.message)
        return HITLActionResponse(success=True, message="Feedback sent")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@router.post("/sessions/{session_id}/hitl/approve", response_model=HITLActionResponse)
async def approve_hitl(
    session_id: str,
    message: Optional[str] = None,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> HITLActionResponse:
    """Quick endpoint to approve pending HITL request."""
    agent = connection_manager.get_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No agent found for session {session_id}")

    hitl = agent.hitl
    if not hitl:
        raise HTTPException(status_code=400, detail="HITL is not enabled for this session")

    if not hitl.is_awaiting_feedback:
        raise HTTPException(status_code=400, detail="Agent is not awaiting feedback")

    hitl.approve(message)
    return HITLActionResponse(success=True, message="Approved")


@router.post("/sessions/{session_id}/hitl/reject", response_model=HITLActionResponse)
async def reject_hitl(
    session_id: str,
    message: Optional[str] = None,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
) -> HITLActionResponse:
    """Quick endpoint to reject pending HITL request."""
    agent = connection_manager.get_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No agent found for session {session_id}")

    hitl = agent.hitl
    if not hitl:
        raise HTTPException(status_code=400, detail="HITL is not enabled for this session")

    if not hitl.is_awaiting_feedback:
        raise HTTPException(status_code=400, detail="Agent is not awaiting feedback")

    hitl.reject(message)
    return HITLActionResponse(success=True, message="Rejected - task aborted")
