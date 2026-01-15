"""API Request and Response Models for DSAgent Server."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class WebSocketEventType(str, Enum):
    """Types of events sent via WebSocket."""

    CONNECTED = "connected"
    THINKING = "thinking"
    PLAN = "plan"
    CODE_EXECUTING = "code_executing"
    CODE_RESULT = "code_result"
    RESPONSE = "response"
    ANSWER = "answer"
    ERROR = "error"
    COMPLETE = "complete"
    DISCONNECTED = "disconnected"
    # HITL events
    HITL_REQUEST = "hitl_request"  # Server requesting human input
    HITL_RESPONSE = "hitl_response"  # Confirmation that feedback was received


class WebSocketMessageType(str, Enum):
    """Types of messages received via WebSocket from client."""

    CHAT = "chat"
    EXECUTE = "execute"
    APPROVE = "approve"
    CANCEL = "cancel"


# =============================================================================
# Request Models
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    name: Optional[str] = Field(None, description="Optional name for the session")
    model: Optional[str] = Field(None, description="LLM model to use")
    hitl_mode: Optional[str] = Field("none", description="Human-in-the-loop mode")


class UpdateSessionRequest(BaseModel):
    """Request to update a session."""

    name: Optional[str] = Field(None, description="New name for the session")
    status: Optional[str] = Field(None, description="New status (active, paused, completed)")
    hitl_mode: Optional[str] = Field(None, description="HITL mode (none, plan_only, on_error, plan_and_answer, full)")


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., description="The message to send to the agent")


class ExecuteCodeRequest(BaseModel):
    """Request to execute code directly in the kernel."""

    code: str = Field(..., description="Python code to execute")


class WebSocketMessage(BaseModel):
    """Message received via WebSocket from client."""

    type: WebSocketMessageType = Field(..., description="Type of message")
    content: Optional[str] = Field(None, description="Message content (for chat)")
    code: Optional[str] = Field(None, description="Code to execute (for execute)")
    approved: Optional[bool] = Field(None, description="Approval status (for approve)")
    feedback: Optional[str] = Field(None, description="Optional feedback message")
    # Extended HITL fields
    action: Optional[str] = Field(
        None,
        description="HITL action: approve, reject, modify, retry, skip, feedback",
    )
    modified_plan: Optional[str] = Field(None, description="Modified plan text")
    modified_code: Optional[str] = Field(None, description="Modified code")


# =============================================================================
# Response Models - Nested Components
# =============================================================================


class PlanStepResponse(BaseModel):
    """A single step in a plan."""

    number: int = Field(..., description="Step number")
    description: str = Field(..., description="Step description")
    completed: bool = Field(False, description="Whether the step is completed")


class PlanResponse(BaseModel):
    """Plan state response."""

    steps: List[PlanStepResponse] = Field(default_factory=list)
    raw_text: str = Field("", description="Raw plan text")
    total_steps: int = Field(0, description="Total number of steps")
    completed_steps: int = Field(0, description="Number of completed steps")
    is_complete: bool = Field(False, description="Whether all steps are completed")


class ExecutionResultResponse(BaseModel):
    """Result of code execution."""

    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    error: Optional[str] = Field(None, description="Error message if any")
    images: List[Dict[str, str]] = Field(
        default_factory=list, description="Base64 encoded images"
    )
    success: bool = Field(True, description="Whether execution was successful")


class KernelVariableResponse(BaseModel):
    """Information about a kernel variable."""

    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Variable type")
    value_preview: Optional[str] = Field(None, description="Preview of value")


class DataFrameInfoResponse(BaseModel):
    """Information about a DataFrame in the kernel."""

    name: str = Field(..., description="DataFrame variable name")
    shape: List[int] = Field(..., description="Shape [rows, columns]")
    columns: List[str] = Field(default_factory=list, description="Column names")
    dtypes: Dict[str, str] = Field(default_factory=dict, description="Column dtypes")
    memory_mb: float = Field(0.0, description="Memory usage in MB")


class KernelStateResponse(BaseModel):
    """State of the kernel."""

    variables: List[KernelVariableResponse] = Field(default_factory=list)
    dataframes: List[DataFrameInfoResponse] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list, description="Imported modules")
    is_running: bool = Field(True, description="Whether kernel is running")


class MessageResponse(BaseModel):
    """A conversation message."""

    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user, assistant, execution, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="When the message was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Response Models - Main
# =============================================================================


class SessionResponse(BaseModel):
    """Response for session information."""

    id: str = Field(..., description="Session ID")
    name: Optional[str] = Field(None, description="Session name")
    status: str = Field(..., description="Session status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(0, description="Number of messages in history")
    kernel_variables: int = Field(0, description="Number of variables in kernel")
    workspace_path: Optional[str] = Field(None, description="Path to session workspace")


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: List[SessionResponse] = Field(default_factory=list)
    total: int = Field(0, description="Total number of sessions")


class ChatResponseModel(BaseModel):
    """Response from chat endpoint."""

    content: str = Field(..., description="Response content")
    code: Optional[str] = Field(None, description="Extracted code if any")
    execution_result: Optional[ExecutionResultResponse] = Field(
        None, description="Code execution result"
    )
    plan: Optional[PlanResponse] = Field(None, description="Plan if extracted")
    has_answer: bool = Field(False, description="Whether response contains final answer")
    answer: Optional[str] = Field(None, description="Final answer if present")
    thinking: Optional[str] = Field(None, description="Agent's thinking process")
    is_complete: bool = Field(False, description="Whether the task is complete")


class MessagesResponse(BaseModel):
    """Response for message history."""

    messages: List[MessageResponse] = Field(default_factory=list)
    total: int = Field(0, description="Total number of messages")
    has_more: bool = Field(False, description="Whether there are more messages")


class ConversationTurn(BaseModel):
    """A single turn in the conversation (user message + agent response).

    This matches the structure of `round_complete` SSE events so the UI
    can render historical messages the same way as live streaming.
    """

    round: int = Field(..., description="Turn/round number")
    timestamp: datetime = Field(..., description="Timestamp of the turn")

    # User message (null for autonomous continuation rounds)
    user_message: Optional[str] = Field(None, description="User's message if any")

    # Agent response - same structure as ChatResponseModel
    content: str = Field(..., description="Full LLM response text")
    code: Optional[str] = Field(None, description="Extracted code if any")
    execution_result: Optional[ExecutionResultResponse] = Field(
        None, description="Code execution result"
    )
    plan: Optional[PlanResponse] = Field(None, description="Plan if extracted")
    has_answer: bool = Field(False, description="Whether response contains final answer")
    answer: Optional[str] = Field(None, description="Final answer if present")
    thinking: Optional[str] = Field(None, description="Agent's thinking process")
    is_complete: bool = Field(False, description="Whether the task is complete")


class TurnsResponse(BaseModel):
    """Response for conversation turns (structured history)."""

    turns: List[ConversationTurn] = Field(default_factory=list)
    total: int = Field(0, description="Total number of turns")
    has_more: bool = Field(False, description="Whether there are more turns")


# =============================================================================
# WebSocket Event Models
# =============================================================================


class WebSocketEvent(BaseModel):
    """Event sent via WebSocket to client."""

    type: WebSocketEventType = Field(..., description="Event type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Event timestamp"
    )
    session_id: Optional[str] = Field(None, description="Session ID")

    @classmethod
    def connected(cls, session_id: str, message: str = "Connected") -> "WebSocketEvent":
        """Create a connected event."""
        return cls(
            type=WebSocketEventType.CONNECTED,
            data={"message": message},
            session_id=session_id,
        )

    @classmethod
    def thinking(cls, session_id: str) -> "WebSocketEvent":
        """Create a thinking event."""
        return cls(
            type=WebSocketEventType.THINKING,
            data={"message": "Processing..."},
            session_id=session_id,
        )

    @classmethod
    def plan(cls, session_id: str, plan: PlanResponse) -> "WebSocketEvent":
        """Create a plan event."""
        return cls(
            type=WebSocketEventType.PLAN,
            data=plan.model_dump(),
            session_id=session_id,
        )

    @classmethod
    def code_executing(cls, session_id: str, code: str) -> "WebSocketEvent":
        """Create a code executing event."""
        return cls(
            type=WebSocketEventType.CODE_EXECUTING,
            data={"code": code},
            session_id=session_id,
        )

    @classmethod
    def code_result(
        cls, session_id: str, result: ExecutionResultResponse
    ) -> "WebSocketEvent":
        """Create a code result event."""
        return cls(
            type=WebSocketEventType.CODE_RESULT,
            data=result.model_dump(),
            session_id=session_id,
        )

    @classmethod
    def response(cls, session_id: str, content: str, is_partial: bool = False) -> "WebSocketEvent":
        """Create a response event."""
        return cls(
            type=WebSocketEventType.RESPONSE,
            data={"content": content, "is_partial": is_partial},
            session_id=session_id,
        )

    @classmethod
    def answer(cls, session_id: str, answer: str) -> "WebSocketEvent":
        """Create an answer event."""
        return cls(
            type=WebSocketEventType.ANSWER,
            data={"answer": answer},
            session_id=session_id,
        )

    @classmethod
    def error(cls, session_id: str, error: str, code: Optional[str] = None) -> "WebSocketEvent":
        """Create an error event."""
        return cls(
            type=WebSocketEventType.ERROR,
            data={"error": error, "code": code},
            session_id=session_id,
        )

    @classmethod
    def complete(cls, session_id: str) -> "WebSocketEvent":
        """Create a complete event."""
        return cls(
            type=WebSocketEventType.COMPLETE,
            data={"message": "Task completed"},
            session_id=session_id,
        )

    @classmethod
    def hitl_request(
        cls,
        session_id: str,
        request_type: str,
        plan: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
        error: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> "WebSocketEvent":
        """Create a HITL request event.

        Args:
            session_id: Session ID
            request_type: Type of request (plan, code, error, answer)
            plan: Plan data if requesting plan approval
            code: Code if requesting code approval or showing error context
            error: Error message if requesting error guidance
            answer: Answer if requesting answer approval
        """
        return cls(
            type=WebSocketEventType.HITL_REQUEST,
            data={
                "request_type": request_type,
                "plan": plan,
                "code": code,
                "error": error,
                "answer": answer,
            },
            session_id=session_id,
        )

    @classmethod
    def hitl_response(
        cls, session_id: str, accepted: bool, message: Optional[str] = None
    ) -> "WebSocketEvent":
        """Create a HITL response confirmation event."""
        return cls(
            type=WebSocketEventType.HITL_RESPONSE,
            data={"accepted": accepted, "message": message},
            session_id=session_id,
        )


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok", description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Check timestamp"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(True, description="Whether the service is ready")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual checks")


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
