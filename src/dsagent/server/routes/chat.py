"""Chat endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from dsagent.server.deps import (
    get_connection_manager,
    get_session_manager,
    verify_api_key,
)
from dsagent.server.manager import AgentConnectionManager
from dsagent.server.models import (
    ChatRequest,
    ChatResponseModel,
    ConversationTurn,
    ErrorResponse,
    ExecutionResultResponse,
    MessageResponse,
    MessagesResponse,
    PlanResponse,
    PlanStepResponse,
    TurnsResponse,
)
from dsagent.session import SessionManager
from dsagent.session.models import MessageRole

if TYPE_CHECKING:
    from dsagent.agents.conversational import ChatResponse
    from dsagent.schema.models import PlanState, ExecutionResult

router = APIRouter(dependencies=[Depends(verify_api_key)])


def _sanitize_arguments(
    arguments: Dict[str, Any],
    max_length: int = 500,
) -> Dict[str, Any]:
    """Sanitize tool arguments by redacting sensitive keys and truncating long values."""
    sensitive_keys = {
        "api_key", "apikey", "api-key",
        "password", "passwd", "pwd",
        "token", "access_token", "auth_token", "bearer",
        "secret", "secret_key", "secretkey",
        "credential", "credentials",
        "private_key", "privatekey",
        "authorization", "auth",
    }

    sanitized = {}
    for key, value in arguments.items():
        key_lower = key.lower()
        is_sensitive = any(sens in key_lower for sens in sensitive_keys)

        if is_sensitive:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > max_length:
            sanitized[key] = value[:max_length] + "..."
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_arguments(value, max_length)
        else:
            sanitized[key] = value

    return sanitized


def _convert_plan_to_dict(plan: "PlanState") -> Dict[str, Any]:
    """Convert PlanState to dictionary for JSON serialization."""
    steps = []
    if hasattr(plan, "steps"):
        for step in plan.steps:
            steps.append({
                "number": step.number,
                "description": step.description,
                "completed": step.completed,
            })
    return {
        "steps": steps,
        "raw_text": getattr(plan, "raw_text", ""),
        "total_steps": getattr(plan, "total_steps", len(steps)),
        "completed_steps": getattr(plan, "completed_steps", 0),
        "is_complete": getattr(plan, "is_complete", False),
    }


def _convert_execution_result_to_dict(result: "ExecutionResult") -> Dict[str, Any]:
    """Convert ExecutionResult to dictionary for JSON serialization."""
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
        "images": result.images or [],
        "success": result.success,
    }


# =============================================================================
# Helper functions for parsing message content
# =============================================================================

import re
from pathlib import Path


# =============================================================================
# Helper functions for reading events.jsonl
# =============================================================================


def _parse_events_jsonl(events_file: Path) -> list[Dict[str, Any]]:
    """Parse events.jsonl file and return list of events."""
    events = []
    if not events_file.exists():
        return events

    with open(events_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _events_to_turns(events: list[Dict[str, Any]]) -> list[ConversationTurn]:
    """Convert events from events.jsonl into conversation turns.

    A turn consists of:
    - Optional user_message (only at start of user interaction)
    - llm_response (the assistant's response)
    - Optional code_execution (if code was executed)
    - Optional plan_update (latest plan state)
    """
    turns: list[ConversationTurn] = []
    round_num = 0
    current_user_message: Optional[str] = None
    current_plan: Optional[PlanResponse] = None

    i = 0
    while i < len(events):
        event = events[i]
        event_type = event.get("type", "")
        data = event.get("data", {})
        timestamp_str = event.get("timestamp", "")

        # Track user messages
        if event_type == "user_message":
            current_user_message = data.get("message", "")
            i += 1
            continue

        # Track plan updates
        if event_type == "plan_update":
            plan_text = data.get("plan_text", "")
            completed_steps = data.get("completed_steps", 0)
            total_steps = data.get("total_steps", 0)

            # Parse plan steps from plan_text
            steps = []
            step_pattern = re.compile(r"(\d+)\.\s*\[([ xX])\]\s*(.+)")
            for line in plan_text.split("\n"):
                step_match = step_pattern.match(line.strip())
                if step_match:
                    number = int(step_match.group(1))
                    completed = step_match.group(2).lower() == "x"
                    description = step_match.group(3).strip()
                    steps.append(PlanStepResponse(
                        number=number,
                        description=description,
                        completed=completed,
                    ))

            if steps:
                current_plan = PlanResponse(
                    steps=steps,
                    raw_text=plan_text,
                    total_steps=total_steps,
                    completed_steps=completed_steps,
                    is_complete=completed_steps == total_steps and total_steps > 0,
                )
            i += 1
            continue

        # Process LLM response - this is the core of a turn
        if event_type == "llm_response":
            round_num += 1
            response_content = data.get("response", "")
            has_code = data.get("has_code", False)
            has_plan = data.get("has_plan", False)
            has_answer = data.get("has_answer", False)

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.utcnow()

            # Extract code from response
            code = _extract_code_from_content(response_content) if has_code else None

            # Extract answer from response
            answer = _extract_answer_from_content(response_content) if has_answer else None

            # Extract plan from response content (most accurate source)
            # Parse from content first, fallback to current_plan if parsing fails
            plan = None
            if has_plan:
                plan = _extract_plan_from_content(response_content)
            if plan is None and current_plan is not None:
                plan = current_plan

            # Look ahead for code_execution event
            execution_result = None
            j = i + 1
            while j < len(events):
                next_event = events[j]
                next_type = next_event.get("type", "")

                # Skip llm_request events
                if next_type == "llm_request":
                    j += 1
                    continue

                # Found code_execution
                if next_type == "code_execution":
                    exec_data = next_event.get("data", {})
                    success = exec_data.get("success", True)
                    output = exec_data.get("output", "")
                    error = exec_data.get("error")
                    images = exec_data.get("images", [])

                    execution_result = ExecutionResultResponse(
                        stdout=output if success else "",
                        stderr="",
                        error=error if not success else None,
                        images=images,
                        success=success,
                    )
                    i = j  # Move past code_execution
                    break

                # Found plan_update - process it and continue looking
                if next_type == "plan_update":
                    plan_data = next_event.get("data", {})
                    plan_text = plan_data.get("plan_text", "")
                    completed_steps = plan_data.get("completed_steps", 0)
                    total_steps = plan_data.get("total_steps", 0)

                    steps = []
                    step_pattern = re.compile(r"(\d+)\.\s*\[([ xX])\]\s*(.+)")
                    for line in plan_text.split("\n"):
                        step_match = step_pattern.match(line.strip())
                        if step_match:
                            number = int(step_match.group(1))
                            completed = step_match.group(2).lower() == "x"
                            description = step_match.group(3).strip()
                            steps.append(PlanStepResponse(
                                number=number,
                                description=description,
                                completed=completed,
                            ))

                    if steps:
                        current_plan = PlanResponse(
                            steps=steps,
                            raw_text=plan_text,
                            total_steps=total_steps,
                            completed_steps=completed_steps,
                            is_complete=completed_steps == total_steps and total_steps > 0,
                        )
                        plan = current_plan
                    j += 1
                    continue

                # Any other event type means no code_execution for this response
                break

            # Determine if this turn completed the task
            is_complete = has_answer and (plan is None or (plan and plan.is_complete))

            turn = ConversationTurn(
                round=round_num,
                timestamp=timestamp,
                user_message=current_user_message,
                content=response_content,
                code=code,
                execution_result=execution_result,
                plan=plan,
                has_answer=has_answer,
                answer=answer,
                thinking=None,
                is_complete=is_complete,
            )
            turns.append(turn)

            # Clear user message after first turn that uses it
            current_user_message = None
            i += 1
            continue

        # Skip other event types
        i += 1

    return turns


def _extract_code_from_content(content: str) -> Optional[str]:
    """Extract code from <code> tags in content."""
    match = re.search(r"<code>(.*?)(?:</code>|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_answer_from_content(content: str) -> Optional[str]:
    """Extract answer from <answer> tags in content."""
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_plan_from_content(content: str) -> Optional[PlanResponse]:
    """Extract and parse plan from <plan> tags in content."""
    match = re.search(r"<plan>(.*?)(?:</plan>|$)", content, re.DOTALL)
    if not match:
        return None

    raw_text = match.group(1).strip()
    steps = []
    completed_count = 0

    # Parse plan steps: "1. [x] Description" or "1. [ ] Description"
    step_pattern = re.compile(r"(\d+)\.\s*\[([ xX])\]\s*(.+)")
    for line in raw_text.split("\n"):
        step_match = step_pattern.match(line.strip())
        if step_match:
            number = int(step_match.group(1))
            completed = step_match.group(2).lower() == "x"
            description = step_match.group(3).strip()
            steps.append(PlanStepResponse(
                number=number,
                description=description,
                completed=completed,
            ))
            if completed:
                completed_count += 1

    if not steps:
        return None

    return PlanResponse(
        steps=steps,
        raw_text=raw_text,
        total_steps=len(steps),
        completed_steps=completed_count,
        is_complete=completed_count == len(steps),
    )


def _has_answer_tag(content: str) -> bool:
    """Check if content contains <answer> tag."""
    return "<answer>" in content


def _convert_chat_response(response: "ChatResponse") -> ChatResponseModel:
    """Convert agent ChatResponse to API ChatResponseModel."""
    # Convert execution result
    execution_result = None
    if response.execution_result:
        execution_result = ExecutionResultResponse(
            stdout=response.execution_result.stdout,
            stderr=response.execution_result.stderr,
            error=response.execution_result.error,
            images=response.execution_result.images,
            success=response.execution_result.success,
        )

    # Convert plan
    plan = None
    if response.plan:
        steps = []
        if hasattr(response.plan, "steps"):
            for step in response.plan.steps:
                steps.append(
                    PlanStepResponse(
                        number=step.number,
                        description=step.description,
                        completed=step.completed,
                    )
                )
        plan = PlanResponse(
            steps=steps,
            raw_text=getattr(response.plan, "raw_text", ""),
            total_steps=getattr(response.plan, "total_steps", len(steps)),
            completed_steps=getattr(response.plan, "completed_steps", 0),
            is_complete=getattr(response.plan, "is_complete", False),
        )

    return ChatResponseModel(
        content=response.content,
        code=response.code,
        execution_result=execution_result,
        plan=plan,
        has_answer=response.has_answer,
        answer=response.answer,
        thinking=response.thinking,
        is_complete=response.is_complete,
    )


@router.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponseModel,
    responses={404: {"model": ErrorResponse}},
)
async def chat(
    session_id: str,
    request: ChatRequest,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> ChatResponseModel:
    """Send a chat message and get the response.

    This is a synchronous endpoint that waits for the full response.
    For streaming responses, use /sessions/{id}/chat/stream.

    Args:
        session_id: Session ID
        request: Chat message
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        Chat response

    Raises:
        HTTPException: If session not found or agent error
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
        # Run chat in thread pool (blocking operation)
        response = await asyncio.to_thread(agent.chat, request.message)
        return _convert_chat_response(response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/chat/stream",
    responses={404: {"model": ErrorResponse}},
)
async def chat_stream(
    session_id: str,
    request: ChatRequest,
    connection_manager: AgentConnectionManager = Depends(get_connection_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """Send a chat message and stream granular SSE events.

    Server-Sent Events emitted (in order):
    - event: thinking        - LLM is processing
    - event: llm_response    - LLM response text received
    - event: plan            - Plan extracted from response
    - event: code_executing  - Code about to execute
    - event: code_result     - Code execution result
    - event: round_complete  - Full round data (for compatibility)
    - event: done            - Stream complete
    - event: error           - Error occurred

    Args:
        session_id: Session ID
        request: Chat message
        connection_manager: Connection manager instance
        session_manager: Session manager instance

    Returns:
        Streaming response with granular SSE events

    Raises:
        HTTPException: If session not found
    """
    # Check session exists
    if session_manager.load_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get or create agent
    agent = await connection_manager.get_or_create_agent(session_id)

    # Thread-safe queue for callback events
    event_queue: queue.Queue[Tuple[str, Dict[str, Any]]] = queue.Queue()
    round_num = 0

    # Define callbacks that put events in the queue
    def on_thinking():
        event_queue.put(("thinking", {"message": "Processing..."}))

    def on_llm_response(content: str):
        event_queue.put(("llm_response", {"content": content}))

    def on_plan_update(plan):
        event_queue.put(("plan", _convert_plan_to_dict(plan)))

    def on_code_executing(code: str):
        event_queue.put(("code_executing", {"code": code}))

    def on_code_result(result):
        event_queue.put(("code_result", _convert_execution_result_to_dict(result)))

    def on_hitl_request(request_type: str, plan, code: Optional[str], error: Optional[str]):
        """Called when HITL approval is needed."""
        event_queue.put(("hitl_request", {
            "request_type": request_type,
            "plan": _convert_plan_to_dict(plan) if plan else None,
            "code": code,
            "error": error,
        }))

    def on_tool_calling(tool_name: str, arguments: Dict[str, Any]):
        """Called when MCP tool is about to be executed."""
        sanitized = _sanitize_arguments(arguments)
        event_queue.put(("tool_calling", {
            "tool_name": tool_name,
            "arguments": sanitized,
        }))

    def on_tool_result(
        tool_name: str,
        success: bool,
        result: Optional[str],
        error: Optional[str],
        execution_time_ms: float,
    ):
        """Called when MCP tool execution completes."""
        event_type = "tool_success" if success else "tool_failed"
        # Truncate result for SSE
        truncated_result = result[:1000] + "..." if result and len(result) > 1000 else result
        event_queue.put((event_type, {
            "tool_name": tool_name,
            "success": success,
            "result": truncated_result,
            "error": error,
            "execution_time_ms": execution_time_ms,
        }))

    # Register callbacks
    agent.set_callbacks(
        on_thinking=on_thinking,
        on_llm_response=on_llm_response,
        on_plan_update=on_plan_update,
        on_code_executing=on_code_executing,
        on_code_result=on_code_result,
        on_hitl_request=on_hitl_request,
        on_tool_calling=on_tool_calling,
        on_tool_result=on_tool_result,
    )

    # Flag to signal completion
    stream_done = threading.Event()
    stream_error: Optional[Exception] = None

    def run_chat_stream():
        """Run chat_stream in a thread and put events in queue."""
        nonlocal round_num, stream_error
        try:
            for response in agent.chat_stream(request.message):
                round_num += 1
                api_response = _convert_chat_response(response)
                event_queue.put(("round_complete", {
                    "round": round_num,
                    **api_response.model_dump(),
                }))
            event_queue.put(("done", {}))
        except Exception as e:
            stream_error = e
            event_queue.put(("error", {"error": str(e)}))
        finally:
            stream_done.set()

    async def generate_events():
        """Generate SSE events from the queue."""
        # Start chat in background thread
        chat_thread = threading.Thread(target=run_chat_stream, daemon=True)
        chat_thread.start()

        try:
            while not stream_done.is_set() or not event_queue.empty():
                try:
                    # Non-blocking get with timeout to check stream_done
                    event_type, data = event_queue.get(timeout=0.1)
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

                    if event_type in ("done", "error"):
                        break
                except queue.Empty:
                    # No event yet, continue waiting
                    await asyncio.sleep(0.01)
                    continue

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

        finally:
            # Clear callbacks to avoid memory leaks
            agent.set_callbacks()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessagesResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    role: Optional[str] = Query(None),
    session_manager: SessionManager = Depends(get_session_manager),
) -> MessagesResponse:
    """Get conversation history for a session.

    Args:
        session_id: Session ID
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        role: Optional role filter (user, assistant, execution, system)
        session_manager: Session manager instance

    Returns:
        List of messages

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get messages from history
    all_messages = session.history.messages if session.history else []

    # Filter by role if specified
    if role:
        all_messages = [m for m in all_messages if m.role.value == role]

    # Calculate total and pagination
    total = len(all_messages)
    has_more = offset + limit < total

    # Apply pagination
    messages = all_messages[offset : offset + limit]

    # Convert to response format
    message_responses = []
    for msg in messages:
        message_responses.append(
            MessageResponse(
                id=str(msg.id),
                role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata or {},
            )
        )

    return MessagesResponse(
        messages=message_responses,
        total=total,
        has_more=has_more,
    )


@router.get(
    "/sessions/{session_id}/turns",
    response_model=TurnsResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_turns(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TurnsResponse:
    """Get conversation history as structured turns.

    Returns conversation turns in the same format as `round_complete` SSE events,
    allowing the UI to render historical messages identically to live streaming.

    This endpoint reads from the events.jsonl log file which contains the complete
    history of all events (user messages, LLM responses, code executions, etc.).

    Each turn contains:
    - User message (if any - autonomous rounds have no user message)
    - Assistant response with parsed code, plan, and execution result

    Args:
        session_id: Session ID
        limit: Maximum number of turns to return
        offset: Number of turns to skip
        session_manager: Session manager instance

    Returns:
        List of conversation turns

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.load_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Get workspace path from session and construct events.jsonl path
    workspace_path = session.workspace_path
    if workspace_path:
        events_file = Path(workspace_path) / "logs" / "events.jsonl"
    else:
        # Fallback: try to find events.jsonl in default location
        events_file = Path(f"workspace/runs/{session_id}/logs/events.jsonl")

    # Parse events from events.jsonl
    events = _parse_events_jsonl(events_file)

    if not events:
        # No events found - return empty response
        return TurnsResponse(
            turns=[],
            total=0,
            has_more=False,
        )

    # Convert events to turns
    turns = _events_to_turns(events)

    # Calculate total and pagination
    total = len(turns)
    has_more = offset + limit < total

    # Apply pagination
    paginated_turns = turns[offset : offset + limit]

    return TurnsResponse(
        turns=paginated_turns,
        total=total,
        has_more=has_more,
    )
