"""Connection and Agent Manager for DSAgent Server."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from fastapi import WebSocket

from dsagent.agents import ConversationalAgent, ConversationalAgentConfig
from dsagent.schema.models import HITLMode
from dsagent.server.models import (
    ExecutionResultResponse,
    PlanResponse,
    PlanStepResponse,
    WebSocketEvent,
)
from dsagent.session import Session, SessionManager

if TYPE_CHECKING:
    from dsagent.session.models import KernelSnapshot

logger = logging.getLogger(__name__)


class AgentConnectionManager:
    """Manages WebSocket connections and agent instances.

    This class handles:
    - Multiple WebSocket connections per session (for dashboards, etc.)
    - Agent lifecycle management (create, start, shutdown)
    - Broadcasting events to all connections in a session
    - Thread-safe agent operations via asyncio.to_thread()
    """

    def __init__(
        self,
        session_manager: SessionManager,
        default_model: Optional[str] = None,
        default_hitl_mode: str = "none",
    ):
        """Initialize the connection manager.

        Args:
            session_manager: SessionManager instance for session persistence
            default_model: Default LLM model to use for new agents
            default_hitl_mode: Default HITL mode for new agents
        """
        self._session_manager = session_manager
        self._default_model = default_model
        self._default_hitl_mode = default_hitl_mode

        # session_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # session_id -> ConversationalAgent instance
        self._agents: Dict[str, ConversationalAgent] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @property
    def session_manager(self) -> SessionManager:
        """Get the session manager."""
        return self._session_manager

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        model: Optional[str] = None,
        hitl_mode: Optional[str] = None,
    ) -> ConversationalAgent:
        """Connect a WebSocket to a session and ensure agent is running.

        Args:
            websocket: The WebSocket connection
            session_id: Session ID to connect to
            model: Optional model override
            hitl_mode: Optional HITL mode override

        Returns:
            The ConversationalAgent for this session
        """
        await websocket.accept()

        async with self._lock:
            # Add connection to set
            if session_id not in self._connections:
                self._connections[session_id] = set()
            self._connections[session_id].add(websocket)

            # Get or create agent
            agent = await self._get_or_create_agent(
                session_id, model=model, hitl_mode=hitl_mode
            )

        logger.info(f"WebSocket connected to session {session_id}")
        return agent

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Disconnect a WebSocket from a session.

        Args:
            websocket: The WebSocket connection to remove
            session_id: Session ID to disconnect from
        """
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)

                # If no more connections, optionally shutdown agent
                if not self._connections[session_id]:
                    del self._connections[session_id]
                    # Note: We keep the agent running for potential reconnection
                    # Use shutdown_agent() explicitly if needed

        logger.info(f"WebSocket disconnected from session {session_id}")

    async def broadcast(
        self,
        session_id: str,
        event: WebSocketEvent,
        exclude: Optional[WebSocket] = None,
    ) -> None:
        """Broadcast an event to all connections in a session.

        Args:
            session_id: Session ID to broadcast to
            event: Event to send
            exclude: Optional WebSocket to exclude from broadcast
        """
        if session_id not in self._connections:
            return

        disconnected = set()
        for websocket in self._connections[session_id]:
            if websocket == exclude:
                continue
            try:
                await websocket.send_json(event.model_dump(mode="json"))
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for websocket in disconnected:
            self._connections[session_id].discard(websocket)

    async def send_to(self, websocket: WebSocket, event: WebSocketEvent) -> bool:
        """Send an event to a specific WebSocket.

        Args:
            websocket: Target WebSocket
            event: Event to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json(event.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")
            return False

    def get_agent(self, session_id: str) -> Optional[ConversationalAgent]:
        """Get the agent for a session if it exists.

        Args:
            session_id: Session ID

        Returns:
            The ConversationalAgent or None if not found
        """
        return self._agents.get(session_id)

    def set_agent_hitl_mode(self, session_id: str, hitl_mode: str) -> bool:
        """Change HITL mode for an agent at runtime.

        Args:
            session_id: Session ID
            hitl_mode: New HITL mode (none, plan_only, on_error, plan_and_answer, full)

        Returns:
            True if successful, False if agent not found
        """
        agent = self._agents.get(session_id)
        if not agent:
            return False

        hitl_mode_map = {
            "none": HITLMode.NONE,
            "plan_only": HITLMode.PLAN_ONLY,
            "plan": HITLMode.PLAN_ONLY,
            "on_error": HITLMode.ON_ERROR,
            "plan_and_answer": HITLMode.PLAN_AND_ANSWER,
            "full": HITLMode.FULL,
        }
        mode = hitl_mode_map.get(hitl_mode, HITLMode.NONE)
        agent.set_hitl_mode(mode)
        logger.info(f"Changed HITL mode for session {session_id} to {hitl_mode}")
        return True

    def set_agent_model(self, session_id: str, model: str) -> bool:
        """Change LLM model for an agent at runtime.

        Args:
            session_id: Session ID
            model: New model to use

        Returns:
            True if successful, False if agent not found
        """
        agent = self._agents.get(session_id)
        if not agent:
            return False

        agent.set_model(model)
        logger.info(f"Changed model for session {session_id} to {model}")
        return True

    async def get_or_create_agent(
        self,
        session_id: str,
        model: Optional[str] = None,
        hitl_mode: Optional[str] = None,
    ) -> ConversationalAgent:
        """Get or create an agent for a session.

        Args:
            session_id: Session ID
            model: Optional model override
            hitl_mode: Optional HITL mode override

        Returns:
            The ConversationalAgent for this session
        """
        async with self._lock:
            return await self._get_or_create_agent(session_id, model, hitl_mode)

    async def _get_or_create_agent(
        self,
        session_id: str,
        model: Optional[str] = None,
        hitl_mode: Optional[str] = None,
    ) -> ConversationalAgent:
        """Internal method to get or create agent (must hold lock).

        Args:
            session_id: Session ID
            model: Optional model override
            hitl_mode: Optional HITL mode override

        Returns:
            The ConversationalAgent for this session
        """
        if session_id in self._agents:
            return self._agents[session_id]

        # Get or create session
        session = self._session_manager.get_or_create(session_id)

        # Create agent config - use session config, then params, then defaults
        import os
        effective_model = (
            model
            or getattr(session, "model", None)
            or self._default_model
            or os.getenv("LLM_MODEL", "gpt-4o")
        )

        # Convert hitl_mode string to HITLMode enum
        # Priority: parameter > session > default
        hitl_mode_str = (
            hitl_mode
            or getattr(session, "hitl_mode", None)
            or self._default_hitl_mode
        )
        hitl_mode_map = {
            "none": HITLMode.NONE,
            "plan_only": HITLMode.PLAN_ONLY,
            "plan": HITLMode.PLAN_ONLY,  # Alias
            "on_error": HITLMode.ON_ERROR,
            "plan_and_answer": HITLMode.PLAN_AND_ANSWER,
            "full": HITLMode.FULL,
        }
        effective_hitl_mode = hitl_mode_map.get(hitl_mode_str, HITLMode.NONE)

        config = ConversationalAgentConfig(
            model=effective_model,
            hitl_mode=effective_hitl_mode,
        )

        # Create agent
        agent = ConversationalAgent(
            config=config,
            session=session,
            session_manager=self._session_manager,
        )

        # Start agent in thread pool (it's synchronous)
        await asyncio.to_thread(agent.start)

        self._agents[session_id] = agent
        logger.info(f"Created and started agent for session {session_id}")

        return agent

    async def shutdown_agent(self, session_id: str, save_notebook: bool = True) -> None:
        """Shutdown an agent for a session.

        Args:
            session_id: Session ID
            save_notebook: Whether to save the notebook before shutdown
        """
        async with self._lock:
            if session_id in self._agents:
                agent = self._agents[session_id]
                await asyncio.to_thread(agent.shutdown, save_notebook)
                del self._agents[session_id]
                logger.info(f"Shutdown agent for session {session_id}")

    async def shutdown_all(self) -> None:
        """Shutdown all agents and close all connections."""
        async with self._lock:
            # Close all WebSocket connections
            for session_id, connections in self._connections.items():
                for websocket in connections:
                    try:
                        await websocket.close()
                    except Exception:
                        pass
            self._connections.clear()

            # Shutdown all agents
            for session_id, agent in self._agents.items():
                try:
                    await asyncio.to_thread(agent.shutdown)
                except Exception as e:
                    logger.error(f"Error shutting down agent {session_id}: {e}")
            self._agents.clear()

            logger.info("All agents and connections shut down")

    def get_connection_count(self, session_id: str) -> int:
        """Get the number of connections for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of active WebSocket connections
        """
        return len(self._connections.get(session_id, set()))

    def get_active_sessions(self) -> list[str]:
        """Get list of session IDs with active agents.

        Returns:
            List of session IDs
        """
        return list(self._agents.keys())

    async def chat(
        self,
        session_id: str,
        message: str,
        websocket: Optional[WebSocket] = None,
    ) -> None:
        """Send a chat message and stream responses via WebSocket.

        Args:
            session_id: Session ID
            message: User message
            websocket: WebSocket to send events to (or broadcast if None)
        """
        agent = self.get_agent(session_id)
        if not agent:
            raise ValueError(f"No agent found for session {session_id}")

        async def send_event(event: WebSocketEvent) -> None:
            if websocket:
                await self.send_to(websocket, event)
            else:
                await self.broadcast(session_id, event)

        # Send thinking event
        await send_event(WebSocketEvent.thinking(session_id))

        # Run chat in thread pool
        try:
            response_gen = await asyncio.to_thread(agent.chat_stream, message)

            for response in response_gen:
                # Send plan if present
                if response.plan:
                    plan_response = self._convert_plan(response.plan)
                    await send_event(WebSocketEvent.plan(session_id, plan_response))

                # Send code executing event
                if response.code:
                    await send_event(
                        WebSocketEvent.code_executing(session_id, response.code)
                    )

                # Send execution result
                if response.execution_result:
                    result_response = ExecutionResultResponse(
                        stdout=response.execution_result.stdout,
                        stderr=response.execution_result.stderr,
                        error=response.execution_result.error,
                        images=response.execution_result.images,
                        success=response.execution_result.success,
                    )
                    await send_event(
                        WebSocketEvent.code_result(session_id, result_response)
                    )

                # Send response content
                await send_event(
                    WebSocketEvent.response(
                        session_id,
                        response.content,
                        is_partial=not response.is_complete,
                    )
                )

                # Send answer if present
                if response.answer:
                    await send_event(WebSocketEvent.answer(session_id, response.answer))

                # Send complete event if done
                if response.is_complete:
                    await send_event(WebSocketEvent.complete(session_id))

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await send_event(WebSocketEvent.error(session_id, str(e)))
            raise

    async def execute_code(
        self,
        session_id: str,
        code: str,
    ) -> ExecutionResultResponse:
        """Execute code directly in the kernel.

        Args:
            session_id: Session ID
            code: Python code to execute

        Returns:
            Execution result
        """
        agent = self.get_agent(session_id)
        if not agent:
            raise ValueError(f"No agent found for session {session_id}")

        result = await asyncio.to_thread(agent.execute_code_directly, code)

        return ExecutionResultResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            images=result.images,
            success=result.success,
        )

    def _convert_plan(self, plan: Any) -> PlanResponse:
        """Convert agent PlanState to API PlanResponse.

        Args:
            plan: PlanState from agent

        Returns:
            PlanResponse for API
        """
        steps = []
        if hasattr(plan, "steps"):
            for step in plan.steps:
                steps.append(
                    PlanStepResponse(
                        number=step.number,
                        description=step.description,
                        completed=step.completed,
                    )
                )

        return PlanResponse(
            steps=steps,
            raw_text=getattr(plan, "raw_text", ""),
            total_steps=getattr(plan, "total_steps", len(steps)),
            completed_steps=getattr(plan, "completed_steps", 0),
            is_complete=getattr(plan, "is_complete", False),
        )
