"""Tests for DSAgent API Server."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dsagent.server.models import (
    ChatRequest,
    ChatResponseModel,
    CreateSessionRequest,
    DataFrameInfoResponse,
    ErrorResponse,
    ExecuteCodeRequest,
    ExecutionResultResponse,
    HealthResponse,
    KernelStateResponse,
    KernelVariableResponse,
    MessageResponse,
    MessagesResponse,
    PlanResponse,
    PlanStepResponse,
    ReadinessResponse,
    SessionListResponse,
    SessionResponse,
    UpdateSessionRequest,
    WebSocketEvent,
    WebSocketEventType,
    WebSocketMessage,
    WebSocketMessageType,
)


# =============================================================================
# Model Tests
# =============================================================================


class TestAPIModels:
    """Tests for API request/response models."""

    def test_create_session_request(self):
        """Test CreateSessionRequest model."""
        request = CreateSessionRequest(
            name="Test Session",
            model="gpt-4o",
            hitl_mode="plan",
        )
        assert request.name == "Test Session"
        assert request.model == "gpt-4o"
        assert request.hitl_mode == "plan"

    def test_create_session_request_defaults(self):
        """Test CreateSessionRequest with defaults."""
        request = CreateSessionRequest()
        assert request.name is None
        assert request.model is None
        assert request.hitl_mode == "none"

    def test_chat_request(self):
        """Test ChatRequest model."""
        request = ChatRequest(message="Hello, world!")
        assert request.message == "Hello, world!"

    def test_execute_code_request(self):
        """Test ExecuteCodeRequest model."""
        request = ExecuteCodeRequest(code="print('hello')")
        assert request.code == "print('hello')"

    def test_session_response(self):
        """Test SessionResponse model."""
        now = datetime.utcnow()
        response = SessionResponse(
            id="test-123",
            name="Test",
            status="active",
            created_at=now,
            updated_at=now,
            message_count=5,
            kernel_variables=3,
            workspace_path="/tmp/test",
        )
        assert response.id == "test-123"
        assert response.status == "active"
        assert response.message_count == 5

    def test_execution_result_response(self):
        """Test ExecutionResultResponse model."""
        response = ExecutionResultResponse(
            stdout="Hello",
            stderr="",
            error=None,
            images=[],
            success=True,
        )
        assert response.stdout == "Hello"
        assert response.success is True

    def test_plan_response(self):
        """Test PlanResponse model."""
        response = PlanResponse(
            steps=[
                PlanStepResponse(number=1, description="Step 1", completed=True),
                PlanStepResponse(number=2, description="Step 2", completed=False),
            ],
            raw_text="1. Step 1\n2. Step 2",
            total_steps=2,
            completed_steps=1,
            is_complete=False,
        )
        assert len(response.steps) == 2
        assert response.total_steps == 2
        assert response.completed_steps == 1

    def test_kernel_state_response(self):
        """Test KernelStateResponse model."""
        response = KernelStateResponse(
            variables=[
                KernelVariableResponse(name="df", type="DataFrame", value_preview="<DataFrame>"),
            ],
            dataframes=[
                DataFrameInfoResponse(
                    name="df",
                    shape=[100, 5],
                    columns=["a", "b", "c", "d", "e"],
                    dtypes={"a": "int64", "b": "float64"},
                    memory_mb=0.5,
                ),
            ],
            imports=["pandas", "numpy"],
            is_running=True,
        )
        assert len(response.variables) == 1
        assert len(response.dataframes) == 1
        assert response.is_running is True

    def test_websocket_message(self):
        """Test WebSocketMessage model."""
        msg = WebSocketMessage(
            type=WebSocketMessageType.CHAT,
            content="Hello",
        )
        assert msg.type == WebSocketMessageType.CHAT
        assert msg.content == "Hello"

    def test_websocket_event_connected(self):
        """Test WebSocketEvent.connected factory."""
        event = WebSocketEvent.connected("session-1", "Connected!")
        assert event.type == WebSocketEventType.CONNECTED
        assert event.session_id == "session-1"
        assert event.data["message"] == "Connected!"

    def test_websocket_event_thinking(self):
        """Test WebSocketEvent.thinking factory."""
        event = WebSocketEvent.thinking("session-1")
        assert event.type == WebSocketEventType.THINKING
        assert event.session_id == "session-1"

    def test_websocket_event_plan(self):
        """Test WebSocketEvent.plan factory."""
        plan = PlanResponse(
            steps=[PlanStepResponse(number=1, description="Step 1", completed=False)],
            raw_text="1. Step 1",
            total_steps=1,
            completed_steps=0,
            is_complete=False,
        )
        event = WebSocketEvent.plan("session-1", plan)
        assert event.type == WebSocketEventType.PLAN
        assert "steps" in event.data

    def test_websocket_event_error(self):
        """Test WebSocketEvent.error factory."""
        event = WebSocketEvent.error("session-1", "Something went wrong", "print('error')")
        assert event.type == WebSocketEventType.ERROR
        assert event.data["error"] == "Something went wrong"
        assert event.data["code"] == "print('error')"

    def test_health_response(self):
        """Test HealthResponse model (version matches package)."""
        from dsagent import __version__
        response = HealthResponse(
            status="ok",
            version=__version__,
        )
        assert response.status == "ok"
        assert response.version == __version__

    def test_readiness_response(self):
        """Test ReadinessResponse model."""
        response = ReadinessResponse(
            ready=True,
            checks={"session_manager": True, "connection_manager": True},
        )
        assert response.ready is True
        assert response.checks["session_manager"] is True


# =============================================================================
# Integration Tests (require FastAPI test client)
# =============================================================================


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = MagicMock()
    manager.exists.return_value = True
    manager.list_sessions.return_value = []
    return manager


@pytest.fixture
def mock_connection_manager(mock_session_manager):
    """Create a mock AgentConnectionManager."""
    manager = MagicMock()
    manager.session_manager = mock_session_manager
    manager.get_agent.return_value = None
    return manager


class TestServerIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self, mock_session_manager, mock_connection_manager):
        """Create a test client."""
        try:
            from fastapi.testclient import TestClient
            from dsagent.server.app import create_app
            from dsagent.server import deps

            # Set up mock managers
            deps._session_manager = mock_session_manager
            deps._connection_manager = mock_connection_manager

            app = create_app()
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_readiness_endpoint(self, client):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data

    def test_list_sessions_empty(self, client, mock_session_manager):
        """Test listing sessions when empty."""
        mock_session_manager.list_sessions.return_value = []
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_create_session(self, client, mock_session_manager, mock_connection_manager):
        """Test creating a new session."""
        # Mock session creation
        mock_session = MagicMock()
        mock_session.id = "test-123"
        mock_session.name = "Test Session"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.updated_at = datetime.utcnow()
        mock_session.history = MagicMock()
        mock_session.history.messages = []
        mock_session.kernel_snapshot = None
        mock_session.workspace_path = "/tmp/test"
        mock_session_manager.create_session.return_value = mock_session

        # Mock agent creation
        mock_connection_manager.get_or_create_agent = AsyncMock()

        response = client.post(
            "/api/sessions",
            json={"name": "Test Session", "model": "gpt-4o"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "test-123"
        assert data["name"] == "Test Session"

    def test_get_session_not_found(self, client, mock_session_manager):
        """Test getting a non-existent session."""
        mock_session_manager.load_session.return_value = None
        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    def test_delete_session_not_found(self, client, mock_session_manager):
        """Test deleting a non-existent session."""
        mock_session_manager.load_session.return_value = None
        response = client.delete("/api/sessions/nonexistent")
        assert response.status_code == 404


# =============================================================================
# Unit Tests for Connection Manager
# =============================================================================


class TestAgentConnectionManager:
    """Tests for AgentConnectionManager."""

    @pytest.fixture
    def connection_manager(self, mock_session_manager):
        """Create a ConnectionManager for testing."""
        from dsagent.server.manager import AgentConnectionManager
        return AgentConnectionManager(
            session_manager=mock_session_manager,
            default_model="gpt-4o",
        )

    def test_init(self, connection_manager):
        """Test ConnectionManager initialization."""
        assert connection_manager._default_model == "gpt-4o"
        assert len(connection_manager._connections) == 0
        assert len(connection_manager._agents) == 0

    def test_get_active_sessions_empty(self, connection_manager):
        """Test getting active sessions when empty."""
        sessions = connection_manager.get_active_sessions()
        assert sessions == []

    def test_get_connection_count_empty(self, connection_manager):
        """Test connection count for non-existent session."""
        count = connection_manager.get_connection_count("nonexistent")
        assert count == 0

    def test_get_agent_none(self, connection_manager):
        """Test getting agent when none exists."""
        agent = connection_manager.get_agent("nonexistent")
        assert agent is None


# =============================================================================
# API Key Authentication Tests
# =============================================================================


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    @pytest.fixture
    def authenticated_client(self, mock_session_manager, mock_connection_manager):
        """Create a test client with API key authentication."""
        try:
            import os
            from fastapi.testclient import TestClient
            from dsagent.server.app import create_app
            from dsagent.server import deps

            # Set API key
            os.environ["DSAGENT_API_KEY"] = "test-api-key"
            from dsagent.config import clear_settings_cache
            clear_settings_cache()

            # Set up mock managers
            deps._session_manager = mock_session_manager
            deps._connection_manager = mock_connection_manager

            app = create_app()
            client = TestClient(app)
            yield client

            # Cleanup
            del os.environ["DSAGENT_API_KEY"]
            clear_settings_cache()
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_missing_api_key(self, authenticated_client):
        """Test request without API key when required."""
        response = authenticated_client.get("/api/sessions")
        assert response.status_code == 401

    def test_invalid_api_key(self, authenticated_client):
        """Test request with invalid API key."""
        response = authenticated_client.get(
            "/api/sessions",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_valid_api_key(self, authenticated_client, mock_session_manager):
        """Test request with valid API key."""
        mock_session_manager.list_sessions.return_value = []
        response = authenticated_client.get(
            "/api/sessions",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    @pytest.fixture
    def ws_client(self, mock_session_manager, mock_connection_manager):
        """Create a test client for WebSocket testing."""
        try:
            from fastapi.testclient import TestClient
            from dsagent.server.app import create_app
            from dsagent.server import deps

            # Set up mock managers
            deps._session_manager = mock_session_manager
            deps._connection_manager = mock_connection_manager

            # Mock connect method
            mock_agent = MagicMock()
            mock_connection_manager.connect = AsyncMock(return_value=mock_agent)
            mock_connection_manager.disconnect = AsyncMock()
            mock_connection_manager.send_to = AsyncMock(return_value=True)

            app = create_app()
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_websocket_connect(self, ws_client):
        """Test WebSocket connection."""
        with ws_client.websocket_connect("/ws/chat/test-session") as websocket:
            # Should receive connected event
            data = websocket.receive_json()
            assert data["type"] == "connected"

    def test_websocket_invalid_message(self, ws_client):
        """Test WebSocket with invalid message format."""
        with ws_client.websocket_connect("/ws/chat/test-session") as websocket:
            # Skip connected event
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("not json")
            data = websocket.receive_json()
            assert data["type"] == "error"


# =============================================================================
# SSE Streaming Tests
# =============================================================================


class TestSSEStreaming:
    """Tests for SSE streaming endpoint."""

    @pytest.fixture
    def sse_client(self, mock_session_manager, mock_connection_manager):
        """Create a test client for SSE testing."""
        try:
            from fastapi.testclient import TestClient
            from dsagent.server.app import create_app
            from dsagent.server import deps

            # Set up mock managers
            deps._session_manager = mock_session_manager
            deps._connection_manager = mock_connection_manager

            # Mock agent
            mock_agent = MagicMock()

            # Create mock response
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.code = None
            mock_response.execution_result = None
            mock_response.plan = None
            mock_response.has_answer = False
            mock_response.answer = None
            mock_response.thinking = None
            mock_response.is_complete = True

            mock_agent.chat_stream.return_value = iter([mock_response])
            mock_connection_manager.get_or_create_agent = AsyncMock(return_value=mock_agent)

            app = create_app()
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_stream_session_not_found(self, sse_client, mock_session_manager):
        """Test SSE streaming with non-existent session."""
        mock_session_manager.exists.return_value = False
        response = sse_client.post(
            "/api/sessions/nonexistent/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 404

    def test_chat_stream_success(self, sse_client, mock_session_manager):
        """Test successful SSE streaming."""
        mock_session_manager.exists.return_value = True
        response = sse_client.post(
            "/api/sessions/test-session/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
