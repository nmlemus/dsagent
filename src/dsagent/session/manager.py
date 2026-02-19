"""Session manager for handling multiple conversation sessions."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dsagent.session.models import (
    ConversationMessage,
    KernelSnapshot,
    MessageRole,
    Session,
    SessionStatus,
)
from dsagent.session.store import SessionStore, _validate_session_id


class SessionManager:
    """Manages conversation sessions with persistence.

    The SessionManager handles creating, loading, saving, and switching
    between conversation sessions. It maintains an in-memory cache of
    active sessions and persists them to storage.

    Usage:
        manager = SessionManager(workspace_path)

        # Create new session
        session = manager.create_session(name="My Analysis")

        # Load existing session
        session = manager.load_session(session_id)

        # Get or create (useful for CLI)
        session = manager.get_or_create(session_id)

        # List sessions
        sessions = manager.list_sessions()

        # Auto-save on changes
        manager.save_session(session)
    """

    def __init__(
        self,
        workspace_path: Path,
        backend: str = "sqlite",
        auto_save: bool = True
    ):
        """Initialize session manager.

        Args:
            workspace_path: Base path for session storage
            backend: Storage backend ("sqlite" or "json")
            auto_save: Whether to auto-save on session changes
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.store = SessionStore(
            self.workspace_path / ".dsagent",
            backend=backend
        )
        self.auto_save = auto_save

        # In-memory cache of active sessions
        self._active_sessions: Dict[str, Session] = {}

        # Session change callbacks
        self._on_change_callbacks: List[Callable[[Session], None]] = []

    def create_session(
        self,
        name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Session:
        """Create a new session.

        Args:
            name: Optional user-friendly name
            session_id: Optional specific ID (auto-generated if not provided)

        Returns:
            Newly created Session
        """
        if session_id is None:
            session_id = self._generate_session_id()
        else:
            _validate_session_id(session_id)

        session = Session(
            id=session_id,
            name=name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            status=SessionStatus.ACTIVE
        )

        # Set up workspace paths
        session_workspace = self.workspace_path / "runs" / session_id
        session.set_workspace(session_workspace)

        # Create workspace directories
        Path(session.data_path).mkdir(parents=True, exist_ok=True)
        Path(session.artifacts_path).mkdir(parents=True, exist_ok=True)
        Path(session.notebooks_path).mkdir(parents=True, exist_ok=True)

        # Cache and persist
        self._active_sessions[session.id] = session
        self.store.save(session)

        return session

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from storage.

        Args:
            session_id: ID of session to load

        Returns:
            Session if found, None otherwise
        """
        # Check cache first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Load from storage
        session = self.store.load(session_id)
        if session:
            self._active_sessions[session.id] = session

        return session

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Session:
        """Get existing session or create new one.

        Args:
            session_id: Optional session ID to load
            name: Name for new session if creating

        Returns:
            Session (existing or newly created)
        """
        if session_id:
            session = self.load_session(session_id)
            if session:
                return session

        return self.create_session(name=name, session_id=session_id)

    def save_session(self, session: Session) -> None:
        """Save a session to storage.

        Args:
            session: Session to save
        """
        session.touch()
        self.store.save(session)
        self._notify_change(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: ID of session to delete

        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        self._active_sessions.pop(session_id, None)

        # Delete from storage
        return self.store.delete(session_id)

    def archive_session(self, session_id: str) -> bool:
        """Archive a session (mark as archived but don't delete).

        Args:
            session_id: ID of session to archive

        Returns:
            True if archived, False if not found
        """
        session = self.load_session(session_id)
        if session:
            session.status = SessionStatus.ARCHIVED
            self.save_session(session)
            # Remove from active cache
            self._active_sessions.pop(session_id, None)
            return True
        return False

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List session summaries.

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of sessions to return

        Returns:
            List of session summary dicts
        """
        return self.store.list_sessions(status=status, limit=limit)

    def get_active_sessions(self) -> List[Session]:
        """Get all currently active sessions in memory.

        Returns:
            List of active Session objects
        """
        return list(self._active_sessions.values())

    def add_message(
        self,
        session: Session,
        message: ConversationMessage
    ) -> None:
        """Add a message to a session and optionally auto-save.

        Args:
            session: Session to add message to
            message: Message to add
        """
        session.add_message(message)

        if self.auto_save:
            self.save_session(session)

    def add_user_message(
        self,
        session: Session,
        content: str,
        **kwargs
    ) -> ConversationMessage:
        """Add a user message to a session.

        Args:
            session: Session to add message to
            content: Message content
            **kwargs: Additional message parameters

        Returns:
            Created ConversationMessage
        """
        msg = ConversationMessage.user(content, **kwargs)
        self.add_message(session, msg)
        return msg

    def add_assistant_message(
        self,
        session: Session,
        content: str,
        **kwargs
    ) -> ConversationMessage:
        """Add an assistant message to a session.

        Args:
            session: Session to add message to
            content: Message content
            **kwargs: Additional message parameters

        Returns:
            Created ConversationMessage
        """
        msg = ConversationMessage.assistant(content, **kwargs)
        self.add_message(session, msg)
        return msg

    def add_execution_result(
        self,
        session: Session,
        code: str,
        output: str,
        success: bool,
        **kwargs
    ) -> ConversationMessage:
        """Add an execution result to a session.

        Args:
            session: Session to add message to
            code: Executed code
            output: Execution output
            success: Whether execution succeeded
            **kwargs: Additional message parameters

        Returns:
            Created ConversationMessage
        """
        msg = ConversationMessage.execution(code, output, success, **kwargs)
        self.add_message(session, msg)
        return msg

    def update_kernel_snapshot(
        self,
        session: Session,
        snapshot: KernelSnapshot
    ) -> None:
        """Update the kernel snapshot for a session.

        Args:
            session: Session to update
            snapshot: New kernel snapshot
        """
        session.kernel_snapshot = snapshot

        if self.auto_save:
            self.save_session(session)

    def set_session_status(
        self,
        session: Session,
        status: SessionStatus
    ) -> None:
        """Update session status.

        Args:
            session: Session to update
            status: New status
        """
        session.status = status

        if self.auto_save:
            self.save_session(session)

    def on_change(self, callback: Callable[[Session], None]) -> None:
        """Register a callback for session changes.

        Args:
            callback: Function to call when a session changes
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self, session: Session) -> None:
        """Notify all callbacks of a session change."""
        for callback in self._on_change_callbacks:
            try:
                callback(session)
            except Exception:
                pass  # Don't let callback errors break the manager

    def _generate_session_id(self) -> str:
        """Generate a unique session ID.

        Format: YYYYMMDD_HHMMSS_XXXXXX
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"{timestamp}_{unique}"

    def close(self) -> None:
        """Close the session manager and save all active sessions."""
        for session in self._active_sessions.values():
            self.store.save(session)
        self._active_sessions.clear()

    def __enter__(self) -> "SessionManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
