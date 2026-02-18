"""Session persistence storage backends."""

from __future__ import annotations

import json
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dsagent.session.models import Session, SessionStatus


class SessionStoreBackend(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Save a session to storage."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]:
        """Load a session from storage by ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session from storage."""
        pass

    @abstractmethod
    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List session summaries."""
        pass

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass


# Session ID must be safe for use in paths (no path traversal).
# Allow alphanumeric, underscore, hyphen; matches _generate_session_id format.
SESSION_ID_SAFE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_session_id(session_id: str) -> None:
    """Validate session_id for path safety (JSON backend and API).

    Raises:
        ValueError: If session_id contains disallowed characters.
    """
    if not session_id or not SESSION_ID_SAFE_PATTERN.match(session_id):
        raise ValueError(
            "session_id must contain only letters, digits, underscore, and hyphen"
        )


class JSONSessionStore(SessionStoreBackend):
    """JSON file-based session storage.

    Simple storage backend that saves each session as a JSON file.
    Suitable for development and single-user scenarios.
    """

    def __init__(self, storage_dir: Path):
        """Initialize JSON session store.

        Args:
            storage_dir: Directory to store session files
        """
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file. Validates session_id to prevent path traversal."""
        _validate_session_id(session_id)
        path = (self.storage_dir / f"{session_id}.json").resolve()
        if not path.is_relative_to(self.storage_dir):
            raise ValueError("session_id would resolve outside storage directory")
        return path

    def save(self, session: Session) -> None:
        """Save session to JSON file."""
        path = self._session_path(session.id)
        data = session.model_dump(mode="json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, session_id: str) -> Optional[Session]:
        """Load session from JSON file."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return Session.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            # Log error but return None
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session JSON file."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List session summaries from JSON files."""
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                session = Session.model_validate(data)
                if status is None or session.status == status:
                    sessions.append(session.to_summary())
            except Exception:
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions[:limit]

    def exists(self, session_id: str) -> bool:
        """Check if session file exists."""
        return self._session_path(session_id).exists()


class SQLiteSessionStore(SessionStoreBackend):
    """SQLite-based session storage.

    Production-ready storage backend using SQLite for persistence.
    Supports concurrent access and efficient querying.
    """

    def __init__(self, db_path: Path):
        """Initialize SQLite session store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC)
            """)
            conn.commit()

    def save(self, session: Session) -> None:
        """Save session to SQLite database."""
        data = session.model_dump_json()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (id, name, status, created_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.name,
                session.status.value,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                data
            ))
            conn.commit()

    def load(self, session_id: str) -> Optional[Session]:
        """Load session from SQLite database."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            if row:
                return Session.model_validate_json(row["data"])
        return None

    def delete(self, session_id: str) -> bool:
        """Delete session from SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List session summaries from SQLite database."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute("""
                    SELECT data FROM sessions
                    WHERE status = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (status.value, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT data FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()

            sessions = []
            for row in rows:
                try:
                    session = Session.model_validate_json(row["data"])
                    sessions.append(session.to_summary())
                except Exception:
                    continue
            return sessions

    def exists(self, session_id: str) -> bool:
        """Check if session exists in database."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            return row is not None

    def cleanup_old_sessions(
        self,
        days: int = 30,
        status: SessionStatus = SessionStatus.ARCHIVED
    ) -> int:
        """Clean up old sessions.

        Args:
            days: Delete sessions older than this many days
            status: Only delete sessions with this status

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Calculate cutoff date
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM sessions
                WHERE status = ? AND updated_at < ?
            """, (status.value, cutoff.isoformat()))
            conn.commit()
            return cursor.rowcount


class SessionStore:
    """Unified session store with pluggable backends.

    Factory class that creates the appropriate storage backend
    based on configuration.
    """

    def __init__(
        self,
        storage_path: Path,
        backend: str = "sqlite"
    ):
        """Initialize session store.

        Args:
            storage_path: Path to storage location
            backend: Backend type ("json" or "sqlite")
        """
        self.storage_path = Path(storage_path)

        if backend == "json":
            self._backend = JSONSessionStore(self.storage_path / "sessions")
        elif backend == "sqlite":
            self._backend = SQLiteSessionStore(
                self.storage_path / "sessions.db"
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def save(self, session: Session) -> None:
        """Save a session."""
        self._backend.save(session)

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        return self._backend.load(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        return self._backend.delete(session_id)

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List session summaries."""
        return self._backend.list_sessions(status=status, limit=limit)

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self._backend.exists(session_id)
