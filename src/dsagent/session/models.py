"""Pydantic models for session management."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    EXECUTION = "execution"  # Code execution results


class ConversationMessage(BaseModel):
    """A single message in a conversation.

    Represents any message in the conversation history, including
    user inputs, assistant responses, and code execution results.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Metadata for rich message content
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # - code: str (for execution messages)
    # - output: str (for execution messages)
    # - images: List[Dict] (for execution with visualizations)
    # - execution_time: float (seconds)
    # - success: bool (for execution messages)
    # - thinking: str (for assistant reasoning)
    # - plan: dict (for plan updates)

    # Threading support for complex conversations
    parent_id: Optional[str] = None

    def to_llm_message(self) -> Dict[str, str]:
        """Convert to LLM-compatible message format."""
        role = self.role.value
        # Map execution role to user for LLM
        if role == "execution":
            role = "user"
        return {"role": role, "content": self.content}

    @classmethod
    def user(cls, content: str, **kwargs) -> "ConversationMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "ConversationMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def system(cls, content: str, **kwargs) -> "ConversationMessage":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def execution(
        cls,
        code: str,
        output: str,
        success: bool,
        images: Optional[List[Dict]] = None,
        execution_time: float = 0.0,
        **kwargs
    ) -> "ConversationMessage":
        """Create an execution result message."""
        # Format content for LLM context
        status = "succeeded" if success else "failed"
        content = f"Code execution {status}.\n\nOutput:\n{output}"
        if not success:
            content = f"Code execution failed.\n\nError:\n{output}"

        return cls(
            role=MessageRole.EXECUTION,
            content=content,
            metadata={
                "code": code,
                "output": output,
                "success": success,
                "images": images or [],
                "execution_time": execution_time,
            },
            **kwargs
        )


class ConversationHistory(BaseModel):
    """Manages conversation history with intelligent truncation.

    Handles the conversation history for a session, providing methods
    for adding messages, truncation, and preparing context for LLM calls.

    Supports automatic summarization when conversations exceed limits.
    """

    messages: List[ConversationMessage] = Field(default_factory=list)

    # Configuration for context management
    max_messages: int = 100  # Max messages before truncation
    max_tokens_estimate: int = 100000  # Approximate token limit
    chars_per_token: int = 4  # Rough estimate for truncation

    # Summary of truncated messages
    summary: Optional[str] = None
    summary_messages_count: int = 0  # How many messages the summary covers
    summary_created_at: Optional[datetime] = None
    truncated_count: int = 0  # Total messages ever truncated

    def add(self, message: ConversationMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        self._maybe_truncate()

    def add_user(self, content: str, **kwargs) -> ConversationMessage:
        """Add a user message and return it."""
        msg = ConversationMessage.user(content, **kwargs)
        self.add(msg)
        return msg

    def add_assistant(self, content: str, **kwargs) -> ConversationMessage:
        """Add an assistant message and return it."""
        msg = ConversationMessage.assistant(content, **kwargs)
        self.add(msg)
        return msg

    def add_execution(
        self,
        code: str,
        output: str,
        success: bool,
        **kwargs
    ) -> ConversationMessage:
        """Add an execution result message and return it."""
        msg = ConversationMessage.execution(code, output, success, **kwargs)
        self.add(msg)
        return msg

    def get_recent(self, n: int = 10) -> List[ConversationMessage]:
        """Get the n most recent messages."""
        return self.messages[-n:]

    def get_by_role(self, role: MessageRole) -> List[ConversationMessage]:
        """Get all messages with a specific role."""
        return [m for m in self.messages if m.role == role]

    def to_llm_messages(
        self,
        include_system: bool = True,
        max_chars: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Convert history to LLM-compatible message list.

        Args:
            include_system: Whether to include system messages
            max_chars: Maximum total characters (for context window management)

        Returns:
            List of message dicts compatible with LiteLLM
        """
        messages = []
        total_chars = 0

        # Process messages from newest to oldest if we have a char limit
        if max_chars:
            for msg in reversed(self.messages):
                if not include_system and msg.role == MessageRole.SYSTEM:
                    continue

                llm_msg = msg.to_llm_message()
                msg_chars = len(llm_msg["content"])

                if total_chars + msg_chars > max_chars:
                    break

                messages.insert(0, llm_msg)
                total_chars += msg_chars
        else:
            for msg in self.messages:
                if not include_system and msg.role == MessageRole.SYSTEM:
                    continue
                messages.append(msg.to_llm_message())

        # Inject summary if we have truncated messages
        if self.summary and messages:
            summary_msg = {
                "role": "system",
                "content": f"[Previous conversation summary: {self.summary}]"
            }
            # Insert after first system message or at beginning
            insert_idx = 1 if messages[0]["role"] == "system" else 0
            messages.insert(insert_idx, summary_msg)

        return messages

    def _maybe_truncate(self) -> None:
        """Truncate history if it exceeds limits."""
        if len(self.messages) <= self.max_messages:
            return

        # Keep system messages and recent messages
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]

        # Calculate how many to keep
        keep_count = self.max_messages - len(system_msgs)

        if len(non_system) > keep_count:
            # Truncate older messages
            to_summarize = non_system[:-keep_count]
            to_keep = non_system[-keep_count:]

            # Update truncated count
            self.truncated_count += len(to_summarize)

            # Generate summary placeholder (actual summarization in manager)
            if not self.summary:
                self.summary = f"[{self.truncated_count} earlier messages truncated]"

            # Rebuild messages
            self.messages = system_msgs + to_keep

    def set_summary(
        self,
        summary_text: str,
        messages_covered: int,
    ) -> None:
        """Set the conversation summary.

        Args:
            summary_text: The summary content
            messages_covered: Number of messages this summary covers
        """
        self.summary = summary_text
        self.summary_messages_count = messages_covered
        self.summary_created_at = datetime.now()

    def needs_summarization(self, threshold: int = 30) -> bool:
        """Check if the conversation needs summarization.

        Args:
            threshold: Message count threshold for triggering summarization

        Returns:
            True if summarization is recommended
        """
        return len(self.messages) > threshold

    def get_messages_for_summary(self, keep_recent: int = 10) -> List[ConversationMessage]:
        """Get messages that should be summarized.

        Args:
            keep_recent: Number of recent messages to exclude from summary

        Returns:
            List of messages to summarize
        """
        if len(self.messages) <= keep_recent:
            return []
        return self.messages[:-keep_recent]

    def apply_summary(self, keep_recent: int = 10) -> int:
        """Apply summarization by removing old messages.

        Call this after setting the summary to actually truncate messages.

        Args:
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages removed
        """
        if len(self.messages) <= keep_recent:
            return 0

        removed_count = len(self.messages) - keep_recent
        self.messages = self.messages[-keep_recent:]
        self.truncated_count += removed_count
        return removed_count

    def clear(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        else:
            self.messages = []
        self.summary = None
        self.truncated_count = 0

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)


class KernelSnapshot(BaseModel):
    """Snapshot of kernel state for persistence.

    Captures the state of the Jupyter kernel at a point in time,
    allowing for session restoration.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    # Variable information
    variables: Dict[str, str] = Field(default_factory=dict)  # name -> type
    dataframes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # dataframe_name -> {"shape": [rows, cols], "columns": [...], "dtypes": {...}}

    # Execution state
    imports: List[str] = Field(default_factory=list)
    executed_code: List[str] = Field(default_factory=list)  # For replay

    # Serialized state (optional, using dill/cloudpickle)
    serialized_state: Optional[bytes] = None

    def get_context_summary(self) -> str:
        """Generate a summary of kernel state for LLM context."""
        lines = []

        if self.variables:
            lines.append("Variables:")
            for name, type_name in self.variables.items():
                if name in self.dataframes:
                    df_info = self.dataframes[name]
                    shape = df_info.get("shape", [0, 0])
                    cols = df_info.get("columns", [])[:5]
                    cols_str = ", ".join(cols)
                    if len(df_info.get("columns", [])) > 5:
                        cols_str += ", ..."
                    lines.append(f"  - {name}: DataFrame ({shape[0]} rows x {shape[1]} cols) [{cols_str}]")
                else:
                    lines.append(f"  - {name}: {type_name}")

        if self.imports:
            lines.append(f"\nImported: {', '.join(self.imports[:10])}")
            if len(self.imports) > 10:
                lines.append(f"  ... and {len(self.imports) - 10} more")

        return "\n".join(lines) if lines else "Kernel is empty"


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"  # Session is active and kernel may be running
    PAUSED = "paused"  # Session is paused (kernel stopped but restorable)
    COMPLETED = "completed"  # Session completed successfully
    ERROR = "error"  # Session ended with error
    ARCHIVED = "archived"  # Session archived (old, may be cleaned up)


class Session(BaseModel):
    """A conversation session with persistent state.

    Represents a complete session including conversation history,
    kernel state, and execution records.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None  # User-friendly name
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE

    # Conversation
    history: ConversationHistory = Field(default_factory=ConversationHistory)

    # Kernel state
    kernel_snapshot: Optional[KernelSnapshot] = None

    # Workspace paths
    workspace_path: Optional[str] = None
    data_path: Optional[str] = None
    artifacts_path: Optional[str] = None
    notebooks_path: Optional[str] = None
    logs_path: Optional[str] = None

    # Agent configuration
    model: Optional[str] = None  # LLM model for this session
    hitl_mode: str = "none"  # HITL mode (none, plan_only, on_error, plan_and_answer, full)

    # Plan state (from existing models)
    plan_raw: Optional[str] = None
    plan_steps_completed: int = 0
    plan_steps_total: int = 0

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # - model: str (LLM model used)
    # - total_tokens: int
    # - total_executions: int
    # - last_error: str

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

    def set_workspace(self, workspace: Path) -> None:
        """Set workspace paths from base workspace path."""
        self.workspace_path = str(workspace)
        self.data_path = str(workspace / "data")
        self.artifacts_path = str(workspace / "artifacts")
        self.notebooks_path = str(workspace / "notebooks")
        self.logs_path = str(workspace / "logs")

    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the session history."""
        self.history.add(message)
        self.touch()

    def get_context_for_llm(self, max_chars: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM."""
        return self.history.to_llm_messages(max_chars=max_chars)

    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of the session for listing."""
        return {
            "id": self.id,
            "name": self.name or f"Session {self.id[:8]}",
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.history),
            "has_kernel_state": self.kernel_snapshot is not None,
        }

    @classmethod
    def new(cls, name: Optional[str] = None, **kwargs) -> "Session":
        """Create a new session with optional name."""
        return cls(name=name, **kwargs)
