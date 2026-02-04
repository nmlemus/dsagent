"""Pydantic models for the AI Planner Agent."""

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HITLMode(str, Enum):
    """Human-in-the-Loop modes for controlling agent autonomy.

    Modes:
        NONE: Fully autonomous, no human intervention (default)
        PLAN_ONLY: Pause after plan generation for approval
        ON_ERROR: Pause only when code execution fails
        PLAN_AND_ANSWER: Pause on plan + before final answer
        FULL: Pause before every code execution
    """

    NONE = "none"
    PLAN_ONLY = "plan_only"
    ON_ERROR = "on_error"
    PLAN_AND_ANSWER = "plan_and_answer"
    FULL = "full"


class HITLAction(str, Enum):
    """Actions a human can take when HITL is triggered.

    Actions:
        APPROVE: Accept and continue execution
        REJECT: Reject and abort execution
        MODIFY: Provide modified plan/code
        RETRY: Retry the failed operation
        SKIP: Skip current step and continue
        FEEDBACK: Provide textual feedback to the agent
    """

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    RETRY = "retry"
    SKIP = "skip"
    FEEDBACK = "feedback"


class HumanFeedback(BaseModel):
    """Feedback from human during HITL interaction."""

    action: HITLAction
    message: Optional[str] = None
    modified_plan: Optional[str] = None
    modified_code: Optional[str] = None


class AgentConfig(BaseSettings):
    """Configuration for the Planner Agent.

    Configuration priority (highest to lowest):
        1. Explicit parameters passed to PlannerAgent()
        2. Environment variables
        3. .env file
        4. Default values

    Model format examples (LiteLLM):
        - OpenAI: "gpt-4o", "gpt-4o-mini"
        - Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
        - Google: "gemini/gemini-pro", "gemini/gemini-1.5-pro"
        - Local: "ollama/llama2", "ollama/mistral"
        - LM Studio: "openai/model-name" (with LLM_API_BASE)
        - Azure: "azure/gpt-4"
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model configuration - checks multiple env var names for compatibility
    # Default is None; resolved via get_default_model() at runtime
    model: Optional[str] = Field(
        default=None,
        validation_alias="LLM_MODEL",
    )
    api_key: str = Field(
        default="",
        validation_alias="LLM_API_KEY",
    )
    api_base: str = Field(
        default="",
        validation_alias="LLM_API_BASE",
    )

    # Generation parameters
    temperature: float = Field(default=0.3, validation_alias="DSAGENT_TEMPERATURE")
    max_tokens: int = Field(default=4096, validation_alias="DSAGENT_MAX_TOKENS")
    max_rounds: int = Field(default=30, validation_alias="DSAGENT_MAX_ROUNDS")

    # Execution settings
    code_timeout: int = Field(default=300, validation_alias="DSAGENT_CODE_TIMEOUT")
    workspace: str = Field(default="./workspace", validation_alias="DSAGENT_WORKSPACE")

    # Session management (for multi-user)
    session_id: Optional[str] = None

    # Stop sequences for tag-based protocol
    stop_sequences: List[str] = Field(default_factory=lambda: ["</code>", "</answer>"])

    @property
    def workspace_path(self) -> Path:
        """Get workspace path, optionally with session isolation."""
        base = Path(self.workspace)
        if self.session_id:
            return base / self.session_id
        return base

    def get_provider(self) -> str:
        """Determine LLM provider from model name."""
        model = self.model.lower()
        if "claude" in model:
            return "Anthropic"
        elif "gemini" in model:
            return "Google"
        elif "ollama" in model:
            return "Ollama"
        elif "azure" in model:
            return "Azure"
        elif "gpt" in model or "o1" in model:
            return "OpenAI"
        else:
            return "Unknown"


class ExecutionResult(BaseModel):
    """Result of a code execution in the Jupyter kernel."""

    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    images: List[Dict[str, str]] = Field(default_factory=list)
    success: bool = True

    @property
    def output(self) -> str:
        """Combined output string."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        if self.error:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts) if parts else "(No output)"

    @property
    def has_images(self) -> bool:
        """Check if execution produced images."""
        return len(self.images) > 0


class ExecutionRecord(BaseModel):
    """Record of a single code execution for notebook generation."""

    code: str
    success: bool
    output: str
    images: List[Dict[str, Any]] = Field(default_factory=list)
    step_description: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class PlanStep(BaseModel):
    """A single step in the agent's plan."""

    number: int
    description: str
    completed: bool = False

    def mark_complete(self) -> None:
        """Mark this step as completed."""
        self.completed = True

    def __str__(self) -> str:
        marker = "[x]" if self.completed else "[ ]"
        return f"{self.number}. {marker} {self.description}"


class PlanState(BaseModel):
    """Current state of the agent's plan."""

    steps: List[PlanStep] = Field(default_factory=list)
    raw_text: str = ""

    @property
    def total_steps(self) -> int:
        """Total number of steps in the plan."""
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        """Number of completed steps."""
        return sum(1 for s in self.steps if s.completed)

    @property
    def pending_steps(self) -> int:
        """Number of pending steps."""
        return self.total_steps - self.completed_steps

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return self.total_steps > 0 and all(s.completed for s in self.steps)

    @property
    def progress(self) -> str:
        """Progress string (e.g., '3/5')."""
        return f"{self.completed_steps}/{self.total_steps}"

    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get the current (first uncompleted) step."""
        for step in self.steps:
            if not step.completed:
                return step
        return None

    def to_markdown(self) -> str:
        """Convert plan to markdown format."""
        lines = []
        for step in self.steps:
            lines.append(str(step))
        return "\n".join(lines)


class EventType(str, Enum):
    """Types of events emitted by the agent."""

    # Lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    AGENT_ERROR = "agent_error"

    # Round events
    ROUND_STARTED = "round_started"
    ROUND_FINISHED = "round_finished"

    # LLM events
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_FINISHED = "llm_call_finished"
    LLM_FALLBACK = "llm_fallback"

    # Plan events
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    PLAN_STEP_COMPLETED = "plan_step_completed"

    # Execution events
    CODE_EXECUTING = "code_executing"
    CODE_SUCCESS = "code_success"
    CODE_FAILED = "code_failed"

    # Tool execution events (MCP tools)
    TOOL_CALLING = "tool_calling"
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILED = "tool_failed"

    # Thinking events
    THINKING = "thinking"

    # Answer events
    ANSWER_REJECTED = "answer_rejected"
    ANSWER_ACCEPTED = "answer_accepted"

    # Notebook events
    NOTEBOOK_SAVED = "notebook_saved"

    # Human-in-the-Loop events
    HITL_AWAITING_PLAN_APPROVAL = "hitl_awaiting_plan_approval"
    HITL_AWAITING_CODE_APPROVAL = "hitl_awaiting_code_approval"
    HITL_AWAITING_ERROR_GUIDANCE = "hitl_awaiting_error_guidance"
    HITL_AWAITING_ANSWER_APPROVAL = "hitl_awaiting_answer_approval"
    HITL_FEEDBACK_RECEIVED = "hitl_feedback_received"
    HITL_PLAN_APPROVED = "hitl_plan_approved"
    HITL_PLAN_MODIFIED = "hitl_plan_modified"
    HITL_PLAN_REJECTED = "hitl_plan_rejected"
    HITL_EXECUTION_ABORTED = "hitl_execution_aborted"


class AgentEvent(BaseModel):
    """Event emitted by the agent for streaming to UI."""

    type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    round_num: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)

    # Optional fields for specific events
    plan: Optional[PlanState] = None
    code: Optional[str] = None
    result: Optional[ExecutionResult] = None
    message: Optional[str] = None
    error: Optional[str] = None

    # HITL-specific fields
    feedback: Optional[HumanFeedback] = None
    awaiting_input: bool = False

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        import json
        data = self.model_dump(mode="json", exclude_none=True)
        return f"event: {self.type.value}\ndata: {json.dumps(data)}\n\n"


class SessionState(BaseModel):
    """Serializable state of an agent session for persistence."""

    session_id: str
    config: AgentConfig
    plan: Optional[PlanState] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    execution_records: List[ExecutionRecord] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
