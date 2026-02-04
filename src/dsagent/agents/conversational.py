"""Conversational Data Science Agent.

This module provides a conversational interface for data science tasks,
maintaining context across messages and code executions. It supports both
interactive chat mode and autonomous execution mode with plan tracking.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from litellm import completion

from dsagent.config import get_default_model
from dsagent.kernel import LocalExecutor, ExecutorConfig, KernelIntrospector
from dsagent.session import Session, SessionManager, ConversationMessage, SessionLogger
from dsagent.utils.validation import validate_configuration, get_proxy_model_name
from dsagent.schema.models import ExecutionResult, AgentConfig, PlanState, PlanStep, HITLMode
from dsagent.core.planner import PlanParser
from dsagent.core.hitl import HITLGateway
from dsagent.utils.notebook import NotebookBuilder, LiveNotebookBuilder, LiveNotebookSync, NotebookChange
from dsagent.memory import ConversationSummarizer, SummaryConfig
from dsagent.prompts import PromptBuilder

if TYPE_CHECKING:
    from dsagent.utils.logger import AgentLogger
    from dsagent.tools.mcp_manager import MCPManager
    from dsagent.tools.config import MCPConfig
    from dsagent.observability import ObservabilityManager


class ExecutionMode(str, Enum):
    """Execution mode for the agent."""
    CONVERSATIONAL = "conversational"  # Single response, wait for user
    AUTONOMOUS = "autonomous"  # Loop until plan complete


# System prompt is now managed by PromptBuilder
# See dsagent/prompts/sections.py for prompt content


@dataclass
class ChatResponse:
    """Response from a chat interaction."""

    content: str  # Full response text
    code: Optional[str] = None  # Extracted code (if any)
    execution_result: Optional[ExecutionResult] = None  # Code execution result
    plan: Optional[PlanState] = None  # Extracted plan (if any)
    has_answer: bool = False  # Whether response contains <answer>
    answer: Optional[str] = None  # Extracted answer text
    thinking: Optional[str] = None  # Extracted thinking/reasoning
    intent: Optional[str] = None  # Classified intent: question, simple, complex
    explanation: Optional[str] = None  # Text between </intent> and <plan>/<code>
    is_complete: bool = False  # Whether task is complete (all steps done or no plan)


@dataclass
class ConversationalAgentConfig:
    """Configuration for the conversational agent."""

    model: Optional[str] = None  # Resolved via get_default_model() if None
    temperature: float = 0.3
    max_tokens: int = 4096
    code_timeout: int = 300
    max_rounds: int = 30  # Max rounds for autonomous execution
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    hitl_mode: HITLMode = field(default_factory=lambda: HITLMode.NONE)
    hitl_timeout: float = 300.0
    enable_live_notebook: bool = False  # Enable real-time notebook sync with Jupyter
    enable_notebook_sync: bool = False  # Enable bidirectional sync (watches for user edits)

    # Summarization settings (Phase 5)
    enable_summarization: bool = True  # Auto-summarize long conversations
    summarization_threshold: int = 30  # Summarize when messages exceed this
    keep_recent_messages: int = 10  # Keep this many recent messages after summarization
    summarization_model: Optional[str] = None  # None = use main model, or specify a cheaper one

    # Logging settings
    enable_logging: bool = True  # Enable event logging to files (run.log, events.jsonl)

    # MCP settings
    mcp_config: Optional[Any] = None  # Path to MCP YAML, dict, or MCPConfig object

    # Observability settings
    observability_config: Optional[Any] = None  # ObservabilityConfig object or None

    def get_effective_model(self) -> str:
        """Get the effective model, using resolution cascade if not set."""
        return get_default_model(explicit=self.model)

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> "ConversationalAgentConfig":
        """Create from AgentConfig."""
        return cls(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            code_timeout=config.code_timeout,
            workspace=Path(config.workspace),
        )


class ConversationalAgent:
    """Conversational Data Science Agent with hybrid execution modes.

    This agent provides both:
    1. Interactive chat mode: User guides each step
    2. Autonomous mode: Agent executes plan automatically

    The mode is determined by whether the LLM generates a <plan>:
    - No plan → Simple response, return and wait
    - With plan → Execute in loop until complete (respecting HITL settings)

    Example - Simple chat:
        agent = ConversationalAgent(workspace="./workspace")
        agent.start()

        response = agent.chat("What is a DataFrame?")
        print(response.content)  # Just an explanation, no code

    Example - Autonomous task:
        response = agent.chat("Analyze sales data and create visualizations")
        # Agent creates plan, executes all steps, saves files
        # Returns final response with answer

    Example - With HITL:
        config = ConversationalAgentConfig(hitl_mode=HITLMode.PLAN_ONLY)
        agent = ConversationalAgent(config=config)

        for response in agent.chat_stream("Build a model"):
            if response.plan and not response.is_complete:
                # User can approve/modify plan here
                pass
    """

    # Stop sequences for LLM
    STOP_SEQUENCES = ["</code>", "</answer>"]

    def __init__(
        self,
        config: Optional[ConversationalAgentConfig] = None,
        session: Optional[Session] = None,
        session_manager: Optional[SessionManager] = None,
        logger: Optional["AgentLogger"] = None,
    ):
        """Initialize the conversational agent.

        Args:
            config: Agent configuration
            session: Existing session to use (or creates new one)
            session_manager: Session manager for persistence
            logger: Optional logger
        """
        self.config = config or ConversationalAgentConfig()
        self.logger = logger

        # Session management
        self._session_manager = session_manager
        self._session = session

        # Execution components (initialized on start)
        self._executor: Optional[LocalExecutor] = None
        self._introspector: Optional[KernelIntrospector] = None
        self._started = False

        # Plan tracking
        self._current_plan: Optional[PlanState] = None
        self._round_num = 0

        # Notebook building (can be NotebookBuilder, LiveNotebookBuilder, or LiveNotebookSync)
        self._notebook_builder: Optional[NotebookBuilder] = None
        self._notebook_sync: Optional[LiveNotebookSync] = None
        self._current_task: Optional[str] = None  # First message becomes the task

        # Callback for external notebook changes (user edits in Jupyter)
        self._on_notebook_change: Optional[Callable[[List[NotebookChange]], None]] = None

        # HITL gateway
        self._hitl_gateway: Optional[HITLGateway] = None
        if self.config.hitl_mode != HITLMode.NONE:
            self._hitl_gateway = HITLGateway(
                mode=self.config.hitl_mode,
                timeout=self.config.hitl_timeout
            )

        # Summarizer for long conversations (Phase 5)
        self._summarizer: Optional[ConversationSummarizer] = None

        # Session logger for event logging
        self._session_logger: Optional[SessionLogger] = None

        # MCP manager for external tools
        self._mcp_manager: Optional["MCPManager"] = None

        # Skills registry for Agent Skills
        self._skill_registry: Optional["SkillRegistry"] = None

        # Observability manager for LLM tracing
        self._observability: Optional["ObservabilityManager"] = None

        # Callbacks for UI updates
        self._on_plan_update: Optional[Callable[[PlanState], None]] = None
        self._on_code_executing: Optional[Callable[[str], None]] = None
        self._on_code_result: Optional[Callable[[ExecutionResult], None]] = None
        self._on_thinking: Optional[Callable[[], None]] = None
        self._on_llm_response: Optional[Callable[[str], None]] = None
        self._on_hitl_request: Optional[Callable[[str, Optional[PlanState], Optional[str], Optional[str]], None]] = None
        # Tool execution callbacks
        self._on_tool_calling: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._on_tool_result: Optional[Callable[[str, bool, Optional[str], Optional[str], float], None]] = None

    @property
    def session(self) -> Optional[Session]:
        """Get the current session."""
        return self._session

    @property
    def current_plan(self) -> Optional[PlanState]:
        """Get the current plan state."""
        return self._current_plan

    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._started and self._executor is not None and self._executor.is_running

    @property
    def hitl(self) -> Optional[HITLGateway]:
        """Access to HITL gateway for providing feedback."""
        return self._hitl_gateway

    def set_hitl_mode(self, mode: HITLMode) -> None:
        """Change HITL mode at runtime.

        Args:
            mode: New HITL mode to use
        """
        if mode == HITLMode.NONE:
            self._hitl_gateway = None
        else:
            self._hitl_gateway = HITLGateway(
                mode=mode,
                timeout=self.config.hitl_timeout,
            )

    def set_model(self, model: str) -> None:
        """Change LLM model at runtime.

        The new model will be used for all subsequent LLM calls.
        Does not affect any in-progress calls.

        Args:
            model: New model to use (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
        """
        from dsagent.llm.config import validate_configuration

        # Validate model configuration
        validate_configuration(model)

        old_model = self.config.model
        self.config.model = model

        # Log the change
        if self._session_logger:
            self._session_logger._file_logger.info(
                f"Model changed from {old_model} to {model}"
            )

    def set_callbacks(
        self,
        on_plan_update: Optional[Callable[[PlanState], None]] = None,
        on_code_executing: Optional[Callable[[str], None]] = None,
        on_code_result: Optional[Callable[[ExecutionResult], None]] = None,
        on_notebook_change: Optional[Callable[[List[NotebookChange]], None]] = None,
        on_thinking: Optional[Callable[[], None]] = None,
        on_llm_response: Optional[Callable[[str], None]] = None,
        on_hitl_request: Optional[Callable[[str, Optional[PlanState], Optional[str], Optional[str]], None]] = None,
        on_tool_calling: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, bool, Optional[str], Optional[str], float], None]] = None,
    ) -> None:
        """Set callbacks for UI updates during autonomous execution.

        Args:
            on_plan_update: Called when plan state changes
            on_code_executing: Called before code execution
            on_code_result: Called after code execution
            on_notebook_change: Called when user edits notebook in Jupyter
            on_thinking: Called before LLM request (for "thinking" indicator)
            on_llm_response: Called when LLM response is received (before code execution)
            on_hitl_request: Called when HITL approval is needed (request_type, plan, code, error)
            on_tool_calling: Called before MCP tool execution (tool_name, arguments)
            on_tool_result: Called after MCP tool execution (tool_name, success, result, error, time_ms)
        """
        self._on_plan_update = on_plan_update
        self._on_code_executing = on_code_executing
        self._on_code_result = on_code_result
        self._on_notebook_change = on_notebook_change
        self._on_thinking = on_thinking
        self._on_llm_response = on_llm_response
        self._on_hitl_request = on_hitl_request
        self._on_tool_calling = on_tool_calling
        self._on_tool_result = on_tool_result

    def start(self, session: Optional[Session] = None) -> None:
        """Start the agent and kernel.

        Args:
            session: Optional session to use
        """
        if self._started:
            return

        # Resolve model if not explicitly set
        if self.config.model is None:
            self.config.model = self.config.get_effective_model()

        # Validate model configuration and apply API base mapping
        validate_configuration(self.config.model)

        # Set up session
        if session:
            self._session = session
        elif not self._session:
            if self._session_manager:
                self._session = self._session_manager.create_session()
            else:
                self._session = Session.new()

        # Set up workspace from session
        workspace = Path(self._session.workspace_path or self.config.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (workspace / "data").mkdir(exist_ok=True)
        (workspace / "artifacts").mkdir(exist_ok=True)
        (workspace / "logs").mkdir(exist_ok=True)

        # Initialize session logger
        if self.config.enable_logging and self._session:
            self._session_logger = SessionLogger(
                session=self._session,
                enabled=True,
            )

        # Initialize executor
        executor_config = ExecutorConfig(
            workspace=workspace,
            timeout=self.config.code_timeout,
        )
        self._executor = LocalExecutor(executor_config)
        self._executor.start()

        # Initialize introspector
        self._introspector = KernelIntrospector.from_executor(self._executor)

        # Initialize summarizer for long conversations (Phase 5)
        if self.config.enable_summarization:
            summary_config = SummaryConfig(
                max_messages=self.config.summarization_threshold,
                keep_recent=self.config.keep_recent_messages,
                model=self.config.summarization_model or self.config.model,  # Use main model if not specified
            )
            self._summarizer = ConversationSummarizer(config=summary_config)

        # Initialize MCP manager for external tools
        if self.config.mcp_config:
            self._init_mcp()

        # Initialize skills registry
        self._init_skills()

        # Initialize observability for LLM tracing
        self._init_observability()

        self._started = True

    def shutdown(self, save_notebook: bool = True) -> Optional[Path]:
        """Shutdown the agent and kernel.

        Args:
            save_notebook: Whether to save notebook on shutdown

        Returns:
            Path to saved notebook if save_notebook=True and there was content
        """
        notebook_path = None

        # Teardown observability
        if self._observability:
            try:
                self._observability.teardown()
            except Exception:
                pass
            self._observability = None

        # Disconnect MCP servers
        if self._mcp_manager:
            try:
                self._mcp_manager.disconnect_all_sync()
            except Exception:
                pass  # Ignore errors on cleanup
            self._mcp_manager = None

        # Close session logger
        if self._session_logger:
            self._session_logger.close()
            self._session_logger = None

        # Stop notebook sync if running
        if self._notebook_sync:
            self._notebook_sync.stop()
            self._notebook_sync = None

        # Save notebook if we have any executions
        if save_notebook and self._notebook_builder:
            notebook_path = self.export_notebook()

        # Save session before shutdown
        if self._session and self._session_manager:
            self._session_manager.save_session(self._session)

        if self._executor:
            self._executor.shutdown()
            self._executor = None
        self._introspector = None
        self._started = False
        self._current_plan = None
        self._round_num = 0
        self._notebook_builder = None
        self._current_task = None

        return notebook_path

    def _init_mcp(self) -> None:
        """Initialize MCP manager from config."""
        try:
            from dsagent.tools.mcp_manager import MCPManager
            from dsagent.tools.config import MCPConfig

            mcp_config = self.config.mcp_config
            if isinstance(mcp_config, (str, Path)):
                self._mcp_manager = MCPManager.from_yaml(mcp_config)
            elif isinstance(mcp_config, dict):
                self._mcp_manager = MCPManager.from_dict(mcp_config)
            elif isinstance(mcp_config, MCPConfig):
                self._mcp_manager = MCPManager(mcp_config)
            else:
                return

            # Connect to all configured servers
            self._mcp_manager.connect_all_sync()

            if self._mcp_manager.connected_servers:
                if self._session_logger:
                    self._session_logger._file_logger.info(
                        f"MCP: Connected to {len(self._mcp_manager.connected_servers)} server(s), "
                        f"{len(self._mcp_manager.available_tools)} tool(s) available"
                    )
            else:
                if self._session_logger:
                    self._session_logger._file_logger.warning("MCP: No servers connected")

        except ImportError:
            # MCP package not installed
            pass
        except Exception as e:
            if self._session_logger:
                self._session_logger.log_error(f"Failed to initialize MCP: {e}", error_type="mcp_error")

    def _init_skills(self) -> None:
        """Initialize skills registry and discover installed skills."""
        try:
            from dsagent.skills import SkillRegistry, SkillLoader

            loader = SkillLoader()
            self._skill_registry = SkillRegistry(loader)
            count = self._skill_registry.discover()

            if count > 0 and self._session_logger:
                self._session_logger._file_logger.info(
                    f"Skills: Discovered {count} skill(s)"
                )

        except ImportError:
            # Skills module not available
            pass
        except Exception as e:
            if self._session_logger:
                self._session_logger.log_error(f"Failed to initialize skills: {e}", error_type="skills_error")

    def _init_observability(self) -> None:
        """Initialize observability for LLM tracing."""
        try:
            from dsagent.observability import ObservabilityManager, ObservabilityConfig

            # Use provided config or load from environment
            if self.config.observability_config:
                obs_config = self.config.observability_config
            else:
                obs_config = ObservabilityConfig.from_env()

            # Set session ID from current session if available
            if self._session and not obs_config.session_id:
                obs_config.session_id = self._session.id

            self._observability = ObservabilityManager(obs_config)
            if self._observability.setup():
                if self._session_logger:
                    status = self._observability.get_status()
                    providers = ", ".join(status["providers"])
                    self._session_logger._file_logger.info(
                        f"Observability: Enabled with providers: {providers}"
                    )
            else:
                self._observability = None

        except ImportError:
            # Observability module not available
            pass
        except Exception as e:
            if self._session_logger:
                self._session_logger.log_error(
                    f"Failed to initialize observability: {e}",
                    error_type="observability_error"
                )

    def _get_kernel_context(self) -> str:
        """Get kernel context for the system prompt."""
        if not self._introspector or not self.is_running:
            return "No kernel state available yet."

        result = self._introspector.introspect()
        if result.success:
            return result.get_summary()
        return "Kernel state: empty"

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context and available tools."""
        kernel_context = self._get_kernel_context()

        # Get tools list if MCP manager is available
        tools = None
        if self._mcp_manager and self._mcp_manager.available_tools:
            tools = self._mcp_manager.available_tools

        # Get skills context if available
        skills_context = None
        if self._skill_registry and self._skill_registry.skills:
            skills_context = self._skill_registry.get_prompt_context()

        # Use PromptBuilder to construct the prompt
        return PromptBuilder.build_conversational_prompt(
            kernel_context=kernel_context,
            tools=tools,
            skills_context=skills_context,
        )

    def _get_tools_for_llm(self) -> Optional[List[Dict[str, Any]]]:
        """Get tool definitions for LLM if MCP is available."""
        if self._mcp_manager and self._mcp_manager.available_tools:
            return self._mcp_manager.get_tools_for_llm()
        return None

    def _handle_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            List of tool result messages
        """
        import json
        import time as _time
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Notify callback that tool is being called
            if self._on_tool_calling:
                self._on_tool_calling(tool_name, arguments)

            start_time = _time.time()
            success = False
            result = None
            error = None

            try:
                # Use the synchronous API which uses MCPManager's dedicated event loop
                result = self._mcp_manager.execute_tool_sync(tool_name, arguments)
                result = result if isinstance(result, str) else str(result)
                success = True

            except Exception as e:
                error = str(e)
                result = f"Error executing tool {tool_name}: {error}"

            execution_time_ms = (_time.time() - start_time) * 1000

            # Log tool execution
            if self._session_logger:
                self._session_logger.log_tool_execution(
                    tool_name=tool_name,
                    arguments=arguments,
                    success=success,
                    result=result if success else None,
                    error=error,
                    execution_time_ms=execution_time_ms,
                )

            # Notify callback with tool result
            if self._on_tool_result:
                self._on_tool_result(
                    tool_name,
                    success,
                    result if success else None,
                    error,
                    execution_time_ms,
                )

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        return results

    def _create_notebook_builder(self, task: str, workspace: Path) -> NotebookBuilder:
        """Create the appropriate notebook builder based on configuration.

        Args:
            task: The user's task description
            workspace: Workspace path

        Returns:
            NotebookBuilder, LiveNotebookBuilder, or LiveNotebookSync.builder
        """
        if self.config.enable_notebook_sync:
            # Full bidirectional sync with Jupyter
            self._notebook_sync = LiveNotebookSync(
                task=task,
                workspace=workspace,
                on_external_change=self._on_notebook_change,
            )
            notebook_path = self._notebook_sync.start()
            # Return the underlying builder so track_execution works
            return self._notebook_sync.builder

        elif self.config.enable_live_notebook:
            # Live updates without bidirectional sync
            return LiveNotebookBuilder(
                task=task,
                workspace=workspace,
                auto_save=True,
            )

        else:
            # Standard notebook builder (saves at end)
            return NotebookBuilder(
                task=task,
                workspace=workspace,
            )

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build messages list for LLM from session history."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        if self._session:
            # Get conversation history (excluding system messages)
            history_messages = self._session.history.to_llm_messages(
                include_system=False,
                max_chars=100000  # Limit context size
            )
            messages.extend(history_messages)

        return messages

    def _maybe_summarize(self) -> bool:
        """Check if summarization is needed and perform it.

        Automatically summarizes conversation history when it exceeds
        the configured threshold, keeping only recent messages.

        Returns:
            True if summarization was performed
        """
        if not self._summarizer or not self._session:
            return False

        history = self._session.history

        # Check if we need to summarize
        if not self._summarizer.should_summarize(history.messages):
            return False

        # Get kernel state for context
        kernel_state = None
        if self._session.kernel_snapshot:
            kernel_state = {
                "variables": self._session.kernel_snapshot.variables,
                "dataframes": self._session.kernel_snapshot.dataframes,
                "imports": self._session.kernel_snapshot.imports,
            }

        # Get existing summary if any
        existing_summary = None
        if history.summary:
            from dsagent.memory import ConversationSummary
            existing_summary = ConversationSummary(
                content=history.summary,
                messages_summarized=history.summary_messages_count,
            )

        # Perform summarization
        try:
            summary = self._summarizer.summarize(
                messages=history.messages,
                kernel_state=kernel_state,
                existing_summary=existing_summary,
            )

            # Apply summary to history
            history.set_summary(summary.content, summary.messages_summarized)
            removed = history.apply_summary(
                keep_recent=self.config.keep_recent_messages
            )

            # Save session with updated summary
            if self._session_manager:
                self._session_manager.save_session(self._session)

            # Log summarization event
            if self._session_logger:
                self._session_logger.log_summarization(
                    messages_summarized=removed,
                    messages_kept=len(history.messages),
                )

            if self.logger:
                self.logger.info(
                    f"Summarized conversation: removed {removed} messages, "
                    f"keeping {len(history.messages)} recent"
                )

            return True

        except Exception as e:
            if self._session_logger:
                self._session_logger.log_error(str(e), error_type="summarization_error")
            if self.logger:
                self.logger.error(f"Summarization failed: {e}")
            return False

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from <code> tags or markdown code blocks."""
        # Try <code> tags first (complete)
        match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try incomplete <code> tag (LLM stopped before closing tag)
        match = re.search(r"<code>(.*?)$", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        # Try markdown code block with python
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic markdown code block
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Skip if it looks like a non-code block (e.g., file structure)
            if code and not code.startswith("./") and not code.startswith("├"):
                return code

        return None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags."""
        # Try complete tag first
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try incomplete tag (LLM stopped before closing tag)
        match = re.search(r"<answer>(.*?)$", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _extract_thinking(self, text: str) -> Optional[str]:
        """Extract thinking from <think> tags in text."""
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_intent(self, text: str) -> Optional[str]:
        """Extract intent classification from <intent> tags.

        Returns:
            One of: 'question', 'simple', 'complex', or None if not found.
        """
        match = re.search(r"<intent>\s*(question|simple|complex)\s*</intent>", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None

    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation text between </intent> and first structural tag.

        This captures the model's reasoning/explanation that appears after
        intent classification but before the plan or code.

        Returns:
            Explanation text if found, None otherwise.
        """
        # Find the end of </intent> tag
        intent_end = re.search(r"</intent>\s*", text, re.IGNORECASE)
        if not intent_end:
            return None

        # Find the start of next structural tag (<plan> or <code>)
        after_intent = text[intent_end.end():]

        # Find where plan or code starts
        next_tag = re.search(r"<(plan|code)>", after_intent, re.IGNORECASE)
        if not next_tag:
            return None

        # Extract text between
        explanation = after_intent[:next_tag.start()].strip()

        # Return only if there's meaningful content
        if explanation and len(explanation) > 10:
            return explanation
        return None

    def _extract_thinking_from_response(self, response: Any) -> Optional[str]:
        """Extract thinking/reasoning from LLM response.

        Checks multiple locations where thinking might be stored:
        - Claude's extended thinking (reasoning_content)
        - Response content blocks with thinking type
        - <think> tags in content
        """
        thinking_parts = []

        try:
            message = response.choices[0].message

            # Check for Claude's reasoning_content (extended thinking)
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                thinking_parts.append(message.reasoning_content)

            # Check for thinking in content blocks (Claude format)
            if hasattr(message, 'content') and isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'type') and block.type == 'thinking':
                        if hasattr(block, 'thinking'):
                            thinking_parts.append(block.thinking)

            # Check for thinking in tool_calls or other attributes
            if hasattr(message, 'thinking') and message.thinking:
                thinking_parts.append(message.thinking)

            # Also extract from <think> tags in content
            content = message.content if isinstance(message.content, str) else ""
            if content:
                think_match = self._extract_thinking(content)
                if think_match:
                    thinking_parts.append(think_match)

        except Exception:
            pass  # Silently ignore extraction errors

        return "\n\n".join(thinking_parts) if thinking_parts else None

    def _extract_plan(self, text: str) -> Optional[PlanState]:
        """Extract plan from <plan> tags."""
        return PlanParser.parse_plan(text)

    def _has_final_answer(self, text: str) -> bool:
        """Check if response contains <answer> tag."""
        return PlanParser.has_final_answer(text)

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        use_stop: bool = True,
        use_temperature: bool = True,
        use_max_tokens: bool = True,
        disable_thinking: bool = False,
    ) -> str:
        """Call the LLM and return response text.

        Handles MCP tool calls if available - executes tools and calls LLM again with results.
        Also handles fallbacks for unsupported parameters (stop, temperature, max_tokens).

        Args:
            messages: Chat messages to send
            use_stop: Whether to use stop sequences
            use_temperature: Whether to use temperature parameter
            use_max_tokens: Whether to use max_tokens (vs max_completion_tokens)
            disable_thinking: Whether to disable thinking mode (for Gemini thought_signature issues)
        """
        # Notify thinking started (for UI updates)
        if self._on_thinking:
            self._on_thinking()

        # Log request
        if self._session_logger:
            self._session_logger.log_llm_request(
                model=self.config.model,
                messages_count=len(messages),
                temperature=self.config.temperature if use_temperature else None,
                max_tokens=self.config.max_tokens,
            )

        start_time = time.time()
        # Transform model name for proxy if LLM_API_BASE is set
        model_name = get_proxy_model_name(self.config.model)

        # Build completion kwargs conditionally based on what's supported
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        if use_temperature:
            kwargs["temperature"] = self.config.temperature
        if use_max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        else:
            # Some providers use max_completion_tokens instead
            kwargs["max_completion_tokens"] = self.config.max_tokens

        # Add tools if available
        tools = self._get_tools_for_llm()
        if tools:
            kwargs["tools"] = tools
        elif use_stop:
            # Only use stop sequences when no tools (tools don't work well with stop)
            kwargs["stop"] = self.STOP_SEQUENCES

        # Disable thinking mode for Gemini if requested (fallback for thought_signature issues)
        if disable_thinking and "gemini" in model_name.lower():
            kwargs["thinking"] = {"type": "disabled", "budget_tokens": 0}

        # Add observability metadata for tracing
        if self._observability and self._observability.is_active():
            kwargs["metadata"] = self._observability.get_call_metadata(
                session_id=self._session.id if self._session else None,
                call_type="chat",
            )

        try:
            response = completion(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = getattr(message, "tool_calls", None)

            # Handle tool calls if present
            if tool_calls and self._mcp_manager:
                if self._session_logger:
                    self._session_logger._file_logger.info(
                        f"[TOOL CALLS] LLM requested {len(tool_calls)} tool(s)"
                    )

                # Add assistant message with tool calls to messages
                # Preserve provider_specific_fields for Gemini thought_signature support
                tool_calls_list = []
                for tc in tool_calls:
                    tc_dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    # Preserve provider_specific_fields if present (for Gemini thought_signature)
                    if hasattr(tc, "provider_specific_fields") and tc.provider_specific_fields:
                        tc_dict["provider_specific_fields"] = tc.provider_specific_fields
                    tool_calls_list.append(tc_dict)

                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls_list,
                })

                # Execute tools and add results
                tool_results = self._handle_tool_calls(tool_calls)
                messages.extend(tool_results)

                # Call LLM again with tool results (using same parameter settings)
                kwargs_retry: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                }
                if use_temperature:
                    kwargs_retry["temperature"] = self.config.temperature
                if use_max_tokens:
                    kwargs_retry["max_tokens"] = self.config.max_tokens
                else:
                    kwargs_retry["max_completion_tokens"] = self.config.max_tokens
                if tools:
                    kwargs_retry["tools"] = tools

                # Add observability metadata for tracing (tool follow-up call)
                if self._observability and self._observability.is_active():
                    kwargs_retry["metadata"] = self._observability.get_call_metadata(
                        session_id=self._session.id if self._session else None,
                        call_type="tool-followup",
                    )

                response = completion(**kwargs_retry)
                content = response.choices[0].message.content or ""

            latency_ms = (time.time() - start_time) * 1000

            # Extract thinking from Claude responses (if present)
            thinking = self._extract_thinking_from_response(response)
            if thinking and self._session_logger:
                self._session_logger.log_thinking(thinking)

            # Log response
            if self._session_logger:
                tokens = getattr(response.usage, 'total_tokens', 0) if response.usage else 0
                self._session_logger.log_llm_response(
                    response=content,
                    tokens_used=tokens,
                    latency_ms=latency_ms,
                    model=self.config.model,
                    has_code="<code>" in content,
                    has_plan="<plan>" in content,
                    has_answer="<answer>" in content,
                )

            # Notify LLM response received (for UI updates)
            if self._on_llm_response:
                self._on_llm_response(content)

            return content

        except Exception as e:
            error_msg = str(e).lower()

            # Handle Gemini thought_signature error - disable thinking mode as fallback
            if not disable_thinking and "thought_signature" in error_msg:
                if self._session_logger:
                    self._session_logger._file_logger.warning(
                        "Gemini thought_signature error, retrying with thinking disabled"
                    )
                return self._call_llm(
                    messages,
                    use_stop=use_stop,
                    use_temperature=use_temperature,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=True,
                )

            # Handle stop parameter not supported
            if use_stop and "stop" in error_msg:
                if self._session_logger:
                    self._session_logger._file_logger.warning(
                        "Provider doesn't support 'stop', retrying without it"
                    )
                return self._call_llm(
                    messages,
                    use_stop=False,
                    use_temperature=use_temperature,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=disable_thinking,
                )

            # Handle temperature not supported
            if use_temperature and "temperature" in error_msg:
                if self._session_logger:
                    self._session_logger._file_logger.warning(
                        "Provider doesn't support 'temperature', retrying without it"
                    )
                return self._call_llm(
                    messages,
                    use_stop=use_stop,
                    use_temperature=False,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=disable_thinking,
                )

            # Handle max_tokens vs max_completion_tokens
            if use_max_tokens and "max_tokens" in error_msg:
                if self._session_logger:
                    self._session_logger._file_logger.warning(
                        "Provider requires 'max_completion_tokens', retrying"
                    )
                return self._call_llm(
                    messages,
                    use_stop=use_stop,
                    use_temperature=use_temperature,
                    use_max_tokens=False,
                    disable_thinking=disable_thinking,
                )

            # Log error
            if self._session_logger:
                self._session_logger.log_error(str(e), error_type="llm_error")
            raise

    def _execute_code(self, code: str, step_desc: str = "") -> ExecutionResult:
        """Execute code and update kernel snapshot.

        Args:
            code: Python code to execute
            step_desc: Description of current plan step (for notebook)

        Returns:
            ExecutionResult with output/error
        """
        if not self._executor:
            return ExecutionResult(error="Kernel not started", success=False)

        if self._on_code_executing:
            self._on_code_executing(code)

        start_time = time.time()
        result = self._executor.execute(code)
        execution_time_ms = (time.time() - start_time) * 1000

        if self._on_code_result:
            self._on_code_result(result)

        # Log code execution
        if self._session_logger:
            self._session_logger.log_code_execution(
                code=code,
                success=result.success,
                output=result.output or "",
                error=result.error,
                images_count=len(result.images) if result.images else 0,
                execution_time_ms=execution_time_ms,
            )

        # Track execution for notebook generation
        if self._notebook_builder:
            self._notebook_builder.track_execution(
                code=code,
                result=result,
                step_desc=step_desc,
            )

        # Update kernel snapshot in session
        if self._session and self._introspector:
            snapshot = self._executor.get_kernel_state()
            self._session.kernel_snapshot = snapshot

            if self._session_manager:
                self._session_manager.save_session(self._session)

        return result

    def _build_context_message(self, code: str, result: ExecutionResult) -> str:
        """Build context message from execution result."""
        output = result.output or ""
        # Clean ANSI codes
        output = PlanParser.clean_ansi(output)

        # Truncate if too long
        max_output = 4000
        if len(output) > max_output:
            output = output[:max_output] + f"\n... (truncated, {len(output)} chars total)"

        parts = [f"Code executed:\n```python\n{code}\n```\n"]

        if result.success:
            parts.append(f"Output:\n{output}")
            if result.images:
                parts.append(f"\n[{len(result.images)} image(s) generated]")
        else:
            parts.append(f"Error:\n{output}")

        return "\n".join(parts)

    def _is_plan_complete(self) -> bool:
        """Check if current plan is complete."""
        if not self._current_plan:
            return True  # No plan means nothing to complete

        pending = [s for s in self._current_plan.steps if not s.completed]
        return len(pending) == 0

    def _get_current_step_desc(self) -> str:
        """Get the description of the current (first incomplete) step."""
        if not self._current_plan:
            return ""

        for step in self._current_plan.steps:
            if not step.completed:
                return f"Step {step.number}: {step.description}"
        return ""

    def chat(self, message: str) -> ChatResponse:
        """Send a message and get a response.

        If the response contains a <plan>, the agent will automatically
        continue executing until the plan is complete (autonomous mode).

        Args:
            message: User message

        Returns:
            ChatResponse with content and optional code execution results
        """
        if not self.is_running:
            raise RuntimeError("Agent not started. Call start() first.")

        # Initialize notebook builder on first message
        if not self._notebook_builder:
            self._current_task = message
            workspace = Path(self._session.workspace_path) if self._session and self._session.workspace_path else self.config.workspace
            self._notebook_builder = self._create_notebook_builder(message, workspace)

        # Add user message to history
        if self._session:
            self._session.history.add_user(message)

        # Log user message
        if self._session_logger:
            self._session_logger.log_user_message(message)

        # Build messages and call LLM
        messages = self._build_messages()
        response_text = self._call_llm(messages)

        # Parse initial response
        response = self._parse_response(response_text)

        # Log plan if present
        if response.plan and self._session_logger:
            completed = sum(1 for s in response.plan.steps if s.completed)
            self._session_logger.log_plan_update(
                plan_text=response.plan.raw_text or "",
                completed_steps=completed,
                total_steps=len(response.plan.steps),
            )

        # HITL: Check if we need plan approval
        # Only block if on_hitl_request callback is set (API/streaming use this)
        # CLI handles HITL externally and doesn't set this callback
        if (response.plan and self._hitl_gateway and
            self._hitl_gateway.should_pause_for_plan() and self._on_hitl_request):
            # Notify external handler (streaming endpoint)
            self._on_hitl_request("plan", response.plan, None, None)

            # Request approval and wait (blocking)
            self._hitl_gateway.request_plan_approval(response.plan)
            feedback = self._hitl_gateway.wait_for_feedback()

            # Handle feedback
            if feedback:
                from dsagent.schema.models import HITLAction
                if feedback.action == HITLAction.REJECT:
                    response.is_complete = True
                    response.answer = "Task aborted by user."
                    if self._session:
                        self._session.history.add_assistant(response_text)
                    return response
                # APPROVE or MODIFY: continue execution

        # Execute code if present
        if response.code:
            step_desc = self._get_current_step_desc()
            result = self._execute_code(response.code, step_desc=step_desc)
            response.execution_result = result

            # Add execution result to history
            if self._session:
                self._session.history.add_execution(
                    code=response.code,
                    output=result.output,
                    success=result.success,
                    images=result.images,
                )

        # Add assistant response to history
        if self._session:
            self._session.history.add_assistant(
                response_text,
                metadata={
                    "code": response.code,
                    "has_answer": response.has_answer,
                    "has_plan": response.plan is not None,
                }
            )

        # If there's a plan and it's not complete, continue autonomously
        if response.plan and not response.is_complete:
            response = self._run_autonomous(response)

        # Check if we need to summarize the conversation (Phase 5)
        self._maybe_summarize()

        return response

    def _parse_response(self, response_text: str) -> ChatResponse:
        """Parse LLM response into ChatResponse."""
        code = self._extract_code(response_text)
        answer = self._extract_answer(response_text)
        thinking = self._extract_thinking(response_text)
        intent = self._extract_intent(response_text)
        explanation = self._extract_explanation(response_text)
        plan = self._extract_plan(response_text)
        has_answer = self._has_final_answer(response_text)

        # Log intent if found
        if intent and self._session_logger:
            self._session_logger._log_event("intent", {"intent": intent})

        # Update current plan
        if plan:
            self._current_plan = plan
            if self._on_plan_update:
                self._on_plan_update(plan)

        # Determine if task is complete
        is_complete = False
        if has_answer:
            # Explicit answer provided
            is_complete = True
        elif self._is_plan_complete() and not code:
            # Plan is complete and no more code to execute
            is_complete = True
        elif not plan and not code:
            # Simple conversational response
            is_complete = True

        return ChatResponse(
            content=response_text,
            code=code,
            plan=plan,
            has_answer=has_answer,
            answer=answer,
            thinking=thinking,
            intent=intent,
            explanation=explanation,
            is_complete=is_complete,
        )

    def _run_autonomous(self, initial_response: ChatResponse) -> ChatResponse:
        """Run autonomous execution loop until plan is complete.

        Args:
            initial_response: The initial response with a plan

        Returns:
            Final ChatResponse when plan is complete or max rounds reached
        """
        self._round_num = 1
        last_response = initial_response

        while self._round_num < self.config.max_rounds:
            self._round_num += 1

            # Check if we're done
            if last_response.is_complete:
                break

            # Build context message from last execution
            if last_response.code and last_response.execution_result:
                context = self._build_context_message(
                    last_response.code,
                    last_response.execution_result
                )
                # Add to session history
                if self._session:
                    self._session.history.add_user(context)
            else:
                # No code executed, prompt to continue
                if self._session:
                    self._session.history.add_user(
                        "Please continue with the next step of your plan."
                    )

            # Call LLM again
            messages = self._build_messages()
            response_text = self._call_llm(messages)

            # Parse response
            last_response = self._parse_response(response_text)

            # Execute code if present
            if last_response.code:
                step_desc = self._get_current_step_desc()
                result = self._execute_code(last_response.code, step_desc=step_desc)
                last_response.execution_result = result

                # Add execution result to history
                if self._session:
                    self._session.history.add_execution(
                        code=last_response.code,
                        output=result.output,
                        success=result.success,
                        images=result.images,
                    )

            # Add assistant response to history
            if self._session:
                self._session.history.add_assistant(
                    response_text,
                    metadata={
                        "code": last_response.code,
                        "has_answer": last_response.has_answer,
                        "has_plan": last_response.plan is not None,
                        "round": self._round_num,
                    }
                )

        # Handle max rounds without completion
        if not last_response.is_complete:
            last_response.content += f"\n\n[Max rounds ({self.config.max_rounds}) reached]"

        return last_response

    def chat_stream(
        self,
        message: str,
        on_code_execute: Optional[Callable[[str], None]] = None,
    ) -> Generator[ChatResponse, None, None]:
        """Send a message and stream responses.

        Yields ChatResponse objects as each step completes.
        Useful for showing progress in a UI.

        Args:
            message: User message
            on_code_execute: Callback when code is about to execute

        Yields:
            ChatResponse objects as they become available
        """
        if not self.is_running:
            raise RuntimeError("Agent not started. Call start() first.")

        # Initialize notebook builder on first message
        if not self._notebook_builder:
            self._current_task = message
            workspace = Path(self._session.workspace_path) if self._session and self._session.workspace_path else self.config.workspace
            self._notebook_builder = self._create_notebook_builder(message, workspace)

        # Add user message to history
        if self._session:
            self._session.history.add_user(message)

        # Log user message
        if self._session_logger:
            self._session_logger.log_user_message(message)

        # Build messages and call LLM
        messages = self._build_messages()
        response_text = self._call_llm(messages)

        # Parse initial response
        response = self._parse_response(response_text)

        # Log plan if present
        if response.plan and self._session_logger:
            completed = sum(1 for s in response.plan.steps if s.completed)
            self._session_logger.log_plan_update(
                plan_text=response.plan.raw_text or "",
                completed_steps=completed,
                total_steps=len(response.plan.steps),
            )

        # HITL: Check if we need plan approval (only for initial plan)
        # Only block if on_hitl_request callback is set (API/streaming use this)
        # CLI handles HITL externally and doesn't set this callback
        if (response.plan and self._hitl_gateway and
            self._hitl_gateway.should_pause_for_plan() and self._on_hitl_request):
            # Notify external handler (streaming endpoint)
            self._on_hitl_request("plan", response.plan, None, None)

            # Request approval and wait (blocking)
            self._hitl_gateway.request_plan_approval(response.plan)
            feedback = self._hitl_gateway.wait_for_feedback()

            # Handle feedback
            if feedback:
                from dsagent.schema.models import HITLAction
                if feedback.action == HITLAction.REJECT:
                    response.is_complete = True
                    response.answer = "Task aborted by user."
                    if self._session:
                        self._session.history.add_assistant(response_text)
                    yield response
                    return
                # APPROVE or MODIFY: continue execution

        # Execute code if present
        if response.code:
            if on_code_execute:
                on_code_execute(response.code)
            step_desc = self._get_current_step_desc()
            result = self._execute_code(response.code, step_desc=step_desc)
            response.execution_result = result

            if self._session:
                self._session.history.add_execution(
                    code=response.code,
                    output=result.output,
                    success=result.success,
                    images=result.images,
                )

        # Add to history
        if self._session:
            self._session.history.add_assistant(response_text)

        yield response

        # If there's a plan, continue yielding responses
        if response.plan and not response.is_complete:
            yield from self._run_autonomous_stream(response, on_code_execute)

        # Check if we need to summarize the conversation (Phase 5)
        self._maybe_summarize()

    def _run_autonomous_stream(
        self,
        initial_response: ChatResponse,
        on_code_execute: Optional[Callable[[str], None]] = None,
    ) -> Generator[ChatResponse, None, None]:
        """Run autonomous execution with streaming."""
        self._round_num = 1
        last_response = initial_response

        while self._round_num < self.config.max_rounds:
            self._round_num += 1

            if last_response.is_complete:
                break

            # Build context
            if last_response.code and last_response.execution_result:
                context = self._build_context_message(
                    last_response.code,
                    last_response.execution_result
                )
                if self._session:
                    self._session.history.add_user(context)
            else:
                if self._session:
                    self._session.history.add_user(
                        "Please continue with the next step of your plan."
                    )

            # Call LLM
            messages = self._build_messages()
            response_text = self._call_llm(messages)

            # Parse response
            last_response = self._parse_response(response_text)

            # Execute code
            if last_response.code:
                if on_code_execute:
                    on_code_execute(last_response.code)
                step_desc = self._get_current_step_desc()
                result = self._execute_code(last_response.code, step_desc=step_desc)
                last_response.execution_result = result

                if self._session:
                    self._session.history.add_execution(
                        code=last_response.code,
                        output=result.output,
                        success=result.success,
                        images=result.images,
                    )

            if self._session:
                self._session.history.add_assistant(response_text)

            yield last_response

    def execute_code_directly(self, code: str) -> ExecutionResult:
        """Execute code directly without LLM.

        Useful for the CLI /exec command or programmatic use.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult
        """
        if not self.is_running:
            raise RuntimeError("Agent not started. Call start() first.")

        result = self._execute_code(code)

        # Add to history
        if self._session:
            self._session.history.add_execution(
                code=code,
                output=result.output,
                success=result.success,
            )

        return result

    def get_kernel_state(self) -> Dict[str, Any]:
        """Get current kernel state.

        Returns:
            Dict with variables, dataframes, imports, etc.
        """
        if not self._introspector or not self.is_running:
            return {}

        result = self._introspector.introspect()
        if result.success:
            return {
                "variables": result.variables,
                "dataframes": result.dataframes,
                "imports": result.imports,
                "functions": result.functions,
            }
        return {}

    def reset_kernel(self) -> None:
        """Reset the kernel to a clean state."""
        if self._executor:
            self._executor.reset()

        self._current_plan = None
        self._round_num = 0

        # Update session snapshot
        if self._session:
            self._session.kernel_snapshot = None
            if self._session_manager:
                self._session_manager.save_session(self._session)

    def export_notebook(self, filename: Optional[str] = None) -> Optional[Path]:
        """Export the current session to a Jupyter notebook.

        Generates a clean notebook with:
        - Consolidated imports at the top
        - Only successful code cells
        - Final answer and plan status

        Args:
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved notebook, or None if no executions to export
        """
        if not self._notebook_builder:
            return None

        # Check if there are any executions to export
        if not self._notebook_builder.tracker.records:
            return None

        # Get final answer if we have one
        answer = None
        if self._session and self._session.history:
            # Look for the last assistant message with an answer
            for msg in reversed(self._session.history.messages):
                if msg.role.value == "assistant" and "<answer>" in msg.content:
                    match = re.search(r'<answer>(.*?)</answer>', msg.content, re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        break

        # Generate clean notebook
        clean_notebook = self._notebook_builder.generate_clean_notebook(
            final_plan=self._current_plan,
            answer=answer,
        )

        # Save to session's notebooks directory
        if self._session and self._session.notebooks_path:
            clean_notebook._notebooks_path = Path(self._session.notebooks_path)

        return clean_notebook.save(filename)

    def get_live_notebook_path(self) -> Optional[Path]:
        """Get the path to the live notebook file if live mode is enabled.

        Returns:
            Path to the live notebook, or None if not in live mode
        """
        if self._notebook_sync:
            return self._notebook_sync.get_notebook_path()
        elif isinstance(self._notebook_builder, LiveNotebookBuilder):
            return self._notebook_builder.get_notebook_path()
        return None

    def __enter__(self) -> "ConversationalAgent":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
