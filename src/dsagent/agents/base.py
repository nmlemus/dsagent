"""Main PlannerAgent class - the primary interface for users."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Optional, Callable, Any, Generator, Dict, Union, TYPE_CHECKING
from datetime import datetime

from dsagent.config import get_default_model
from dsagent.schema.models import (
    AgentConfig,
    AgentEvent,
    EventType,
    SessionState,
    PlanState,
    HITLMode,
    HITLAction,
    HumanFeedback,
)
from dsagent.core.executor import JupyterExecutor
from dsagent.core.engine import AgentEngine
from dsagent.core.hitl import HITLGateway
from dsagent.utils.logger import AgentLogger
from dsagent.utils.notebook import NotebookBuilder
from dsagent.utils.validation import validate_configuration

if TYPE_CHECKING:
    from dsagent.core.context import RunContext
    from dsagent.utils.run_logger import RunLogger
    from dsagent.tools.mcp_manager import MCPManager
    from dsagent.tools.config import MCPConfig


class PlannerAgent:
    """High-level planner agent for data analysis tasks.

    This is the main class users interact with. It orchestrates:
    - Jupyter kernel for code execution
    - LLM communication via LiteLLM
    - Plan tracking and management
    - Notebook generation
    - Event streaming for UI integration

    Example - Basic usage:
        agent = PlannerAgent(
            model="gpt-4o",
            workspace="./workspace",
        )
        result = agent.run("Analyze sales_data.csv and find trends")
        print(result.answer)
        print(f"Notebook saved to: {result.notebook_path}")

    Example - With streaming:
        agent = PlannerAgent(model="claude-3-sonnet-20240229")

        for event in agent.run_stream("Build a predictive model"):
            if event.type == EventType.CODE_COMPLETED:
                print(f"Code executed: {event.success}")
            elif event.type == EventType.ANSWER:
                print(f"Answer: {event.message}")

    Example - FastAPI integration:
        @app.post("/analyze")
        async def analyze(task: str):
            agent = PlannerAgent(
                session_id=str(uuid4()),
                event_callback=send_sse_event,
            )
            return agent.run(task)
    """

    def __init__(
        self,
        model: Optional[str] = None,  # Resolved via get_default_model() if None
        workspace: str | Path = "./workspace",
        data: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        max_rounds: int = 30,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout: int = 300,
        verbose: bool = True,
        event_callback: Optional[Callable[[AgentEvent], Any]] = None,
        context: Optional["RunContext"] = None,
        hitl: HITLMode = HITLMode.NONE,
        hitl_timeout: float = 300.0,
        mcp_config: Optional[Union[str, Path, Dict, "MCPConfig"]] = None,
    ) -> None:
        """Initialize the planner agent.

        Args:
            model: LLM model to use (any LiteLLM-supported model)
            workspace: Working directory for files and notebooks
            data: Path to data file or directory (optional, copied to workspace/data/)
            session_id: Unique session identifier (for multi-user scenarios)
            max_rounds: Maximum agent loop iterations
            max_tokens: Max tokens per LLM response
            temperature: LLM temperature (0.0-1.0)
            timeout: Code execution timeout in seconds
            verbose: Print to console
            event_callback: Callback for streaming events to UI
            context: RunContext for organized workspace structure
            hitl: Human-in-the-Loop mode (NONE, PLAN_ONLY, ON_ERROR, PLAN_AND_ANSWER, FULL)
            hitl_timeout: Max seconds to wait for human feedback (default 5 min)
            mcp_config: MCP configuration (path to YAML, dict, or MCPConfig object)

        Raises:
            ConfigurationError: If the model or API key configuration is invalid
            FileNotFoundError: If data path does not exist
        """
        # Resolve model if not explicitly set
        effective_model = get_default_model(explicit=model)

        # Validate model and API key before anything else
        validate_configuration(effective_model)

        # Store or create context
        self.context = context
        if context:
            self.workspace = context.run_path
        else:
            self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Create workspace structure (data/ and artifacts/ folders)
        self._data_path = self.workspace / "data"
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._artifacts_path = self.workspace / "artifacts"
        self._artifacts_path.mkdir(parents=True, exist_ok=True)
        self._has_data = False

        # Copy data if provided
        if data:
            data_path = Path(data).resolve()
            if not data_path.exists():
                raise FileNotFoundError(f"Data path does not exist: {data_path}")

            if data_path.is_file():
                shutil.copy2(data_path, self._data_path / data_path.name)
            else:
                # Copy directory contents
                for item in data_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, self._data_path / item.name)
            self._has_data = True

        # Create configuration
        self.config = AgentConfig(
            model=effective_model,
            session_id=session_id,
            max_rounds=max_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
            workspace=str(self.workspace),
        )

        # Initialize run logger if context provided
        self._run_logger: Optional["RunLogger"] = None
        if context:
            from dsagent.utils.run_logger import RunLogger
            self._run_logger = RunLogger(context)

        # Initialize components
        self.executor = JupyterExecutor(
            workspace=self.workspace,
            timeout=timeout,
        )

        self.logger = AgentLogger(
            verbose=verbose,
            event_callback=event_callback,
        )

        # Initialize HITL gateway if enabled
        self._hitl_gateway: Optional[HITLGateway] = None
        if hitl != HITLMode.NONE:
            self._hitl_gateway = HITLGateway(mode=hitl, timeout=hitl_timeout)

        # Initialize MCP manager if config provided
        self._mcp_manager: Optional["MCPManager"] = None
        self._mcp_config = mcp_config

        self.event_callback = event_callback
        self._engine: Optional[AgentEngine] = None
        self._notebook: Optional[NotebookBuilder] = None
        self._started = False

    def start(self) -> None:
        """Start the agent (initializes Jupyter kernel and MCP connections).

        Call this before running tasks, or use the context manager.
        """
        if self._started:
            return

        self.executor.start()

        # Initialize MCP manager if config provided
        if self._mcp_config:
            self._init_mcp()

        self._started = True
        self.logger.info("PlannerAgent started")

    def _init_mcp(self) -> None:
        """Initialize MCP manager from config."""
        try:
            from dsagent.tools.mcp_manager import MCPManager
            from dsagent.tools.config import MCPConfig

            # Create manager based on config type
            if isinstance(self._mcp_config, (str, Path)):
                self._mcp_manager = MCPManager.from_yaml(self._mcp_config)
            elif isinstance(self._mcp_config, dict):
                self._mcp_manager = MCPManager.from_dict(self._mcp_config)
            elif isinstance(self._mcp_config, MCPConfig):
                self._mcp_manager = MCPManager(self._mcp_config)
            else:
                self.logger.warning(f"Unknown MCP config type: {type(self._mcp_config)}")
                return

            # Connect using the synchronous API (keeps event loop alive for tool calls)
            self._mcp_manager.connect_all_sync()

            if self._mcp_manager.connected_servers:
                self.logger.info(
                    f"MCP: Connected to {len(self._mcp_manager.connected_servers)} server(s), "
                    f"{len(self._mcp_manager.available_tools)} tool(s) available"
                )
            else:
                self.logger.warning("MCP: No servers connected")

        except ImportError:
            self.logger.warning(
                "MCP package not installed. Install with: pip install 'datascience-agent[mcp]'"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize MCP: {e}")

    def shutdown(self) -> None:
        """Shutdown the agent (stops Jupyter kernel and MCP connections)."""
        if not self._started:
            return

        # Close run logger if present
        if self._run_logger:
            self._run_logger.close()

        # Disconnect MCP servers using sync API
        if self._mcp_manager:
            try:
                self._mcp_manager.disconnect_all_sync()
            except Exception as e:
                self.logger.warning(f"Error disconnecting MCP: {e}")

        self.executor.shutdown()
        self._started = False
        self.logger.info("PlannerAgent stopped")

    def __enter__(self) -> "PlannerAgent":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()

    # ==================== HITL Methods ====================

    @property
    def hitl(self) -> Optional[HITLGateway]:
        """Access to HITL gateway for providing feedback."""
        return self._hitl_gateway

    @property
    def is_awaiting_input(self) -> bool:
        """Check if agent is waiting for human input."""
        return self._hitl_gateway is not None and self._hitl_gateway.is_awaiting_feedback

    def approve(self, message: Optional[str] = None) -> None:
        """Approve the current pending item (plan, code, or answer).

        Args:
            message: Optional message to include
        """
        if self._hitl_gateway:
            self._hitl_gateway.approve(message)

    def reject(self, message: Optional[str] = None) -> None:
        """Reject and abort execution.

        Args:
            message: Optional rejection reason
        """
        if self._hitl_gateway:
            self._hitl_gateway.reject(message)

    def modify_plan(self, new_plan: str, message: Optional[str] = None) -> None:
        """Provide a modified plan.

        Args:
            new_plan: The modified plan text
            message: Optional message
        """
        if self._hitl_gateway:
            self._hitl_gateway.modify_plan(new_plan, message)

    def modify_code(self, new_code: str, message: Optional[str] = None) -> None:
        """Provide modified code.

        Args:
            new_code: The modified code
            message: Optional message
        """
        if self._hitl_gateway:
            self._hitl_gateway.modify_code(new_code, message)

    def skip(self, message: Optional[str] = None) -> None:
        """Skip the current step.

        Args:
            message: Optional message
        """
        if self._hitl_gateway:
            self._hitl_gateway.skip(message)

    def send_feedback(self, message: str) -> None:
        """Send feedback to guide the agent.

        Args:
            message: Feedback message
        """
        if self._hitl_gateway:
            self._hitl_gateway.send_feedback(message)

    def provide_feedback(self, feedback: HumanFeedback) -> None:
        """Provide complete feedback object.

        Args:
            feedback: HumanFeedback object
        """
        if self._hitl_gateway:
            self._hitl_gateway.provide_feedback(feedback)

    def run(self, task: str) -> "AgentResult":
        """Run the agent on a task synchronously.

        Args:
            task: The user's task description

        Returns:
            AgentResult with answer, notebook path, etc.
        """
        # Consume the generator
        events = list(self.run_stream(task))
        return self._build_result(events)

    def run_stream(self, task: str) -> Generator[AgentEvent, None, None]:
        """Run the agent with streaming events.

        Args:
            task: The user's task description

        Yields:
            AgentEvent for each significant event
        """
        # Auto-start if needed
        if not self._started:
            self.start()

        # Add data context to task if data was provided
        if self._has_data:
            task_with_context = f"""{task}

The data is available in the 'data/' subdirectory of the current working directory.
List files in 'data/' first to see what's available."""
        else:
            task_with_context = task

        # Initialize notebook builder with context if available
        self._notebook = NotebookBuilder(
            task=task,  # Use original task for notebook title
            workspace=self.workspace,
            context=self.context,
        )

        # Create engine with run logger, HITL gateway, and MCP manager if available
        self._engine = AgentEngine(
            config=self.config,
            executor=self.executor,
            logger=self.logger,
            notebook_builder=self._notebook,
            event_callback=self.event_callback,
            run_logger=self._run_logger,
            hitl_gateway=self._hitl_gateway,
            mcp_manager=self._mcp_manager,
        )

        # Run the agent loop
        self.logger.print_header(f"Task: {task}")

        yield from self._engine.run_stream(task_with_context)

        # Generate clean notebook at the end
        self.logger.print_header("Task Complete")

    def get_result(self) -> "AgentResult":
        """Get the result after streaming completes.

        Call this after consuming all events from run_stream() to get
        the final result including the generated notebook.

        Returns:
            AgentResult with answer, notebook path, etc.
        """
        if not self._engine:
            raise RuntimeError("No run in progress. Call run_stream() first.")

        answer = self._engine.answer
        notebook_path = None

        # Generate and save clean notebook
        if self._notebook:
            clean_notebook = self._notebook.generate_clean_notebook(
                final_plan=self._engine.current_plan,
                answer=answer,
            )
            notebook_path = clean_notebook.save()

        return AgentResult(
            answer=answer or "No answer generated",
            notebook_path=notebook_path,
            events=[],  # Events were already consumed
            plan=self._engine.current_plan,
            rounds=self._engine.round_num,
        )

    def _build_result(self, events: list[AgentEvent]) -> "AgentResult":
        """Build result object from events.

        Args:
            events: List of all events from the run

        Returns:
            AgentResult object
        """
        answer = None
        notebook_path = None

        # Find answer event
        for event in events:
            if event.type == EventType.ANSWER_ACCEPTED:
                answer = event.message
                break

        # Generate and save clean notebook
        if self._engine and self._notebook:
            clean_notebook = self._notebook.generate_clean_notebook(
                final_plan=self._engine.current_plan,
                answer=answer,
            )
            notebook_path = clean_notebook.save()

        return AgentResult(
            answer=answer or "No answer generated",
            notebook_path=notebook_path,
            events=events,
            plan=self._engine.current_plan if self._engine else None,
            rounds=self._engine.round_num if self._engine else 0,
        )

    def serialize_state(self) -> str:
        """Serialize agent state for persistence.

        TODO: This is experimental and incomplete. Current limitations:
        - SessionState model fields don't match what's being serialized
        - Jupyter kernel state (variables in memory) is NOT preserved
        - Full session restore would require re-executing all cells

        Returns:
            JSON string of agent state (partial, for inspection only)
        """
        state = SessionState(
            session_id=self.config.session_id or "default",
            config=self.config,
            messages=self._engine.messages if self._engine else [],
        )
        return state.model_dump_json(indent=2)

    def restore_state(self, state_json: str) -> None:
        """Restore agent state from persistence.

        TODO: This is experimental and incomplete. Does NOT restore:
        - Jupyter kernel state (variables, dataframes, models in memory)
        - Would need to re-execute all previous cells to fully restore

        Args:
            state_json: JSON string of agent state
        """
        state = SessionState.model_validate_json(state_json)
        self.config = state.config

        if self._engine:
            self._engine.messages = state.messages


class AgentResult:
    """Result from running the planner agent.

    Attributes:
        answer: The final answer text
        notebook_path: Path to generated Jupyter notebook
        events: All events from the run
        plan: Final plan state
        rounds: Number of rounds executed
    """

    def __init__(
        self,
        answer: str,
        notebook_path: Optional[Path],
        events: list[AgentEvent],
        plan: Optional[PlanState],
        rounds: int,
    ) -> None:
        """Initialize the result.

        Args:
            answer: Final answer text
            notebook_path: Path to notebook
            events: All events
            plan: Final plan state
            rounds: Number of rounds
        """
        self.answer = answer
        self.notebook_path = notebook_path
        self.events = events
        self.plan = plan
        self.rounds = rounds

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AgentResult(answer={self.answer[:50]}..., "
            f"notebook={self.notebook_path}, rounds={self.rounds})"
        )

    @property
    def success(self) -> bool:
        """Check if the run was successful."""
        return self.answer and "error" not in self.answer.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "notebook_path": str(self.notebook_path) if self.notebook_path else None,
            "rounds": self.rounds,
            "plan": self.plan.raw_text if self.plan else None,
            "success": self.success,
        }
