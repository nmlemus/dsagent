"""Main agent engine for orchestrating the planning loop."""

from __future__ import annotations

import json
import re
from typing import Optional, Callable, Any, Generator, List, Dict, TYPE_CHECKING

from litellm import completion

from dsagent.schema.models import (
    AgentConfig,
    ExecutionResult,
    EventType,
    AgentEvent,
    PlanState,
    HITLMode,
    HITLAction,
    HumanFeedback,
)
from dsagent.core.planner import PlanParser
from dsagent.core.executor import JupyterExecutor
from dsagent.core.hitl import HITLGateway
from dsagent.utils.notebook import NotebookBuilder
from dsagent.utils.logger import AgentLogger, Colors
from dsagent.prompts import PromptBuilder

if TYPE_CHECKING:
    from dsagent.schema.models import Message
    from dsagent.utils.run_logger import RunLogger
    from dsagent.tools.mcp_manager import MCPManager
    from dsagent.observability import ObservabilityManager


# System prompt is now managed by PromptBuilder
# See dsagent/prompts/sections.py for prompt content


class AgentEngine:
    """Core engine that runs the agent loop.

    Handles:
    - LLM communication with fallbacks
    - Plan state management
    - Code execution via Jupyter kernel
    - Event streaming for UI updates

    Example:
        engine = AgentEngine(config, executor, logger)

        # Run with streaming
        for event in engine.run_stream("Analyze sales data"):
            print(event.type, event.message)

        # Or run synchronously
        result = engine.run("Analyze sales data")
    """

    # Stop sequences for LLM
    STOP_SEQUENCES = ["</code>", "</answer>"]

    def __init__(
        self,
        config: AgentConfig,
        executor: JupyterExecutor,
        logger: AgentLogger,
        notebook_builder: Optional[NotebookBuilder] = None,
        event_callback: Optional[Callable[[AgentEvent], Any]] = None,
        run_logger: Optional["RunLogger"] = None,
        hitl_gateway: Optional[HITLGateway] = None,
        mcp_manager: Optional["MCPManager"] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Agent configuration
            executor: Jupyter executor for code execution
            logger: Logger instance
            notebook_builder: Optional notebook builder for tracking
            event_callback: Optional callback for streaming events
            run_logger: Optional run logger for comprehensive logging
            hitl_gateway: Optional HITL gateway for human intervention
            mcp_manager: Optional MCP manager for external tools
        """
        self.config = config
        self.executor = executor
        self.logger = logger
        self.notebook = notebook_builder
        self.event_callback = event_callback
        self.run_logger = run_logger
        self.hitl = hitl_gateway
        self.mcp = mcp_manager

        self.messages: list[Message] = []
        self.current_plan: Optional[PlanState] = None
        self.round_num = 0
        self.answer: Optional[str] = None
        self._initial_plan_approved = False  # Track if first plan was approved

        # Initialize observability from environment
        self._observability: Optional["ObservabilityManager"] = None
        self._init_observability()

    def _emit(
        self,
        event_type: EventType,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentEvent:
        """Emit an event to the callback and logger."""
        event = self.logger.emit_event(event_type, message, **kwargs)
        if self.event_callback:
            self.event_callback(event)
        return event

    def _init_observability(self) -> None:
        """Initialize observability from environment variables."""
        try:
            from dsagent.observability import ObservabilityManager, ObservabilityConfig

            obs_config = ObservabilityConfig.from_env()
            if obs_config.enabled:
                self._observability = ObservabilityManager(obs_config)
                if self._observability.setup():
                    self.logger.debug("Observability initialized")
                else:
                    self._observability = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize observability: {e}")
            self._observability = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt with tools section if MCP is available."""
        tools = None
        if self.mcp and self.mcp.available_tools:
            tools = self.mcp.available_tools

        return PromptBuilder.build_engine_prompt(tools=tools)

    def _get_tools_for_llm(self) -> Optional[List[Dict[str, Any]]]:
        """Get tool definitions for LLM if MCP is available."""
        if self.mcp and self.mcp.available_tools:
            return self.mcp.get_tools_for_llm()
        return None

    def _handle_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            List of tool result messages
        """
        import time as _time
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Emit tool calling event
            self._emit(
                EventType.TOOL_CALLING,
                f"Calling tool: {tool_name}",
                data={"tool_name": tool_name, "arguments": arguments},
            )
            self.logger.print_status("🔧", f"Calling tool: {tool_name}")

            start_time = _time.time()
            success = False
            error = None

            try:
                # Use the synchronous API which uses MCPManager's dedicated event loop
                result = self.mcp.execute_tool_sync(tool_name, arguments)
                success = True
                self.logger.print_status("✅", f"Tool {tool_name} completed")

            except Exception as e:
                error = str(e)
                result = f"Error executing tool {tool_name}: {error}"
                self.logger.print_error(f"Tool error: {result}")

            execution_time_ms = (_time.time() - start_time) * 1000

            # Emit tool result event
            if success:
                self._emit(
                    EventType.TOOL_SUCCESS,
                    f"Tool {tool_name} succeeded",
                    data={
                        "tool_name": tool_name,
                        "result": result[:500] if len(result) > 500 else result,
                        "execution_time_ms": execution_time_ms,
                    },
                )
            else:
                self._emit(
                    EventType.TOOL_FAILED,
                    f"Tool {tool_name} failed: {error}",
                    data={
                        "tool_name": tool_name,
                        "error": error,
                        "execution_time_ms": execution_time_ms,
                    },
                )

            # Log tool execution if run_logger available
            if self.run_logger:
                self.run_logger.log_tool_execution(
                    tool_name=tool_name,
                    arguments=arguments,
                    success=success,
                    result=result if success else None,
                    error=error,
                    execution_time_ms=execution_time_ms,
                )

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        return results

    def _wait_for_hitl(
        self,
        event_type: EventType,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[AgentEvent, None, Optional[HumanFeedback]]:
        """Wait for HITL feedback (generator that yields awaiting event).

        Args:
            event_type: The HITL event type to emit
            message: Message for the event
            **kwargs: Additional event data

        Yields:
            AgentEvent indicating waiting for input

        Returns:
            Human feedback or None if HITL not enabled
        """
        if not self.hitl:
            return None

        # Emit awaiting event
        yield self._emit(event_type, message, awaiting_input=True, **kwargs)

        # Wait for feedback (blocking)
        feedback = self.hitl.wait_for_feedback()

        # Emit feedback received event
        if feedback:
            yield self._emit(
                EventType.HITL_FEEDBACK_RECEIVED,
                f"Received: {feedback.action.value}",
                feedback=feedback,
            )

        return feedback

    def _call_llm(self, messages: list[dict]) -> tuple[str, Optional[List[Any]]]:
        """Call the LLM with automatic fallbacks.

        Handles various provider-specific issues:
        - stop parameter not supported
        - temperature not supported
        - max_tokens vs max_completion_tokens

        Args:
            messages: Chat messages

        Returns:
            Tuple of (response text, tool_calls or None)
        """
        return self._call_llm_with_fallbacks(
            messages,
            use_stop=True,
            use_temperature=True,
            use_max_tokens=True,
        )

    def _call_llm_with_fallbacks(
        self,
        messages: list[dict],
        use_stop: bool = True,
        use_temperature: bool = True,
        use_max_tokens: bool = True,
        disable_thinking: bool = False,
    ) -> tuple[str, Optional[List[Any]]]:
        """Call LLM with recursive fallbacks for parameter issues.

        Args:
            messages: Chat messages
            use_stop: Whether to use stop sequences
            use_temperature: Whether to use temperature
            use_max_tokens: Whether to use max_tokens (vs max_completion_tokens)
            disable_thinking: Whether to disable thinking mode (for Gemini thought_signature issues)

        Returns:
            Tuple of (response text, tool_calls or None)
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        # Add tools if available
        tools = self._get_tools_for_llm()
        if tools:
            kwargs["tools"] = tools

        if use_stop and not tools:  # Don't use stop with tools
            kwargs["stop"] = self.STOP_SEQUENCES
        if use_temperature:
            kwargs["temperature"] = self.config.temperature
        if use_max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        else:
            kwargs["max_completion_tokens"] = self.config.max_tokens

        # Disable thinking mode for Gemini if requested (fallback for thought_signature issues)
        if disable_thinking and "gemini" in self.config.model.lower():
            kwargs["thinking"] = {"type": "disabled", "budget_tokens": 0}

        # Add observability metadata for tracing
        if self._observability and self._observability.is_active():
            kwargs["metadata"] = self._observability.get_call_metadata(
                call_type="engine",
            )

        try:
            response = completion(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = getattr(message, "tool_calls", None)

            # If stop was disabled, manually truncate at stop sequences
            if not use_stop and not tools:
                for stop_seq in self.STOP_SEQUENCES:
                    if stop_seq in content:
                        idx = content.index(stop_seq)
                        content = content[: idx + len(stop_seq)]
                        break

            return content, tool_calls

        except Exception as e:
            error_msg = str(e).lower()

            # Handle Gemini thought_signature error - disable thinking mode as fallback
            if not disable_thinking and "thought_signature" in error_msg:
                self.logger.warning(
                    "Gemini thought_signature error, retrying with thinking disabled"
                )
                return self._call_llm_with_fallbacks(
                    messages,
                    use_stop=use_stop,
                    use_temperature=use_temperature,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=True,
                )

            # Handle stop parameter not supported
            if use_stop and "stop" in error_msg:
                self.logger.warning(
                    f"Provider doesn't support 'stop', retrying without it"
                )
                return self._call_llm_with_fallbacks(
                    messages,
                    use_stop=False,
                    use_temperature=use_temperature,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=disable_thinking,
                )

            # Handle temperature not supported
            if use_temperature and "temperature" in error_msg:
                self.logger.warning(
                    f"Provider doesn't support 'temperature', retrying without it"
                )
                return self._call_llm_with_fallbacks(
                    messages,
                    use_stop=use_stop,
                    use_temperature=False,
                    use_max_tokens=use_max_tokens,
                    disable_thinking=disable_thinking,
                )

            # Handle max_tokens vs max_completion_tokens
            if use_max_tokens and "max_tokens" in error_msg:
                self.logger.warning(
                    f"Provider requires 'max_completion_tokens', retrying"
                )
                return self._call_llm_with_fallbacks(
                    messages,
                    use_stop=use_stop,
                    use_temperature=use_temperature,
                    use_max_tokens=False,
                    disable_thinking=disable_thinking,
                )

            raise

    def _should_reject_answer(self, plan: Optional[PlanState]) -> bool:
        """Check if the answer should be rejected because plan isn't complete.

        Args:
            plan: Current plan state

        Returns:
            True if answer should be rejected
        """
        if not plan:
            return False

        pending = [s for s in plan.steps if not s.completed]
        if pending:
            self.logger.warning(
                f"Rejecting early answer: {len(pending)} steps still pending"
            )
            return True
        return False

    def _execute_code(self, code: str, step_desc: str = "") -> ExecutionResult:
        """Execute code and track for notebook generation.

        Args:
            code: Python code to execute
            step_desc: Description of current step

        Returns:
            Execution result
        """
        result = self.executor.execute(code)

        # Track for notebook
        if self.notebook:
            self.notebook.track_execution(code, result, step_desc)

        return result

    def _build_context_message(
        self,
        code: str,
        result: ExecutionResult,
    ) -> str:
        """Build context message from execution result.

        Args:
            code: Executed code
            result: Execution result

        Returns:
            Formatted context message
        """
        output = result.output
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

    def run(self, task: str) -> str:
        """Run the agent loop synchronously.

        Args:
            task: User task to accomplish

        Returns:
            Final answer string
        """
        # Consume the generator
        for _ in self.run_stream(task):
            pass
        return self.answer or "No answer generated"

    def run_stream(self, task: str) -> Generator[AgentEvent, None, None]:
        """Run the agent loop with streaming events.

        Args:
            task: User task to accomplish

        Yields:
            AgentEvent for each significant event
        """
        self._emit(EventType.AGENT_STARTED, f"Starting task: {task}")

        # Initialize messages with dynamic system prompt
        self.messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Task: {task}"},
        ]

        self.round_num = 0
        self.answer = None

        while self.round_num < self.config.max_rounds:
            self.round_num += 1
            self.logger.set_round(self.round_num)

            yield self._emit(
                EventType.ROUND_STARTED,
                f"Round {self.round_num}/{self.config.max_rounds}",
            )

            # Get LLM response
            yield self._emit(EventType.LLM_CALL_STARTED)

            # Log LLM request
            if self.run_logger:
                self.run_logger.log_round_start(self.round_num)
                prompt = self.messages[-1].get("content", "") if self.messages else ""
                self.run_logger.log_llm_request(
                    prompt=prompt,
                    model=self.config.model,
                    messages=self.messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            try:
                import time
                start_time = time.time()
                response, tool_calls = self._call_llm(self.messages)
                latency_ms = (time.time() - start_time) * 1000

                # Handle tool calls if present
                if tool_calls:
                    self.logger.print_status("🔧", f"LLM requested {len(tool_calls)} tool(s)")

                    # Add assistant message with tool calls
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

                    self.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tool_calls": tool_calls_list,
                    })

                    # Execute tools and add results
                    tool_results = self._handle_tool_calls(tool_calls)
                    self.messages.extend(tool_results)

                    # Call LLM again with tool results
                    response, tool_calls = self._call_llm(self.messages)
                    latency_ms += (time.time() - start_time) * 1000

                # Log LLM response
                if self.run_logger:
                    self.run_logger.log_llm_response(
                        response=response,
                        latency_ms=latency_ms,
                        model=self.config.model,
                    )
            except Exception as e:
                if self.run_logger:
                    self.run_logger.log_error(str(e), error_type="llm_error")
                yield self._emit(EventType.AGENT_ERROR, f"LLM error: {e}")
                break

            yield self._emit(EventType.LLM_CALL_FINISHED, response[:200] + "...")

            # Parse response
            new_plan = PlanParser.parse_plan(response)
            code = PlanParser.extract_code(response)
            thinking = PlanParser.extract_thinking(response)
            has_answer = PlanParser.has_final_answer(response)

            # Update plan state
            if new_plan:
                plan_update = PlanParser.extract_plan_update(response)
                self.current_plan = new_plan
                yield self._emit(
                    EventType.PLAN_UPDATED,
                    plan_update,
                    plan=new_plan,
                )

                # Log plan update
                if self.run_logger:
                    self.run_logger.log_plan_update(
                        plan_text=new_plan.raw_text,
                        completed_steps=new_plan.completed_steps,
                        total_steps=new_plan.total_steps,
                        reason=plan_update,
                    )

                if self.logger.verbose:
                    self.logger.print_plan(new_plan.raw_text)

                # HITL: Check if we need plan approval (only for initial plan)
                if self.hitl and self.hitl.should_pause_for_plan() and not self._initial_plan_approved:
                    self.hitl.request_plan_approval(new_plan)
                    yield self._emit(
                        EventType.HITL_AWAITING_PLAN_APPROVAL,
                        "Waiting for plan approval",
                        plan=new_plan,
                        awaiting_input=True,
                    )

                    feedback = self.hitl.wait_for_feedback()

                    if feedback:
                        yield self._emit(
                            EventType.HITL_FEEDBACK_RECEIVED,
                            f"Received: {feedback.action.value}",
                            feedback=feedback,
                        )

                        if feedback.action == HITLAction.REJECT:
                            yield self._emit(EventType.HITL_PLAN_REJECTED, feedback.message)
                            yield self._emit(EventType.HITL_EXECUTION_ABORTED, "Plan rejected by user")
                            return

                        elif feedback.action == HITLAction.MODIFY and feedback.modified_plan:
                            # Inject modified plan into conversation
                            yield self._emit(EventType.HITL_PLAN_MODIFIED, "Plan modified by user")
                            self.messages.append({"role": "assistant", "content": response})
                            self.messages.append({
                                "role": "user",
                                "content": f"Please use this modified plan instead:\n\n{feedback.modified_plan}",
                            })
                            self._initial_plan_approved = True
                            continue

                        elif feedback.action == HITLAction.APPROVE:
                            yield self._emit(EventType.HITL_PLAN_APPROVED, feedback.message)

                    self._initial_plan_approved = True

            # Log thinking
            if thinking:
                if self.run_logger:
                    self.run_logger.log_thinking(thinking)
                if self.logger.verbose:
                    self.logger.print_status("💭", f"Thinking: {thinking[:100]}...")

            # Check for final answer
            if has_answer:
                # Validate that plan is complete
                if self._should_reject_answer(self.current_plan):
                    # Log rejection
                    if self.run_logger:
                        self.run_logger.log_answer(
                            answer=PlanParser.extract_answer(response) or "",
                            accepted=False,
                            rejection_reason="Plan not complete - pending steps remain",
                        )
                    # Ask agent to continue
                    self.messages.append({"role": "assistant", "content": response})
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Please complete all remaining plan steps before providing "
                            "the final answer. Some steps are still marked as [ ]."
                        ),
                    })
                    continue

                self.answer = PlanParser.extract_answer(response)

                # HITL: Check if we need answer approval (PLAN_AND_ANSWER or FULL mode)
                if self.hitl and self.hitl.should_pause_for_answer():
                    self.hitl.request_answer_approval(self.answer or "")
                    yield self._emit(
                        EventType.HITL_AWAITING_ANSWER_APPROVAL,
                        "Waiting for answer approval",
                        message=self.answer,
                        awaiting_input=True,
                    )

                    feedback = self.hitl.wait_for_feedback()

                    if feedback:
                        yield self._emit(
                            EventType.HITL_FEEDBACK_RECEIVED,
                            f"Received: {feedback.action.value}",
                            feedback=feedback,
                        )

                        if feedback.action == HITLAction.REJECT:
                            yield self._emit(EventType.HITL_EXECUTION_ABORTED, "Answer rejected by user")
                            return

                        elif feedback.action == HITLAction.FEEDBACK and feedback.message:
                            # Request more analysis based on feedback
                            self.messages.append({"role": "assistant", "content": response})
                            self.messages.append({
                                "role": "user",
                                "content": f"Please revise your answer based on this feedback: {feedback.message}",
                            })
                            self.answer = None
                            continue

                # Log accepted answer
                if self.run_logger:
                    self.run_logger.log_answer(answer=self.answer or "", accepted=True)
                yield self._emit(EventType.ANSWER_ACCEPTED, self.answer)
                break

            # Execute code if present
            if code:
                # HITL: Check if we need code approval before execution (FULL mode)
                if self.hitl and self.hitl.should_pause_for_code():
                    self.hitl.request_code_approval(code)
                    yield self._emit(
                        EventType.HITL_AWAITING_CODE_APPROVAL,
                        "Waiting for code approval",
                        code=code,
                        awaiting_input=True,
                    )

                    feedback = self.hitl.wait_for_feedback()

                    if feedback:
                        yield self._emit(
                            EventType.HITL_FEEDBACK_RECEIVED,
                            f"Received: {feedback.action.value}",
                            feedback=feedback,
                        )

                        if feedback.action == HITLAction.REJECT:
                            yield self._emit(EventType.HITL_EXECUTION_ABORTED, "Code rejected by user")
                            return

                        elif feedback.action == HITLAction.SKIP:
                            # Skip this code execution
                            self.messages.append({"role": "assistant", "content": response})
                            self.messages.append({
                                "role": "user",
                                "content": "Code execution was skipped by user. Please continue with the next step.",
                            })
                            continue

                        elif feedback.action == HITLAction.MODIFY and feedback.modified_code:
                            code = feedback.modified_code

                yield self._emit(EventType.CODE_EXECUTING, code[:100] + "...")

                if self.logger.verbose:
                    self.logger.print_code(code)

                # Get current step description
                step_desc = ""
                if self.current_plan:
                    current_step = self.current_plan.current_step
                    if current_step:
                        step_desc = current_step.description

                import time
                exec_start = time.time()
                result = self._execute_code(code, step_desc)
                exec_time_ms = (time.time() - exec_start) * 1000

                # Log code execution
                if self.run_logger:
                    self.run_logger.log_code_execution(
                        code=code,
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        images_count=len(result.images),
                        execution_time_ms=exec_time_ms,
                    )

                if result.success:
                    yield self._emit(
                        EventType.CODE_SUCCESS,
                        result.output[:500] if result.output else "(no output)",
                    )
                else:
                    yield self._emit(
                        EventType.CODE_FAILED,
                        result.output[:500] if result.output else "(no output)",
                    )

                    # HITL: Check if we need error guidance (ON_ERROR or FULL mode)
                    if self.hitl and self.hitl.should_pause_on_error():
                        self.hitl.request_error_guidance(code, result.output)
                        yield self._emit(
                            EventType.HITL_AWAITING_ERROR_GUIDANCE,
                            "Waiting for error guidance",
                            code=code,
                            error=result.output,
                            awaiting_input=True,
                        )

                        feedback = self.hitl.wait_for_feedback()

                        if feedback:
                            yield self._emit(
                                EventType.HITL_FEEDBACK_RECEIVED,
                                f"Received: {feedback.action.value}",
                                feedback=feedback,
                            )

                            if feedback.action == HITLAction.REJECT:
                                yield self._emit(EventType.HITL_EXECUTION_ABORTED, "Aborted by user after error")
                                return

                            elif feedback.action == HITLAction.SKIP:
                                self.messages.append({"role": "assistant", "content": response})
                                self.messages.append({
                                    "role": "user",
                                    "content": "Error was acknowledged. Please skip this step and continue with the next one.",
                                })
                                continue

                            elif feedback.action == HITLAction.FEEDBACK and feedback.message:
                                # Inject user feedback into conversation
                                self.messages.append({"role": "assistant", "content": response})
                                self.messages.append({
                                    "role": "user",
                                    "content": f"Code failed with error:\n{result.output}\n\nUser guidance: {feedback.message}",
                                })
                                continue

                if self.logger.verbose:
                    if result.success:
                        self.logger.print_output(result.output)
                    else:
                        self.logger.print_error(result.output)

                # Add context to conversation
                context = self._build_context_message(code, result)
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({"role": "user", "content": context})

                # Save notebook incrementally
                if self.notebook:
                    self.notebook.save_incremental()

            else:
                # No code, just add response and prompt to continue
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": "Please continue with the next step of your plan.",
                })

            # Log round end
            if self.run_logger:
                self.run_logger.log_round_end(self.round_num)

            yield self._emit(EventType.ROUND_FINISHED)

        # Handle max rounds reached
        if self.round_num >= self.config.max_rounds and not self.answer:
            self.answer = f"Max rounds ({self.config.max_rounds}) reached without completion."
            yield self._emit(EventType.AGENT_ERROR, self.answer)

        yield self._emit(EventType.AGENT_FINISHED, self.answer)

    def get_state(self) -> dict[str, Any]:
        """Get current engine state for persistence.

        Returns:
            State dictionary
        """
        return {
            "messages": self.messages,
            "round_num": self.round_num,
            "current_plan": self.current_plan.model_dump() if self.current_plan else None,
            "answer": self.answer,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore engine state from persistence.

        Args:
            state: State dictionary
        """
        self.messages = state.get("messages", [])
        self.round_num = state.get("round_num", 0)
        if state.get("current_plan"):
            self.current_plan = PlanState(**state["current_plan"])
        self.answer = state.get("answer")

    def shutdown(self) -> None:
        """Clean up resources.

        Call this when done with the engine to clean up observability and other resources.
        """
        if self._observability:
            try:
                self._observability.teardown()
            except Exception:
                pass
            self._observability = None
