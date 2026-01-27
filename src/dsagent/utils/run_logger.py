"""Run logger for comprehensive execution logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dsagent.core.context import RunContext


class RunLogger:
    """Comprehensive logger for agent runs.

    Logs to two files:
    - run.log: Human-readable log with timestamps
    - events.jsonl: Structured JSON events for ML training

    All events are logged, including failures, for later analysis
    and model retraining.

    Example:
        context = RunContext(workspace="./workspace")
        logger = RunLogger(context)

        logger.log_llm_request(prompt="...", model="gpt-4o")
        logger.log_llm_response(response="...", tokens=150, latency=1.2)
        logger.log_code_execution(code="...", success=True, output="...")
    """

    def __init__(self, context: "RunContext") -> None:
        """Initialize the run logger.

        Args:
            context: Run context with log paths
        """
        self.context = context
        self.run_id = context.run_id

        # Setup file logger for human-readable logs
        self._file_logger = logging.getLogger(f"run_{self.run_id}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.handlers.clear()

        # File handler for run.log
        file_handler = logging.FileHandler(
            context.run_log_path, mode="w", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)

        # Events file for JSONL
        self._events_file = open(context.events_log_path, "w", encoding="utf-8")

        # Metadata
        self._round_num = 0
        self._event_count = 0

        # Log session start
        self._log_event("session_start", {
            "run_id": self.run_id,
            "workspace": str(context.workspace),
            "start_time": context.start_time.isoformat(),
        })
        self._file_logger.info(f"=== Run Started: {self.run_id} ===")

    def set_round(self, round_num: int) -> None:
        """Set the current round number."""
        self._round_num = round_num

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event to events.jsonl.

        Args:
            event_type: Type of event
            data: Event data
        """
        self._event_count += 1
        event = {
            "event_id": self._event_count,
            "timestamp": datetime.now().isoformat(),
            "round": self._round_num,
            "type": event_type,
            "data": data,
        }
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def log_llm_request(
        self,
        prompt: str,
        model: str,
        messages: Optional[list] = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
    ) -> None:
        """Log an LLM request.

        Args:
            prompt: The user prompt
            model: Model name
            messages: Full message history
            temperature: LLM temperature
            max_tokens: Max tokens setting
        """
        self._file_logger.info(f"[LLM REQUEST] Model: {model}")
        self._file_logger.debug(f"Prompt: {prompt[:200]}...")

        self._log_event("llm_request", {
            "model": model,
            "prompt": prompt,
            "messages_count": len(messages) if messages else 0,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

    def log_llm_response(
        self,
        response: str,
        tokens_used: int = 0,
        latency_ms: float = 0,
        model: str = "",
    ) -> None:
        """Log an LLM response.

        Args:
            response: The LLM response text
            tokens_used: Total tokens used
            latency_ms: Response latency in milliseconds
            model: Model that generated the response
        """
        self._file_logger.info(
            f"[LLM RESPONSE] Tokens: {tokens_used}, Latency: {latency_ms:.0f}ms"
        )
        self._file_logger.debug(f"Response: {response[:200]}...")

        self._log_event("llm_response", {
            "response": response,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "model": model,
            "response_length": len(response),
        })

    def log_code_execution(
        self,
        code: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
        images_count: int = 0,
        execution_time_ms: float = 0,
    ) -> None:
        """Log a code execution.

        Args:
            code: The executed code
            success: Whether execution succeeded
            output: Execution output
            error: Error message if failed
            images_count: Number of images generated
            execution_time_ms: Execution time in milliseconds
        """
        status = "SUCCESS" if success else "FAILED"
        self._file_logger.info(f"[CODE {status}] Lines: {len(code.splitlines())}")

        if success:
            self._file_logger.debug(f"Output: {output[:200]}...")
        else:
            self._file_logger.warning(f"Error: {error}")

        self._log_event("code_execution", {
            "code": code,
            "success": success,
            "output": output,
            "error": error,
            "images_count": images_count,
            "execution_time_ms": execution_time_ms,
            "code_lines": len(code.splitlines()),
        })

    def log_plan_update(
        self,
        plan_text: str,
        completed_steps: int = 0,
        total_steps: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        """Log a plan update.

        Args:
            plan_text: The full plan text
            completed_steps: Number of completed steps
            total_steps: Total number of steps
            reason: Reason for update
        """
        self._file_logger.info(
            f"[PLAN UPDATE] Progress: {completed_steps}/{total_steps}"
        )
        self._file_logger.debug(f"Plan:\n{plan_text}")

        self._log_event("plan_update", {
            "plan_text": plan_text,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "reason": reason,
        })

    def log_answer(
        self,
        answer: str,
        accepted: bool,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Log an answer attempt.

        Args:
            answer: The answer text
            accepted: Whether the answer was accepted
            rejection_reason: Reason for rejection if not accepted
        """
        status = "ACCEPTED" if accepted else "REJECTED"
        self._file_logger.info(f"[ANSWER {status}]")
        self._file_logger.info(f"Answer: {answer[:500]}...")

        if not accepted and rejection_reason:
            self._file_logger.warning(f"Rejection reason: {rejection_reason}")

        self._log_event("answer", {
            "answer": answer,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "answer_length": len(answer),
        })

    def log_thinking(self, thinking: str) -> None:
        """Log agent thinking/reasoning.

        Args:
            thinking: The thinking text
        """
        self._file_logger.debug(f"[THINKING] {thinking[:200]}...")

        self._log_event("thinking", {
            "thinking": thinking,
            "length": len(thinking),
        })

    def log_error(
        self,
        error: str,
        error_type: str = "general",
        traceback: Optional[str] = None,
    ) -> None:
        """Log an error.

        Args:
            error: Error message
            error_type: Type of error
            traceback: Full traceback if available
        """
        self._file_logger.error(f"[ERROR] {error_type}: {error}")
        if traceback:
            self._file_logger.error(f"Traceback:\n{traceback}")

        self._log_event("error", {
            "error": error,
            "error_type": error_type,
            "traceback": traceback,
        })

    def log_round_start(self, round_num: int) -> None:
        """Log the start of a round.

        Args:
            round_num: Round number
        """
        self._round_num = round_num
        self._file_logger.info(f"{'='*50}")
        self._file_logger.info(f"ROUND {round_num}")
        self._file_logger.info(f"{'='*50}")

        self._log_event("round_start", {"round": round_num})

    def log_round_end(self, round_num: int) -> None:
        """Log the end of a round.

        Args:
            round_num: Round number
        """
        self._file_logger.info(f"--- End of Round {round_num} ---\n")
        self._log_event("round_end", {"round": round_num})

    def log_artifact_saved(
        self,
        artifact_type: str,
        path: str,
        size_bytes: int = 0,
    ) -> None:
        """Log when an artifact is saved.

        Args:
            artifact_type: Type of artifact (image, notebook, etc.)
            path: Path where saved
            size_bytes: Size in bytes
        """
        self._file_logger.info(f"[ARTIFACT] {artifact_type}: {path}")

        self._log_event("artifact_saved", {
            "artifact_type": artifact_type,
            "path": path,
            "size_bytes": size_bytes,
        })

    def _sanitize_arguments(
        self,
        arguments: Dict[str, Any],
        max_length: int = 500,
    ) -> Dict[str, Any]:
        """Sanitize tool arguments by redacting sensitive keys and truncating.

        Args:
            arguments: Original arguments dictionary
            max_length: Maximum length for string values

        Returns:
            Sanitized arguments dictionary
        """
        sensitive_keys = {
            "api_key", "apikey", "password", "token", "secret",
            "credential", "private_key", "authorization", "auth",
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
                sanitized[key] = self._sanitize_arguments(value, max_length)
            else:
                sanitized[key] = value

        return sanitized

    def log_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        result: Optional[str] = None,
        error: Optional[str] = None,
        execution_time_ms: float = 0,
    ) -> None:
        """Log a tool (MCP) execution.

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments (will be sanitized)
            success: Whether the tool execution succeeded
            result: Result from the tool (if success)
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
        """
        status = "SUCCESS" if success else "FAILED"
        self._file_logger.info(
            f"[TOOL {status}] {tool_name}, Time: {execution_time_ms:.0f}ms"
        )

        sanitized_args = self._sanitize_arguments(arguments)

        if success:
            result_preview = result[:200] + "..." if result and len(result) > 200 else result
            self._file_logger.debug(f"Result: {result_preview}")
        else:
            self._file_logger.warning(f"Error: {error}")

        self._log_event("tool_execution", {
            "tool_name": tool_name,
            "arguments": sanitized_args,
            "success": success,
            "result": result[:1000] if result and len(result) > 1000 else result,
            "error": error,
            "execution_time_ms": execution_time_ms,
        })

    def close(self) -> None:
        """Close the logger and flush all buffers."""
        self._log_event("session_end", {
            "run_id": self.run_id,
            "total_events": self._event_count,
            "end_time": datetime.now().isoformat(),
        })
        self._file_logger.info(f"=== Run Completed: {self.run_id} ===")
        self._file_logger.info(f"Total events logged: {self._event_count}")

        # Close file handlers
        for handler in self._file_logger.handlers[:]:
            handler.close()
            self._file_logger.removeHandler(handler)

        self._events_file.close()

    def __enter__(self) -> "RunLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type:
            self.log_error(
                str(exc_val),
                error_type=exc_type.__name__,
                traceback=str(exc_tb),
            )
        self.close()
