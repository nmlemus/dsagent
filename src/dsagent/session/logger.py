"""Session logger for comprehensive event logging.

Logs events during conversational sessions to:
- run.log: Human-readable log
- events.jsonl: Structured JSON events for debugging and analysis
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dsagent.session.models import Session


class SessionLogger:
    """Logger for conversational sessions.

    Logs to two files in the session's logs directory:
    - run.log: Human-readable log with timestamps
    - events.jsonl: Structured JSON events for debugging

    Example:
        logger = SessionLogger(session)
        logger.log_llm_request(model="gpt-4o", messages=[...])
        logger.log_code_execution(code="...", success=True, output="...")
        logger.close()
    """

    def __init__(
        self,
        session: "Session",
        enabled: bool = True,
    ) -> None:
        """Initialize the session logger.

        Args:
            session: The session to log for
            enabled: Whether logging is enabled
        """
        self.session = session
        self.session_id = session.id
        self.enabled = enabled

        if not enabled or not session.logs_path:
            self._file_logger = None
            self._events_file = None
            return

        # Create logs directory
        logs_path = Path(session.logs_path)
        logs_path.mkdir(parents=True, exist_ok=True)

        # Setup file logger for human-readable logs
        self._file_logger = logging.getLogger(f"session_{self.session_id}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.handlers.clear()

        # File handler for run.log
        run_log_path = logs_path / "run.log"
        file_handler = logging.FileHandler(run_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)

        # Events file for JSONL (append mode)
        events_path = logs_path / "events.jsonl"
        self._events_file = open(events_path, "a", encoding="utf-8")

        # Metadata
        self._round_num = 0
        self._event_count = 0

        # Log session resumed/started
        self._log_event("session_resumed", {
            "session_id": self.session_id,
            "session_name": session.name,
            "timestamp": datetime.now().isoformat(),
        })
        self._file_logger.info(f"=== Session Resumed: {self.session_id} ===")

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event to events.jsonl."""
        if not self._events_file:
            return

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

    def set_round(self, round_num: int) -> None:
        """Set the current round number."""
        self._round_num = round_num

    def log_user_message(self, message: str) -> None:
        """Log a user message."""
        if not self._file_logger:
            return

        self._file_logger.info(f"[USER] {message[:200]}...")
        self._log_event("user_message", {
            "message": message,
            "length": len(message),
        })

    def log_llm_request(
        self,
        model: str,
        messages_count: int,
        temperature: float = 0.0,
        max_tokens: int = 0,
    ) -> None:
        """Log an LLM request."""
        if not self._file_logger:
            return

        self._file_logger.info(f"[LLM REQUEST] Model: {model}, Messages: {messages_count}")
        self._log_event("llm_request", {
            "model": model,
            "messages_count": messages_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

    def log_llm_response(
        self,
        response: str,
        tokens_used: int = 0,
        latency_ms: float = 0,
        model: str = "",
        has_code: bool = False,
        has_plan: bool = False,
        has_answer: bool = False,
    ) -> None:
        """Log an LLM response."""
        if not self._file_logger:
            return

        extras = []
        if has_code:
            extras.append("code")
        if has_plan:
            extras.append("plan")
        if has_answer:
            extras.append("answer")
        extras_str = f" [{', '.join(extras)}]" if extras else ""

        self._file_logger.info(
            f"[LLM RESPONSE] Tokens: {tokens_used}, Latency: {latency_ms:.0f}ms{extras_str}"
        )
        self._file_logger.debug(f"Response: {response[:300]}...")

        self._log_event("llm_response", {
            "response": response,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "model": model,
            "response_length": len(response),
            "has_code": has_code,
            "has_plan": has_plan,
            "has_answer": has_answer,
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
        """Log a code execution."""
        if not self._file_logger:
            return

        status = "SUCCESS" if success else "FAILED"
        self._file_logger.info(
            f"[CODE {status}] Lines: {len(code.splitlines())}, "
            f"Time: {execution_time_ms:.0f}ms"
        )

        if success:
            self._file_logger.debug(f"Output: {output[:200]}...")
        else:
            self._file_logger.warning(f"Error: {error or output}")

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
    ) -> None:
        """Log a plan update."""
        if not self._file_logger:
            return

        self._file_logger.info(
            f"[PLAN] Progress: {completed_steps}/{total_steps}"
        )
        self._file_logger.debug(f"Plan:\n{plan_text}")

        self._log_event("plan_update", {
            "plan_text": plan_text,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
        })

    def log_answer(self, answer: str) -> None:
        """Log a final answer."""
        if not self._file_logger:
            return

        self._file_logger.info(f"[ANSWER] Length: {len(answer)}")
        self._file_logger.info(f"Answer: {answer[:500]}...")

        self._log_event("answer", {
            "answer": answer,
            "answer_length": len(answer),
        })

    def log_thinking(self, thinking: str) -> None:
        """Log LLM thinking/reasoning (e.g., from Claude extended thinking)."""
        if not self._file_logger:
            return

        self._file_logger.debug(f"[THINKING] Length: {len(thinking)}")
        self._file_logger.debug(f"Thinking: {thinking[:500]}...")

        self._log_event("thinking", {
            "thinking": thinking,
            "thinking_length": len(thinking),
        })

    def log_round_start(self, round_num: int) -> None:
        """Log the start of a round."""
        if not self._file_logger:
            return

        self._round_num = round_num
        self._file_logger.info(f"{'='*50}")
        self._file_logger.info(f"ROUND {round_num}")
        self._file_logger.info(f"{'='*50}")

        self._log_event("round_start", {"round": round_num})

    def log_round_end(self, round_num: int) -> None:
        """Log the end of a round."""
        if not self._file_logger:
            return

        self._file_logger.info(f"--- End of Round {round_num} ---\n")
        self._log_event("round_end", {"round": round_num})

    def log_summarization(
        self,
        messages_summarized: int,
        messages_kept: int,
    ) -> None:
        """Log a conversation summarization event."""
        if not self._file_logger:
            return

        self._file_logger.info(
            f"[SUMMARIZATION] Summarized {messages_summarized} messages, "
            f"kept {messages_kept}"
        )

        self._log_event("summarization", {
            "messages_summarized": messages_summarized,
            "messages_kept": messages_kept,
        })

    def log_error(
        self,
        error: str,
        error_type: str = "general",
        traceback: Optional[str] = None,
    ) -> None:
        """Log an error."""
        if not self._file_logger:
            return

        self._file_logger.error(f"[ERROR] {error_type}: {error}")
        if traceback:
            self._file_logger.error(f"Traceback:\n{traceback}")

        self._log_event("error", {
            "error": error,
            "error_type": error_type,
            "traceback": traceback,
        })

    def log_artifact_saved(
        self,
        artifact_type: str,
        path: str,
        size_bytes: int = 0,
    ) -> None:
        """Log when an artifact is saved."""
        if not self._file_logger:
            return

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
        """Sanitize tool arguments by redacting sensitive keys and truncating long values.

        Args:
            arguments: Original arguments dictionary
            max_length: Maximum length for string values before truncation

        Returns:
            Sanitized arguments dictionary
        """
        # Keys that should be redacted
        sensitive_keys = {
            "api_key", "apikey", "api-key",
            "password", "passwd", "pwd",
            "token", "access_token", "auth_token", "bearer",
            "secret", "secret_key", "secretkey",
            "credential", "credentials",
            "private_key", "privatekey",
            "authorization", "auth",
        }

        sanitized = {}
        for key, value in arguments.items():
            key_lower = key.lower()

            # Check if key contains any sensitive pattern
            is_sensitive = any(sens in key_lower for sens in sensitive_keys)

            if is_sensitive:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > max_length:
                sanitized[key] = value[:max_length] + f"... (truncated, {len(value)} chars)"
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
        if not self._file_logger:
            return

        status = "SUCCESS" if success else "FAILED"
        self._file_logger.info(
            f"[TOOL {status}] {tool_name}, Time: {execution_time_ms:.0f}ms"
        )

        # Sanitize arguments for logging
        sanitized_args = self._sanitize_arguments(arguments)

        if success:
            # Truncate result for log file
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
        if not self._file_logger:
            return

        self._log_event("session_paused", {
            "session_id": self.session_id,
            "total_events": self._event_count,
            "end_time": datetime.now().isoformat(),
        })
        self._file_logger.info(f"=== Session Paused: {self.session_id} ===")
        self._file_logger.info(f"Events logged this session: {self._event_count}")

        # Close file handlers
        for handler in self._file_logger.handlers[:]:
            handler.close()
            self._file_logger.removeHandler(handler)

        if self._events_file:
            self._events_file.close()

    def __enter__(self) -> "SessionLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type and self._file_logger:
            self.log_error(
                str(exc_val),
                error_type=exc_type.__name__,
                traceback=str(exc_tb),
            )
        self.close()
