"""Jupyter Kernel Executor for persistent code execution.

Security note: Generated code runs in the kernel with the same user and no sandbox.
Malicious or buggy code can access the filesystem, network, and other resources.
For multi-user or untrusted environments, consider ephemeral containers per session
or resource limits (CPU/memory) at deployment level.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from jupyter_client import KernelManager

from dsagent.schema.models import ExecutionResult

if TYPE_CHECKING:
    from dsagent.utils.logger import AgentLogger


class JupyterExecutor:
    """Executes Python code in a persistent Jupyter kernel.

    The kernel maintains state between executions, allowing variables,
    imports, and objects to persist across multiple code blocks. There is
    no global kernel lifetime or memory limit; long sessions may accumulate
    resource use. Consider kernel recycling or time limits in shared deployments.

    Example:
        executor = JupyterExecutor(workspace="./workspace")
        executor.start()

        result1 = executor.execute("import pandas as pd")
        result2 = executor.execute("df = pd.DataFrame({'a': [1,2,3]})")
        result3 = executor.execute("print(df.shape)")  # (3, 1)

        executor.shutdown()
    """

    def __init__(
        self,
        workspace: str | Path,
        timeout: int = 300,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            workspace: Working directory for the kernel
            timeout: Maximum execution time in seconds (default: 5 minutes)
            logger: Optional logger for events
        """
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.logger = logger
        self.km: Optional[KernelManager] = None
        self.kc = None
        self._started = False

    @property
    def is_running(self) -> bool:
        """Check if the kernel is running."""
        return self._started and self.km is not None

    def start(self) -> None:
        """Start the Jupyter kernel."""
        if self._started:
            return

        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Start kernel
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel(cwd=str(self.workspace))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)

        # Initialize kernel environment
        workspace_path = str(self.workspace).replace("\\", "\\\\")
        setup_code = f"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import os
os.chdir('{workspace_path}')
"""
        self._execute_silent(setup_code)
        self._started = True

        if self.logger:
            self.logger.info(f"Jupyter kernel started in {self.workspace}")

    def _drain_iopub(self) -> None:
        """Drain any pending messages from the iopub channel."""
        if not self.kc:
            return
        while True:
            try:
                self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                break

    def _execute_silent(self, code: str) -> None:
        """Execute code without capturing output."""
        if not self.kc:
            return
        msg_id = self.kc.execute(code)

        # Wait for shell reply
        while True:
            try:
                msg = self.kc.get_shell_msg(timeout=30)
                if msg["msg_type"] == "execute_reply":
                    break
            except queue.Empty:
                break

        # Drain iopub to clear any output
        self._drain_iopub()

    def execute(self, code: str) -> ExecutionResult:
        """Execute code and capture all outputs.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with stdout, stderr, error, and images
        """
        if not self.kc:
            return ExecutionResult(error="Kernel not started", success=False)

        # Clear any stale messages
        self._drain_iopub()

        result = ExecutionResult()
        msg_id = self.kc.execute(code)

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
                msg_type = msg["msg_type"]
                content = msg["content"]

                # Only process messages for THIS execution
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")
                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    if content["name"] == "stdout":
                        result.stdout += content["text"]
                    elif content["name"] == "stderr":
                        text = content["text"]
                        # Filter out common warnings
                        if not text.startswith("WARNING") and "UserWarning" not in text:
                            result.stderr += text

                elif msg_type == "execute_result":
                    data = content.get("data", {})
                    if "text/plain" in data:
                        result.stdout += data["text/plain"] + "\n"
                    # Capture images
                    for mime in ["image/png", "image/jpeg", "image/svg+xml"]:
                        if mime in data:
                            result.images.append({"mime": mime, "data": data[mime]})

                elif msg_type == "display_data":
                    data = content.get("data", {})
                    if "text/plain" in data:
                        result.stdout += data["text/plain"] + "\n"
                    for mime in ["image/png", "image/jpeg", "image/svg+xml"]:
                        if mime in data:
                            result.images.append({"mime": mime, "data": data[mime]})

                elif msg_type == "error":
                    result.error = "\n".join(content.get("traceback", []))
                    result.success = False

                elif msg_type == "status":
                    if content["execution_state"] == "idle":
                        break

            except queue.Empty:
                result.error = f"Timeout after {self.timeout}s"
                result.success = False
                break
            except Exception as e:
                result.error = str(e)
                result.success = False
                break

        # Check shell reply for execution status
        try:
            reply = self.kc.get_shell_msg(timeout=10)
            if reply["content"]["status"] == "error":
                result.success = False
        except Exception:
            pass

        return result

    def get_variables(self) -> list[str]:
        """Get list of defined variables in the kernel."""
        result = self.execute("print([v for v in dir() if not v.startswith('_')])")
        if result.success:
            try:
                import ast
                return ast.literal_eval(result.stdout.strip())
            except Exception:
                pass
        return []

    def shutdown(self) -> None:
        """Shutdown the Jupyter kernel."""
        if self.kc:
            self.kc.stop_channels()
        if self.km:
            self.km.shutdown_kernel(now=True)
        self._started = False

        if self.logger:
            self.logger.info("Jupyter kernel stopped")

    def __enter__(self) -> "JupyterExecutor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
