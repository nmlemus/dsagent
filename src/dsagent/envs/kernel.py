"""Kernel env: local shell rooted at the workspace + a persistent Jupyter kernel.

Ported (conceptually) from dsagent v1 `kernel/local.py`: variables survive across
tool calls and across workflow steps that share the env, which is what EDA and
reporting want. Shell commands go through Deep Agents' `LocalShellBackend`, so
the agent gets `execute`, `read_file`, `write_file`, `ls`, `glob`, `grep` for free.
"""

from __future__ import annotations

import queue
from pathlib import Path

from deepagents.backends import LocalShellBackend
from langchain_core.tools import tool

from dsagent import requirements
from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env, EnvRequirementsError


class JupyterKernel:
    """Minimal persistent IPython kernel wrapper."""

    def __init__(self, cwd: Path, timeout: int = 600) -> None:
        from jupyter_client import KernelManager

        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel(cwd=str(cwd))
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)
        self.timeout = timeout

    def run(self, code: str) -> str:
        msg_id = self._kc.execute(code)
        out: list[str] = []
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=self.timeout)
            except queue.Empty:
                out.append(f"[timeout after {self.timeout}s]")
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "stream":
                out.append(c["text"])
            elif t in ("execute_result", "display_data"):
                data = c.get("data", {})
                out.append(data.get("text/plain", ""))
                if "image/png" in data:
                    out.append("[image/png output — save it to a file to keep it]")
            elif t == "error":
                out.append("\n".join(c.get("traceback", [])))
            elif t == "status" and c.get("execution_state") == "idle":
                break
        return "".join(out).strip() or "(no output)"

    def close(self) -> None:
        try:
            self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        # A dying kernel must not mask the error that got us here.
        except Exception:  # noqa: BLE001, S110  # pragma: no cover
            pass


def _verify_requirements(spec: EnvSpec) -> None:
    """Fail before the kernel exists rather than inside the persona's first call."""
    missing = requirements.missing(spec.requirements)
    if not missing:
        return
    raise EnvRequirementsError(
        f"env '{spec.name}' is missing {len(missing)} of its {len(spec.requirements)} "
        f"declared requirements: {', '.join(missing)}\n"
        f"install them with:\n"
        f"  {requirements.pip_install_command(missing)}\n"
        f"or, for every kernel env of the cartridge:\n"
        f"  dsagent cartridge install <cartridge path>"
    )


def make_kernel_env(spec: EnvSpec, workspace: Path) -> Env:
    _verify_requirements(spec)
    backend = LocalShellBackend(root_dir=workspace, virtual_mode=True, inherit_env=True)
    kernel: JupyterKernel | None = None

    @tool
    def run_python(code: str) -> str:
        """Run Python in a persistent kernel rooted at the workspace.

        Variables, imports and DataFrames persist between calls. Use this for
        analysis; use `execute` for shell commands. Save figures/tables to files
        under the workspace so later steps and the report can use them.
        """
        nonlocal kernel
        if kernel is None:
            kernel = JupyterKernel(workspace)
        return kernel.run(code)

    def _close() -> None:
        if kernel is not None:
            kernel.close()

    return Env(spec=spec, workspace=workspace, backend=backend, tools=[run_python], closers=[_close])
