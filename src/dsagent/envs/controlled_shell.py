"""Local shell whose process group is durably owned by a workflow worker."""

from __future__ import annotations

import os
import subprocess
import tempfile

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from dsagent.execution_storage import durable_json
from dsagent.processes import PROCESS_DIR_ENV, identity, kill_group, resource_intent


class ControlledShellBackend(LocalShellBackend):
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not os.environ.get(PROCESS_DIR_ENV):
            return super().execute(command, timeout=timeout)
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
            # Journaling an intent before creation closes the unrecorded-spawn
            # gap conservatively: an incomplete intent blocks unsafe retries.
            resource = resource_intent("process")
            child = subprocess.Popen(command, shell=True, cwd=str(self.cwd),
                                     env=self._env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=output, start_new_session=True)
            token = identity(child.pid)
            durable_json(resource, {"kind": "process", "identity": token})
            try:
                code = child.wait(timeout=timeout or self._default_timeout)
            except subprocess.TimeoutExpired:
                kill_group(token)
                child.wait(timeout=3)
                code = 124
            finally:
                # Includes background children left by a shell that exited.
                kill_group(token)
            output.seek(0)
            text = output.read(self._max_output_bytes + 1)
            return ExecuteResponse(output=text[:self._max_output_bytes], exit_code=code,
                                   truncated=len(text) > self._max_output_bytes)
