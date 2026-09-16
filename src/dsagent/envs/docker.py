"""Docker env: one container per run, workspace bind-mounted at /workspace.

Implements Deep Agents' `BaseSandbox`, which only needs `execute()`,
`upload_files()` and `download_files()`; ls/read/write/edit/glob/grep are
derived from those. This is the M2 piece — it is functional but untested
until a Docker daemon is available.
"""

from __future__ import annotations

import base64
import shlex
import subprocess
import uuid
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env


class DockerBackend(BaseSandbox):
    def __init__(self, image: str, workspace: Path, *, gpu: bool = False, timeout: int = 3600) -> None:
        self._id = f"dsagent-{uuid.uuid4().hex[:12]}"
        self.timeout = timeout
        cmd = ["docker", "run", "-d", "--name", self._id, "-v", f"{workspace}:/workspace", "-w", "/workspace"]
        if gpu:
            cmd += ["--gpus", "all"]
        cmd += [image, "sleep", "infinity"]
        subprocess.run(cmd, check=True, capture_output=True)

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            r = subprocess.run(
                ["docker", "exec", self._id, "bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                check=False,
            )
            return ExecuteResponse(output=r.stdout + r.stderr, exit_code=r.returncode, truncated=False)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"[timeout after {timeout or self.timeout}s]", exit_code=124, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        out: list[FileUploadResponse] = []
        for path, data in files:
            b64 = base64.b64encode(data).decode()
            r = self.execute(f"mkdir -p $(dirname {shlex.quote(path)}) && echo {b64} | base64 -d > {shlex.quote(path)}")
            out.append(FileUploadResponse(path=path, error=None if r.exit_code == 0 else "permission_denied"))
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        out: list[FileDownloadResponse] = []
        for path in paths:
            r = self.execute(f"base64 -w0 {shlex.quote(path)}")
            if r.exit_code == 0:
                out.append(FileDownloadResponse(path=path, content=base64.b64decode(r.output.strip()), error=None))
            else:
                out.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
        return out

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self._id], capture_output=True, check=False)


def ensure_image(spec: EnvSpec, cartridge_name: str) -> str:
    if spec.image:
        return spec.image
    assert spec.build is not None
    tag = f"dsagent/{cartridge_name}-{spec.name}:latest"
    subprocess.run(["docker", "build", "-t", tag, str(spec.build)], check=True)
    return tag


def make_docker_env(spec: EnvSpec, workspace: Path, cartridge_name: str = "cartridge") -> Env:
    image = ensure_image(spec, cartridge_name)
    backend = DockerBackend(image, workspace, gpu=spec.gpu == "required")
    return Env(spec=spec, workspace=workspace, backend=backend)
