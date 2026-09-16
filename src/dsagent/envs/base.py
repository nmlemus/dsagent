"""Execution environments.

An Env owns a Deep Agents backend (where `execute`, `read_file`, `write_file`
land) plus any extra tools it contributes (the kernel env adds a persistent
`run_python`). The run workspace is always mounted at the backend root, so
step instructions can use workspace-relative paths regardless of the env kind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol

from dsagent.cartridge.models import EnvSpec


class EnvRequirementsError(RuntimeError):
    """An env was provisioned without the packages its cartridge declared."""


@dataclass
class Env:
    spec: EnvSpec
    workspace: Path
    backend: BackendProtocol
    tools: list[Callable[..., Any]] = field(default_factory=list)
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:  # pragma: no cover - trivial
        for c in self.closers:
            c()
        closer = getattr(self.backend, "close", None)
        if closer:
            closer()


def make_env(spec: EnvSpec, workspace: Path) -> Env:
    workspace.mkdir(parents=True, exist_ok=True)
    if spec.kind == "kernel":
        from dsagent.envs.kernel import make_kernel_env

        return make_kernel_env(spec, workspace)
    if spec.kind == "docker":
        from dsagent.envs.docker import make_docker_env

        return make_docker_env(spec, workspace)
    raise ValueError(f"unknown env kind: {spec.kind}")
