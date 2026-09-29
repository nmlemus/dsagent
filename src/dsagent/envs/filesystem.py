"""Workspace-only environment for personas that cannot execute code."""

from pathlib import Path

from deepagents.backends import FilesystemBackend

from dsagent.cartridge.models import EnvSpec
from dsagent.envs.base import Env


def make_filesystem_env(spec: EnvSpec, workspace: Path) -> Env:
    return Env(
        spec=spec,
        workspace=workspace,
        backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        runtime_identity={"filesystem_boundary": "workspace-v1"},
    )
