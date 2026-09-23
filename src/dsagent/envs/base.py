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

SKILLS_DIR = Path(".dsagent") / "skills"
"""Where a run's skills are materialized, relative to the workspace.

The harness knows this path without knowing what any skill *is*: it is workspace
layout, the same way `artifacts/` would be. It lives here rather than beside the
code that fills it because the Docker env has to bind-mount the directory and the
host builder has to write it, and two spellings of one path is how they drift.
"""


class EnvRequirementsError(RuntimeError):
    """An env was provisioned without the packages its cartridge declared."""


@dataclass
class Env:
    spec: EnvSpec
    workspace: Path
    backend: BackendProtocol
    tools: list[Callable[..., Any]] = field(default_factory=list)
    closers: list[Callable[[], None]] = field(default_factory=list)
    python: str = "python3"
    """The interpreter that runs inside this env.

    The kernel env is the interpreter running DSAgent, not whatever `python3`
    resolves to on PATH — those differ whenever DSAgent runs from a virtualenv,
    and the cartridge's requirements are installed in the former.
    """
    capabilities: Any = None
    """Frozen host composition that created or adopted this environment."""

    def close(self) -> None:  # pragma: no cover - trivial
        # Resources unwind in the opposite order from acquisition.
        for c in reversed(self.closers):
            c()
        closer = getattr(self.backend, "close", None)
        if closer:
            closer()


def make_env(spec: EnvSpec, workspace: Path, capabilities: Any = None) -> Env:
    """Provision an environment through its selected capability provider.

    ``capabilities`` is optional for API compatibility.  Direct callers receive
    a minimal built-in composition containing only this environment provider;
    runners and servers pass the broader, already-frozen product composition.
    """
    from dsagent.capabilities import builtin_registry

    workspace.mkdir(parents=True, exist_ok=True)
    composition = capabilities or builtin_registry().compose([spec.capability_id])
    provider = composition.provider(spec.capability_id, kind="environment")
    env = provider.factory(spec, workspace)
    env.capabilities = composition
    return env
