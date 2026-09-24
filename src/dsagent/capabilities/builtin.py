"""Built-in providers and cartridge-to-composition resolution."""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from pathlib import Path

from dsagent.capabilities.core import (
    CapabilityError,
    CapabilityProvider,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilitySet,
)
from dsagent.cartridge.models import Cartridge, EnvSpec


def _kernel(spec: EnvSpec, workspace: Path):
    from dsagent.envs.kernel import make_kernel_env

    return make_kernel_env(spec, workspace)


def _docker(spec: EnvSpec, workspace: Path):
    from dsagent.envs.docker import make_docker_env

    return make_docker_env(spec, workspace)


def _execute(env, argv: list[str]):
    return env.backend.execute(" ".join(shlex.quote(value) for value in argv))


def _python_script(env, target: Path, argv: list[str]):
    return _execute(env, [env.python, target.as_posix(), *argv])


def _shell_script(env, target: Path, argv: list[str]):
    return _execute(env, ["bash", target.as_posix(), *argv])


def builtin_registry() -> CapabilityRegistry:
    """A new registry containing providers shipped with this DSAgent build."""
    registry = CapabilityRegistry()
    registry.register(CapabilityProvider(
        id="environment.kernel", kind="environment", version="1",
        factory=_kernel,
    ))
    registry.register(CapabilityProvider(
        id="environment.docker", kind="environment", version="1",
        factory=_docker,
    ))
    registry.register(CapabilityProvider(
        id="script.python", kind="script-runner", version="1",
        factory=_python_script, metadata={"extension": ".py"},
    ))
    registry.register(CapabilityProvider(
        id="script.shell", kind="script-runner", version="1",
        factory=_shell_script, metadata={"extension": ".sh"},
    ))
    return registry


def compose_cartridges(
    cartridges: Iterable[Cartridge], registry: CapabilityRegistry | CapabilitySet | None = None
) -> CapabilitySet:
    """Resolve every explicit and environment capability before work starts."""
    registry = registry if registry is not None else builtin_registry()
    cartridges = list(cartridges)
    try:
        requirements = requirements_for_cartridges(cartridges)
        resolve = registry.restrict if isinstance(registry, CapabilitySet) else registry.compose
        composition = resolve(requirements)
        # Validate routing within each consumer, not across a multi-domain host:
        # two cartridges may intentionally use different executors for .py.
        for cartridge in cartridges:
            scoped = composition.restrict(requirements_for_cartridges([cartridge]))
            for entry in scoped.manifest():
                extension = entry['metadata'].get('extension')
                if entry['kind'] == 'script-runner' and extension:
                    scoped.select(kind='script-runner', metadata=('extension', extension))
        return composition
    except (CapabilityError, ValueError) as error:
        names = ", ".join(cartridge.name for cartridge in cartridges)
        raise CapabilityError(f"cannot compose cartridge(s) {names}: {error}") from error


def requirements_for_cartridges(
    cartridges: Iterable[Cartridge],
) -> list[CapabilityRequirement]:
    """The executable host contracts declared or implied by cartridges."""
    requirements: list[CapabilityRequirement] = []
    for cartridge in cartridges:
        requirements.extend(
            CapabilityRequirement(item.id, item.version) for item in cartridge.capabilities
        )
        requirements.extend(
            CapabilityRequirement(spec.capability_id, kind="environment")
            for spec in cartridge.envs.values()
        )
        requirements.extend(
            CapabilityRequirement(step.gate.runner, kind="script-runner")
            for workflow in cartridge.workflows.values()
            for step in workflow.steps
            if step.gate is not None and step.gate.kind == "auto"
        )
    return requirements
