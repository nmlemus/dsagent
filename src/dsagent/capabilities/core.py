"""Small, explicit dependency container for trusted host capabilities.

This is deliberately not a dynamic plugin loader.  Provider code is registered
by the application at boot, dependencies are checked, and a frozen composition
is handed to the execution path.  Installing arbitrary code and changing a
running job's providers are outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")


class CapabilityError(RuntimeError):
    """A requested composition cannot be built safely."""


@dataclass(frozen=True)
class CapabilityRequirement:
    """A provider id and, optionally, the exact contract version required."""

    id: str
    version: str | None = None

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError(f"invalid capability id: {self.id!r}")


@dataclass(frozen=True)
class CapabilityProvider:
    """One trusted implementation registered by the host at boot.

    ``factory`` is intentionally opaque here.  The consumer owns the typed
    boundary: environment consumers pass ``(EnvSpec, workspace)`` and script
    runners pass ``(Env, script_path, argv)``.  ``kind`` prevents accidentally
    wiring one surface into another even though both are callables.
    """

    id: str
    kind: str
    version: str
    factory: Callable[..., Any] = field(repr=False, compare=False)
    requires: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        CapabilityRequirement(self.id)
        if not self.kind.strip():
            raise ValueError(f"capability {self.id!r} has an empty kind")
        if not self.version.strip():
            raise ValueError(f"capability {self.id!r} has an empty version")
        for dependency in self.requires:
            CapabilityRequirement(dependency)
        metadata = dict(self.metadata)
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items()):
            raise ValueError(f"capability {self.id!r} metadata must contain only strings")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


class CapabilityRegistry:
    """Mutable only during application assembly; ``compose`` returns a snapshot."""

    def __init__(self) -> None:
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, provider: CapabilityProvider) -> CapabilityRegistry:
        if provider.id in self._providers:
            old = self._providers[provider.id]
            raise CapabilityError(
                f"capability '{provider.id}' is already registered by version {old.version}"
            )
        self._providers[provider.id] = provider
        return self

    def compose(
        self, requirements: Iterable[CapabilityRequirement | str]
    ) -> CapabilitySet:
        requested = [
            r if isinstance(r, CapabilityRequirement) else CapabilityRequirement(r)
            for r in requirements
        ]
        resolved: dict[str, CapabilityProvider] = {}
        visiting: list[str] = []

        def visit(requirement: CapabilityRequirement, consumer: str = "host") -> None:
            provider = self._providers.get(requirement.id)
            if provider is None:
                raise CapabilityError(
                    f"{consumer} requires capability '{requirement.id}', but no provider is registered"
                )
            if requirement.version is not None and requirement.version != provider.version:
                raise CapabilityError(
                    f"{consumer} requires capability '{requirement.id}' version "
                    f"{requirement.version}, but version {provider.version} is registered"
                )
            if requirement.id in resolved:
                return
            if requirement.id in visiting:
                cycle = " -> ".join([*visiting[visiting.index(requirement.id):], requirement.id])
                raise CapabilityError(f"capability dependency cycle: {cycle}")
            visiting.append(requirement.id)
            for dependency in provider.requires:
                visit(CapabilityRequirement(dependency), provider.id)
            visiting.pop()
            resolved[provider.id] = provider

        for requirement in requested:
            visit(requirement)
        return CapabilitySet(resolved)


class CapabilitySet:
    """Immutable provider snapshot used by one execution composition."""

    def __init__(self, providers: Mapping[str, CapabilityProvider]) -> None:
        self._providers = MappingProxyType(dict(providers))
        manifest = [
            {
                "id": p.id,
                "kind": p.kind,
                "version": p.version,
                "requires": tuple(p.requires),
                "metadata": tuple(sorted(p.metadata.items())),
            }
            for p in sorted(self._providers.values(), key=lambda item: item.id)
        ]
        self._manifest = tuple(MappingProxyType(item) for item in manifest)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def manifest(self) -> list[dict[str, Any]]:
        """Return a JSON-safe copy suitable for ``run.json`` and diagnostics."""
        return [
            {
                **dict(item),
                "requires": list(item["requires"]),
                "metadata": dict(item["metadata"]),
            }
            for item in self._manifest
        ]

    def has(self, capability_id: str) -> bool:
        return capability_id in self._providers

    def restrict(
        self, requirements: Iterable[CapabilityRequirement | str]
    ) -> CapabilitySet:
        """Derive a smaller frozen composition from this trusted snapshot."""
        registry = CapabilityRegistry()
        for provider in self._providers.values():
            registry.register(provider)
        return registry.compose(requirements)

    def provider(self, capability_id: str, *, kind: str | None = None) -> CapabilityProvider:
        provider = self._providers.get(capability_id)
        if provider is None:
            raise CapabilityError(
                f"capability '{capability_id}' is not enabled in this execution composition"
            )
        if kind is not None and provider.kind != kind:
            raise CapabilityError(
                f"capability '{capability_id}' is a {provider.kind!r} provider, not {kind!r}"
            )
        return provider

    def select(self, *, kind: str, metadata: tuple[str, Any]) -> CapabilityProvider:
        matches = [
            provider
            for provider in self._providers.values()
            if provider.kind == kind and provider.metadata.get(metadata[0]) == metadata[1]
        ]
        if not matches:
            raise CapabilityError(
                f"no enabled {kind!r} capability handles {metadata[0]}={metadata[1]!r}"
            )
        if len(matches) > 1:
            ids = ", ".join(sorted(provider.id for provider in matches))
            raise CapabilityError(
                f"ambiguous {kind!r} capability for {metadata[0]}={metadata[1]!r}: {ids}"
            )
        return matches[0]
