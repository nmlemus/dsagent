"""Versioned capability composition for the DSAgent host.

Cartridges consume capabilities; providers implement them.  The resulting
``CapabilitySet`` is resolved once and kept unchanged for the lifetime of a
run or chat agent.
"""

from dsagent.capabilities.builtin import builtin_registry, compose_cartridges
from dsagent.capabilities.core import (
    CapabilityError,
    CapabilityProvider,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilitySet,
)

__all__ = [
    "CapabilityError",
    "CapabilityProvider",
    "CapabilityRegistry",
    "CapabilityRequirement",
    "CapabilitySet",
    "builtin_registry",
    "compose_cartridges",
]
