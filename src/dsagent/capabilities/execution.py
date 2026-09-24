"""Detect changes to executable cartridge configuration and source on resume.

This records hashes, not an executable archive. A mismatch requires restoring
the recorded version or starting a new run. Provider code and installed packages
remain deployment responsibilities; this is not a hermetic environment lock.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dsagent.capabilities.core import CapabilityError, CapabilitySet
from dsagent.cartridge.models import Cartridge
from dsagent.models import default_model


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execution_manifest(cartridge: Cartridge, workflow: str, capabilities: CapabilitySet) -> dict:
    """Hash effective definitions and declared source trees without absolute root paths."""
    root = cartridge.root.resolve()

    def normalize(value):
        if isinstance(value, Path):
            path = value.resolve()
            return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    definition = cartridge.model_dump(exclude={"root", "experience"})
    definition["default_model"] = default_model()
    files: dict[str, str] = {}

    def record(label: str, path: Path) -> None:
        files[label] = hashlib.sha256(path.read_bytes()).hexdigest()

    def tree(label: str, directory: Path) -> None:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if any(part in {"__pycache__", ".git"} for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
                continue
            if path.is_symlink() and path.is_dir():
                raise CapabilityError(f"execution snapshot cannot follow directory symlink: {path}")
            if path.is_file():
                record(f"{label}/{relative.as_posix()}", path)

    try:
        for name, persona in cartridge.personas.items():
            record(f"personas/{name}", persona.path)
        for name, skill in cartridge.skills.items():
            tree(f"skills/{name}", skill.path)
        for name, declared in cartridge.workflows.items():
            tree(f"workflows/{name}", declared.path)
        for name, env in cartridge.envs.items():
            if env.build is not None:
                tree(f"build/{name}", env.build)
        # Detect source edits even when a long-lived server retains loaded models.
        if (root / "cartridge.yaml").is_file():
            record("cartridge.yaml", root / "cartridge.yaml")
    except OSError as error:
        raise CapabilityError(f"cannot read execution snapshot: {error}") from error
    manifest = {
        "schema": 1,
        "cartridge": cartridge.name,
        "workflow": workflow,
        "definition": _hash(normalize(definition)),
        "providers": capabilities.fingerprint,
        "files": files,
    }
    return {**manifest, "fingerprint": _hash(manifest)}
