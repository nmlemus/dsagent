"""Project datasets — files a project owns, reused across its runs.

A dataset is uploaded once to `<project>/data/` and seeded into each new run's
`workspace/data/` when the run is created, so the workflow's `data_path` (e.g.
`data/sales.csv`) resolves without uploading per run. Generic on purpose
(invariant 1): the harness copies files; that they are *datasets* is the
cartridge's framing.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(name: str) -> str:
    """A filename that cannot mean anything to a filesystem — last path segment,
    stripped to safe characters, never a dotfile or empty."""
    base = Path(name or "").name
    cleaned = _UNSAFE.sub("_", base).lstrip(".")
    return cleaned or "file"


def store_dataset(data_dir: Path, name: str, payload: bytes) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(name)
    (data_dir / safe).write_bytes(payload)
    return safe


def list_datasets(data_dir: Path) -> list[dict[str, Any]]:
    if not data_dir.is_dir():
        return []
    return [
        {"name": f.name, "bytes": f.stat().st_size}
        for f in sorted(data_dir.iterdir())
        if f.is_file()
    ]


def delete_dataset(data_dir: Path, name: str) -> bool:
    target = data_dir / safe_name(name)
    if target.is_file():
        target.unlink()
        return True
    return False


def seed_run(project_data_dir: Path, run_workspace: Path) -> list[str]:
    """Copy a project's datasets into a new run's `workspace/data/`. Returns the
    names copied. Existing files are overwritten (a re-seed is idempotent); a
    resumed run should not call this."""
    if not project_data_dir.is_dir():
        return []
    dest = run_workspace / "data"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for f in project_data_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            copied.append(f.name)
    return copied
