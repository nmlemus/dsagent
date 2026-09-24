"""Local POSIX execution ownership and durable control records.

Lock files must never be replaced/unlinked while a run exists. These locks are
for a local disk on one host, not leases for a distributed filesystem.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class RunBusy(RuntimeError):
    pass


def acquire(run_dir: Path, name: str = "owner", *, blocking: bool = False) -> Any:
    control = run_dir / ".execution"
    control.mkdir(parents=True, exist_ok=True)
    handle = (control / f"{name}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError as exc:
        handle.close()
        raise RunBusy("this run has an active owner") from exc
    return handle


@contextmanager
def locked(run_dir: Path, name: str = "owner", *, blocking: bool = False):
    handle = acquire(run_dir, name, blocking=blocking)
    try:
        yield
    finally:
        handle.close()


def durable_json(path: Path, value: Any) -> None:
    """Publish only complete, fsynced records; retain no truncated journal entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
