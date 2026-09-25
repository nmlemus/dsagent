"""Minimal environment for generated code; the model client keeps host secrets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def compute_environment(workspace: Path) -> dict[str, str]:
    """An allowlist, not a growing blacklist of credential variable names.

    Do not include PYTHONPATH, shell startup hooks, provider credentials,
    process-control paths, cloud profiles or tracing configuration. This only
    prevents environment inheritance; it does not restrict the host filesystem.
    """
    config = workspace.resolve() / ".dsagent" / "runtime"
    result = {
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    }
    for key, folder in {
        "IPYTHONDIR": "ipython", "JUPYTER_CONFIG_DIR": "jupyter",
        "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "cache",
    }.items():
        path = config / folder
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    return result
