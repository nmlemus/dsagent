"""Python requirement strings, handled without knowing what any of them are.

Invariant 1 says the harness is domain-agnostic, so a requirement here is an
opaque pip requirement string a cartridge declares on one of its envs. This
module only ever *names* it, checks whether it is installed, and says how to
install it — it never maps, special-cases or imports any particular package.

The check is distribution metadata rather than `find_spec`: what a cartridge
declares is what pip installs, and the import name is frequently not the
distribution name (`scikit-learn` imports as `sklearn`). Resolving that would
mean the harness carrying a table of package knowledge, which is exactly what
invariant 1 forbids. The trade-off is that version specifiers are not enforced
here — pip enforces them at install time.
"""

from __future__ import annotations

import importlib.metadata
import re
import shlex
import sys

_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_AFTER_NAME = re.compile(r"[\[<>=!~;@\s]")


def distribution_name(requirement: str) -> str:
    """`pandas>=2` -> `pandas`. Raises ValueError if it is not a requirement."""
    name = _AFTER_NAME.split(requirement.strip(), 1)[0]
    if not _NAME.match(name):
        raise ValueError(f"not a pip requirement: {requirement!r}")
    return name


def is_installed(requirement: str) -> bool:
    try:
        importlib.metadata.distribution(distribution_name(requirement))
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def missing(requirements: list[str]) -> list[str]:
    """The declared requirements this interpreter cannot satisfy, in order."""
    return [r for r in requirements if not is_installed(r)]


def pip_install_command(requirements: list[str]) -> str:
    """The exact command that installs `requirements` into this interpreter."""
    argv = [sys.executable, "-m", "pip", "install", *requirements]
    return " ".join(shlex.quote(a) for a in argv)
