"""Shared helpers for the fake agents.

Every fake satisfies a step by reading the `produces` block out of its prompt and
writing what it names. Since `produces` entries may be globs, "what it names" is
no longer a filename, and one definition of the expansion beats three.
"""

from __future__ import annotations

from dsagent.runner import is_pattern

PRODUCES_PREFIX = "- `"


def produces_of(prompt: str) -> list[str]:
    """The `produces` entries a step prompt declares, patterns included.

    The runner annotates a pattern as ``- `a/*.png` (one or more matching
    files)``, so the entry is what sits between the first pair of backticks.
    """
    out = []
    for line in prompt.splitlines():
        if line.startswith(PRODUCES_PREFIX) and "`" in line[len(PRODUCES_PREFIX):]:
            out.append(line[len(PRODUCES_PREFIX):].split("`")[0])
    return out


def paths_for(entry: str, count: int = 2) -> list[str]:
    """Concrete workspace-relative paths that satisfy one `produces` entry.

    A literal is itself. A pattern becomes `count` files that match it — the
    point of a pattern is that the step decides how many, so a fake has to
    decide too. `count=0` is how a test makes a pattern go unsatisfied.
    """
    if not is_pattern(entry):
        return [entry]
    stem, _, suffix = entry.rpartition("*")
    return [f"{stem}{i:02d}{suffix}" for i in range(1, count + 1)]


def expand(prompt: str, count: int = 2) -> list[str]:
    """Every path a fake should write to satisfy a step prompt."""
    return [p for entry in produces_of(prompt) for p in paths_for(entry, count)]
