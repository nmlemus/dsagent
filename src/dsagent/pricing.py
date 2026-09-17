"""What a run cost, from a table the repository ships and anyone can edit.

The harness does not know any prices. It reads them from `prices.yaml` — rates
per million tokens, by model — and multiplies them by the usage each step already
records. A model with no entry costs `None`, which the UI prints as "—": an
unknown price is not zero, and a number nobody can source is worse than a dash.

Rates change. The file is the place to change them, and `dsagent serve --prices`
points at a different one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PRICES_FILE = "prices.yaml"
"""Default name, looked for beside the working directory."""

PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Rate:
    """Dollars per million tokens, by how the tokens were spent.

    `cached` is a cache *read*, which is the cheap one, and `cache_write` is what
    it costs to put them there. Both are sub-counts of input, so the uncached
    input is what is left after subtracting them — see `cost`.
    """

    input: float = 0.0
    output: float = 0.0
    cached: float = 0.0
    cache_write: float = 0.0

    @classmethod
    def parse(cls, raw: Any) -> Rate | None:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                input=float(raw.get("input", 0) or 0),
                output=float(raw.get("output", 0) or 0),
                cached=float(raw.get("cached", 0) or 0),
                cache_write=float(raw.get("cache_write", 0) or 0),
            )
        except (TypeError, ValueError):
            return None


class Prices:
    """A model → `Rate` table, and the arithmetic that uses it."""

    def __init__(self, rates: dict[str, Rate] | None = None) -> None:
        self.rates = rates or {}

    @classmethod
    def load(cls, path: Path | None) -> Prices:
        """Read a `prices.yaml`. A missing or unreadable file means no prices."""
        if path is None or not path.is_file():
            return cls()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return cls()
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, dict):
            return cls()
        rates: dict[str, Rate] = {}
        for key, value in models.items():
            rate = Rate.parse(value)
            if rate is None:
                continue
            name = str(key)
            rates[name] = rate
            # Indexed under the bare name too: a persona's frontmatter may say
            # `claude-sonnet-5` where the default says `anthropic:claude-sonnet-5`,
            # and they are one model to everyone but a string comparison.
            rates.setdefault(name.split(":", 1)[-1], rate)
        return cls(rates)

    def rate_for(self, model: str) -> Rate | None:
        """The rate for a model id, tolerating the provider prefix either way.

        `anthropic:claude-sonnet-5` and `claude-sonnet-5` are the same model to a
        person, and which one reaches here depends on whether a persona overrode
        it in frontmatter.
        """
        if model in self.rates:
            return self.rates[model]
        bare = model.split(":", 1)[-1]
        return self.rates.get(bare)

    def cost(self, model: str, usage: dict[str, int]) -> float | None:
        """What one step's usage cost, or None when the model has no rate.

        `input_tokens` is the whole input, with `cache_read` and `cache_creation`
        as sub-counts of it — that is what the provider reports and what
        `docs/runs/eda-to-report-003.md` adds up. So the uncached input is the
        remainder, and a negative remainder (a provider counting differently)
        floors at zero rather than paying us back.
        """
        rate = self.rate_for(model)
        if rate is None or not usage:
            return None
        cached = int(usage.get("cache_read", 0) or 0)
        written = int(usage.get("cache_creation", 0) or 0)
        uncached = max(0, int(usage.get("input_tokens", 0) or 0) - cached - written)
        output = int(usage.get("output_tokens", 0) or 0)
        total = (
            uncached * rate.input
            + cached * rate.cached
            + written * rate.cache_write
            + output * rate.output
        ) / PER_MILLION
        return round(total, 6)


def find_prices(start: Path | None = None) -> Path | None:
    """`prices.yaml` in the working directory, or the repository above it."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / PRICES_FILE
        if candidate.is_file():
            return candidate
    return None
